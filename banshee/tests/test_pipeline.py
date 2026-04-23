"""Tests for the ingest pipeline's error-handling contract.

The pipeline has three storage sinks (MinIO, SQLite, InfluxDB) and
three failure modes that each needed a standing decision:

- MinIO upload fails  → drop the event (no orphan possible).
- SQLite insert fails → upload succeeded, so the blob is an orphan;
                        delete it and drop the event.
- Influx write fails  → event is already indexed in SQLite, so we
                        log and continue. Metrics are recoverable.

These tests pin that behavior using constructor injection — the real
``Pipeline.__init__`` runs with fakes passed in for every sink. No
monkey-patching, no ``_FakePipeline`` stand-in.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from banshee.config import (
    BansheeConfig,
    InfluxConfig,
    MqttConfig,
    NotifyConfig,
    ThothStorageConfig,
)
from banshee.events import ImageEvent, StatusEvent
from banshee.main import Pipeline
from banshee.minio_store import MinioConfig


def _make_image_event() -> ImageEvent:
    body = b"\xff\xd8\xff\xe0fake"
    return ImageEvent(
        schema_version=1,
        station="horus",
        captured_at=datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(2328, 1748),
        content_type="image/jpeg",
        image_bytes=body,
        size_bytes=len(body),
        changed_fraction=0.04,
        sha256=hashlib.sha256(body).hexdigest(),
    )


def _make_status_event() -> StatusEvent:
    return StatusEvent(
        schema_version=1,
        station="horus",
        ts=datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        raw={"camera_ok": True},
    )


def _make_cfg(tmp_path: Path) -> BansheeConfig:
    return BansheeConfig(
        mqtt=MqttConfig(host="unused"),
        storage=ThothStorageConfig(
            db_path=tmp_path / "events.db",
            minio=MinioConfig(
                endpoint="unused:9000",
                access_key="k",
                secret_key="s",
            ),
        ),
        influx=InfluxConfig(token=""),  # disabled
        notify=NotifyConfig(),
    )


class _FakeMinio:
    """Exposes the same methods ``MinioStore`` does, minus the real client."""

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.removed: list[str] = []
        self.put_raises: Exception | None = None

    def put_image(self, event: ImageEvent, event_id: str) -> str:
        if self.put_raises is not None:
            raise self.put_raises
        key = f"{event.station}/image/2026/04/17/{event_id}.jpg"
        self.uploaded.append(key)
        return key

    def remove_object(self, key: str) -> None:
        self.removed.append(key)


class _FakeEventstore:
    """Matches ``EventStore.record_image`` — no DB, no disk."""

    def __init__(self) -> None:
        # Tracks (event_id, media_key, sharpness) tuples so tests can
        # assert the pipeline propagated the computed sharpness into the
        # eventstore insert.
        self.recorded: list[tuple[str, str, float | None]] = []
        self.record_raises: Exception | None = None

    def record_image(
        self,
        event: ImageEvent,
        media_key: str,
        event_id: str | None = None,
        *,
        sharpness: float | None = None,
    ) -> str:
        if self.record_raises is not None:
            raise self.record_raises
        self.recorded.append((event_id or "", media_key, sharpness))
        return event_id or "generated"


class _FakeInflux:
    """Matches ``InfluxWriter`` — captures calls, can be armed to raise."""

    def __init__(self) -> None:
        self.image_writes: list[tuple[str, str]] = []
        self.status_writes: list[str] = []
        self.image_raises: Exception | None = None
        self.status_raises: Exception | None = None

    def write_image_event(
        self, event: ImageEvent, event_id: str, media_key: str
    ) -> None:
        if self.image_raises is not None:
            raise self.image_raises
        self.image_writes.append((event_id, media_key))

    def write_status(self, event: StatusEvent) -> None:
        if self.status_raises is not None:
            raise self.status_raises
        self.status_writes.append(event.station)


class _FakeSubscriber:
    """Stand-in so ``Pipeline.__init__`` doesn't try to create a paho client."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def run_forever(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture
def fakes():
    return {
        "minio": _FakeMinio(),
        "eventstore": _FakeEventstore(),
        "influx": _FakeInflux(),
    }


@pytest.fixture
def pipeline(tmp_path: Path, fakes) -> Pipeline:
    """A real Pipeline with every sink injected.

    This exercises ``Pipeline.__init__`` end-to-end (the whole point of
    PR #7 / issue #8). If injection regresses, this fixture fails to
    construct and every test below reports it.
    """
    cfg = _make_cfg(tmp_path)
    return Pipeline(
        cfg,
        eventstore=fakes["eventstore"],  # type: ignore[arg-type]
        minio=fakes["minio"],  # type: ignore[arg-type]
        influx=fakes["influx"],  # type: ignore[arg-type]
        subscriber=_FakeSubscriber(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Constructor-injection guarantees
# ---------------------------------------------------------------------------


def test_pipeline_accepts_all_injected_sinks(pipeline: Pipeline, fakes) -> None:
    """Every sink attribute is the fake we injected — no silent real-client
    construction."""
    assert pipeline.minio is fakes["minio"]
    assert pipeline.eventstore is fakes["eventstore"]
    assert pipeline.influx is fakes["influx"]


# ---------------------------------------------------------------------------
# Error-handling contract (PR #6 decisions, now re-pinned against the real
# Pipeline surface instead of a _FakePipeline stand-in).
# ---------------------------------------------------------------------------


def test_happy_path_writes_all_three_stores(pipeline: Pipeline, fakes) -> None:
    pipeline._handle_image(_make_image_event())
    assert len(fakes["minio"].uploaded) == 1
    assert len(fakes["eventstore"].recorded) == 1
    assert len(fakes["influx"].image_writes) == 1
    assert fakes["minio"].removed == []


def test_minio_failure_drops_event(pipeline: Pipeline, fakes) -> None:
    fakes["minio"].put_raises = RuntimeError("minio down")
    pipeline._handle_image(_make_image_event())
    # Nothing was recorded anywhere — no orphan, no phantom row.
    assert fakes["eventstore"].recorded == []
    assert fakes["influx"].image_writes == []
    assert fakes["minio"].removed == []


def test_eventstore_failure_removes_orphan_blob(pipeline: Pipeline, fakes) -> None:
    fakes["eventstore"].record_raises = RuntimeError("sqlite locked")
    pipeline._handle_image(_make_image_event())
    # MinIO received the upload, SQLite rejected the insert, so the
    # pipeline must clean up the orphan and NOT write metrics.
    assert len(fakes["minio"].uploaded) == 1
    assert fakes["minio"].removed == fakes["minio"].uploaded
    assert fakes["influx"].image_writes == []


def test_influx_failure_keeps_indexed_event(pipeline: Pipeline, fakes) -> None:
    """Influx is best-effort — event stays in SQLite, blob stays in MinIO."""
    fakes["influx"].image_raises = RuntimeError("influx down")
    pipeline._handle_image(_make_image_event())
    assert len(fakes["minio"].uploaded) == 1
    assert len(fakes["eventstore"].recorded) == 1
    # Influx raised, so no successful write was recorded.
    assert fakes["influx"].image_writes == []
    # And crucially, we did NOT remove the blob.
    assert fakes["minio"].removed == []


def test_status_influx_failure_does_not_raise(pipeline: Pipeline, fakes) -> None:
    fakes["influx"].status_raises = RuntimeError("influx down")
    # Must not bubble — MQTT loop would crash otherwise.
    pipeline._handle_status(_make_status_event())
