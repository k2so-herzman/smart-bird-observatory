"""Tests for the ingest pipeline's error-handling contract.

The pipeline has three storage sinks (MinIO, SQLite, InfluxDB) and
three failure modes that each needed a standing decision:

- MinIO upload fails  → drop the event (no orphan possible).
- SQLite insert fails → upload succeeded, so the blob is an orphan;
                        delete it and drop the event.
- Influx write fails  → event is already indexed in SQLite, so we
                        log and continue. Metrics are recoverable.

These tests pin that behavior — the real concern on PR #6.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from banshee.events import ImageEvent, StatusEvent
from banshee.main import Pipeline


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


class _FakeMinio:
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
    def __init__(self) -> None:
        self.recorded: list[tuple[str, str]] = []
        self.record_raises: Exception | None = None

    def record_image(
        self, event: ImageEvent, media_key: str, event_id: str | None = None
    ) -> str:
        if self.record_raises is not None:
            raise self.record_raises
        self.recorded.append((event_id or "", media_key))
        return event_id or "generated"


class _FakeInflux:
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


@dataclass
class _FakePipeline:
    """Pipeline with storage sinks swapped for fakes.

    We reach into the Pipeline methods directly rather than constructing
    a real one, because Pipeline.__init__ wires real MQTT/MinIO clients.
    """

    minio: _FakeMinio
    eventstore: _FakeEventstore
    influx: _FakeInflux

    def handle_image(self, event: ImageEvent) -> None:
        Pipeline._handle_image(self, event)  # type: ignore[arg-type]

    def handle_status(self, event: StatusEvent) -> None:
        Pipeline._handle_status(self, event)  # type: ignore[arg-type]


@pytest.fixture
def pipeline() -> _FakePipeline:
    return _FakePipeline(
        minio=_FakeMinio(),
        eventstore=_FakeEventstore(),
        influx=_FakeInflux(),
    )


def test_happy_path_writes_all_three_stores(pipeline: _FakePipeline) -> None:
    pipeline.handle_image(_make_image_event())
    assert len(pipeline.minio.uploaded) == 1
    assert len(pipeline.eventstore.recorded) == 1
    assert len(pipeline.influx.image_writes) == 1
    assert pipeline.minio.removed == []


def test_minio_failure_drops_event(pipeline: _FakePipeline) -> None:
    pipeline.minio.put_raises = RuntimeError("minio down")

    pipeline.handle_image(_make_image_event())

    # Nothing was recorded anywhere — no orphan, no phantom row.
    assert pipeline.eventstore.recorded == []
    assert pipeline.influx.image_writes == []
    assert pipeline.minio.removed == []


def test_eventstore_failure_removes_orphan_blob(pipeline: _FakePipeline) -> None:
    pipeline.eventstore.record_raises = RuntimeError("sqlite locked")

    pipeline.handle_image(_make_image_event())

    # MinIO received the upload, SQLite rejected the insert, so the
    # pipeline must clean up the orphan and NOT write metrics.
    assert len(pipeline.minio.uploaded) == 1
    assert pipeline.minio.removed == pipeline.minio.uploaded
    assert pipeline.influx.image_writes == []


def test_influx_failure_keeps_indexed_event(pipeline: _FakePipeline) -> None:
    """Influx is best-effort — event stays in SQLite, blob stays in MinIO."""
    pipeline.influx.image_raises = RuntimeError("influx down")

    pipeline.handle_image(_make_image_event())

    assert len(pipeline.minio.uploaded) == 1
    assert len(pipeline.eventstore.recorded) == 1
    # Influx raised, so no successful write was recorded.
    assert pipeline.influx.image_writes == []
    # And crucially, we did NOT remove the blob.
    assert pipeline.minio.removed == []


def test_status_influx_failure_does_not_raise(pipeline: _FakePipeline) -> None:
    pipeline.influx.status_raises = RuntimeError("influx down")

    # Must not bubble — MQTT loop would crash otherwise.
    pipeline.handle_status(_make_status_event())
