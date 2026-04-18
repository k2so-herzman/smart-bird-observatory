"""Tests for :class:`banshee.classifier.worker.ClassifierWorker`.

Exercises the pipeline (fetch → classify → record) end-to-end with an
in-memory event store, a fake MinIO, and a stub classifier. The real
TFLite path is out of scope; see ``test_classifier_model`` for the
model surface.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pytest

from banshee.classifier.model import ClassificationResult, DummyClassifier
from banshee.classifier.worker import ClassifierWorker, WorkerConfig
from banshee.events import ImageEvent
from banshee.eventstore import EventStore


def _make_image_event(station: str = "horus") -> ImageEvent:
    body = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    return ImageEvent(
        schema_version=1,
        station=station,
        captured_at=datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(2328, 1748),
        content_type="image/jpeg",
        image_bytes=body,
        size_bytes=len(body),
        changed_fraction=0.042,
        sha256=hashlib.sha256(body).hexdigest(),
    )


class _FakeMinio:
    """Minimal stand-in for :class:`MinioStore` — serves bytes from a dict."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs
        self.fetched: list[str] = []

    def get_object_stream(self, key: str) -> tuple[Iterator[bytes], int | None]:
        self.fetched.append(key)
        blob = self._blobs[key]
        return iter([blob]), len(blob)


class _RecordingClassifier:
    """Classifier that returns a canned result and records what it saw."""

    def __init__(
        self,
        species: str = "american goldfinch",
        confidence: float = 0.87,
    ) -> None:
        self._result = ClassificationResult(species=species, confidence=confidence)
        self.seen_bytes: list[bytes] = []

    def classify(self, image_bytes: bytes) -> ClassificationResult:
        self.seen_bytes.append(image_bytes)
        return self._result


def test_tick_classifies_pending_row_and_marks_it_done(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()
    store.record_image(_make_image_event(), media_key="horus/image/a.jpg")
    minio = _FakeMinio({"horus/image/a.jpg": b"IMG"})
    classifier = _RecordingClassifier()

    worker = ClassifierWorker(store, minio, classifier, cfg=WorkerConfig(batch_size=5))
    processed = worker.tick()

    assert processed == 1
    assert minio.fetched == ["horus/image/a.jpg"]
    assert classifier.seen_bytes == [b"IMG"]

    # The row is now out of the pending set.
    assert store.fetch_pending_classification() == []


def test_tick_returns_zero_when_no_pending(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()
    worker = ClassifierWorker(store, _FakeMinio({}), DummyClassifier())
    assert worker.tick() == 0


def test_minio_fetch_failure_leaves_row_pending(tmp_path: Path) -> None:
    """A bad media_key should not mark the row classified — so the
    worker retries after operator intervention (e.g. backfilling MinIO)."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    store.record_image(_make_image_event(), media_key="missing.jpg")

    # Fake MinIO has no blobs → KeyError on fetch.
    worker = ClassifierWorker(store, _FakeMinio({}), _RecordingClassifier())
    # Should not raise.
    worker.tick()

    # Still pending.
    assert len(store.fetch_pending_classification()) == 1


def test_classifier_exception_leaves_row_pending(tmp_path: Path) -> None:
    """A crash inside classify() must not be silently marked done."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    store.record_image(_make_image_event(), media_key="horus/image/a.jpg")
    minio = _FakeMinio({"horus/image/a.jpg": b"IMG"})

    class _Boom:
        def classify(self, _: bytes) -> ClassificationResult:
            raise RuntimeError("model exploded")

    worker = ClassifierWorker(store, minio, _Boom())
    worker.tick()
    assert len(store.fetch_pending_classification()) == 1


def test_dummy_classifier_marks_row_classified(tmp_path: Path) -> None:
    """The sentinel label flows through: row is marked done, species
    stored as 'unclassified' so downstream can filter on it."""
    import sqlite3

    store = EventStore(tmp_path / "events.db")
    store.init()
    event_id = store.record_image(_make_image_event(), media_key="horus/image/a.jpg")
    minio = _FakeMinio({"horus/image/a.jpg": b"IMG"})

    worker = ClassifierWorker(store, minio, DummyClassifier())
    worker.tick()

    with sqlite3.connect(tmp_path / "events.db") as conn:
        row = conn.execute(
            "SELECT species, confidence, classified_at FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
    assert row[0] == "unclassified"
    assert row[1] == pytest.approx(0.0)
    assert row[2] is not None


def test_stop_exits_run_forever(tmp_path: Path) -> None:
    """Setting the stop flag before run_forever starts returns immediately."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    worker = ClassifierWorker(
        store,
        _FakeMinio({}),
        DummyClassifier(),
        cfg=WorkerConfig(poll_interval_seconds=60.0),  # would block forever
    )
    worker.stop()
    worker.run_forever()  # should return immediately, not hang
