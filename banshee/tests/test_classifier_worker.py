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
from banshee.classifier.worker import ClassifierWorker
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
    """Minimal stand-in for :class:`MinioStore` — serves bytes from a dict.

    Returns a real generator (not ``iter([...])``) so the worker's
    ``contextlib.closing(stream)`` has a ``.close()`` to call, mirroring
    production where MinIO's response object is generator-shaped and
    needs explicit release.
    """

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs
        self.fetched: list[str] = []
        self.closed_streams: int = 0

    def get_object_stream(self, key: str) -> tuple[Iterator[bytes], int | None]:
        self.fetched.append(key)
        blob = self._blobs[key]

        def _gen() -> Iterator[bytes]:
            try:
                yield blob
            finally:
                self.closed_streams += 1

        return _gen(), len(blob)


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

    worker = ClassifierWorker(store, minio, classifier, batch_size=5)
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
    # Should not raise, and — importantly — must return 0 so the outer
    # ``run_forever`` loop backs off instead of hot-looping during a
    # MinIO outage.
    assert worker.tick() == 0

    # Still pending.
    assert len(store.fetch_pending_classification()) == 1


def test_tick_returns_zero_when_whole_batch_fails(tmp_path: Path) -> None:
    """Regression test for the hot-loop fix: if every row in the batch
    fails its MinIO fetch, ``tick()`` must return 0 (not ``len(pending)``)
    so the caller sleeps on ``poll_interval_seconds`` instead of
    hammering the DB + MinIO during an outage."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    for i in range(3):
        store.record_image(_make_image_event(), media_key=f"missing-{i}.jpg")

    # Empty MinIO — every fetch raises KeyError.
    worker = ClassifierWorker(store, _FakeMinio({}), _RecordingClassifier())
    assert worker.tick() == 0  # would be 3 under the old len(pending) semantic
    assert len(store.fetch_pending_classification()) == 3  # all still pending


def test_fetch_image_closes_stream_on_success(tmp_path: Path) -> None:
    """``contextlib.closing`` guarantees the MinIO response is released
    after the read — not left dangling on GC."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    store.record_image(_make_image_event(), media_key="horus/image/a.jpg")
    minio = _FakeMinio({"horus/image/a.jpg": b"IMG"})

    worker = ClassifierWorker(store, minio, _RecordingClassifier())
    worker.tick()

    assert minio.closed_streams == 1


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
        poll_interval_seconds=60.0,  # would block forever
    )
    worker.stop()
    worker.run_forever()  # should return immediately, not hang
