"""Tests for the SQLite event store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

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


def test_init_creates_schema(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()

    # Connect independently and verify tables + indexes exist.
    conn = sqlite3.connect(tmp_path / "events.db")
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        )
    }
    conn.close()

    assert "events" in names
    assert "idx_events_captured" in names
    assert "idx_events_station" in names
    assert "idx_events_species" in names


def test_init_is_idempotent(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()
    # Second call must not raise.
    store.init()


def test_record_image_inserts_row(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()

    event = _make_image_event()
    event_id = store.record_image(event, media_key="horus/image/2026/04/17/abc.jpg")

    assert event_id  # UUID generated
    with sqlite3.connect(tmp_path / "events.db") as conn:
        row = conn.execute(
            "SELECT id, station, event_type, media_key, payload_json FROM events"
        ).fetchone()

    assert row[0] == event_id
    assert row[1] == "horus"
    assert row[2] == "image"
    assert row[3] == "horus/image/2026/04/17/abc.jpg"

    payload = json.loads(row[4])
    assert payload["camera"] == "imx519"
    assert payload["resolution"] == [2328, 1748]
    assert payload["sha256"] == event.sha256


def test_record_image_accepts_explicit_id(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()

    event = _make_image_event()
    returned = store.record_image(event, media_key="k", event_id="deadbeef")

    assert returned == "deadbeef"


def test_record_image_before_init_autocreates_parent(tmp_path: Path) -> None:
    """Lazy init + missing parent dir still works."""
    nested = tmp_path / "deep" / "nested" / "events.db"
    store = EventStore(nested)
    store.init()

    store.record_image(_make_image_event(), media_key="k")
    assert nested.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
