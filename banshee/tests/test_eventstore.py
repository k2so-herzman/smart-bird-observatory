"""Tests for the SQLite event store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from banshee.events import ImageEvent
from banshee.eventstore import EventStore, PendingClassification  # noqa: F401  # re-exported for downstream tests


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


def test_record_image_omits_optional_fields_when_absent(tmp_path: Path) -> None:
    """Older horus builds (pre-observability PR) publish events without
    bbox_fraction/af.  We must NOT insert null keys for those — the
    payload JSON should stay clean."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    event = _make_image_event()  # bbox_fraction + af both None by default
    store.record_image(event, media_key="k")
    with sqlite3.connect(tmp_path / "events.db") as conn:
        payload = json.loads(
            conn.execute("SELECT payload_json FROM events").fetchone()[0]
        )
    assert "bbox_fraction" not in payload
    assert "af" not in payload


def test_record_image_preserves_bbox_fraction_when_present(tmp_path: Path) -> None:
    """New observability fields must round-trip through the store so the
    API can surface them and downstream tooling can filter on them."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    body = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    event = ImageEvent(
        schema_version=1,
        station="horus",
        captured_at=datetime(2026, 4, 18, 21, 9, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(896, 504),
        content_type="image/jpeg",
        image_bytes=body,
        size_bytes=len(body),
        changed_fraction=0.024,
        sha256=hashlib.sha256(body).hexdigest(),
        bbox_fraction=(0.1, 0.2, 0.5, 0.8),
        af={"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234},
    )
    store.record_image(event, media_key="k")
    with sqlite3.connect(tmp_path / "events.db") as conn:
        payload = json.loads(
            conn.execute("SELECT payload_json FROM events").fetchone()[0]
        )
    assert payload["bbox_fraction"] == [0.1, 0.2, 0.5, 0.8]
    assert payload["af"] == {"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234}


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


def test_fetch_pending_classification_returns_unclassified_images(
    tmp_path: Path,
) -> None:
    """Only image rows with a media_key and NULL classified_at come back."""
    store = EventStore(tmp_path / "events.db")
    store.init()

    # Two unclassified images — should both come back.
    older = store.record_image(
        _make_image_event("horus"), media_key="horus/image/2026/04/17/a.jpg"
    )
    newer = store.record_image(
        _make_image_event("horus"), media_key="horus/image/2026/04/17/b.jpg"
    )

    # Mark `older` as classified — should drop out of the pending set.
    store.record_classification(older, species="american goldfinch", confidence=0.91)

    pending = store.fetch_pending_classification(limit=10)
    ids = {row.event_id for row in pending}
    assert newer in ids
    assert older not in ids
    # Populated fields round-trip.
    only = next(row for row in pending if row.event_id == newer)
    assert only.station == "horus"
    assert only.media_key == "horus/image/2026/04/17/b.jpg"


def test_fetch_pending_classification_respects_limit(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()

    for i in range(5):
        store.record_image(
            _make_image_event("horus"), media_key=f"horus/image/2026/04/17/{i}.jpg"
        )

    assert len(store.fetch_pending_classification(limit=2)) == 2


def test_fetch_pending_skips_rows_without_media_key(tmp_path: Path) -> None:
    """A row without a media_key (shouldn't happen in prod, but defend)
    must never enter the pending set — the worker can't fetch bytes."""
    store = EventStore(tmp_path / "events.db")
    store.init()

    # Direct insert to bypass record_image's media_key requirement.
    with sqlite3.connect(tmp_path / "events.db") as conn:
        conn.execute(
            """
            INSERT INTO events (
              id, station, event_type, captured_at, received_at,
              schema_version, payload_json, media_key
            ) VALUES ('bare', 'horus', 'image', ?, ?, 1, '{}', NULL)
            """,
            (
                datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc).isoformat(),
                datetime(2026, 4, 17, 22, 0, 1, tzinfo=timezone.utc).isoformat(),
            ),
        )

    pending = store.fetch_pending_classification()
    assert pending == []


def test_record_classification_sets_all_three_columns(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()

    event_id = store.record_image(_make_image_event(), media_key="k")
    store.record_classification(event_id, species="dark-eyed junco", confidence=0.83)

    with sqlite3.connect(tmp_path / "events.db") as conn:
        row = conn.execute(
            "SELECT species, confidence, classified_at FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
    assert row[0] == "dark-eyed junco"
    assert row[1] == pytest.approx(0.83)
    assert row[2] is not None  # timestamp set


def test_record_classification_rejects_empty_species(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()
    event_id = store.record_image(_make_image_event(), media_key="k")
    with pytest.raises(ValueError):
        store.record_classification(event_id, species="", confidence=0.5)


def test_record_classification_raises_on_unknown_event_id(tmp_path: Path) -> None:
    """A mistyped / stale event_id must not silently succeed — that would
    mask a caller bug (e.g. worker handing us the wrong id)."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    with pytest.raises(LookupError):
        store.record_classification(
            "nonexistent-uuid", species="fallback", confidence=0.1
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
