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


# ---- burst metadata + hero_score (PR-B) ------------------------------------


def _burst_event(
    station: str = "horus",
    *,
    burst_id: str | None = "horus-1776898133321-de38",
    burst_seq: int | None = 1,
    bird_score: float | None = 0.8,
    bbox_fraction: tuple[float, float, float, float] | None = (0.1, 0.1, 0.4, 0.4),
) -> ImageEvent:
    body = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    return ImageEvent(
        schema_version=1,
        station=station,
        captured_at=datetime(2026, 4, 18, 21, 9, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(2304, 1296),
        content_type="image/jpeg",
        image_bytes=body,
        size_bytes=len(body),
        changed_fraction=0.042,
        sha256=hashlib.sha256(body).hexdigest(),
        bbox_fraction=bbox_fraction,
        bird_score=bird_score,
        burst_id=burst_id,
        burst_seq=burst_seq,
    )


def test_migration_adds_burst_and_hero_columns(tmp_path: Path) -> None:
    """init() must run the column migration so burst_id/burst_seq/
    sharpness/hero_score land on the events table. Without this the
    record_image insert would throw ``no such column``."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    with sqlite3.connect(tmp_path / "events.db") as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    assert {"burst_id", "burst_seq", "sharpness", "hero_score"}.issubset(cols)


def test_migration_is_idempotent_on_upgraded_schema(tmp_path: Path) -> None:
    """A second init() on an already-migrated DB must be a no-op."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    # Second call — issues ALTER TABLE only for missing columns; all are
    # already present, so no statements execute and nothing raises.
    store.init()


def test_migration_from_legacy_schema_adds_columns(tmp_path: Path) -> None:
    """Simulate a Thoth DB that predates PR-B: legacy SCHEMA applied,
    then init() upgrades it without losing data."""
    from banshee.eventstore import SCHEMA

    db_path = tmp_path / "events.db"
    # Build the pre-migration shape and insert one row the old way.
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO events (
              id, station, event_type, captured_at, received_at,
              schema_version, payload_json, media_key
            ) VALUES ('legacy-1', 'horus', 'image', ?, ?, 1, '{}', 'k')
            """,
            (
                datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc).isoformat(),
                datetime(2026, 4, 17, 22, 0, 1, tzinfo=timezone.utc).isoformat(),
            ),
        )

    # Run the full init — migration must add the new columns and keep
    # the legacy row intact with NULL burst metadata.
    store = EventStore(db_path)
    store.init()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, burst_id, burst_seq, sharpness, hero_score "
            "FROM events WHERE id = 'legacy-1'"
        ).fetchone()
    assert row == ("legacy-1", None, None, None, None)


def test_record_image_persists_burst_metadata(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()
    event_id = store.record_image(
        _burst_event(burst_id="horus-abc-1", burst_seq=3),
        media_key="k",
    )
    with sqlite3.connect(tmp_path / "events.db") as conn:
        row = conn.execute(
            "SELECT burst_id, burst_seq FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
    assert row == ("horus-abc-1", 3)


def test_record_image_persists_sharpness_and_hero_score(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    store.init()
    event_id = store.record_image(
        _burst_event(bird_score=0.8, bbox_fraction=(0.0, 0.0, 0.5, 0.5)),
        media_key="k",
        sharpness=500.0,  # → normalized 0.5 on sharpness axis
    )
    with sqlite3.connect(tmp_path / "events.db") as conn:
        row = conn.execute(
            "SELECT sharpness, hero_score FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
    sharpness, hero = row
    assert sharpness == pytest.approx(500.0)
    # 0.4*0.8 + 0.3*0.5 + 0.2*0.25 + 0.1*0 = 0.32 + 0.15 + 0.05 = 0.52
    assert hero == pytest.approx(0.52)


def test_record_image_omits_sharpness_leaves_columns_null(tmp_path: Path) -> None:
    """A caller that can't compute sharpness (e.g. a future non-image
    event path) must be able to insert without breaking — sharpness and
    hero_score stay NULL / 0 as appropriate."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    event_id = store.record_image(
        _burst_event(bird_score=None, bbox_fraction=None),
        media_key="k",
        # sharpness intentionally omitted
    )
    with sqlite3.connect(tmp_path / "events.db") as conn:
        row = conn.execute(
            "SELECT sharpness, hero_score FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
    assert row[0] is None
    # hero_score is still computed (from zeros) — equals 0.0.
    assert row[1] == pytest.approx(0.0)


def test_record_classification_recomputes_hero_score(tmp_path: Path) -> None:
    """The classifier term (0.1*confidence) must fold into hero_score
    when record_classification runs — otherwise post-classify reranking
    is impossible and the Tier-1 composite is incomplete."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    event_id = store.record_image(
        _burst_event(bird_score=0.8, bbox_fraction=(0.0, 0.0, 0.5, 0.5)),
        media_key="k",
        sharpness=500.0,
    )

    store.record_classification(event_id, species="steller's jay", confidence=0.9)

    with sqlite3.connect(tmp_path / "events.db") as conn:
        hero = conn.execute(
            "SELECT hero_score FROM events WHERE id = ?", (event_id,)
        ).fetchone()[0]
    # Base (0.52) + 0.1*0.9 = 0.61
    assert hero == pytest.approx(0.61)


def test_record_classification_still_raises_on_missing_id(tmp_path: Path) -> None:
    """Regression: the hero-score recompute path must still raise
    LookupError on an unknown id rather than silently succeeding on the
    SELECT."""
    store = EventStore(tmp_path / "events.db")
    store.init()
    with pytest.raises(LookupError):
        store.record_classification("missing", species="x", confidence=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
