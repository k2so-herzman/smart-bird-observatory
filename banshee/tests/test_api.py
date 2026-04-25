"""Tests for the thoth-api read endpoints.

Uses FastAPI's TestClient against an in-memory SQLite + a fake MinIO
store so the suite never touches the network or the real DB path.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from banshee.api import create_app
from banshee.config import (
    BansheeConfig,
    InfluxConfig,
    MqttConfig,
    NotifyConfig,
    ThothStorageConfig,
)
from banshee.eventstore import SCHEMA, _migrate
from banshee.minio_store import MinioConfig


# ---- fixtures --------------------------------------------------------------


def _make_cfg() -> BansheeConfig:
    """Minimal config for the test app — values don't need to reach a real host."""
    return BansheeConfig(
        mqtt=MqttConfig(host="test"),
        storage=ThothStorageConfig(
            db_path=Path("/unused/in/tests.db"),
            minio=MinioConfig(
                endpoint="http://test:9000",
                access_key="ak",
                secret_key="sk",
                bucket="thoth",
            ),
        ),
        influx=InfluxConfig(),
        notify=NotifyConfig(),
    )


class _FakeMinio:
    """Dead-simple MinioStore stand-in that serves from a dict."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def get_object_stream(self, key: str) -> tuple[Iterator[bytes], int]:
        if key not in self._objects:
            raise KeyError(key)
        data = self._objects[key]

        def _iter() -> Iterator[bytes]:
            yield data

        return _iter(), len(data)


@pytest.fixture
def seeded_db() -> sqlite3.Connection:
    """In-memory SQLite with a few canned events spanning two stations."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Mirror production EventStore.init() so the columns added in
    # later migrations (burst_id, burst_seq, sharpness, hero_score) are
    # present for the API queries. Without this the grouped-events path
    # trips ``no such column: burst_id`` on the in-memory fixture.
    _migrate(conn)

    rows = [
        (
            "evt-1",
            "horus",
            "image",
            "2026-04-18T12:00:00+00:00",
            "2026-04-18T12:00:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg", "size_bytes": 100}),
            "horus/image/2026/04/18/evt-1.jpg",
            None,
            None,
            None,
            None,
        ),
        (
            "evt-2",
            "horus",
            "image",
            "2026-04-18T12:05:00+00:00",
            "2026-04-18T12:05:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg", "size_bytes": 200}),
            "horus/image/2026/04/18/evt-2.jpg",
            None,
            "Cyanocitta stelleri",
            0.92,
            "2026-04-18T12:05:02+00:00",
        ),
        (
            "evt-3",
            "thoth-test",
            "image",
            "2026-04-18T11:00:00+00:00",
            "2026-04-18T11:00:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg", "size_bytes": 150}),
            "thoth-test/image/2026/04/18/evt-3.jpg",
            None,
            None,
            None,
            None,
        ),
        # evt-4: classified with low confidence (noise). Placed
        # at the earliest timestamp so its ordering is deterministic
        # when the floor is disabled.
        (
            "evt-4",
            "horus",
            "image",
            "2026-04-18T09:00:00+00:00",
            "2026-04-18T09:00:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg", "size_bytes": 120}),
            "horus/image/2026/04/18/evt-4.jpg",
            None,
            "Haemorhous mexicanus",
            0.05,
            "2026-04-18T09:00:02+00:00",
        ),
    ]
    conn.executemany(
        "INSERT INTO events ("
        "id, station, event_type, captured_at, received_at, schema_version, "
        "payload_json, media_key, thumb_key, species, confidence, classified_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


@pytest.fixture
def client(seeded_db: sqlite3.Connection) -> Iterator[TestClient]:
    """TestClient wired to the in-memory DB + fake MinIO."""
    fake_minio = _FakeMinio(
        {
            "horus/image/2026/04/18/evt-1.jpg": b"\xff\xd8\xff jpeg bytes for evt-1",
            "horus/image/2026/04/18/evt-2.jpg": b"\xff\xd8\xff jpeg bytes for evt-2",
            # evt-3 intentionally missing to exercise the 502 path
        }
    )

    app = create_app(
        cfg=_make_cfg(),
        db_connection=seeded_db,
        minio_store=fake_minio,  # type: ignore[arg-type]
    )
    with TestClient(app) as tc:
        yield tc


# ---- tests -----------------------------------------------------------------


def test_health_reports_event_count(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # /health reports the unfiltered row count — it's a DB reachability
    # probe, not a dashboard view. Floor applies only to /events.
    assert body["event_count"] == 4


def test_list_events_sorted_desc_by_captured_at(client: TestClient) -> None:
    resp = client.get("/events")
    assert resp.status_code == 200
    body = resp.json()
    ids = [e["id"] for e in body["events"]]
    # evt-2 (12:05) > evt-1 (12:00) > evt-3 (11:00).
    # evt-4 (09:00, confidence=0.05) is hidden by the default floor.
    assert ids == ["evt-2", "evt-1", "evt-3"]
    assert body["count"] == 3


def test_list_events_filters_by_station(client: TestClient) -> None:
    resp = client.get("/events", params={"station": "horus"})
    body = resp.json()
    assert {e["id"] for e in body["events"]} == {"evt-1", "evt-2"}


def test_list_events_filters_by_species(client: TestClient) -> None:
    resp = client.get("/events", params={"species": "Cyanocitta stelleri"})
    body = resp.json()
    assert [e["id"] for e in body["events"]] == ["evt-2"]


def test_list_events_filters_by_since(client: TestClient) -> None:
    resp = client.get("/events", params={"since": "2026-04-18T12:03:00+00:00"})
    body = resp.json()
    assert [e["id"] for e in body["events"]] == ["evt-2"]


def test_list_events_respects_limit_and_offset(client: TestClient) -> None:
    resp = client.get("/events", params={"limit": 1, "offset": 1})
    body = resp.json()
    assert [e["id"] for e in body["events"]] == ["evt-1"]
    assert body["limit"] == 1
    assert body["offset"] == 1


def test_get_event_by_id_returns_row(client: TestClient) -> None:
    resp = client.get("/events/evt-2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "evt-2"
    assert body["species"] == "Cyanocitta stelleri"
    assert body["confidence"] == 0.92
    assert body["payload"]["content_type"] == "image/jpeg"


def test_get_event_by_id_404s_on_miss(client: TestClient) -> None:
    resp = client.get("/events/nope")
    assert resp.status_code == 404


def test_get_image_streams_jpeg(client: TestClient) -> None:
    resp = client.get("/images/evt-1")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"\xff\xd8\xff jpeg bytes for evt-1"


def test_get_image_404s_when_event_missing(client: TestClient) -> None:
    resp = client.get("/images/nope")
    assert resp.status_code == 404


def test_get_image_502s_when_minio_fetch_fails(client: TestClient) -> None:
    # evt-3's media_key is not present in the fake store
    resp = client.get("/images/evt-3")
    assert resp.status_code == 502


def test_limit_validated(client: TestClient) -> None:
    # limit=0 should be rejected by Query(ge=1, le=500)
    resp = client.get("/events", params={"limit": 0})
    assert resp.status_code == 422


# ---- min_confidence floor --------------------------------------------------
#
# The /events endpoint hides classified-but-low-confidence rows from the
# default listing so the UI doesn't surface "House Finch · 5%" tiles that
# are almost certainly misclassified noise. Unclassified rows (NULL
# confidence) always pass through — the pipeline's in-flight events stay
# visible regardless of the floor.


def test_list_events_hides_low_confidence_by_default(client: TestClient) -> None:
    # Default floor is 0.10; evt-4 at confidence 0.05 should not appear.
    resp = client.get("/events")
    body = resp.json()
    ids = {e["id"] for e in body["events"]}
    assert "evt-4" not in ids


def test_list_events_min_confidence_zero_returns_everything(
    client: TestClient,
) -> None:
    # Explicit override to 0 disables the floor — dashboards tuning the
    # classifier need to see the full noise population.
    resp = client.get("/events", params={"min_confidence": 0})
    body = resp.json()
    ids = {e["id"] for e in body["events"]}
    assert ids == {"evt-1", "evt-2", "evt-3", "evt-4"}
    assert body["count"] == 4


def test_list_events_min_confidence_override_hides_mid_confidence(
    client: TestClient,
) -> None:
    # Raising the floor above evt-2's 0.92 should hide it, but NULLs
    # (evt-1, evt-3) must still come back.
    resp = client.get("/events", params={"min_confidence": 0.99})
    body = resp.json()
    ids = {e["id"] for e in body["events"]}
    assert "evt-2" not in ids
    assert {"evt-1", "evt-3"}.issubset(ids)


def test_list_events_unclassified_rows_bypass_floor(client: TestClient) -> None:
    # NULL confidence means "classifier hasn't run yet" — those must
    # pass through any floor so the in-flight pipeline remains visible.
    resp = client.get("/events", params={"min_confidence": 0.5})
    ids = {e["id"] for e in resp.json()["events"]}
    assert {"evt-1", "evt-3"}.issubset(ids)
    # evt-2 (0.92) passes, evt-4 (0.05) does not.
    assert "evt-2" in ids
    assert "evt-4" not in ids


def test_list_events_min_confidence_out_of_range_rejected(
    client: TestClient,
) -> None:
    # Query(ge=0.0, le=1.0) should 422 on > 1.0 and < 0.
    assert client.get("/events", params={"min_confidence": 1.5}).status_code == 422
    assert client.get("/events", params={"min_confidence": -0.1}).status_code == 422


def test_create_app_honors_min_confidence_override(
    seeded_db: sqlite3.Connection,
) -> None:
    """Tests that construct-time ``min_confidence`` replaces the default."""
    fake_minio = _FakeMinio({})
    app = create_app(
        cfg=_make_cfg(),
        db_connection=seeded_db,
        minio_store=fake_minio,  # type: ignore[arg-type]
        min_confidence=0.0,
    )
    with TestClient(app) as tc:
        body = tc.get("/events").json()
    ids = {e["id"] for e in body["events"]}
    # With no floor, evt-4 (0.05) must be visible.
    assert "evt-4" in ids


# ---- burst grouping (PR-B) -------------------------------------------------
#
# /events?group=burst collapses frames sharing a burst_id to a single
# hero row. The hero is MAX(hero_score) within the burst; the other
# frames are exposed via alternate_ids / alternate_count so the UI can
# render a "N more" expander.


@pytest.fixture
def burst_db() -> sqlite3.Connection:
    """In-memory DB with a 3-frame burst + two singletons.

    burst-A: three frames, hero is b-2 (highest hero_score).
    single-1: no burst_id (legacy/singleton).
    single-2: later singleton, captured after the burst completed.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)

    rows = [
        # (id, station, event_type, captured_at, received_at, sv,
        #  payload_json, media_key, thumb, species, confidence,
        #  classified_at, burst_id, burst_seq, sharpness, hero_score)
        (
            "b-1",
            "horus",
            "image",
            "2026-04-18T12:00:00+00:00",
            "2026-04-18T12:00:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-b-1",
            None,
            None,
            None,
            None,
            "burst-A",
            1,
            200.0,
            0.40,
        ),
        (
            "b-2",
            "horus",
            "image",
            "2026-04-18T12:00:01+00:00",
            "2026-04-18T12:00:02+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-b-2",
            None,
            None,
            None,
            None,
            "burst-A",
            2,
            900.0,
            0.75,  # highest — hero
        ),
        (
            "b-3",
            "horus",
            "image",
            "2026-04-18T12:00:02+00:00",
            "2026-04-18T12:00:03+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-b-3",
            None,
            None,
            None,
            None,
            "burst-A",
            3,
            400.0,
            0.55,
        ),
        (
            "single-1",
            "horus",
            "image",
            "2026-04-18T11:55:00+00:00",
            "2026-04-18T11:55:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-single-1",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "single-2",
            "horus",
            "image",
            "2026-04-18T12:10:00+00:00",
            "2026-04-18T12:10:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-single-2",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ]
    conn.executemany(
        "INSERT INTO events ("
        "id, station, event_type, captured_at, received_at, schema_version, "
        "payload_json, media_key, thumb_key, species, confidence, classified_at, "
        "burst_id, burst_seq, sharpness, hero_score"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


@pytest.fixture
def burst_client(burst_db: sqlite3.Connection) -> Iterator[TestClient]:
    fake_minio = _FakeMinio({})
    app = create_app(
        cfg=_make_cfg(),
        db_connection=burst_db,
        minio_store=fake_minio,  # type: ignore[arg-type]
    )
    with TestClient(app) as tc:
        yield tc


def test_group_burst_collapses_frames_to_hero(burst_client: TestClient) -> None:
    resp = burst_client.get("/events")  # default group=burst
    assert resp.status_code == 200
    body = resp.json()
    # Three groups: single-2 (newest), burst-A (collapsed), single-1.
    ids = [e["id"] for e in body["events"]]
    assert ids == ["single-2", "b-2", "single-1"]
    assert body["count"] == 3
    assert body["group"] == "burst"


def test_group_burst_exposes_alternates_in_seq_order(
    burst_client: TestClient,
) -> None:
    body = burst_client.get("/events").json()
    hero_row = next(e for e in body["events"] if e["id"] == "b-2")
    # Hero is b-2, alternates are b-1 + b-3 in burst_seq order.
    assert hero_row["alternate_ids"] == ["b-1", "b-3"]
    assert hero_row["alternate_count"] == 2


def test_group_burst_singletons_have_no_alternates(burst_client: TestClient) -> None:
    body = burst_client.get("/events").json()
    singleton = next(e for e in body["events"] if e["id"] == "single-2")
    assert singleton["alternate_ids"] == []
    assert singleton["alternate_count"] == 0


def test_group_none_returns_every_frame(burst_client: TestClient) -> None:
    resp = burst_client.get("/events", params={"group": "none"})
    body = resp.json()
    assert body["group"] == "none"
    ids = [e["id"] for e in body["events"]]
    # All 5 rows in captured_at DESC.
    assert ids == ["single-2", "b-3", "b-2", "b-1", "single-1"]
    assert body["count"] == 5


def test_group_burst_pagination_counts_bursts_not_frames(
    burst_client: TestClient,
) -> None:
    # limit=1 at offset=0 returns exactly one group (single-2) even
    # though the underlying window includes the three burst frames.
    body = burst_client.get("/events", params={"limit": 1, "offset": 0}).json()
    ids = [e["id"] for e in body["events"]]
    assert ids == ["single-2"]
    # Offset by 1 group → the burst hero.
    body = burst_client.get("/events", params={"limit": 1, "offset": 1}).json()
    ids = [e["id"] for e in body["events"]]
    assert ids == ["b-2"]


def test_group_burst_rejects_unknown_value(burst_client: TestClient) -> None:
    # Pattern constraint on the Query — anything other than burst/none
    # comes back 422 rather than silently falling through.
    resp = burst_client.get("/events", params={"group": "banana"})
    assert resp.status_code == 422


# ---- deep-offset pagination regression (post-R2 review) --------------------
#
# An earlier implementation clamped the candidate-row window to 500 rows,
# which silently truncated any ``?group=burst`` request where
# ``(limit + offset) * _MAX_BURST_FANOUT > 500``.  For limit=10 offset=20
# the caller needed ~900 rows but received 500 → the grouped slice came
# back empty or partial with no error signal. This fixture + test seeds
# enough singletons to reproduce that failure mode under the old cap,
# and asserts the new unbounded window resolves the page correctly.


@pytest.fixture
def deep_singletons_db() -> sqlite3.Connection:
    """In-memory DB with 600 singleton events in strict DESC order.

    Size chosen to exceed the prior 500-row window cap so the deep-offset
    pagination test below actually exercises the bug.  Each row is a
    singleton (``burst_id`` NULL), so every row is its own group — the
    test can assert group-position == row-position without worrying
    about burst collapsing.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)

    base = datetime(2026, 4, 18, 12, 0, 0, tzinfo=timezone.utc)
    rows: list[tuple[Any, ...]] = []
    for i in range(600):
        # i=0 is the newest, i=599 the oldest, so DESC sort yields
        # ``deep-0000`` first. Per-row offset is 1s so every row has a
        # unique captured_at and the ordering is stable.
        ts = (base - timedelta(seconds=i)).isoformat()
        rows.append(
            (
                f"deep-{i:04d}",
                "horus",
                "image",
                ts,
                ts,
                1,
                json.dumps({"content_type": "image/jpeg"}),
                f"k-deep-{i}",
                None,
                None,
                None,
                None,
                None,  # burst_id
                None,  # burst_seq
                None,  # sharpness
                None,  # hero_score
            )
        )
    conn.executemany(
        "INSERT INTO events ("
        "id, station, event_type, captured_at, received_at, schema_version, "
        "payload_json, media_key, thumb_key, species, confidence, classified_at, "
        "burst_id, burst_seq, sharpness, hero_score"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


@pytest.fixture
def deep_client(deep_singletons_db: sqlite3.Connection) -> Iterator[TestClient]:
    fake_minio = _FakeMinio({})
    app = create_app(
        cfg=_make_cfg(),
        db_connection=deep_singletons_db,
        minio_store=fake_minio,  # type: ignore[arg-type]
    )
    with TestClient(app) as tc:
        yield tc


def test_group_burst_pagination_beyond_old_window_cap(
    deep_client: TestClient,
) -> None:
    """Deep offset in group=burst must return the correct rows, not silently
    empty. Under the previous 500-row window cap this page came back [].
    """
    body = deep_client.get(
        "/events", params={"limit": 10, "offset": 500}
    ).json()
    ids = [e["id"] for e in body["events"]]
    # Rows 500..509 in captured_at DESC order.
    assert ids == [f"deep-{i:04d}" for i in range(500, 510)]
    assert body["count"] == 10


def test_group_burst_pagination_past_end_returns_empty(
    deep_client: TestClient,
) -> None:
    """Offset past the end still returns an empty page without error —
    distinguishes 'legitimately nothing here' from the old silent-cap bug
    which returned empty *despite* having rows available.
    """
    body = deep_client.get(
        "/events", params={"limit": 10, "offset": 600}
    ).json()
    assert body["events"] == []
    assert body["count"] == 0


# ---- concurrent bursts on different stations -------------------------------
#
# Two stations can fire bursts at the same wall-clock instant (the feeder
# camera and a sibling unit on the same household get a goldfinch + a
# squirrel within the same second). The grouped listing must keep those
# bursts separate even when their captured_at values overlap — the
# segregation key is burst_id, not time.


@pytest.fixture
def concurrent_bursts_db() -> sqlite3.Connection:
    """Two same-time bursts on different stations.

    burst-A on station ``horus``: 2 frames, hero is a-2 (higher score).
    burst-B on station ``horus2``: 2 frames, hero is b-1 (higher score).
    Frames interleave in captured_at to simulate the wire ordering when
    both stations publish concurrently.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)

    rows = [
        # (id, station, event_type, captured_at, received_at, sv,
        #  payload_json, media_key, thumb, species, confidence,
        #  classified_at, burst_id, burst_seq, sharpness, hero_score)
        (
            "a-1",
            "horus",
            "image",
            "2026-04-18T12:00:00+00:00",
            "2026-04-18T12:00:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-a-1",
            None,
            None,
            None,
            None,
            "burst-A",
            1,
            300.0,
            0.40,
        ),
        (
            "b-1",
            "horus2",
            "image",
            "2026-04-18T12:00:00+00:00",  # same wall-clock as a-1
            "2026-04-18T12:00:01+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-b-1",
            None,
            None,
            None,
            None,
            "burst-B",
            1,
            800.0,
            0.70,  # B's hero
        ),
        (
            "a-2",
            "horus",
            "image",
            "2026-04-18T12:00:01+00:00",
            "2026-04-18T12:00:02+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-a-2",
            None,
            None,
            None,
            None,
            "burst-A",
            2,
            900.0,
            0.65,  # A's hero
        ),
        (
            "b-2",
            "horus2",
            "image",
            "2026-04-18T12:00:01+00:00",  # same wall-clock as a-2
            "2026-04-18T12:00:02+00:00",
            1,
            json.dumps({"content_type": "image/jpeg"}),
            "k-b-2",
            None,
            None,
            None,
            None,
            "burst-B",
            2,
            200.0,
            0.30,
        ),
    ]
    conn.executemany(
        "INSERT INTO events ("
        "id, station, event_type, captured_at, received_at, schema_version, "
        "payload_json, media_key, thumb_key, species, confidence, classified_at, "
        "burst_id, burst_seq, sharpness, hero_score"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


@pytest.fixture
def concurrent_bursts_client(
    concurrent_bursts_db: sqlite3.Connection,
) -> Iterator[TestClient]:
    fake_minio = _FakeMinio({})
    app = create_app(
        cfg=_make_cfg(),
        db_connection=concurrent_bursts_db,
        minio_store=fake_minio,  # type: ignore[arg-type]
    )
    with TestClient(app) as tc:
        yield tc


def test_concurrent_bursts_on_different_stations_segregate(
    concurrent_bursts_client: TestClient,
) -> None:
    """Two bursts overlapping in captured_at on different stations must
    surface as two distinct groups in /events?group=burst, each with the
    correct hero. Regression guard: the grouping key is burst_id, never
    captured_at, so concurrent feeder visits don't bleed into one group.
    """
    body = concurrent_bursts_client.get("/events").json()
    assert body["group"] == "burst"
    # Two bursts → two groups.
    groups_by_burst = {e["burst_id"]: e for e in body["events"]}
    assert set(groups_by_burst) == {"burst-A", "burst-B"}
    assert body["count"] == 2

    # burst-A hero is a-2 (hero_score 0.65 > 0.40); a-1 is its alternate.
    a = groups_by_burst["burst-A"]
    assert a["id"] == "a-2"
    assert a["station"] == "horus"
    assert a["alternate_ids"] == ["a-1"]
    assert a["alternate_count"] == 1

    # burst-B hero is b-1 (hero_score 0.70 > 0.30); b-2 is its alternate.
    b = groups_by_burst["burst-B"]
    assert b["id"] == "b-1"
    assert b["station"] == "horus2"
    assert b["alternate_ids"] == ["b-2"]
    assert b["alternate_count"] == 1
