"""Tests for the thoth-api read endpoints.

Uses FastAPI's TestClient against an in-memory SQLite + a fake MinIO
store so the suite never touches the network or the real DB path.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
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
from banshee.eventstore import SCHEMA
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
