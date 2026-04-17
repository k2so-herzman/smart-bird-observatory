"""Tests for the MinIO store — covers the parts we can test without a
live MinIO: endpoint parsing and deterministic key generation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from banshee.events import ImageEvent
from banshee.minio_store import MinioConfig, MinioStore, _split_endpoint


def test_split_endpoint_bare_hostport() -> None:
    host, secure = _split_endpoint("minio:9000")
    assert host == "minio:9000"
    assert secure is False


def test_split_endpoint_http_url() -> None:
    host, secure = _split_endpoint("http://192.168.1.65:9000")
    assert host == "192.168.1.65:9000"
    assert secure is False


def test_split_endpoint_https_url() -> None:
    host, secure = _split_endpoint("https://minio.example.com")
    assert host == "minio.example.com"
    assert secure is True


def _make_image_event(station: str = "horus", ts: datetime | None = None) -> ImageEvent:
    body = b"\xff\xd8\xff\xe0fake"
    return ImageEvent(
        schema_version=1,
        station=station,
        captured_at=ts or datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(2328, 1748),
        content_type="image/jpeg",
        image_bytes=body,
        size_bytes=len(body),
        changed_fraction=0.04,
        sha256=hashlib.sha256(body).hexdigest(),
    )


def test_image_key_layout() -> None:
    # No network calls — MinioStore constructor is pure client init.
    store = MinioStore(
        MinioConfig(
            endpoint="http://192.168.1.65:9000",
            access_key="ak",
            secret_key="sk",
            bucket="thoth",
        )
    )
    event = _make_image_event(ts=datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc))
    key = store.image_key(event, event_id="deadbeef")
    assert key == "horus/image/2026/04/17/deadbeef.jpg"


def test_image_key_per_station() -> None:
    store = MinioStore(
        MinioConfig(endpoint="minio:9000", access_key="a", secret_key="b")
    )
    key_horus = store.image_key(_make_image_event(station="horus"), "id-1")
    key_seth = store.image_key(_make_image_event(station="seth"), "id-1")
    assert key_horus.startswith("horus/")
    assert key_seth.startswith("seth/")
