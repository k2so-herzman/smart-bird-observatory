"""Tests for the MinIO store — covers the parts we can test without a
live MinIO: endpoint parsing and deterministic key generation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from banshee.events import ImageEvent
from banshee.minio_store import (
    MinioConfig,
    MinioStore,
    _extension_for,
    _split_endpoint,
)


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


def _make_image_event(
    station: str = "horus",
    ts: datetime | None = None,
    content_type: str = "image/jpeg",
) -> ImageEvent:
    body = b"\xff\xd8\xff\xe0fake"
    return ImageEvent(
        schema_version=1,
        station=station,
        captured_at=ts or datetime(2026, 4, 17, 22, 0, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(2328, 1748),
        content_type=content_type,
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


def test_extension_for_jpeg_normalized() -> None:
    # image/jpeg can resolve to .jpe on some platforms; we normalize to .jpg.
    assert _extension_for("image/jpeg") == ".jpg"


def test_extension_for_png() -> None:
    assert _extension_for("image/png") == ".png"


def test_extension_for_unknown_falls_back_to_jpg() -> None:
    assert _extension_for("application/x-weird") == ".jpg"
    assert _extension_for("") == ".jpg"


def test_image_key_uses_content_type_extension() -> None:
    store = MinioStore(
        MinioConfig(endpoint="minio:9000", access_key="a", secret_key="b")
    )
    event = _make_image_event(content_type="image/png")
    key = store.image_key(event, "abc123")
    assert key.endswith(".png")
    assert "abc123" in key


class _FakeMinioClient:
    """Minimal Minio() stand-in for exercising put/remove paths."""

    def __init__(self) -> None:
        self.uploaded: list[tuple[str, str]] = []
        self.removed: list[tuple[str, str]] = []
        self.remove_raises: Exception | None = None

    def put_object(
        self, bucket: str, key: str, body, length: int, content_type: str
    ) -> None:
        self.uploaded.append((bucket, key))

    def remove_object(self, bucket: str, key: str) -> None:
        if self.remove_raises is not None:
            raise self.remove_raises
        self.removed.append((bucket, key))


def test_put_image_delegates_to_client() -> None:
    store = MinioStore(
        MinioConfig(endpoint="minio:9000", access_key="a", secret_key="b")
    )
    fake = _FakeMinioClient()
    store._client = fake  # type: ignore[assignment]

    event = _make_image_event()
    key = store.put_image(event, "evt-1")

    assert fake.uploaded == [("thoth", key)]
    assert key.endswith("/evt-1.jpg")


def test_remove_object_calls_client() -> None:
    store = MinioStore(
        MinioConfig(endpoint="minio:9000", access_key="a", secret_key="b")
    )
    fake = _FakeMinioClient()
    store._client = fake  # type: ignore[assignment]

    store.remove_object("horus/image/2026/04/17/orphan.jpg")

    assert fake.removed == [("thoth", "horus/image/2026/04/17/orphan.jpg")]


def test_remove_object_swallows_client_errors() -> None:
    """Orphan cleanup is best-effort — a failing remove must not bubble."""
    store = MinioStore(
        MinioConfig(endpoint="minio:9000", access_key="a", secret_key="b")
    )
    fake = _FakeMinioClient()
    fake.remove_raises = RuntimeError("minio down")
    store._client = fake  # type: ignore[assignment]

    # Should NOT raise — caller shouldn't crash on cleanup failure.
    store.remove_object("horus/image/2026/04/17/orphan.jpg")
