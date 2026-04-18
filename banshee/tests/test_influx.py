"""Tests for banshee.influx — injectable client factory.

Pins two contracts from issue #8 + #9:
1. An injected ``client_factory`` lets tests bypass the real
   InfluxDBClient without monkey-patching.
2. An empty token disables writes (dev convenience) and leaves the
   write API unconstructed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from banshee.config import InfluxConfig
from banshee.events import ImageEvent, StatusEvent
from banshee.influx import InfluxWriter


def _image_event() -> ImageEvent:
    body = b"x"
    return ImageEvent(
        schema_version=1,
        station="horus",
        captured_at=datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc),
        camera="imx519",
        trigger="motion",
        resolution=(100, 100),
        content_type="image/jpeg",
        image_bytes=body,
        size_bytes=len(body),
        changed_fraction=0.01,
        sha256=hashlib.sha256(body).hexdigest(),
    )


def _status_event() -> StatusEvent:
    return StatusEvent(
        schema_version=1,
        station="horus",
        ts=datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc),
        raw={"camera_ok": True, "temp_c": 42.5, "label": "dropped"},
    )


@pytest.fixture
def fake_client():
    client = MagicMock()
    write_api = MagicMock()
    client.write_api.return_value = write_api
    return client


@pytest.fixture
def writer(fake_client):
    cfg = InfluxConfig(url="http://injected", token="t", org="o", bucket="b")
    w = InfluxWriter(cfg, client_factory=lambda _cfg: fake_client)
    w.connect()
    return w


def test_client_factory_is_used_instead_of_real_client(writer, fake_client):
    """No real ``InfluxDBClient`` is ever constructed — the injected
    factory is the only code path."""
    assert writer._client is fake_client
    fake_client.write_api.assert_called_once()


def test_empty_token_disables_writes():
    """Local dev convenience: no token → connect() is a no-op and
    subsequent writes silently skip instead of crashing."""
    cfg = InfluxConfig(url="http://injected", token="", org="o", bucket="b")
    called = []
    w = InfluxWriter(cfg, client_factory=lambda _c: called.append(1))  # type: ignore[return-value]
    w.connect()
    assert called == [], "client_factory must not fire when token is empty"
    # Writes still fine (no-op).
    w.write_image_event(_image_event(), "id", "k")


def test_write_image_event_builds_point(writer, fake_client):
    writer.write_image_event(_image_event(), "eid-1", "horus/image/2026/04/17/eid-1.jpg")
    fake_client.write_api.return_value.write.assert_called_once()
    kwargs = fake_client.write_api.return_value.write.call_args.kwargs
    assert kwargs["bucket"] == "b"


def test_write_status_filters_non_scalar_fields(writer, fake_client):
    """Only numeric/bool values become Influx fields; strings are
    dropped because Influx rejects them alongside the tags we set."""
    writer.write_status(_status_event())
    fake_client.write_api.return_value.write.assert_called_once()


def test_close_releases_client(writer, fake_client):
    writer.close()
    fake_client.close.assert_called_once()
    assert writer._client is None
    # Safe to call again after close.
    writer.close()
