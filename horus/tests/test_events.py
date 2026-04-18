"""Tests for horus.events — PUBACK-aware publish path.

The bug being regressed: ``EventBus._publish`` used to check
``info.rc`` *after* ``wait_for_publish``. ``rc`` is the enqueue result,
not the broker ack, so a silently-dropped message (e.g. broker timeout
without PUBACK) would log nothing. The current contract:

* ``rc != MQTT_ERR_SUCCESS`` at enqueue → warn + return.
* ``wait_for_publish`` raises → warn + return.
* ``wait_for_publish`` returns but ``is_published()`` is False
  (timeout) → warn.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest

from horus.config import (
    CaptureConfig,
    HorusConfig,
    MotionConfig,
    MqttConfig,
    StorageConfig,
)
from horus.events import EventBus


@pytest.fixture
def cfg(tmp_path: Path) -> HorusConfig:
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(),
        motion=MotionConfig(),
        storage=StorageConfig(local_dir=tmp_path),
    )


def _make_bus(cfg: HorusConfig, info: MagicMock) -> EventBus:
    """Build an EventBus whose paho client returns ``info`` from publish()."""
    bus = EventBus(cfg)
    bus._client = MagicMock()
    bus._client.publish.return_value = info
    return bus


def _ack_info() -> MagicMock:
    info = MagicMock()
    info.rc = mqtt.MQTT_ERR_SUCCESS
    info.wait_for_publish.return_value = None
    info.is_published.return_value = True
    return info


def test_publish_success_returns_true_and_is_silent(cfg, caplog):
    info = _ack_info()
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is True
    assert bus.dropped_publishes == 0
    assert caplog.records == []
    info.wait_for_publish.assert_called_once()


def test_publish_enqueue_failure_short_circuits(cfg, caplog):
    """If the enqueue fails we must NOT wait for a PUBACK that never comes."""
    info = _ack_info()
    info.rc = mqtt.MQTT_ERR_NO_CONN
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    info.wait_for_publish.assert_not_called()
    assert any("enqueue" in r.message for r in caplog.records)


def test_publish_wait_raises_returns_false(cfg, caplog):
    info = _ack_info()
    info.wait_for_publish.side_effect = RuntimeError("disconnected mid-publish")
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    assert any("awaiting ack" in r.message for r in caplog.records)


def test_publish_no_puback_within_timeout_returns_false(cfg, caplog):
    """This is the regression: wait_for_publish returns (no raise) but the
    broker never acked. ``rc`` still shows ENQUEUE success (0), so the old
    check passed silently. We now require is_published() to be True and
    return False so callers can skip success side-effects."""
    info = _ack_info()
    info.is_published.return_value = False  # no PUBACK
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    assert any("timed out" in r.message for r in caplog.records)


def test_publish_is_published_raises_returns_false(cfg, caplog):
    """Guard against the race where the paho loop thread flips rc to an
    error between wait_for_publish returning and is_published() being
    called. We treat this as a drop rather than letting the exception
    propagate into the capture loop."""
    info = _ack_info()
    info.is_published.side_effect = ValueError("publish not complete")
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    assert any("ack check" in r.message for r in caplog.records)


def test_dropped_publishes_counter_accumulates(cfg):
    info = _ack_info()
    info.is_published.return_value = False
    bus = _make_bus(cfg, info)
    bus._publish("sbo/horus-test/status", {"ok": True})
    bus._publish("sbo/horus-test/status", {"ok": True})
    bus._publish("sbo/horus-test/status", {"ok": True})
    assert bus.dropped_publishes == 3


def test_status_payload_carries_camera_label(cfg):
    info = _ack_info()
    bus = _make_bus(cfg, info)
    bus.publish_status({"camera_ok": True})
    # Assert the status topic + retain flag were used and payload is JSON.
    args, kwargs = bus._client.publish.call_args
    topic, payload = args[0], args[1]
    assert topic == "sbo/horus-test/status"
    assert kwargs.get("retain") is True
    assert '"station": "horus-test"' in payload


def test_image_event_uses_cfg_camera_label(cfg, tmp_path):
    """The event payload's ``camera`` field must reflect cfg — that's the
    only place downstream services learn which sensor produced the frame."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-ish bytes
    ok = bus.publish_image_event(img, changed_fraction=0.03)
    assert ok is True
    payload = bus._client.publish.call_args.args[1]
    assert '"camera": "imx519"' in payload


def test_image_event_returns_false_on_drop(cfg, tmp_path):
    """Callers (main._tick) rely on this return to skip advancing the
    cooldown on a failed publish."""
    info = _ack_info()
    info.is_published.return_value = False
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    ok = bus.publish_image_event(img, changed_fraction=0.03)
    assert ok is False
    assert bus.dropped_publishes == 1


def test_image_event_uses_cfg_resolution_by_default(cfg, tmp_path):
    """No resolution_override → published ``resolution`` matches capture cfg."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03)
    payload = bus._client.publish.call_args.args[1]
    # CaptureConfig default is 2304 x 1296.
    assert '"resolution": [2304, 1296]' in payload


def test_image_event_resolution_override_is_used(cfg, tmp_path):
    """When the caller passes resolution_override (crop path) the published
    ``resolution`` reflects the actual image bytes, not the sensor config."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03, resolution_override=(640, 640))
    payload = bus._client.publish.call_args.args[1]
    assert '"resolution": [640, 640]' in payload
