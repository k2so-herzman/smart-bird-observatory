"""Tests for horus.main._tick cooldown discipline.

The specific contract being pinned: a dropped publish (broker never
acked) must NOT advance `_last_event_ts`. Otherwise a silent drop
would suppress the next real motion event for `cooldown_s` seconds.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from horus.config import (
    CaptureConfig,
    HorusConfig,
    MotionConfig,
    MqttConfig,
    StorageConfig,
)
from horus.main import Daemon


@pytest.fixture
def cfg(tmp_path: Path) -> HorusConfig:
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(interval_s=0.1),
        motion=MotionConfig(cooldown_s=10.0),
        storage=StorageConfig(local_dir=tmp_path),
    )


def _motion_result(fraction: float = 0.05) -> MagicMock:
    r = MagicMock()
    r.motion = True
    r.changed_fraction = fraction
    return r


def test_tick_advances_cooldown_on_successful_publish(cfg, tmp_path):
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result()

    capture_path = tmp_path / "cap.jpg"
    capture_path.write_bytes(b"\xff\xd8\xff\xd9")

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"):
        assert daemon._last_event_ts == 0.0
        daemon._tick()
        assert daemon._last_event_ts > 0.0, "cooldown must advance on ack'd publish"


def test_tick_does_not_advance_cooldown_on_dropped_publish(cfg, tmp_path):
    """Regression: silent-drop publishes used to tick the cooldown,
    suppressing the next real event for cooldown_s seconds."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = False  # no PUBACK
    daemon.bus.dropped_publishes = 1
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result()

    capture_path = tmp_path / "cap.jpg"
    capture_path.write_bytes(b"\xff\xd8\xff\xd9")

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"):
        daemon._tick()
        assert daemon._last_event_ts == 0.0, (
            "cooldown must NOT advance when publish_image_event returned False — "
            "otherwise a dropped event gates out the next real one"
        )


def test_tick_does_not_advance_cooldown_on_publish_exception(cfg, tmp_path):
    """Belt-and-suspenders: an exception from publish_image_event (bug,
    not silent drop) also must not tick the cooldown."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.side_effect = RuntimeError("boom")
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result()

    capture_path = tmp_path / "cap.jpg"
    capture_path.write_bytes(b"\xff\xd8\xff\xd9")

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"):
        daemon._tick()
        assert daemon._last_event_ts == 0.0
