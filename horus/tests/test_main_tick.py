"""Tests for horus.main._tick cooldown discipline and crop integration.

Two contracts are pinned here:

1. A dropped publish (broker never acked) must NOT advance
   ``_last_event_ts``. Otherwise a silent drop would suppress the next
   real motion event for ``cooldown_s`` seconds.

2. When the motion gate returns a bbox, ``_tick`` must publish the
   **cropped** image (not the full frame) and pass the crop's actual
   dimensions as ``resolution_override`` — this is what gets the
   classifier out of the low-confidence full-frame regime.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

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


def _motion_result(
    fraction: float = 0.05,
    bbox_fraction: tuple[float, float, float, float] | None = None,
) -> MagicMock:
    r = MagicMock()
    r.motion = True
    r.changed_fraction = fraction
    r.bbox_fraction = bbox_fraction
    return r


def _write_real_jpeg(path: Path, size: tuple[int, int] = (1000, 1000)) -> None:
    """Write a real JPEG Pillow can decode — ``crop_to_bbox`` requires a valid image."""
    Image.new("RGB", size, color=(120, 120, 120)).save(path, format="JPEG", quality=90)


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


# ---------------------------------------------------------------------------
# Crop integration — bbox from MotionGate → bird-centered crop → publish
# ---------------------------------------------------------------------------


def test_tick_publishes_crop_when_bbox_present(cfg, tmp_path):
    """When motion result has a bbox, publish the cropped image (not the full frame)."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(
        bbox_fraction=(0.2, 0.2, 0.6, 0.6),
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"):
        daemon._tick()

    assert daemon.bus.publish_image_event.called
    args, kwargs = daemon.bus.publish_image_event.call_args
    published_path = args[0]

    # The published path must be the crop, NOT the original — the whole
    # point of this feature is that the classifier sees the cropped bird.
    assert published_path != capture_path, (
        "expected the crop path to be published, not the full-frame path"
    )
    assert published_path.exists(), "crop file must have been written to disk"
    assert published_path.suffix == ".jpg"

    # resolution_override must match the crop's real dimensions.
    resolution = kwargs.get("resolution_override")
    assert resolution is not None, "crop path must pass resolution_override"
    with Image.open(published_path) as saved:
        assert saved.size == resolution


def test_tick_publishes_full_frame_when_bbox_missing(cfg, tmp_path):
    """No bbox → fall back to the original full-frame publish (backwards compatible)."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"):
        daemon._tick()

    args, kwargs = daemon.bus.publish_image_event.call_args
    assert args[0] == capture_path, "should publish the original path when no bbox"
    assert kwargs.get("resolution_override") is None


def test_tick_falls_back_to_full_frame_when_crop_raises(cfg, tmp_path):
    """Crop failure (unreadable image, etc.) must not drop the motion event."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(
        bbox_fraction=(0.2, 0.2, 0.6, 0.6),
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.crop_to_bbox", side_effect=RuntimeError("unreadable")):
        daemon._tick()

    # Fallback: publish the original.
    args, kwargs = daemon.bus.publish_image_event.call_args
    assert args[0] == capture_path
    assert kwargs.get("resolution_override") is None
    # And the cooldown must still advance — this is a real motion event.
    assert daemon._last_event_ts > 0.0


# ---------------------------------------------------------------------------
# Observability — bbox + AF metadata ride on the published event
# ---------------------------------------------------------------------------


def test_tick_passes_bbox_fraction_to_publish(cfg, tmp_path):
    """bbox_fraction from MotionGate must propagate to publish_image_event so
    Thoth can see where in the frame the motion happened (focal vs
    distributed). Without this, post-mortem debugging of false positives is
    guesswork."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    bbox = (0.2, 0.3, 0.6, 0.7)
    daemon.gate.check.return_value = _motion_result(bbox_fraction=bbox)

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("bbox_fraction") == bbox


def test_tick_passes_af_metadata_from_sidecar_to_publish(cfg, tmp_path):
    """AF summary from the rpicam sidecar must ride on the payload. Thoth
    uses this to ask 'was the lens actually focused when we triggered?'"""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))
    af = {"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234}

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=af) as mock_read:
        daemon._tick()

    # Must be called with the capture path (not the crop path) so the sidecar
    # path resolution lines up with what rpicam-still wrote.
    mock_read.assert_called_once_with(capture_path)
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("af") == af


def test_tick_tolerates_missing_af_sidecar(cfg, tmp_path):
    """read_af_fields returning None (missing/corrupt sidecar) must not
    abort the publish — we still want the event, just without AF context."""
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    assert daemon.bus.publish_image_event.called
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("af") is None
    assert daemon._last_event_ts > 0.0
