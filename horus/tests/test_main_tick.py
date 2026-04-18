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

from horus.classifier import ClassificationResult
from horus.config import (
    CaptureConfig,
    ClassifierConfig,
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


# ---------------------------------------------------------------------------
# On-device bird classifier gate
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_with_classifier(tmp_path: Path) -> HorusConfig:
    """Baseline config with classifier enabled at a 0.30 threshold.

    Tests swap in a MagicMock classifier on the Daemon so the real
    tflite runtime never runs — we're exercising the gate logic, not
    the model.
    """
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(interval_s=0.1),
        motion=MotionConfig(cooldown_s=10.0),
        storage=StorageConfig(local_dir=tmp_path),
        classifier=ClassifierConfig(enabled=True, min_confidence=0.30),
    )


def test_tick_publishes_with_bird_score_when_above_threshold(cfg_with_classifier, tmp_path):
    """Confidence 0.55 > 0.30 → publish, attach bird_score + bird_label."""
    daemon = Daemon(cfg_with_classifier)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=(0.2, 0.2, 0.6, 0.6))
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="carolina chickadee", confidence=0.55
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    assert daemon.bus.publish_image_event.called
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("bird_score") == pytest.approx(0.55)
    assert kwargs.get("bird_label") == "carolina chickadee"
    assert daemon._last_event_ts > 0.0


def test_tick_gates_out_low_confidence(cfg_with_classifier, tmp_path):
    """Confidence 0.18 < 0.30 → do NOT publish. Capture is discarded and
    the cooldown advances so we don't reclassify the same wind-sway burst."""
    daemon = Daemon(cfg_with_classifier)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True  # ignored — publish never called
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=(0.0, 0.0, 0.9, 1.0))
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="new zealand pigeon", confidence=0.18
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    daemon.bus.publish_image_event.assert_not_called()
    assert not capture_path.exists(), "capture must be discarded when gated out"
    # The crop sibling (if any) should also be cleaned up.
    crop_path = capture_path.with_name(capture_path.stem + "_crop.jpg")
    assert not crop_path.exists(), "crop sibling must be cleaned up when gated out"
    # Cooldown advances: prevents rapid re-gate of the same wind gust.
    assert daemon._last_event_ts > 0.0


def test_tick_publishes_anyway_when_classifier_raises(cfg_with_classifier, tmp_path):
    """Degraded behavior: inference failure falls through to the legacy
    'publish every motion event' path. We prefer false positives to
    silently dropping a bird because a model crashed."""
    daemon = Daemon(cfg_with_classifier)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.classifier = MagicMock()
    daemon.classifier.classify.side_effect = RuntimeError("CUDA missing")

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    assert daemon.bus.publish_image_event.called
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("bird_score") is None
    assert kwargs.get("bird_label") is None
    assert daemon._last_event_ts > 0.0


def test_tick_without_classifier_omits_bird_score(cfg, tmp_path):
    """Classifier disabled (the default) → bird_score absent from payload.
    This is the backwards-compatible path for stations without a model."""
    daemon = Daemon(cfg)
    assert daemon.classifier is None
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

    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("bird_score") is None
    assert kwargs.get("bird_label") is None


# ---------------------------------------------------------------------------
# Gated-sample archive — classifier drops should land in the review archive
# so we can measure false-negative rate, not just false positives.
# ---------------------------------------------------------------------------


def _cfg_with_archive(tmp_path: Path, archive_dir: Path | None) -> HorusConfig:
    """Classifier-enabled config with an optional gated archive root."""
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(interval_s=0.1),
        motion=MotionConfig(cooldown_s=10.0),
        storage=StorageConfig(local_dir=tmp_path),
        classifier=ClassifierConfig(
            enabled=True,
            min_confidence=0.30,
            gated_archive_dir=archive_dir,
        ),
    )


def test_tick_archives_gated_capture_when_archive_dir_set(tmp_path):
    """Sub-threshold classifier result must invoke save_gated_sample with
    the score and label so the reviewer can spot false negatives."""
    archive = tmp_path / "gated"
    daemon = Daemon(_cfg_with_archive(tmp_path, archive))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="new zealand pigeon", confidence=0.12
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None), \
         patch("horus.main.storage.save_gated_sample") as mock_save:
        daemon._tick()

    daemon.bus.publish_image_event.assert_not_called()
    mock_save.assert_called_once()
    args, kwargs = mock_save.call_args
    assert args[0] == archive
    # The second positional arg is the path the classifier actually scored.
    # No bbox here → full frame.
    assert args[1] == capture_path
    assert kwargs.get("score") == pytest.approx(0.12)
    assert kwargs.get("label") == "new zealand pigeon"


def test_tick_does_not_archive_when_archive_dir_is_none(tmp_path):
    """Backwards-compat: stations without gated_archive_dir configured
    must not call save_gated_sample at all — the gate drops silently."""
    daemon = Daemon(_cfg_with_archive(tmp_path, archive_dir=None))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="junco", confidence=0.10
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None), \
         patch("horus.main.storage.save_gated_sample") as mock_save:
        daemon._tick()

    mock_save.assert_not_called()


def test_tick_archives_cropped_bytes_when_crop_exists(tmp_path):
    """When MotionGate returns a bbox we crop and classify the crop —
    the archive must save the exact bytes the classifier saw (the crop),
    not the full frame. Otherwise reviewer-visible images would differ
    from model-visible images and we'd lose the ground-truth signal."""
    archive = tmp_path / "gated"
    daemon = Daemon(_cfg_with_archive(tmp_path, archive))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(
        bbox_fraction=(0.2, 0.2, 0.6, 0.6)
    )
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="unknown", confidence=0.08
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None), \
         patch("horus.main.storage.save_gated_sample") as mock_save:
        daemon._tick()

    mock_save.assert_called_once()
    args, _ = mock_save.call_args
    saved_path = args[1]
    # The saved path must be the crop sibling, not the original frame.
    expected_crop = capture_path.with_name(capture_path.stem + "_crop.jpg")
    assert saved_path == expected_crop, (
        "archive must store the cropped bytes the classifier actually saw, "
        "not the full-frame original"
    )
