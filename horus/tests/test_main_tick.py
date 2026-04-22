"""Tests for horus.main._tick cooldown discipline and crop integration.

Three contracts are pinned here:

1. A dropped publish (broker never acked) must NOT advance
   ``_last_event_ts``. Otherwise a silent drop would suppress the next
   real motion event for ``cooldown_s`` seconds.

2. When the motion gate returns a bbox, ``_tick`` generates a bird-
   centered crop **internally** and feeds it to the on-device
   detector/classifier — but publishes the **full frame** to MQTT so
   Thoth's UI shows the bird in context.  thoth-classify reproduces
   the same crop (via ``sbo_shared.imaging.crop_to_bbox_bytes``)
   before running its own inference, keeping scores comparable.

3. On-device gates that drop an event (low classifier score, no-bird
   detector) must save the bytes the model actually scored — i.e. the
   crop — to the gated archive so the reviewer sees the same thing
   the model did.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from horus.classifier import ClassificationResult
from horus.config import (
    BurstConfig,
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


def test_tick_publishes_full_frame_and_writes_internal_crop(cfg, tmp_path):
    """When the motion gate returns a bbox, the FULL frame is published
    to MQTT (Thoth shows bird-in-context) while a crop sibling is written
    as a side-effect for any on-device gates.  Consumers re-crop from
    the full frame using the bbox_fraction field — see the Option A
    design note in horus/main.py::_publish_flow."""
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

    # The published path must be the full-frame capture, NOT the crop.
    assert published_path == capture_path, (
        "full-frame publish: MQTT payload must carry the uncropped image "
        "so Thoth can display bird-in-context"
    )
    # resolution_override is not passed on the full-frame path (bus uses
    # the configured capture resolution).
    assert kwargs.get("resolution_override") is None

    # The crop sibling still exists on disk as a side-effect of the
    # on-device gate path (detector/classifier run on the crop).  It's
    # not published — storage.prune cleans it up on a later tick.
    crop_path = capture_path.with_name(capture_path.stem + "_crop.jpg")
    assert crop_path.exists(), (
        "internal crop must have been written for the on-device gates"
    )
    with Image.open(crop_path) as crop_img:
        assert crop_img.format == "JPEG"


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
    """Crop failure (unreadable image, etc.) must not drop the motion event.

    Patched at the import site — horus.main does
    ``from sbo_shared.imaging import crop_to_bbox_bytes``, so the
    name to mock is ``horus.main.crop_to_bbox_bytes`` (not the
    origin module's binding).
    """
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
         patch(
             "horus.main.crop_to_bbox_bytes",
             side_effect=RuntimeError("unreadable"),
         ):
        daemon._tick()

    # Fallback: publish the original.
    args, kwargs = daemon.bus.publish_image_event.call_args
    assert args[0] == capture_path
    assert kwargs.get("resolution_override") is None
    # And the cooldown must still advance — this is a real motion event.
    assert daemon._last_event_ts > 0.0
    # No crop sibling should exist when the crop helper raised.
    crop_path = capture_path.with_name(capture_path.stem + "_crop.jpg")
    assert not crop_path.exists()


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


# ---------------------------------------------------------------------------
# Object-detector gate — COCO bird/no-bird replaces the species-classifier
# threshold as the gate decision. Classifier, when still configured, drops
# to label-only duty (attach species string, no threshold gating).
# ---------------------------------------------------------------------------


from horus.config import DetectorConfig  # noqa: E402 — keep test imports grouped
from horus.detector import DetectionResult  # noqa: E402


def _cfg_detector(
    tmp_path: Path,
    *,
    archive_dir: Path | None = None,
    classifier_enabled: bool = False,
    classifier_min_confidence: float = 0.30,
) -> HorusConfig:
    """Detector-enabled config. Classifier optional, archive optional."""
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(interval_s=0.1),
        motion=MotionConfig(cooldown_s=10.0),
        storage=StorageConfig(local_dir=tmp_path),
        classifier=ClassifierConfig(
            enabled=classifier_enabled,
            min_confidence=classifier_min_confidence,
            gated_archive_dir=archive_dir,
        ),
        detector=DetectorConfig(enabled=True, min_score=0.30),
    )


def test_tick_publishes_when_detector_finds_bird(tmp_path):
    """Detector says bird → publish. Detector score + bbox ride on payload."""
    daemon = Daemon(_cfg_detector(tmp_path))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.detector = MagicMock()
    det_bbox = (0.3, 0.2, 0.4, 0.5)
    daemon.detector.detect.return_value = DetectionResult(
        has_bird=True, score=0.82, bbox_fraction=det_bbox
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    assert daemon.bus.publish_image_event.called
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("detector_score") == pytest.approx(0.82)
    assert kwargs.get("detector_bbox_fraction") == det_bbox
    assert daemon._last_event_ts > 0.0


def test_tick_drops_when_detector_finds_no_bird(tmp_path):
    """Detector says no bird → discard capture, advance cooldown, no publish."""
    archive = tmp_path / "gated"
    daemon = Daemon(_cfg_detector(tmp_path, archive_dir=archive))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.detector = MagicMock()
    daemon.detector.detect.return_value = DetectionResult(
        has_bird=False, score=0.05, bbox_fraction=None
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None), \
         patch("horus.main.storage.save_gated_sample") as mock_save:
        daemon._tick()

    daemon.bus.publish_image_event.assert_not_called()
    assert not capture_path.exists(), "capture must be discarded when gated out"
    # Archive gets the detector drop labelled so review can distinguish
    # detector-gated from classifier-gated samples.
    mock_save.assert_called_once()
    _, kwargs = mock_save.call_args
    assert kwargs.get("label") == "detector-no-bird"
    assert kwargs.get("score") == pytest.approx(0.05)
    assert daemon._last_event_ts > 0.0


def test_tick_classifier_floor_does_not_gate_when_threshold_is_zero(tmp_path):
    """With min_confidence=0.0 the classifier floor is disabled — any
    score publishes, even when the detector is live.  This preserves
    the pre-2026-04-22 "label-only" semantics for anyone who wants
    them: just zero out `classifier.min_confidence` in horus.yaml."""
    cfg = _cfg_detector(
        tmp_path, classifier_enabled=True, classifier_min_confidence=0.0
    )
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.detector = MagicMock()
    daemon.detector.detect.return_value = DetectionResult(
        has_bird=True, score=0.75, bbox_fraction=(0.1, 0.1, 0.3, 0.3)
    )
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="junco", confidence=0.08
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    assert daemon.bus.publish_image_event.called
    _, kwargs = daemon.bus.publish_image_event.call_args
    # Species label attached for Thoth:
    assert kwargs.get("bird_label") == "junco"
    assert kwargs.get("bird_score") == pytest.approx(0.08)
    # Detector info also attached:
    assert kwargs.get("detector_score") == pytest.approx(0.75)


def test_tick_classifier_floor_gates_low_score_even_with_detector_bird(tmp_path):
    """Apple-on-the-feeder case: detector says ~bird-shaped (score 0.40),
    classifier can't commit to a species (score 0.05, below floor of 0.10).
    Event must be dropped — otherwise Thoth shows "House Finch · 5%" on
    apple pictures, which is nonsense signal.

    Covers the behavior change from 2026-04-22: previously the classifier
    floor was skipped whenever the detector was enabled, so low-confidence
    species labels passed through unconditionally."""
    archive = tmp_path / "gated"
    cfg = _cfg_detector(
        tmp_path,
        archive_dir=archive,
        classifier_enabled=True,
        classifier_min_confidence=0.10,  # the apple-cutting floor
    )
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.detector = MagicMock()
    daemon.detector.detect.return_value = DetectionResult(
        has_bird=True, score=0.40, bbox_fraction=(0.1, 0.1, 0.9, 0.9)
    )
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="Haemorhous mexicanus (House Finch)", confidence=0.05
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None), \
         patch("horus.main.storage.save_gated_sample") as mock_save:
        daemon._tick()

    daemon.bus.publish_image_event.assert_not_called()
    # Archive drop labelled with the classifier's best-guess species so
    # review can see what the false positive got called.
    mock_save.assert_called_once()
    _, save_kwargs = mock_save.call_args
    assert save_kwargs.get("score") == pytest.approx(0.05)
    assert save_kwargs.get("label") == "Haemorhous mexicanus (House Finch)"
    # Cooldown advances so we don't rapid-fire on the same apple.
    assert daemon._last_event_ts > 0.0


def test_tick_classifier_floor_publishes_when_above_threshold_with_detector(tmp_path):
    """Real bird case: detector confident (0.75), classifier confident
    (0.55, above 0.10 floor) → publish.  Both signals agree."""
    cfg = _cfg_detector(
        tmp_path, classifier_enabled=True, classifier_min_confidence=0.10
    )
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.detector = MagicMock()
    daemon.detector.detect.return_value = DetectionResult(
        has_bird=True, score=0.75, bbox_fraction=(0.1, 0.1, 0.3, 0.3)
    )
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="Junco hyemalis", confidence=0.55
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
    assert kwargs.get("bird_label") == "Junco hyemalis"
    assert kwargs.get("detector_score") == pytest.approx(0.75)


def test_tick_publishes_when_detector_inference_raises(tmp_path):
    """Detector failure falls through to 'publish anyway' (same degraded
    philosophy as the classifier). An inference crash must not black-hole
    a real bird; we'd rather get false positives and sort them at Thoth."""
    daemon = Daemon(_cfg_detector(tmp_path))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    daemon.detector = MagicMock()
    daemon.detector.detect.side_effect = RuntimeError("tflite blew up")

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    assert daemon.bus.publish_image_event.called
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("detector_score") is None
    assert daemon._last_event_ts > 0.0


def test_tick_without_detector_uses_classifier_gate(tmp_path):
    """Backwards-compat sanity: when detector is None and classifier is
    enabled, the classifier still gates on min_confidence (legacy path)."""
    # Classifier enabled, no detector. Classifier returns below-threshold.
    cfg = HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(interval_s=0.1),
        motion=MotionConfig(cooldown_s=10.0),
        storage=StorageConfig(local_dir=tmp_path),
        classifier=ClassifierConfig(enabled=True, min_confidence=0.30),
        detector=DetectorConfig(enabled=False),
    )
    daemon = Daemon(cfg)
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)
    assert daemon.detector is None
    daemon.classifier = MagicMock()
    daemon.classifier.classify.return_value = ClassificationResult(
        species="junco", confidence=0.10
    )

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    daemon.bus.publish_image_event.assert_not_called()
    assert daemon._last_event_ts > 0.0


# ---------------------------------------------------------------------------
# Picamera2 / lores preview path — when Daemon.camera is set, _tick routes
# to _tick_lores(): motion runs on the lores numpy array pulled from the
# preview thread, and stills are captured against the running picamera2
# pipeline (no rpicam subprocess).  When camera is None the daemon must
# fall back to the legacy rpicam-still path.
# ---------------------------------------------------------------------------


import numpy as np  # noqa: E402 — grouped with other test-only imports


def _lores_cfg(
    tmp_path: Path,
    *,
    lores_enabled: bool = True,
    burst_enabled: bool = True,
) -> HorusConfig:
    """Build a HorusConfig with the lores preview stream enabled by default.

    ``burst_enabled`` toggles the session-based capture path.  Most tests
    leave it on (the default production mode); a handful of legacy
    cooldown-discipline tests flip it off to verify the fallback path
    still honors motion.cooldown_s.
    """
    capture = CaptureConfig(
        interval_s=0.1,
        lores_width=320 if lores_enabled else 0,
        lores_height=180 if lores_enabled else 0,
        preview_fps=15.0,
    )
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=capture,
        motion=MotionConfig(cooldown_s=10.0),
        storage=StorageConfig(local_dir=tmp_path),
        burst=BurstConfig(enabled=burst_enabled),
    )


def test_tick_with_lores_camera_uses_array_motion_path(tmp_path):
    """When Daemon.camera is set, _tick must pull latest_lores() and call
    gate.check_array() — NOT the file-based gate.check().  Otherwise the
    spike's whole point (cheap motion on the preview stream) is lost."""
    daemon = Daemon(_lores_cfg(tmp_path))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check_array.return_value = _motion_result(bbox_fraction=None)
    # daemon.camera = the picamera2-backed session.  Test shim mocks it
    # so we exercise the dispatch without needing real libcamera.
    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))
    thumb = np.full((180, 320), 128, dtype=np.uint8)

    daemon.camera = MagicMock()
    daemon.camera.latest_lores.return_value = (1.0, thumb)
    daemon.camera.capture.return_value = capture_path

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    daemon.gate.check_array.assert_called_once()
    daemon.gate.check.assert_not_called()  # file-based gate must NOT run
    # Capture happens through the persistent session, not rpicam.
    daemon.camera.capture.assert_called_once_with(capture_path)
    assert daemon.bus.publish_image_event.called
    assert daemon._last_event_ts > 0.0


def test_tick_with_lores_skips_when_preview_not_ready(tmp_path):
    """Preview thread hasn't produced a frame yet → skip tick silently.
    No capture, no publish, no cooldown bump.  This happens on the first
    few ticks after start while the ISP warms up."""
    daemon = Daemon(_lores_cfg(tmp_path))
    daemon.bus = MagicMock()
    daemon.gate = MagicMock()
    daemon.camera = MagicMock()
    daemon.camera.latest_lores.return_value = None

    daemon._tick()

    daemon.gate.check_array.assert_not_called()
    daemon.camera.capture.assert_not_called()
    daemon.bus.publish_image_event.assert_not_called()
    assert daemon._last_event_ts == 0.0


def test_tick_with_lores_skips_capture_when_no_motion(tmp_path):
    """No motion on the lores frame → no full-res capture at all.
    This is the bandwidth win: we don't burn a 4MB capture per tick
    when nothing's happening."""
    daemon = Daemon(_lores_cfg(tmp_path))
    daemon.bus = MagicMock()
    daemon.gate = MagicMock()
    no_motion = MagicMock()
    no_motion.motion = False
    no_motion.changed_fraction = 0.001
    no_motion.bbox_fraction = None
    daemon.gate.check_array.return_value = no_motion
    daemon.camera = MagicMock()
    daemon.camera.latest_lores.return_value = (
        1.0, np.full((180, 320), 128, dtype=np.uint8)
    )

    daemon._tick()

    daemon.camera.capture.assert_not_called()
    daemon.bus.publish_image_event.assert_not_called()


def test_tick_with_lores_respects_cooldown_when_burst_disabled(tmp_path):
    """Motion on the lores frame during cooldown → no capture, no publish.

    Only applies with ``burst.enabled=False`` (the legacy singleton
    path). In burst mode (default) the cooldown gate is intentionally
    bypassed so frames from the same motion session can cluster — see
    the companion ``test_tick_with_lores_ignores_cooldown_when_burst_enabled``.
    """
    daemon = Daemon(_lores_cfg(tmp_path, burst_enabled=False))
    daemon.bus = MagicMock()
    daemon.gate = MagicMock()
    daemon.gate.check_array.return_value = _motion_result(bbox_fraction=None)
    daemon.camera = MagicMock()
    daemon.camera.latest_lores.return_value = (
        1.0, np.full((180, 320), 128, dtype=np.uint8)
    )
    # Pretend we just published a frame.
    daemon._last_event_ts = time.monotonic()

    daemon._tick()

    daemon.camera.capture.assert_not_called()
    daemon.bus.publish_image_event.assert_not_called()


def test_tick_with_lores_ignores_cooldown_when_burst_enabled(tmp_path):
    """Regression: burst capture intentionally skips the per-tick
    cooldown so frames from the same motion session can cluster into
    a burst. Setting ``_last_event_ts`` to "just now" must NOT block
    the next publish when ``burst.enabled=True`` (the default)."""
    daemon = Daemon(_lores_cfg(tmp_path, burst_enabled=True))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check_array.return_value = _motion_result(bbox_fraction=None)
    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))
    daemon.camera = MagicMock()
    daemon.camera.latest_lores.return_value = (
        1.0, np.full((180, 320), 128, dtype=np.uint8)
    )
    daemon.camera.capture.return_value = capture_path
    # Would normally block a legacy cooldown — must NOT block in burst mode.
    daemon._last_event_ts = time.monotonic()

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    daemon.camera.capture.assert_called_once()
    daemon.bus.publish_image_event.assert_called_once()


def test_tick_with_lores_capture_error_aborts_cleanly(tmp_path):
    """If the persistent picamera2 session fails mid-capture (rare, but
    possible on stream reconfigures), the tick logs and returns —
    cooldown does NOT advance so the next real motion still publishes."""
    daemon = Daemon(_lores_cfg(tmp_path))
    daemon.bus = MagicMock()
    daemon.gate = MagicMock()
    daemon.gate.check_array.return_value = _motion_result(bbox_fraction=None)
    daemon.camera = MagicMock()
    daemon.camera.latest_lores.return_value = (
        1.0, np.full((180, 320), 128, dtype=np.uint8)
    )
    # camera.CameraError lives in the horus.camera module — import via
    # horus.main.camera to use the same reference the daemon sees.
    from horus import camera as camera_module
    daemon.camera.capture.side_effect = camera_module.CameraError("sensor reset")

    capture_path = tmp_path / "cap.jpg"
    with patch("horus.main.storage.next_capture_path", return_value=capture_path):
        daemon._tick()

    daemon.bus.publish_image_event.assert_not_called()
    assert daemon._last_event_ts == 0.0


def test_tick_routes_to_legacy_when_camera_is_none(tmp_path):
    """Daemon.camera = None (lores disabled, or start failed) → fall
    back to rpicam-still path unchanged.  This is the zero-downtime
    rollout invariant: a broken picamera2 must not take the station
    offline."""
    daemon = Daemon(_lores_cfg(tmp_path, lores_enabled=False))
    assert daemon.camera is None

    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    capture_path = tmp_path / "cap.jpg"
    _write_real_jpeg(capture_path, (1000, 1000))

    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture") as mock_capture, \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()

    # rpicam-still free function called with (path, CaptureConfig) —
    # this is the legacy invocation signature.
    mock_capture.assert_called_once()
    assert daemon.gate.check.called
    assert daemon.bus.publish_image_event.called


def test_maybe_start_camera_noop_when_lores_disabled(tmp_path):
    """Configs without lores set must not open picamera2 at all —
    preserves the "pure rpicam-still" deployment story."""
    daemon = Daemon(_lores_cfg(tmp_path, lores_enabled=False))
    daemon._maybe_start_camera()
    assert daemon.camera is None


def test_maybe_start_camera_falls_back_on_start_failure(tmp_path):
    """Camera.start() raising CameraError must NOT crash run() — the
    daemon leaves self.camera=None and _tick routes to the legacy
    rpicam path.  This is what keeps a broken picamera2 wheel from
    taking the station offline."""
    daemon = Daemon(_lores_cfg(tmp_path))

    with patch("horus.main.Camera") as mock_cls:
        from horus import camera as camera_module
        mock_instance = MagicMock()
        mock_instance.start.side_effect = camera_module.CameraError("no libcamera")
        mock_cls.return_value = mock_instance
        daemon._maybe_start_camera()

    assert daemon.camera is None


def test_maybe_start_camera_succeeds_when_lores_configured(tmp_path):
    """Happy path: lores set + Camera.start() returns cleanly →
    daemon holds a live Camera instance and _tick will route to the
    lores path."""
    daemon = Daemon(_lores_cfg(tmp_path))

    with patch("horus.main.Camera") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        daemon._maybe_start_camera()

    assert daemon.camera is mock_instance
    mock_instance.start.assert_called_once()


# ---------------------------------------------------------------------------
# Session-based burst capture
# ---------------------------------------------------------------------------
#
# PR-A lifecycle contract:
#
#  * First publish in a fresh burst → burst_seq=1, fresh burst_id.
#  * Subsequent publish within idle_close_s → same burst_id, seq+=1.
#  * Gap > idle_close_s → new burst_id, seq resets to 1.
#  * Elapsed > max_duration_s since the burst started → new burst_id,
#    even if frames were still arriving under idle_close_s cadence.
#  * Publish failure (broker drop, exception) → burst state DOES NOT
#    advance, so the next successful publish gets a clean seq=1 rather
#    than leaving a gap in the sequence Thoth sees.
#  * ``burst.enabled=False`` → publish_image_event receives
#    ``burst_id=None`` and ``burst_seq=None`` (absence is the legacy
#    signal to downstream) AND the legacy cooldown gate re-activates.


def _burst_cfg(
    tmp_path: Path,
    *,
    enabled: bool = True,
    idle_close_s: float = 3.0,
    max_duration_s: float = 30.0,
    cooldown_s: float = 10.0,
) -> HorusConfig:
    """HorusConfig with the burst knobs exposed for per-test tuning."""
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(interval_s=0.1),
        motion=MotionConfig(cooldown_s=cooldown_s),
        storage=StorageConfig(local_dir=tmp_path),
        burst=BurstConfig(
            enabled=enabled,
            idle_close_s=idle_close_s,
            max_duration_s=max_duration_s,
        ),
    )


def _run_publishing_tick(daemon: Daemon, tmp_path: Path, idx: int = 0) -> None:
    """Drive one _tick that is guaranteed to reach publish_image_event.

    Each call uses a distinct capture path so the crop-discard code
    path doesn't collide across ticks in a single test.
    """
    capture_path = tmp_path / f"cap-{idx}.jpg"
    _write_real_jpeg(capture_path, (500, 500))
    with patch("horus.main.storage.next_capture_path", return_value=capture_path), \
         patch("horus.main.camera.capture"), \
         patch("horus.main.camera.read_af_fields", return_value=None):
        daemon._tick()


def test_burst_first_publish_starts_fresh_session(tmp_path):
    """Cold daemon → first successful publish opens burst #1 with seq=1."""
    daemon = Daemon(_burst_cfg(tmp_path))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    _run_publishing_tick(daemon, tmp_path, idx=0)

    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("burst_seq") == 1
    assert kwargs.get("burst_id") is not None
    # Daemon state committed to match what we published.
    assert daemon._burst_seq == 1
    assert daemon._burst_id == kwargs.get("burst_id")


def test_burst_continuation_shares_id_and_increments_seq(tmp_path):
    """Two successful publishes back-to-back (well within idle_close_s
    of 3.0s) → same burst_id, seq=1 then seq=2."""
    daemon = Daemon(_burst_cfg(tmp_path))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    _run_publishing_tick(daemon, tmp_path, idx=0)
    first_id = daemon._burst_id
    first_seq = daemon._burst_seq

    _run_publishing_tick(daemon, tmp_path, idx=1)
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("burst_id") == first_id, (
        "within idle_close_s the burst_id must be stable — Thoth uses this "
        "as the foreign key for grouping"
    )
    assert kwargs.get("burst_seq") == first_seq + 1


def test_burst_idle_close_gap_starts_new_session(tmp_path):
    """Gap longer than idle_close_s between publishes → new burst_id, seq=1.

    Simulated by backdating ``_burst_last_frame_ts`` past the idle
    window.  Time patching is avoided because the publish path calls
    ``time.monotonic()`` multiple times and brittle call counts are
    hostile to future-proofing.
    """
    daemon = Daemon(_burst_cfg(tmp_path, idle_close_s=3.0))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    _run_publishing_tick(daemon, tmp_path, idx=0)
    first_id = daemon._burst_id

    # Jump the clock forward by pushing last-frame into the past.
    daemon._burst_last_frame_ts -= 100.0  # way past idle_close_s=3.0

    _run_publishing_tick(daemon, tmp_path, idx=1)
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("burst_id") != first_id, (
        "a gap past idle_close_s must roll to a new burst — Thoth treats "
        "these as distinct motion sessions (separate tiles)"
    )
    assert kwargs.get("burst_seq") == 1


def test_burst_max_duration_forces_new_session_even_when_continuous(tmp_path):
    """Even if frames arrive under idle_close_s cadence, once the burst
    has been running longer than max_duration_s the next frame must
    start a fresh burst.  Safety rail against "wind-swaying feeder"
    or "cat camped on pole" producing a single 4-hour burst."""
    daemon = Daemon(_burst_cfg(tmp_path, idle_close_s=3.0, max_duration_s=30.0))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    _run_publishing_tick(daemon, tmp_path, idx=0)
    first_id = daemon._burst_id

    # Backdate burst_started_at past max_duration, keep last_frame_ts
    # fresh so the idle check would otherwise say "continuation".
    daemon._burst_started_at -= 100.0

    _run_publishing_tick(daemon, tmp_path, idx=1)
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("burst_id") != first_id
    assert kwargs.get("burst_seq") == 1


def test_burst_failed_publish_does_not_commit_state(tmp_path):
    """A dropped publish must leave burst state unchanged so the next
    successful publish starts from seq=1 (no seq-gap holes in Thoth's
    view) rather than seq=2 against a burst_id the consumer never saw."""
    daemon = Daemon(_burst_cfg(tmp_path))
    daemon.bus = MagicMock()
    # First call: drop (no PUBACK). Second call: succeed.
    daemon.bus.publish_image_event.side_effect = [False, True]
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    _run_publishing_tick(daemon, tmp_path, idx=0)
    assert daemon._burst_id is None, (
        "burst state must stay uninitialized until a publish actually acks"
    )
    assert daemon._burst_seq == 0

    _run_publishing_tick(daemon, tmp_path, idx=1)
    # Grab the SECOND call's kwargs (the successful one).
    second_call = daemon.bus.publish_image_event.call_args_list[1]
    assert second_call.kwargs.get("burst_seq") == 1
    assert second_call.kwargs.get("burst_id") is not None
    assert daemon._burst_id == second_call.kwargs.get("burst_id")


def test_burst_failed_publish_mid_session_keeps_existing_burst(tmp_path):
    """Drop mid-burst: burst state stays anchored on the last acked
    frame, so the retry lands as seq=prev+1 with the same burst_id —
    NOT seq=prev+2 with a gap, and NOT a brand-new burst_id."""
    daemon = Daemon(_burst_cfg(tmp_path))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.side_effect = [True, False, True]
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    _run_publishing_tick(daemon, tmp_path, idx=0)
    burst_id = daemon._burst_id
    assert daemon._burst_seq == 1

    # Failed publish attempt.
    _run_publishing_tick(daemon, tmp_path, idx=1)
    # State must not have advanced.
    assert daemon._burst_id == burst_id
    assert daemon._burst_seq == 1

    # Third publish (success).
    _run_publishing_tick(daemon, tmp_path, idx=2)
    third_call = daemon.bus.publish_image_event.call_args_list[2]
    # Continuation: same burst_id, seq=2 (not 3 — the drop didn't count).
    assert third_call.kwargs.get("burst_id") == burst_id
    assert third_call.kwargs.get("burst_seq") == 2
    assert daemon._burst_seq == 2


def test_burst_disabled_omits_burst_fields_and_honors_cooldown(tmp_path):
    """Legacy fallback: ``burst.enabled=False`` → publish receives
    ``burst_id=None``/``burst_seq=None`` (so Thoth treats each frame as
    a singleton), and the motion.cooldown_s gate suppresses back-to-
    back captures in the same cooldown window."""
    daemon = Daemon(_burst_cfg(tmp_path, enabled=False, cooldown_s=10.0))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    # First publish lands.
    _run_publishing_tick(daemon, tmp_path, idx=0)
    first_call = daemon.bus.publish_image_event.call_args_list[0]
    assert first_call.kwargs.get("burst_id") is None
    assert first_call.kwargs.get("burst_seq") is None
    assert daemon._burst_id is None  # no burst bookkeeping when disabled
    # Cooldown state bumped on success (legacy invariant).
    assert daemon._last_event_ts > 0.0

    # Second publish attempt immediately after — cooldown must suppress.
    before_count = daemon.bus.publish_image_event.call_count
    _run_publishing_tick(daemon, tmp_path, idx=1)
    assert daemon.bus.publish_image_event.call_count == before_count, (
        "legacy cooldown must still gate when burst.enabled=False"
    )


def test_burst_duration_beats_idle_when_both_conditions_tight(tmp_path):
    """Precedence guard: when a burst is *almost* over on max_duration_s
    but the idle-close window would still say "continuation", the
    duration cap must win — i.e. the AND in the continuation predicate
    means both conditions have to hold.  Regression fence against a
    future refactor that might, say, flip the predicate to OR."""
    daemon = Daemon(_burst_cfg(tmp_path, idle_close_s=3.0, max_duration_s=30.0))
    daemon.bus = MagicMock()
    daemon.bus.publish_image_event.return_value = True
    daemon.bus.dropped_publishes = 0
    daemon.gate = MagicMock()
    daemon.gate.check.return_value = _motion_result(bbox_fraction=None)

    _run_publishing_tick(daemon, tmp_path, idx=0)
    first_id = daemon._burst_id

    # Backdate started_at past max_duration (roll should fire) but keep
    # last_frame_ts fresh (idle check would otherwise say continuation).
    daemon._burst_started_at -= 100.0
    # _burst_last_frame_ts left alone — within idle_close_s of "now".

    _run_publishing_tick(daemon, tmp_path, idx=1)
    _, kwargs = daemon.bus.publish_image_event.call_args
    assert kwargs.get("burst_id") != first_id, (
        "duration cap must win over idle-close continuation — both "
        "conditions have to hold for continuation to be picked"
    )
    assert kwargs.get("burst_seq") == 1


def test_make_burst_id_is_unique_across_calls(tmp_path):
    """Regression guard against a bug where two bursts started in the
    same ms (plausible at 15fps tick cadence) would collide on burst_id.
    The 4-hex random suffix provides the disambiguation."""
    from horus.main import _make_burst_id

    ids = {_make_burst_id("horus-test") for _ in range(200)}
    assert len(ids) == 200, "burst_id must be unique across rapid-fire starts"
    for bid in ids:
        # Sanity: format is {station}-{wall_ms}-{4hex}
        assert bid.startswith("horus-test-")
        tail = bid.rsplit("-", 1)[1]
        assert len(tail) == 4
        int(tail, 16)  # raises if not hex
