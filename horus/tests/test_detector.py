"""Tests for horus.detector — BirdDetector post-processing.

We stub the TFLite interpreter with a MagicMock and fake output tensors
so these run on any dev laptop without the runtime wheel. What we're
testing is the detector's *decision logic*, not tflite: label
resolution, shape-based output identification, bird-class filtering,
bbox clamping, and the bird / no-bird verdict.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from horus.detector import BirdDetector, DetectionResult


COCO_LABELS = (
    "person\nbicycle\ncar\nmotorcycle\nairplane\nbus\ntrain\ntruck\nboat\n"
    "traffic light\nfire hydrant\nstop sign\nparking meter\nbench\nbird\ncat\n"
    "dog\nhorse\nsheep\ncow\n"
)


def _write_labels(path: Path, text: str = COCO_LABELS) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _jpeg_bytes(size: tuple[int, int] = (320, 320)) -> bytes:
    """Produce a tiny real JPEG — BirdDetector.detect decodes via Pillow."""
    buf = BytesIO()
    Image.new("RGB", size, color=(100, 100, 100)).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _fake_interpreter(
    *,
    input_shape=(1, 320, 320, 3),
    input_dtype=np.uint8,
    boxes=None,
    classes=None,
    scores=None,
    num=None,
):
    """Build a MagicMock interpreter that returns the given detections.

    Output indices are assigned:
      boxes=0, classes=1, scores=2, num=3
    — chosen so ``_resolve_output_indices`` picks them by shape the way
    it would for a real EfficientDet-Lite0 export.
    """
    interp = MagicMock()
    input_details = [{"index": 99, "shape": np.array(input_shape), "dtype": input_dtype}]
    output_details = [
        {"index": 0, "shape": np.array([1, 25, 4])},  # boxes
        {"index": 1, "shape": np.array([1, 25])},     # classes
        {"index": 2, "shape": np.array([1, 25])},     # scores
        {"index": 3, "shape": np.array([1])},         # num_detections
    ]
    interp.get_input_details.return_value = input_details
    interp.get_output_details.return_value = output_details

    # Default: no detections.
    boxes = np.zeros((1, 25, 4), dtype=np.float32) if boxes is None else boxes
    classes = np.zeros((1, 25), dtype=np.float32) if classes is None else classes
    scores = np.zeros((1, 25), dtype=np.float32) if scores is None else scores
    num = np.array([0], dtype=np.float32) if num is None else num

    tensors = {0: boxes, 1: classes, 2: scores, 3: num}
    interp.get_tensor.side_effect = lambda idx: tensors[idx]
    interp.set_tensor = MagicMock()
    interp.invoke = MagicMock()
    interp.allocate_tensors = MagicMock()
    return interp


@pytest.fixture
def labels_file(tmp_path: Path) -> Path:
    return _write_labels(tmp_path / "coco_labels.txt")


def _build_detector(
    tmp_path: Path, labels_file: Path, interpreter, min_score: float = 0.30
) -> BirdDetector:
    """Bypass the real Interpreter import + construct a BirdDetector with our fake."""
    model_path = tmp_path / "fake.tflite"
    model_path.write_bytes(b"")  # path must exist for the Path repr; contents unused
    with patch("builtins.__import__", side_effect=ImportError) as _, \
         patch.object(BirdDetector, "__init__", lambda self, *a, **k: None):
        det = BirdDetector.__new__(BirdDetector)
    # Manually set everything __init__ would have. This lets us test the
    # public surface without spinning up a real tflite runtime.
    det._interpreter = interpreter
    det._input_index = 99
    det._input_shape = np.array([1, 320, 320, 3])
    det._input_dtype = np.uint8
    det._output_indices = {"boxes": 0, "classes": 1, "scores": 2, "num": 3}
    det._labels = [
        line.strip() for line in labels_file.read_text().splitlines()
    ]
    det._bird_class_indices = BirdDetector._resolve_bird_indices(
        det._labels, labels_file
    )
    det._min_score = float(min_score)
    return det


# --- label / output resolution --------------------------------------------


def test_resolve_bird_indices_finds_bird(labels_file):
    labels = [line.strip() for line in labels_file.read_text().splitlines()]
    indices = BirdDetector._resolve_bird_indices(labels, labels_file)
    # COCO class "bird" is index 14 in the fixture above.
    assert 14 in indices


def test_resolve_bird_indices_is_case_insensitive(tmp_path):
    labels_file = _write_labels(tmp_path / "mixed.txt", "person\nBIRD\ncat\n")
    labels = ["person", "BIRD", "cat"]
    indices = BirdDetector._resolve_bird_indices(labels, labels_file)
    assert indices == (1,)


def test_resolve_bird_indices_raises_on_missing_bird(tmp_path):
    """A labels file with no bird class would silently return 'no bird'
    on every frame. Fail loudly at startup instead."""
    labels_file = _write_labels(tmp_path / "no_bird.txt", "person\ncat\ndog\n")
    labels = ["person", "cat", "dog"]
    with pytest.raises(RuntimeError, match="no 'bird' entry"):
        BirdDetector._resolve_bird_indices(labels, labels_file)


def test_resolve_output_indices_by_shape():
    """Order in get_output_details is NOT guaranteed — resolve by shape.
    Here we hand 'boxes' in last position to confirm shape-based mapping."""
    details = [
        {"index": 10, "shape": np.array([1, 25])},     # classes
        {"index": 11, "shape": np.array([1])},         # num
        {"index": 12, "shape": np.array([1, 25])},     # scores
        {"index": 13, "shape": np.array([1, 25, 4])},  # boxes
    ]
    roles = BirdDetector._resolve_output_indices(details, Path("m.tflite"))
    assert roles["boxes"] == 13
    assert roles["num"] == 11
    # Two (1, N) outputs: classes first by appearance, scores second.
    assert roles["classes"] == 10
    assert roles["scores"] == 12


def test_resolve_output_indices_raises_when_layout_mismatch():
    details = [
        {"index": 0, "shape": np.array([1, 25, 4])},  # only boxes
    ]
    # Which field is reported first depends on iteration order; any of
    # the four is acceptable — we just want a useful failure.
    with pytest.raises(RuntimeError, match="missing a '.+' output"):
        BirdDetector._resolve_output_indices(details, Path("m.tflite"))


# --- detect() post-processing ---------------------------------------------


def test_detect_returns_no_bird_when_all_detections_below_threshold(tmp_path, labels_file):
    # One bird-class detection, score 0.10 < threshold 0.30.
    boxes = np.zeros((1, 25, 4), dtype=np.float32)
    boxes[0, 0] = [0.1, 0.2, 0.5, 0.7]  # ymin, xmin, ymax, xmax
    classes = np.zeros((1, 25), dtype=np.float32)
    classes[0, 0] = 14  # bird
    scores = np.zeros((1, 25), dtype=np.float32)
    scores[0, 0] = 0.10
    num = np.array([1], dtype=np.float32)

    det = _build_detector(
        tmp_path,
        labels_file,
        _fake_interpreter(boxes=boxes, classes=classes, scores=scores, num=num),
    )
    result = det.detect(_jpeg_bytes())

    assert result == DetectionResult(has_bird=False, score=0.0, bbox_fraction=None)


def test_detect_returns_bird_when_above_threshold(tmp_path, labels_file):
    boxes = np.zeros((1, 25, 4), dtype=np.float32)
    # ymin=0.20, xmin=0.30, ymax=0.80, xmax=0.90 → (x=0.30, y=0.20, w=0.60, h=0.60)
    boxes[0, 0] = [0.20, 0.30, 0.80, 0.90]
    classes = np.zeros((1, 25), dtype=np.float32)
    classes[0, 0] = 14  # bird
    scores = np.zeros((1, 25), dtype=np.float32)
    scores[0, 0] = 0.72
    num = np.array([1], dtype=np.float32)

    det = _build_detector(
        tmp_path,
        labels_file,
        _fake_interpreter(boxes=boxes, classes=classes, scores=scores, num=num),
    )
    result = det.detect(_jpeg_bytes())

    assert result.has_bird is True
    assert result.score == pytest.approx(0.72)
    assert result.bbox_fraction is not None
    x, y, w, h = result.bbox_fraction
    assert (x, y, w, h) == pytest.approx((0.30, 0.20, 0.60, 0.60), abs=1e-6)


def test_detect_ignores_non_bird_classes(tmp_path, labels_file):
    """A confident 'person' detection must NOT trip the bird gate."""
    boxes = np.zeros((1, 25, 4), dtype=np.float32)
    boxes[0, 0] = [0.1, 0.1, 0.5, 0.5]
    classes = np.zeros((1, 25), dtype=np.float32)
    classes[0, 0] = 0  # person
    scores = np.zeros((1, 25), dtype=np.float32)
    scores[0, 0] = 0.95  # very confident person
    num = np.array([1], dtype=np.float32)

    det = _build_detector(
        tmp_path,
        labels_file,
        _fake_interpreter(boxes=boxes, classes=classes, scores=scores, num=num),
    )
    result = det.detect(_jpeg_bytes())

    assert result.has_bird is False


def test_detect_picks_highest_confidence_bird(tmp_path, labels_file):
    """When multiple bird detections are present, the best one wins."""
    boxes = np.zeros((1, 25, 4), dtype=np.float32)
    boxes[0, 0] = [0.1, 0.1, 0.3, 0.3]  # small bbox, low conf
    boxes[0, 1] = [0.4, 0.4, 0.8, 0.8]  # big bbox, high conf
    classes = np.zeros((1, 25), dtype=np.float32)
    classes[0, 0] = 14
    classes[0, 1] = 14
    scores = np.zeros((1, 25), dtype=np.float32)
    scores[0, 0] = 0.40
    scores[0, 1] = 0.85  # winner
    num = np.array([2], dtype=np.float32)

    det = _build_detector(
        tmp_path,
        labels_file,
        _fake_interpreter(boxes=boxes, classes=classes, scores=scores, num=num),
    )
    result = det.detect(_jpeg_bytes())

    assert result.score == pytest.approx(0.85)
    # bbox comes from the second detection (xmin=0.4, ymin=0.4, ...).
    x, y, w, h = result.bbox_fraction
    assert (x, y, w, h) == pytest.approx((0.4, 0.4, 0.4, 0.4), abs=1e-6)


def test_detect_respects_num_detections_padding(tmp_path, labels_file):
    """Models zero-pad the tail of the boxes array past num_detections.
    A stale high-score bird in the padding must not fire the gate."""
    boxes = np.zeros((1, 25, 4), dtype=np.float32)
    classes = np.zeros((1, 25), dtype=np.float32)
    scores = np.zeros((1, 25), dtype=np.float32)
    # Only slot 0 is valid. Slot 5 is leftover from a previous frame.
    scores[0, 0] = 0.10  # below threshold → no bird detection at slot 0
    classes[0, 5] = 14
    scores[0, 5] = 0.99  # would-be stale bird past num
    num = np.array([1], dtype=np.float32)

    det = _build_detector(
        tmp_path,
        labels_file,
        _fake_interpreter(boxes=boxes, classes=classes, scores=scores, num=num),
    )
    result = det.detect(_jpeg_bytes())

    assert result.has_bird is False, "must not read past num_detections"


def test_detect_clamps_out_of_range_bbox(tmp_path, labels_file):
    """EfficientDet occasionally emits coordinates slightly outside [0, 1]
    for edge-of-frame detections. Must clamp before returning, else a
    downstream crop gets a negative origin and blows up."""
    boxes = np.zeros((1, 25, 4), dtype=np.float32)
    # ymin=-0.05, xmin=-0.10, ymax=1.08, xmax=1.02 — anchor slop
    boxes[0, 0] = [-0.05, -0.10, 1.08, 1.02]
    classes = np.zeros((1, 25), dtype=np.float32)
    classes[0, 0] = 14
    scores = np.zeros((1, 25), dtype=np.float32)
    scores[0, 0] = 0.50
    num = np.array([1], dtype=np.float32)

    det = _build_detector(
        tmp_path,
        labels_file,
        _fake_interpreter(boxes=boxes, classes=classes, scores=scores, num=num),
    )
    result = det.detect(_jpeg_bytes())

    assert result.has_bird is True
    x, y, w, h = result.bbox_fraction
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert x + w <= 1.0 + 1e-6
    assert y + h <= 1.0 + 1e-6
