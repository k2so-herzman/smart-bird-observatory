"""Tests for horus.classifier — BirdClassifier.

The real ``tflite_runtime`` wheel only ships to the Pi, so these tests
substitute a hand-rolled fake Interpreter that mimics the surface the
classifier touches: allocate_tensors, get_input_details,
get_output_details, set_tensor, invoke, get_tensor.

Two contracts under test:
* ``classify`` picks the top-1 label by argmax and rescales quantized
  output into ``[0, 1]``.
* Label-list shape mismatches, input-rank mismatches, and a missing
  tflite-runtime install fail loudly at construction, not deep inside
  the hot loop.
"""

from __future__ import annotations

import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _install_fake_tflite(
    *,
    input_shape=(1, 224, 224, 3),
    input_dtype=np.uint8,
    output=None,  # np.ndarray, optional
) -> None:
    """Stub ``tflite_runtime.interpreter`` with a deterministic fake.

    The fake Interpreter ignores the model file on disk and returns
    whatever ``output`` the test asks for. Callers wanting to test
    argmax / rescaling just hand a tiny 1D vector.
    """
    class _FakeInterpreter:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def allocate_tensors(self) -> None:
            pass

        def get_input_details(self):
            return [{"index": 0, "shape": np.array(input_shape), "dtype": input_dtype}]

        def get_output_details(self):
            dtype = output.dtype if output is not None else np.uint8
            shape = output.shape if output is not None else (1, 3)
            return [{"index": 0, "shape": np.array(shape), "dtype": dtype}]

        def set_tensor(self, index, value):
            self._last_input = value

        def invoke(self):
            pass

        def get_tensor(self, index):
            # Return a batched (1, N) output so classify's [0] index works.
            if output is None:
                return np.array([[200, 20, 10]], dtype=np.uint8)
            return np.expand_dims(output, axis=0)

    module = types.ModuleType("tflite_runtime")
    interp_module = types.ModuleType("tflite_runtime.interpreter")
    interp_module.Interpreter = _FakeInterpreter
    module.interpreter = interp_module
    sys.modules["tflite_runtime"] = module
    sys.modules["tflite_runtime.interpreter"] = interp_module


def _small_jpeg_bytes(color=(100, 100, 100), size=(64, 64)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _labels_file(tmp_path: Path, labels: list[str]) -> Path:
    p = tmp_path / "labels.txt"
    p.write_text("\n".join(labels), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _cleanup_tflite_stubs():
    yield
    # Purge the fake tflite module after each test so stubs don't leak.
    for name in ("tflite_runtime", "tflite_runtime.interpreter"):
        sys.modules.pop(name, None)


def test_classify_picks_top_label_from_quantized_output(tmp_path):
    """Quantized uint8 output with class 0 = 200, class 1 = 20 → top is class 0
    with confidence 200/255."""
    _install_fake_tflite(output=np.array([200, 20, 10], dtype=np.uint8))
    from horus.classifier import BirdClassifier

    clf = BirdClassifier(
        tmp_path / "fake.tflite",
        _labels_file(tmp_path, ["american goldfinch", "house sparrow", "dark-eyed junco"]),
    )
    result = clf.classify(_small_jpeg_bytes())
    assert result.species == "american goldfinch"
    # 200 / 255 ≈ 0.784
    assert result.confidence == pytest.approx(200 / 255, rel=1e-6)


def test_classify_rescales_quantized_output_to_unit_interval(tmp_path):
    """Quantized output should always land in [0, 1] so threshold config
    is model-invariant."""
    _install_fake_tflite(output=np.array([255, 0, 0], dtype=np.uint8))
    from horus.classifier import BirdClassifier

    clf = BirdClassifier(
        tmp_path / "fake.tflite",
        _labels_file(tmp_path, ["a", "b", "c"]),
    )
    result = clf.classify(_small_jpeg_bytes())
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == pytest.approx(1.0)


def test_classify_handles_float_output(tmp_path):
    """Float-output model path: no rescaling, direct pass-through."""
    _install_fake_tflite(
        input_dtype=np.float32,
        output=np.array([0.2, 0.7, 0.1], dtype=np.float32),
    )
    from horus.classifier import BirdClassifier

    clf = BirdClassifier(
        tmp_path / "fake.tflite",
        _labels_file(tmp_path, ["robin", "sparrow", "finch"]),
    )
    result = clf.classify(_small_jpeg_bytes())
    assert result.species == "sparrow"
    assert result.confidence == pytest.approx(0.7)


def test_classify_falls_back_when_top_index_has_no_label(tmp_path, caplog):
    """Labels file shorter than the model's output classes must not crash —
    we emit ``class_N`` and log a warning. A mismatch in production means
    someone shipped stale labels; we want to notice, not stall."""
    _install_fake_tflite(output=np.array([10, 10, 200], dtype=np.uint8))
    from horus.classifier import BirdClassifier

    clf = BirdClassifier(
        tmp_path / "fake.tflite",
        _labels_file(tmp_path, ["a", "b"]),  # only 2 labels for 3-class output
    )
    caplog.set_level("WARNING", logger="horus.classifier")
    result = clf.classify(_small_jpeg_bytes())
    assert result.species == "class_2"
    assert any("no matching label" in r.message for r in caplog.records)


def test_classify_strips_whitespace_from_labels(tmp_path):
    """Labels file may have trailing whitespace from a broken export — we
    mustn't carry it into MQTT payloads or downstream string compares."""
    _install_fake_tflite(output=np.array([0, 200, 0], dtype=np.uint8))
    from horus.classifier import BirdClassifier

    clf = BirdClassifier(
        tmp_path / "fake.tflite",
        _labels_file(tmp_path, ["  robin  ", "  house sparrow  ", "finch"]),
    )
    result = clf.classify(_small_jpeg_bytes())
    assert result.species == "house sparrow"


def test_constructor_rejects_wrong_input_rank(tmp_path):
    """A 2D input tensor (no batch+channel dim) means the wrong model
    artifact — fail at boot, not inside ``classify``."""
    _install_fake_tflite(input_shape=(224, 224))
    from horus.classifier import BirdClassifier

    with pytest.raises(RuntimeError, match="unsupported input shape"):
        BirdClassifier(tmp_path / "fake.tflite", _labels_file(tmp_path, ["a"]))
