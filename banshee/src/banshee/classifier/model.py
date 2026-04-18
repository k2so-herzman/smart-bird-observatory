"""Pluggable image classifier.

The :class:`Classifier` protocol isolates the model from the pipeline.
Implementations receive raw image bytes (whatever format the station
emitted) and return a :class:`ClassificationResult`. This lets the
pipeline be tested and deployed without pinning to a specific model
runtime.

Two implementations ship in this module:

* :class:`DummyClassifier` — returns ``("unclassified", 0.0)``. Used by
  default so operators can deploy the classifier service and confirm
  the pipeline end-to-end before a real model artifact is staged. The
  label ``"unclassified"`` is a deliberate sentinel that downstream
  filters (e.g. Telegram confidence gate) will reject on confidence
  alone — no risk of spurious alerts.

* :class:`TFLiteClassifier` — loads a TensorFlow Lite model from disk.
  The ``tflite-runtime`` import is lazy so the rest of the package
  imports cleanly in environments (CI, dev laptops) where the wheel
  is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationResult:
    """Output of a single classifier invocation.

    Attributes
    ----------
    species:
        Predicted label. Use ``"unclassified"`` when no model is loaded
        (see :class:`DummyClassifier`). Never ``None`` — a NULL species
        in the DB means "classifier hasn't run yet," which is a distinct
        state from "classifier ran, had no model."
    confidence:
        Softmax-style confidence in ``[0.0, 1.0]``. Downstream
        notification gates compare against
        ``NotifyConfig.telegram_min_confidence``.
    """

    species: str
    confidence: float


class Classifier(Protocol):
    """Contract every model implementation must satisfy.

    Implementations must be **thread-safe for concurrent classify
    calls** or the :class:`~.worker.ClassifierWorker` must call them
    serially. The shipped worker today runs single-threaded, so a
    thread-unsafe model (e.g. a bare TFLite interpreter) is fine.
    """

    def classify(self, image_bytes: bytes) -> ClassificationResult:
        """Run inference on a single image.

        Parameters
        ----------
        image_bytes:
            The raw encoded image (JPEG/PNG/etc) exactly as ingested.
            The implementation is responsible for decoding.

        Returns
        -------
        ClassificationResult
            A populated result. Implementations must not return ``None``
            or raise for ordinary "low-confidence" cases — that information
            should be encoded in the ``confidence`` field so the worker
            can persist a row and the notification layer can decide.
        """
        ...


class DummyClassifier:
    """No-op classifier used when no real model is configured.

    Returns ``("unclassified", 0.0)`` for every image. Lets the
    pipeline run end-to-end — rows get marked ``classified_at`` so
    the worker doesn't re-process them, but downstream confidence
    gates will reject every dummy output.
    """

    LABEL = "unclassified"

    def classify(self, image_bytes: bytes) -> ClassificationResult:
        # image_bytes intentionally unused — we're not running inference.
        # Reading len(image_bytes) would give a nicer log line but the
        # extra work is not worth it on the hot path.
        return ClassificationResult(species=self.LABEL, confidence=0.0)


class TFLiteClassifier:
    """TensorFlow Lite image classifier.

    Loads an ``.tflite`` model file and a newline-separated labels
    file. Each call:

    1. Decodes the input bytes with Pillow.
    2. Resizes to the model's expected input shape.
    3. Runs ``interpreter.invoke()``.
    4. Picks the top-1 label.

    The ``tflite_runtime`` module is imported lazily in
    :meth:`__init__` so users who never instantiate this class don't
    need the wheel installed. The dev/CI image does not ship it; the
    Thoth LXC does (see ``[classify]`` optional dependency in
    ``pyproject.toml``).

    Parameters
    ----------
    model_path:
        Filesystem path to the ``.tflite`` model file.
    labels_path:
        Filesystem path to a UTF-8 text file with one label per line.
        ``labels[output_index]`` is the human-readable name for that
        class. Leading/trailing whitespace is stripped.
    """

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        # Lazy import so dev/CI environments without tflite-runtime
        # can still import this module. The worker builds a
        # TFLiteClassifier only when THOTH_MODEL_PATH is set.
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as exc:  # pragma: no cover — environment-dependent
            raise RuntimeError(
                "tflite-runtime is required for TFLiteClassifier. "
                "Install the `classify` extra: `pip install -e .[classify]`"
            ) from exc

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        if not input_details or not output_details:
            raise RuntimeError(
                f"tflite model {model_path} reports no input/output tensors"
            )
        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]
        self._input_shape = input_details[0]["shape"]  # e.g. [1, 224, 224, 3]
        self._input_dtype = input_details[0]["dtype"]
        self._labels = labels_path.read_text(encoding="utf-8").splitlines()
        log.info(
            "loaded tflite model %s with %d labels, input shape %s",
            model_path,
            len(self._labels),
            tuple(self._input_shape),
        )

    def classify(self, image_bytes: bytes) -> ClassificationResult:
        # Lazy import of numpy + PIL for the same reason as tflite_runtime.
        import numpy as np
        from PIL import Image

        # Model input is typically (1, H, W, 3). Resize + cast.
        _, height, width, _ = self._input_shape
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((int(width), int(height)))
        arr = np.asarray(image, dtype=self._input_dtype)
        # Most bird models expect float32 in [0, 1]. Quantized uint8
        # models want the raw 0–255 values. Detect by dtype.
        if arr.dtype == np.float32:
            arr = arr / 255.0
        batch = np.expand_dims(arr, axis=0)
        self._interpreter.set_tensor(self._input_index, batch)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)[0]
        # If the output is quantized, rescale to [0, 1] best-effort so
        # confidence values are comparable across model types.
        if output.dtype != np.float32:
            output = output.astype(np.float32) / float(np.iinfo(output.dtype).max)
        top_index = int(np.argmax(output))
        confidence = float(output[top_index])
        if top_index >= len(self._labels):
            # Defensive: a model/labels mismatch shouldn't crash the worker.
            species = f"class_{top_index}"
            log.warning(
                "tflite top index %d has no matching label (have %d labels)",
                top_index,
                len(self._labels),
            )
        else:
            species = self._labels[top_index].strip() or f"class_{top_index}"
        return ClassificationResult(species=species, confidence=confidence)
