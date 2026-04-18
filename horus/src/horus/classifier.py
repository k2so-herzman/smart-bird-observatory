"""On-device bird/no-bird gate for horus.

Runs an iNaturalist MobileNet-V2 (quantized uint8) TFLite model against
a cropped capture and returns a confidence score. Used by ``main._tick``
to drop wind-sway and lighting-flicker false positives before they hit
MQTT, saving Thoth ingest + classification cycles on frames a model can
already tell aren't birds.

Design notes
------------
* The interpreter is loaded once at daemon startup. The hot path is a
  pure-inference call (decode, resize, invoke, argmax) with no allocation
  on the interpreter side after ``allocate_tensors``.
* ``tflite_runtime`` is a lazy import so this module can be imported on a
  dev laptop without the Pi wheel installed — tests monkey-patch the
  interpreter and never hit the real import.
* We return the *top-1 confidence*, not a bird-specific score. The
  iNaturalist bird model only has bird classes, so top-1 is already
  "most likely bird." If we ever swap to a multi-class model with
  non-bird labels, gate semantics will have to change.
* Pillow decodes the JPEG. This duplicates Thoth's classifier runtime
  path intentionally: we want the same input pipeline so on-device
  gating scores are comparable to the post-ingest scores Thoth records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationResult:
    """Output of a single classify() call.

    Attributes
    ----------
    species:
        Top-1 label from the model's labels file. Informational — the
        gate decision uses ``confidence`` alone.
    confidence:
        Model confidence in ``[0.0, 1.0]``. Quantized models get their
        raw uint8 output rescaled to ``[0, 1]`` so downstream thresholds
        are comparable across model variants.
    """

    species: str
    confidence: float


class BirdClassifier:
    """Wraps a TFLite interpreter + label list behind a simple API.

    One instance per daemon. :meth:`classify` is synchronous and
    thread-unsafe — the capture loop calls it serially from the main
    thread, which matches the single-threaded paho-mqtt loop.

    Parameters
    ----------
    model_path:
        Path to the ``.tflite`` file on disk.
    labels_path:
        Path to a UTF-8 text file with one label per output index.
        Blank lines and leading/trailing whitespace are tolerated.
    """

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        # Lazy import: only the Pi needs tflite-runtime. Dev laptops
        # and CI can import this module (for tests) without the wheel.
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as exc:  # pragma: no cover — environment-dependent
            raise RuntimeError(
                "tflite-runtime is required for BirdClassifier. "
                "Install it on the capture host: `pip install tflite-runtime`"
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
        # Validate rank up front so a mismatched model fails at startup
        # rather than with a cryptic traceback on the first bird.
        if len(self._input_shape) != 4:
            raise RuntimeError(
                f"tflite model {model_path} has unsupported input shape "
                f"{tuple(self._input_shape)}; expected (1, H, W, C)"
            )
        self._labels = [
            line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()
        ]
        log.info(
            "loaded tflite model %s with %d labels, input shape %s",
            model_path,
            len(self._labels),
            tuple(self._input_shape),
        )

    def classify(self, image_bytes: bytes) -> ClassificationResult:
        """Run inference on a single encoded image.

        Parameters
        ----------
        image_bytes:
            Raw JPEG bytes exactly as they'll be published to MQTT —
            the whole point of gating at this layer is that the model
            sees the same pixels the downstream classifier would.
        """
        import numpy as np
        from PIL import Image

        _, height, width, _ = self._input_shape
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((int(width), int(height)))
        arr = np.asarray(image, dtype=self._input_dtype)
        # Float models want [0, 1]; quantized uint8 models expect raw 0-255.
        if arr.dtype == np.float32:
            arr = arr / 255.0
        batch = np.expand_dims(arr, axis=0)
        self._interpreter.set_tensor(self._input_index, batch)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_index)[0]
        if output.dtype != np.float32:
            # Quantized output → rescale to [0, 1] so thresholds stay
            # comparable regardless of whether the model is quant or float.
            output = output.astype(np.float32) / float(np.iinfo(output.dtype).max)
        top_index = int(np.argmax(output))
        confidence = float(output[top_index])
        if top_index >= len(self._labels):
            species = f"class_{top_index}"
            log.warning(
                "tflite top index %d has no matching label (have %d labels)",
                top_index,
                len(self._labels),
            )
        else:
            species = self._labels[top_index] or f"class_{top_index}"
        return ClassificationResult(species=species, confidence=confidence)
