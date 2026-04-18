"""On-device bird object detector for horus.

Purpose
-------
The species classifier (``horus.classifier.BirdClassifier``) was pressed
into service as a bird/no-bird gate by thresholding its top-1 confidence.
That's fragile: a 965-class species model has "favorite fallback" classes
(e.g. Great Egret, New Zealand Pigeon) that it returns with moderate
confidence on *any* textured scene — leaves, shadows, a bare feeder.
Wind-sway events come back labeled "Great Egret 0.24", which is a
threshold-dependent false positive that varies with weather.

A COCO-trained object detector ("is there a bird here?") is the right
tool for the gate. It returns a clean zero when no bird is present
instead of the classifier's least-wrong-bird fallback. We keep the
species classifier downstream for labeling — but only on frames the
detector has vouched for.

Model assumptions
-----------------
Designed for EfficientDet-Lite0 (COCO-90) or any TFLite object
detection model with the standard four-output layout:

* ``detection_boxes``    — ``[1, N, 4]`` normalized ``[ymin, xmin, ymax, xmax]``
* ``detection_classes``  — ``[1, N]``    class indices (float, cast to int)
* ``detection_scores``   — ``[1, N]``    confidence ``[0, 1]``
* ``num_detections``     — ``[1]``       how many of the N slots are valid

Other architectures (YOLO, SSD variants with different output layouts)
can be swapped in by writing a different post-processor — the
:class:`BirdDetector` class here is the reference implementation.

Design notes
------------
* The interpreter is loaded once at daemon startup. The hot path
  (decode → resize → invoke → post-process) allocates nothing new on
  the interpreter.
* The TFLite runtime import is lazy + backend-portable: we try
  ``tflite_runtime`` first (historical name, small wheel) and fall
  back to ``ai_edge_litert`` (Google's rebrand with Python 3.13
  wheels). Tests monkey-patch the interpreter, so this file is
  importable on any laptop without a wheel installed.
* The class index for "bird" is resolved by label string at startup,
  not hardcoded — different COCO variants number classes differently
  (90-class vs 80-class, 0-indexed vs 1-indexed-with-background).
  Hardcoding a number would silently return "person" detections on
  the wrong model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    """Output of a single :meth:`BirdDetector.detect` call.

    Attributes
    ----------
    has_bird:
        ``True`` iff the detector returned at least one bird-class
        detection above the caller-specified threshold. This is the
        binary gate signal — the caller decides what to do with it.
    score:
        Confidence of the *best* bird detection, or ``0.0`` if no
        bird was found. Use this for logging / ranking gated samples,
        not for gating (the gating decision is baked into
        ``has_bird``).
    bbox_fraction:
        Tight bounding box of the best bird detection in
        ``(x, y, width, height)`` fractional coordinates (same layout
        as ``MotionGate.bbox_fraction``). ``None`` when ``has_bird``
        is False. Intended for a future detector-crop-then-classify
        stage; today's caller can ignore it.
    """

    has_bird: bool
    score: float
    bbox_fraction: tuple[float, float, float, float] | None


class BirdDetector:
    """Wraps a TFLite object-detection interpreter behind a simple API.

    One instance per daemon. :meth:`detect` is synchronous and not
    thread-safe — the capture loop calls it serially from the main
    thread.

    Parameters
    ----------
    model_path:
        Path to the ``.tflite`` file on disk. EfficientDet-Lite0 or
        any 4-output COCO detector with the standard layout.
    labels_path:
        UTF-8 text file with one COCO class per line. Used to resolve
        the "bird" class index at startup — the label string must be
        exactly ``"bird"`` (case-insensitive).
    min_score:
        Per-detection score threshold. Detections below this are
        discarded before deciding the bird/no-bird verdict. Default
        0.30 is a reasonable starting point for EfficientDet-Lite0
        — tune from the gated-archive ground truth.
    """

    # Candidate label strings that count as "bird" in a COCO-style
    # labels file. Different exporters name classes slightly differently;
    # case-insensitive exact match against these covers the common ones.
    _BIRD_ALIASES: frozenset[str] = frozenset({"bird", "birds"})

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        *,
        min_score: float = 0.30,
    ) -> None:
        # Lazy import: only the Pi needs the runtime. Dev laptops and
        # CI can import this module for tests without a wheel installed.
        Interpreter = None
        for module_name in ("tflite_runtime.interpreter", "ai_edge_litert.interpreter"):
            try:
                module = __import__(module_name, fromlist=["Interpreter"])
                Interpreter = module.Interpreter
                break
            except ImportError:
                continue
        if Interpreter is None:  # pragma: no cover — environment-dependent
            raise RuntimeError(
                "No TFLite runtime available for BirdDetector. Install "
                "one on the capture host: `pip install ai-edge-litert` "
                "(or `tflite-runtime` on older Pythons)."
            )

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        if not input_details or not output_details:
            raise RuntimeError(
                f"tflite model {model_path} reports no input/output tensors"
            )
        self._input_index = input_details[0]["index"]
        self._input_shape = input_details[0]["shape"]  # e.g. [1, 320, 320, 3]
        self._input_dtype = input_details[0]["dtype"]
        if len(self._input_shape) != 4:
            raise RuntimeError(
                f"tflite model {model_path} has unsupported input shape "
                f"{tuple(self._input_shape)}; expected (1, H, W, C)"
            )

        # Map the four output tensors by shape. EfficientDet-Lite0
        # doesn't guarantee a fixed order in get_output_details — we
        # pick them out by rank/shape to stay robust across converters.
        #   boxes: rank-3, last dim=4
        #   classes: rank-2 (equal length to scores)
        #   scores: rank-2
        #   num_detections: rank-1
        self._output_indices = self._resolve_output_indices(output_details, model_path)

        self._labels = [
            line.strip()
            for line in labels_path.read_text(encoding="utf-8").splitlines()
        ]
        self._bird_class_indices = self._resolve_bird_indices(self._labels, labels_path)
        self._min_score = float(min_score)
        log.info(
            "loaded tflite detector %s (input shape %s, %d labels, bird classes %s, min_score %.2f)",
            model_path,
            tuple(self._input_shape),
            len(self._labels),
            self._bird_class_indices,
            self._min_score,
        )

    @staticmethod
    def _resolve_output_indices(
        output_details: list[dict], model_path: Path
    ) -> dict[str, int]:
        """Identify which tensor in ``output_details`` is boxes/classes/scores/num.

        We do this by shape rather than trusting list order — different
        converters emit the outputs in different positions. Raises
        ``RuntimeError`` with a useful message if the shapes don't
        match the 4-output COCO detector layout, so a misconfigured
        model fails at startup rather than silently reading the wrong
        tensor on every bird.
        """
        by_role: dict[str, int] = {}
        candidates_score_class: list[dict] = []
        for out in output_details:
            shape = tuple(out["shape"])
            # boxes: (1, N, 4)
            if len(shape) == 3 and shape[-1] == 4:
                by_role["boxes"] = out["index"]
            # num_detections: (1,) — scalar batch
            elif len(shape) == 1:
                by_role["num"] = out["index"]
            # scores and classes are both (1, N); distinguish below.
            elif len(shape) == 2:
                candidates_score_class.append(out)
            else:
                log.warning(
                    "detector output with unexpected shape %s ignored", shape
                )
        # Among the two (1, N) outputs, classes are almost always
        # integer-valued (but stored as float32 post-quant). Post-quant
        # EfficientDet keeps both as float32, so we can't discriminate
        # by dtype alone. Fall back to list order: the TFLite task
        # library convention is (boxes, classes, scores, num) — so if
        # both (1, N) tensors are present we assign in appearance order.
        if len(candidates_score_class) == 2:
            by_role["classes"] = candidates_score_class[0]["index"]
            by_role["scores"] = candidates_score_class[1]["index"]
        for role in ("boxes", "classes", "scores", "num"):
            if role not in by_role:
                raise RuntimeError(
                    f"tflite model {model_path} is missing a '{role}' output — "
                    f"expected the standard 4-output COCO detector layout. "
                    f"Got shapes: {[tuple(o['shape']) for o in output_details]}"
                )
        return by_role

    @classmethod
    def _resolve_bird_indices(
        cls, labels: list[str], labels_path: Path
    ) -> tuple[int, ...]:
        """Find the class indices whose label is 'bird' (case-insensitive).

        Raises if nothing matches — a model/labels mismatch would otherwise
        silently return "no bird detected" on every frame, which looks
        like a dead gate rather than a misconfiguration.
        """
        hits = tuple(
            i for i, lab in enumerate(labels)
            if lab and lab.strip().lower() in cls._BIRD_ALIASES
        )
        if not hits:
            raise RuntimeError(
                f"labels file {labels_path} contains no 'bird' entry — "
                f"BirdDetector needs the COCO 'bird' class to gate on. "
                f"First few labels: {labels[:10]}"
            )
        return hits

    def detect(self, image_bytes: bytes) -> DetectionResult:
        """Run detection on a single encoded image.

        Parameters
        ----------
        image_bytes:
            Raw JPEG bytes exactly as they'd be published — see the
            classifier's matching note about input-pipeline alignment.

        Returns
        -------
        :class:`DetectionResult` — ``has_bird`` is the gate signal, the
        rest is metadata for logging / downstream cropping.
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

        boxes = self._interpreter.get_tensor(self._output_indices["boxes"])[0]
        classes = self._interpreter.get_tensor(self._output_indices["classes"])[0]
        scores = self._interpreter.get_tensor(self._output_indices["scores"])[0]
        num_raw = self._interpreter.get_tensor(self._output_indices["num"])
        num = int(num_raw.flatten()[0])

        bird_set = set(self._bird_class_indices)
        best_score = 0.0
        best_bbox: tuple[float, float, float, float] | None = None
        # Iterate up to num to respect models that zero-pad the tail of
        # the boxes/scores arrays. Clip to array length for safety.
        limit = min(num, len(scores), len(classes), len(boxes))
        for i in range(limit):
            score = float(scores[i])
            if score < self._min_score:
                continue
            cls_idx = int(classes[i])
            if cls_idx not in bird_set:
                continue
            if score > best_score:
                best_score = score
                ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
                # Clamp in case the model emits slightly-negative or
                # >1 coordinates (happens on EfficientDet under rare
                # anchor conditions).
                ymin = max(0.0, min(1.0, ymin))
                xmin = max(0.0, min(1.0, xmin))
                ymax = max(0.0, min(1.0, ymax))
                xmax = max(0.0, min(1.0, xmax))
                # Convert to (x, y, w, h) fraction to match MotionGate.
                best_bbox = (
                    xmin,
                    ymin,
                    max(0.0, xmax - xmin),
                    max(0.0, ymax - ymin),
                )

        return DetectionResult(
            has_bird=best_bbox is not None,
            score=best_score,
            bbox_fraction=best_bbox,
        )
