"""Hero-frame scoring for burst sessions.

When horus publishes a burst (multiple frames from one feeder visit),
Thoth picks *one* canonical "hero" for the UI and keeps the rest as
browsable alternates. This module owns the two numeric building
blocks used to rank frames:

* :func:`laplacian_variance` — a PIL-only sharpness proxy.  No cv2
  dependency; PIL + ImageFilter + ImageStat is enough for a relative
  ranking within a burst.
* :func:`hero_score` — the Tier-1 composite (detector + sharpness +
  bbox area + post-classify confidence).  Weights match the SBO
  hero-selection design note (referenced as SBO-Hero-Selection).

Why these weights (0.4 / 0.3 / 0.2 / 0.1)
------------------------------------------

The composite is::

    0.4 * bird_score
  + 0.3 * sharpness_norm
  + 0.2 * bbox_area
  + 0.1 * classifier_confidence

Each weight earns its slot from a different failure mode:

* **bird_score (0.4) — the most direct signal of "is this a bird at
  all".** horus runs an on-device detector against the motion crop and
  publishes its top-bird confidence with every frame. A frame that
  scores high here is almost certainly a real bird; a frame that
  scores low is almost certainly leaves, shadow, or a squirrel. We
  weight it highest because the cost of picking a non-bird hero is
  worse than picking a slightly-blurry bird — the UI surfaces this
  frame as the canonical card for the burst, so "wrong subject" is the
  most visible failure mode.

* **sharpness (0.3) — mid-weight because motion blur is the dominant
  *quality* failure once subject is right.** Within a burst the
  detector typically agrees on every frame (same bird visiting the
  same feeder for ~1s), so bird_score alone won't separate the
  in-focus shutter release from the wing-beat smear. The Laplacian
  variance discriminates those reliably for the same scene + exposure.
  We don't push it higher because absolute sharpness varies with
  lighting and crop size — within-burst ordering is reliable, but
  cross-burst comparisons aren't, and the weight reflects that.

* **bbox_area (0.2) — mild preference for subject-filling frames.**
  A small bbox usually means the bird is at the edge of frame or
  partially behind a branch; a near-full-frame bbox usually means a
  subject-filling shot the UI can crop tightly. The weight is
  deliberately gentle — a bird perched at 8% of the frame can still
  be a great hero if it's tack-sharp and the detector is confident,
  so we don't want this term to dominate. (Out-of-range or absent
  bbox contributes 0; see :func:`_bbox_area`.)

* **classifier_confidence (0.1) — last because it's often missing and
  always trails ingest.** The Thoth-side classifier runs after the
  frame is already in SQLite and MinIO, and it can be unavailable for
  long stretches (worker outage, model swap). When it does land, it
  confirms or denies the species — useful as a tiebreaker but not
  load-bearing for hero selection. Weighting it low means hero picks
  don't change wildly when the classifier comes back online and
  re-scores a backlog. (NULL confidence contributes 0; see
  :func:`record_classification` for the recompute path.)

The weights sum to 1.0 so the composite stays in ``[0, 1]`` when each
component is normalized — useful for analytics that want to compare
hero quality across bursts or stations.

Design constraints
------------------

* **Relative ordering, not absolute quality.** We care whether frame
  A beats frame B inside the same burst, not whether the score is
  calibrated across cameras or lighting conditions. The normalization
  constants below are tuned for that goal, not peer-reviewed sensor
  physics.

* **Graceful degradation.** Missing inputs (bbox=None, bird_score=None,
  decode failure) contribute 0 rather than propagating NaN — a frame
  with no detector metadata can still win on pure sharpness, and a
  corrupt JPEG lands at the bottom of the pack instead of crashing
  ingest. The "all four inputs missing" case is handled one layer up
  in :func:`eventstore._hero_score_or_null` — that path persists NULL
  rather than 0.0 so unscored frames stay distinguishable from
  scored-at-zero ones.

* **Cheap at ingest time.** Called on the paho callback thread for
  every image; the PIL filter runs on a 640-wide thumbnail so the
  per-frame cost is ~5-10ms on the Thoth LXC.
"""

from __future__ import annotations

import io
import logging

from PIL import Image, ImageFilter, ImageStat

log = logging.getLogger(__name__)


# Discrete 3x3 Laplacian kernel. The ``offset=128`` shifts the filtered
# output into mid-gray so small negative gradients that would otherwise
# clip at 0 are preserved. The mitigation is partial — gradients with
# magnitude > 128 still clip at 0 / 255 — but the variance computed on
# the offset response remains a monotonic sharpness proxy within a
# burst (same scene, same exposure), which is all hero selection needs.
# Do not repurpose this as a cross-camera or cross-exposure sharpness
# metric without accounting for that clipping. ``scale=1`` keeps the
# raw response — we want absolute edge magnitude, not a normalized one.
_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    size=(3, 3),
    kernel=[0, -1, 0, -1, 4, -1, 0, -1, 0],
    scale=1,
    offset=128,
)

# Downsample to 640-wide before the filter pass. A 2304x1296 frame is
# ~3M pixels; 640-wide is ~250k and preserves the relative sharpness
# ordering that hero selection depends on (empirically verified on
# IMX519 feeder captures — see the design note). The thumbnail() call
# preserves aspect ratio, so the second dimension is a no-op ceiling.
_SHARPNESS_THUMB = (640, 640)

# Normalization divisor for sharpness → [0, 1]. A 640-wide feeder crop
# on the IMX519 produces Laplacian variance roughly in [50, 1500]; the
# 1000.0 floor maps "moderately sharp" (variance ~1000) to a full
# unity contribution and clips everything above. Not calibrated across
# cameras — the goal is monotonic per-burst ordering, not an absolute
# scale.
_SHARPNESS_NORM = 1000.0

# Composite-score weights, ordered (detector, sharpness, bbox_area,
# classifier). See the module docstring for why each weight earned its
# slot. Must sum to 1.0 so :func:`hero_score` lands in ``[0, 1]`` when
# every component is normalized.
_WEIGHTS: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)


def laplacian_variance(image_bytes: bytes) -> float:
    """Return the Laplacian variance of a downscaled grayscale copy.

    Parameters
    ----------
    image_bytes:
        Raw JPEG/PNG bytes from the MQTT payload (the same bytes
        ``ImageEvent.image_bytes`` holds).

    Returns
    -------
    float
        Non-negative variance. Higher means sharper. ``0.0`` on any
        decode or filter failure — the caller treats that as a
        "definitely not the hero" signal rather than crashing ingest
        on a corrupt image.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.thumbnail(_SHARPNESS_THUMB)
            gray = im.convert("L")
    except Exception:
        # Log + return 0 so a single bad JPEG doesn't poison the loop.
        log.warning("laplacian_variance: image decode failed", exc_info=True)
        return 0.0

    try:
        lap = gray.filter(_LAPLACIAN_KERNEL)
        return float(ImageStat.Stat(lap).var[0])
    except Exception:
        log.warning("laplacian_variance: filter/stat failed", exc_info=True)
        return 0.0


def _bbox_area(
    bbox_fraction: tuple[float, float, float, float] | None,
) -> float:
    """Area of the motion bbox as a fraction of the full frame.

    Returns ``0.0`` when no bbox is known or the values are out of
    range. A frame with no bbox gets zero credit on this axis; it can
    still win a burst via sharpness + detector confidence.
    """
    if bbox_fraction is None or len(bbox_fraction) != 4:
        return 0.0
    try:
        w = float(bbox_fraction[2])
        h = float(bbox_fraction[3])
    except (TypeError, ValueError):
        return 0.0
    # Clamp into [0, 1]. A malformed payload with w>1 shouldn't let the
    # bbox term blow out the total score.
    return max(0.0, min(1.0, w)) * max(0.0, min(1.0, h))


def _normalize_sharpness(raw_variance: float | None) -> float:
    """Map a Laplacian variance onto ``[0, 1]`` for the composite."""
    if raw_variance is None or raw_variance <= 0.0:
        return 0.0
    return min(1.0, float(raw_variance) / _SHARPNESS_NORM)


def _clamp01(value: float | None) -> float:
    """Clamp a numeric input to ``[0, 1]``, mapping None to 0."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def hero_score(
    *,
    bird_score: float | None,
    sharpness: float | None,
    bbox_fraction: tuple[float, float, float, float] | None,
    classifier_confidence: float | None,
) -> float:
    """Tier-1 composite hero rank; higher is better.

    Composition
    -----------
    ``0.4 * detector + 0.3 * sharpness_norm + 0.2 * bbox_area
     + 0.1 * classifier``

    where:

    * ``detector`` is horus's on-device ``bird_score`` in ``[0, 1]``.
    * ``sharpness_norm`` is the Laplacian variance divided by
      :data:`_SHARPNESS_NORM`, clipped to ``[0, 1]``.
    * ``bbox_area`` is ``w * h`` of the motion bbox in full-frame
      coords, ``[0, 1]``.
    * ``classifier`` is the Thoth-side TFLite confidence, ``[0, 1]``.
      Zero at ingest time (classification hasn't run yet) and filled
      in on the post-classify recompute path.

    Parameters
    ----------
    bird_score, classifier_confidence:
        Both in ``[0, 1]``; ``None`` maps to ``0.0``.
    sharpness:
        Raw Laplacian variance as returned by
        :func:`laplacian_variance`. Normalized here (not by the caller)
        so the weight math stays in one place.
    bbox_fraction:
        ``(x, y, w, h)`` in full-frame coords, or ``None`` for frames
        that lack motion metadata.

    Returns
    -------
    float
        Composite score in ``[0, 1]`` (assuming every input is in its
        documented range). Out-of-range inputs are clamped, so an
        adversarial ``bird_score=5.0`` cannot drag the total above 1.0.
    """
    components: tuple[float, float, float, float] = (
        _clamp01(bird_score),
        _normalize_sharpness(sharpness),
        _bbox_area(bbox_fraction),
        _clamp01(classifier_confidence),
    )
    return sum(w * v for w, v in zip(_WEIGHTS, components))
