"""Shared imaging helpers — bytes-in/bytes-out crop for the subject bbox.

Both horus (publisher, runs its on-device detector/classifier on the
crop before publishing) and thoth-classify (consumer, crops from the
published full frame before running the species classifier) must
produce **byte-identical crops** from the same ``(image_bytes,
bbox_fraction)`` input. Otherwise the on-device score printed in the
horus log will not match the score thoth writes to the DB, and
debugging classifier confidence becomes a nightmare of "which crop
did the model actually see?"

This module is the single source of truth for that crop. Horus's
legacy file-based :func:`horus.motion.crop_to_bbox` delegates here
for the shared geometry, and the horus daemon's on-device gate now
feeds :func:`crop_to_bbox_bytes` output directly to its detector and
classifier so the scored bytes match what thoth-classify re-derives.

Byte-identity preconditions
---------------------------
To keep on-device and post-ingest scores directly comparable, the
JPEG encode step on both sides MUST use the same quality. That
quality is fixed at :data:`CLASSIFIER_JPEG_QUALITY` and is the
default for :func:`crop_to_bbox_bytes`. Callers that need the
inference-comparable bytes must not override ``jpeg_quality``.
The parameter is kept so the rare caller who wants a hi-fi
archive crop (e.g. reviewer gallery) can opt out, but anything
that passes through a model should stick with the default.

Known limitation: non-square crops at image edges
-------------------------------------------------
The pipeline squares the padded bbox from its center, then clamps
to the image bounds. For a bbox near a frame edge the clamp step
truncates one side, so the final crop is a rectangle (typically
off-square by 10–20%). Species classifiers trained on square
inputs may see a small confidence hit on edge-of-frame detections.
This was measured as "small" (single-digit confidence delta) during
phase-1 tuning and accepted over the alternatives (black-fill
padding changes model context; re-centering away from the edge
loses the subject). Revisit if edge-of-frame false negatives
become a real problem in the reviewer queue.

Pillow is required — declared as an optional ``imaging`` extra on
``sbo_shared`` so the core package stays dep-free. Every consumer
that already crops (horus, banshee) already pulls Pillow, so the
extra is documentation; the import here does the real enforcement.
``sbo_shared/__init__.py`` does NOT re-export this module, so
importing the top-level ``sbo_shared`` package does not pull
Pillow into pure-event consumers.
"""

from __future__ import annotations

import io
from typing import Tuple

from PIL import Image

BboxFraction = Tuple[float, float, float, float]
"""``(x0, y0, x1, y1)`` in ``[0.0, 1.0]`` half-open, relative to the
source image's dimensions. Matches :class:`horus.motion.MotionResult.bbox_fraction`.
"""


CLASSIFIER_JPEG_QUALITY: int = 90
"""Canonical JPEG quality used for every classifier-visible crop.

Horus's on-device detector+classifier score the bytes produced by
:func:`crop_to_bbox_bytes` with this default, and thoth-classify
re-derives the same bytes from the published full frame using the
same default. If these two paths ever use a different quality, the
"byte-identical crops" guarantee breaks and on-device vs post-ingest
scores will disagree in hard-to-debug ways.

Do not thread horus's ``capture.jpeg_quality`` (archive-fidelity
knob) into this helper — they serve different purposes.
"""


def crop_to_bbox_bytes(
    image_bytes: bytes,
    bbox_fraction: BboxFraction,
    *,
    padding: float = 0.30,
    min_side_px: int = 224,
    max_side_px: int = 896,
    jpeg_quality: int = CLASSIFIER_JPEG_QUALITY,
) -> bytes:
    """Crop a full-frame JPEG to a padded, squared region around ``bbox_fraction``.

    Algorithm (every step matters for byte-identity):

    1. Scale ``bbox_fraction`` to pixel coords in the full-res image.
    2. Pad by ``padding`` × longer-bbox-side on each edge.
    3. Square from the bbox center, floored to ``min_side_px``.
    4. Clamp to image bounds (may leave a near-edge bbox rectangular —
       see the module docstring's "Known limitation" note).
    5. Downscale with LANCZOS if the longer side exceeds ``max_side_px``.
    6. Convert to RGB (JPEG can't carry alpha) and encode at
       ``jpeg_quality``.

    ``jpeg_quality`` defaults to :data:`CLASSIFIER_JPEG_QUALITY`. Callers
    that need bytes comparable to what horus's on-device gate scored
    (i.e. the classifier worker) MUST keep the default. Callers that
    are producing an archive-only crop (reviewer gallery, forensics)
    may override.

    Returns the JPEG-encoded crop bytes. Size information is available
    by decoding the return value (``Image.open(io.BytesIO(out)).size``)
    or by using :func:`crop_to_bbox_image` if the caller already needs
    a ``PIL.Image``.
    """
    cropped = _crop_image(
        Image.open(io.BytesIO(image_bytes)),
        bbox_fraction,
        padding=padding,
        min_side_px=min_side_px,
        max_side_px=max_side_px,
    )
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


def crop_to_bbox_image(
    image: Image.Image,
    bbox_fraction: BboxFraction,
    *,
    padding: float = 0.30,
    min_side_px: int = 224,
    max_side_px: int = 896,
) -> Image.Image:
    """Crop a PIL :class:`~PIL.Image.Image` to the padded, squared subject region.

    Exposed for callers that already hold a decoded ``Image`` (e.g.
    the file-based :func:`horus.motion.crop_to_bbox` that opens
    the source via a context manager). Same math as
    :func:`crop_to_bbox_bytes` but skips the decode/encode round-trip.
    """
    return _crop_image(
        image,
        bbox_fraction,
        padding=padding,
        min_side_px=min_side_px,
        max_side_px=max_side_px,
    )


def _crop_image(
    im: Image.Image,
    bbox_fraction: BboxFraction,
    *,
    padding: float,
    min_side_px: int,
    max_side_px: int,
) -> Image.Image:
    """Core crop routine shared by the bytes and Image entrypoints.

    Kept private so callers can't accidentally depend on an intermediate
    form; the public entrypoints choose whether to take/return bytes or
    an :class:`~PIL.Image.Image`. Both call into this one implementation
    so the padded-squared-clamped-clipped-resized output is byte-stable
    given the same inputs.
    """
    im.load()  # force read while any source file handle is still open
    W, H = im.size
    x0, y0, x1, y1 = bbox_fraction

    # Step 1: fractional bbox → pixel bbox.
    px0 = int(round(x0 * W))
    py0 = int(round(y0 * H))
    px1 = int(round(x1 * W))
    py1 = int(round(y1 * H))

    # Step 2: pad by `padding` × longer_side on each edge.
    bw = max(1, px1 - px0)
    bh = max(1, py1 - py0)
    pad = int(round(padding * max(bw, bh)))
    px0 -= pad
    py0 -= pad
    px1 += pad
    py1 += pad

    # Step 3: square from the bbox center, floored to min_side_px
    # (prevents classifier-resize from upscaling 20×20 garbage).
    cx = (px0 + px1) / 2
    cy = (py0 + py1) / 2
    side = max(px1 - px0, py1 - py0, min_side_px)
    half = side / 2
    px0 = int(round(cx - half))
    py0 = int(round(cy - half))
    px1 = int(round(cx + half))
    py1 = int(round(cy + half))

    # Step 4: clamp to image bounds. For birds near a frame edge this
    # truncates one side, leaving a rectangular crop rather than a
    # square — see the module docstring's "Known limitation" note.
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(W, px1)
    py1 = min(H, py1)

    cropped = im.crop((px0, py0, px1, py1))

    # Step 5: downscale with LANCZOS if the longer side exceeds the cap.
    cw, ch = cropped.size
    if max(cw, ch) > max_side_px:
        scale = max_side_px / max(cw, ch)
        cropped = cropped.resize(
            (int(round(cw * scale)), int(round(ch * scale))),
            Image.LANCZOS,
        )

    # Step 6: JPEG can't carry alpha; normalise here so the bytes path
    # and the Image path produce equivalent output.
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
    return cropped
