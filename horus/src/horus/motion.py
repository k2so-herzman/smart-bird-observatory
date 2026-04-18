"""Frame-differencing motion gate for the horus capture daemon.

Horus samples frames from the Pi camera at a configurable interval (typically
1 Hz). Before deciding whether to publish a capture event to Banshee, it runs
each frame through :class:`MotionGate`, which implements a lightweight
pixel-difference algorithm cheap enough for a Raspberry Pi 3 B+.

**Algorithm — frame differencing with a slow exponential baseline:**
Each frame is converted to grayscale and downscaled to a fixed 160×90
thumbnail.  The thumbnail is compared against a running *baseline* thumbnail
kept in memory.  For every pixel, the absolute difference is computed; pixels
whose difference exceeds ``MotionConfig.pixel_threshold`` are counted as
*changed*.  If the fraction of changed pixels meets or exceeds
``MotionConfig.frame_fraction`` the frame is flagged as containing motion.
After each comparison the baseline is updated with a slow exponential moving
average (α = 0.1) so gradual lighting drift — sunrise, clouds — does not
trigger false positives.

**Integration with Daemon._tick:**
``Daemon._tick`` calls :meth:`MotionGate.check` after every capture.  When
``MotionResult.motion`` is ``True`` **and** the cooldown window has elapsed,
the daemon publishes the frame to Banshee via ``EventBus.publish_image_event``
and attaches ``MotionResult.changed_fraction`` as metadata.  Frames without
motion are deleted immediately to avoid filling local storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import MotionConfig

log = logging.getLogger(__name__)


@dataclass
class MotionResult:
    """Result returned by :meth:`MotionGate.check` for a single frame.

    Fields
    ------
    motion:
        ``True`` when the changed-pixel fraction meets or exceeds
        ``MotionConfig.frame_fraction``; ``False`` otherwise.  A ``False``
        result on the very first call (no baseline yet) is always returned so
        that an initial frame is never mis-published as a motion event.
    changed_fraction:
        Fraction of pixels (in the 160×90 thumbnail) whose absolute difference
        from the baseline exceeded ``MotionConfig.pixel_threshold``.  Range
        ``[0.0, 1.0]``; ``0.0`` on the first call when no baseline exists yet.
        Passed verbatim to ``EventBus.publish_image_event`` as event metadata.
    bbox_fraction:
        Axis-aligned bounding box of the changed-pixel region, expressed in
        **fractional coordinates** ``(x0, y0, x1, y1)`` where each component is
        in ``[0.0, 1.0]`` relative to the original frame's width/height.
        ``None`` when no pixels cleared ``pixel_threshold`` (including the
        first-frame baseline case).  Half-open: ``x1`` and ``y1`` are
        exclusive, so ``(0, 0, 1, 1)`` means the full frame.

        Downstream consumers multiply these fractions by the full-resolution
        image dimensions to get pixel coords for a crop.  Fractional units
        are chosen so the bbox survives the thumbnail-vs-full-res scale
        factor automatically — ``Image.thumbnail`` preserves aspect ratio,
        so fractional coords in thumb space map one-to-one onto full-res.
    """

    motion: bool
    changed_fraction: float
    bbox_fraction: tuple[float, float, float, float] | None = None


class MotionGate:
    """Stateful frame-differencing gate that decides whether a frame contains motion.

    Instantiate once per daemon run and call :meth:`check` for every captured
    frame.  The gate maintains an internal baseline thumbnail across calls; the
    baseline is initialised lazily on the first :meth:`check` call and updated
    with every subsequent frame via an exponential moving average.

    **Threshold knobs (from** :class:`~horus.config.MotionConfig` **):**

    ``pixel_threshold`` (int, default 25)
        Minimum absolute per-pixel difference (0–255) to count a pixel as
        *changed*.  Lower values increase sensitivity; values below ~10 cause
        noise-triggered false positives under stable lighting.

    ``frame_fraction`` (float, default 0.02)
        Fraction of thumbnail pixels that must be *changed* for the frame to
        be flagged as motion.  0.02 means 2 % of the 160×90 = 288 pixels.
        Raise this to suppress small movements (leaves, insects); lower it to
        catch subtle activity.

    ``cooldown_s`` (float, default 5.0)
        Not enforced by the gate itself — enforced by ``Daemon._tick``.
        Documented here because it is the primary rate-limiter on publish
        frequency and pairs with the gate thresholds when tuning sensitivity.
    """

    THUMB_SIZE = (160, 90)

    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self._baseline: np.ndarray | None = None

    def _thumb(self, path: Path) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("L")
            im.thumbnail(self.THUMB_SIZE)
            return np.asarray(im, dtype=np.int16)

    def check(self, frame_path: Path) -> MotionResult:
        """Compare *frame_path* against the internal baseline and report motion.

        Loads the image at *frame_path*, converts it to a grayscale 160×90
        thumbnail, and diffs it against the stored baseline.  On the first
        call (no baseline) or whenever the thumbnail shape changes (resolution
        switch), the frame is adopted as the new baseline and ``MotionResult``
        is returned with ``motion=False`` and ``changed_fraction=0.0``.

        After a successful diff the baseline is blended toward the current
        thumbnail (``baseline = 0.9 * baseline + 0.1 * thumb``), so slow
        lighting changes do not accumulate into false positives.

        Args:
            frame_path: Path to a JPEG (or any Pillow-readable) image file.
                The file must exist and be readable; no format validation is
                performed — Pillow raises ``UnidentifiedImageError`` on failure.

        Returns:
            A :class:`MotionResult` with ``motion=True`` when the
            changed-pixel fraction meets ``MotionConfig.frame_fraction``, and
            ``changed_fraction`` set to the exact fraction for that frame.

        Side effects:
            Mutates ``self._baseline`` in place on every call — either to
            adopt the first frame or to apply the exponential moving-average
            update.  Does **not** open, write, delete, or otherwise modify
            *frame_path*.
        """
        thumb = self._thumb(frame_path)

        if self._baseline is None or self._baseline.shape != thumb.shape:
            self._baseline = thumb
            return MotionResult(False, 0.0, None)

        diff = np.abs(thumb - self._baseline)
        mask = diff > self.cfg.pixel_threshold
        changed = int(mask.sum())
        fraction = changed / thumb.size

        # Slow baseline update so lighting drift doesn't trigger.
        self._baseline = (0.9 * self._baseline + 0.1 * thumb).astype(np.int16)

        bbox_fraction = _bbox_fraction_from_mask(mask) if changed else None

        return MotionResult(
            motion=fraction >= self.cfg.frame_fraction,
            changed_fraction=float(fraction),
            bbox_fraction=bbox_fraction,
        )


def _bbox_fraction_from_mask(
    mask: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute the tight axis-aligned bbox of ``True`` pixels in *mask*.

    Returned as fractional ``(x0, y0, x1, y1)`` relative to ``mask.shape`` —
    so ``(0, 0, 1, 1)`` covers the whole thumb.  Half-open: ``x1`` and ``y1``
    are **exclusive**, matching the Pillow ``Image.crop`` convention.

    Uses the axis-reduction trick (``np.any`` along each axis, then
    ``argmax`` from both ends) — O(H+W) extra work on top of the diff.
    Callers **must** ensure ``mask.any()`` is true; an all-False mask will
    return a degenerate ``(0, 0, 0, 0)`` bbox and is the caller's bug to
    avoid (we don't check, to keep this hot-path tight).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    # argmax on a bool array returns the first True index; reversed gives
    # the last True. np.argmax on an all-False array returns 0 — the caller
    # contract prevents that case, so we don't guard against it here.
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - np.argmax(cols[::-1]))
    h, w = mask.shape
    return (x0 / w, y0 / h, x1 / w, y1 / h)


def crop_to_bbox(
    src_path: Path,
    dst_path: Path,
    bbox_fraction: tuple[float, float, float, float],
    *,
    padding: float = 0.30,
    min_side_px: int = 224,
    max_side_px: int = 896,
    jpeg_quality: int = 90,
) -> tuple[int, int]:
    """Crop *src_path* to *bbox_fraction* (with padding + squaring) and save as JPEG.

    The motion bbox returned by :class:`MotionGate` hugs the changed pixels
    tightly — a bird's silhouette, typically.  Image classifiers trained on
    iNaturalist-style portraits want the subject centered with a bit of
    context, so this helper:

    1. Scales the fractional bbox to pixel coords in the full-res image.
    2. Expands the bbox by ``padding`` (fraction of the longer bbox side)
       on each side, to include the subject's margin / context.
    3. **Squares** the crop by expanding the shorter axis from the center,
       because the classifier's input tensor is square (typically 224×224).
       Feeding a non-square region would be letterboxed by the resize and
       waste resolution on black bars.
    4. Clamps the crop rectangle to the image bounds.
    5. If the resulting crop is smaller than ``min_side_px`` on either
       axis, expands from the center to at least that size (clamping again
       to image bounds).  A too-small crop is worse than a too-large one —
       upscaling garbage gives the classifier no extra signal.
    6. If the crop is larger than ``max_side_px``, downscales with
       Pillow's LANCZOS filter.  Keeps the published blob under MQTT's
       payload limits and matches the classifier input order of magnitude.
    7. Writes JPEG to *dst_path* at ``jpeg_quality``.

    Parameters
    ----------
    src_path:
        Path to the full-resolution source JPEG.
    dst_path:
        Path to write the cropped JPEG.  Parent directories are not
        created.  Safe to reuse ``src_path`` — the write happens through
        Pillow so in-place overwrite is fine.
    bbox_fraction:
        ``(x0, y0, x1, y1)`` in ``[0.0, 1.0]``, half-open; typically
        straight from :attr:`MotionResult.bbox_fraction`.
    padding:
        Fractional expansion of the bbox's **longer** side, applied on
        each edge.  ``0.30`` (the default) enlarges a 200×200 bbox by
        60 px on each side → 320×320 before squaring.
    min_side_px:
        Enforce a minimum crop side, in pixels.  Default matches the
        iNaturalist classifier input resolution (224) so we never resize
        *up* into the classifier.
    max_side_px:
        Enforce a maximum crop side, in pixels.  Default 896 is 4× the
        classifier input — generous headroom for future higher-res
        models, still small enough to fit MQTT payload budgets.
    jpeg_quality:
        Pillow JPEG quality.  Default 90 matches Horus's capture-side
        quality so we don't recompress at lower fidelity.

    Returns
    -------
    ``(width, height)`` of the written crop, in pixels.  Always square
    (``width == height``) unless the source image itself is smaller than
    ``min_side_px`` on one axis.
    """
    with Image.open(src_path) as im:
        im.load()  # force read while the file handle is open
        W, H = im.size
        x0, y0, x1, y1 = bbox_fraction
        # Step 1: fractional bbox -> pixel bbox in the full-res frame.
        px0 = int(round(x0 * W))
        py0 = int(round(y0 * H))
        px1 = int(round(x1 * W))
        py1 = int(round(y1 * H))

        # Step 2: pad by `padding` * longer_side on each edge.
        bw = max(1, px1 - px0)
        bh = max(1, py1 - py0)
        pad = int(round(padding * max(bw, bh)))
        px0 -= pad
        py0 -= pad
        px1 += pad
        py1 += pad

        # Step 3: square from the center.
        cx = (px0 + px1) / 2
        cy = (py0 + py1) / 2
        side = max(px1 - px0, py1 - py0)
        # Step 5 (min) folded in here — cheaper than re-centering twice.
        side = max(side, min_side_px)
        half = side / 2
        px0 = int(round(cx - half))
        py0 = int(round(cy - half))
        px1 = int(round(cx + half))
        py1 = int(round(cy + half))

        # Step 4: clamp to image bounds. Preserves square-ness only when
        # the center isn't pinned to an edge; a small non-squareness is
        # acceptable — the classifier's resize will handle it.
        px0 = max(0, px0)
        py0 = max(0, py0)
        px1 = min(W, px1)
        py1 = min(H, py1)

        cropped = im.crop((px0, py0, px1, py1))

        # Step 6: downscale if too large.
        cw, ch = cropped.size
        if max(cw, ch) > max_side_px:
            scale = max_side_px / max(cw, ch)
            cropped = cropped.resize(
                (int(round(cw * scale)), int(round(ch * scale))),
                Image.LANCZOS,
            )

        # Step 7: save. Convert to RGB in case source had an alpha channel
        # (rpicam-still emits JPEG so it won't, but the helper is reused
        # in tests that use PNG fixtures).
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        cropped.save(dst_path, format="JPEG", quality=jpeg_quality)
        return cropped.size
