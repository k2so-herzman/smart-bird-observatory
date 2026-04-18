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
    """

    motion: bool
    changed_fraction: float


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
            return MotionResult(False, 0.0)

        diff = np.abs(thumb - self._baseline)
        changed = (diff > self.cfg.pixel_threshold).sum()
        fraction = changed / thumb.size

        # Slow baseline update so lighting drift doesn't trigger.
        self._baseline = (0.9 * self._baseline + 0.1 * thumb).astype(np.int16)

        return MotionResult(
            motion=fraction >= self.cfg.frame_fraction,
            changed_fraction=float(fraction),
        )
