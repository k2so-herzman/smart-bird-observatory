"""Simple frame-difference motion gate.

Holds a running baseline of the last grayscale thumbnail and compares
each new frame against it. Cheap enough for a Pi 3 B+ at 1Hz.
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
    motion: bool
    changed_fraction: float


class MotionGate:
    """Stateful frame-diff gate. Call `check(path)` for each new frame."""

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
