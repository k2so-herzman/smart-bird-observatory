"""Tests for :mod:`horus.motion`: :class:`MotionGate` and :func:`crop_to_bbox`.

The bbox path is the bird-centered crop pipeline's foundation — if bbox
extraction or cropping is wrong, the classifier sees garbage and
confidence collapses.  These tests pin the contract on synthetic images
so regressions surface in CI rather than at 5am on the feeder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from horus.config import MotionConfig
from horus.motion import MotionGate, crop_to_bbox


def _write_gray(path: Path, size: tuple[int, int], fill: int) -> None:
    """Write a solid-gray JPEG of *size* filled with *fill* (0–255)."""
    Image.new("L", size, color=fill).save(path, format="JPEG", quality=95)


def _write_with_rect(
    path: Path,
    size: tuple[int, int],
    rect: tuple[int, int, int, int],
    bg: int = 128,
    fg: int = 255,
) -> None:
    """Write a JPEG that's solid *bg* everywhere except *rect* (x0,y0,x1,y1) which is *fg*."""
    img = Image.new("L", size, color=bg)
    arr = np.asarray(img, dtype=np.uint8).copy()
    x0, y0, x1, y1 = rect
    arr[y0:y1, x0:x1] = fg
    Image.fromarray(arr).save(path, format="JPEG", quality=95)


# ---------------------------------------------------------------------------
# MotionGate.check — baseline + bbox behavior
# ---------------------------------------------------------------------------


def test_first_call_establishes_baseline_and_has_no_bbox(tmp_path: Path) -> None:
    """First call returns motion=False, bbox=None — nothing to compare against yet."""
    frame = tmp_path / "first.jpg"
    _write_gray(frame, (320, 180), fill=100)

    gate = MotionGate(MotionConfig())
    result = gate.check(frame)

    assert result.motion is False
    assert result.changed_fraction == 0.0
    assert result.bbox_fraction is None


def test_unchanged_second_frame_has_no_bbox(tmp_path: Path) -> None:
    """Identical second frame → no changed pixels → bbox is None."""
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_gray(first, (320, 180), fill=100)
    _write_gray(second, (320, 180), fill=100)

    gate = MotionGate(MotionConfig())
    gate.check(first)
    result = gate.check(second)

    assert result.motion is False
    assert result.bbox_fraction is None


def test_motion_in_top_left_produces_bbox_in_top_left(tmp_path: Path) -> None:
    """A bright square in the top-left quadrant → bbox fraction in the top-left."""
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_gray(first, (320, 180), fill=100)
    # Rectangle covering the top-left ~quarter: (0,0) to (160,90).
    _write_with_rect(second, (320, 180), (0, 0, 160, 90), bg=100, fg=250)

    gate = MotionGate(MotionConfig(pixel_threshold=25, frame_fraction=0.01))
    gate.check(first)
    result = gate.check(second)

    assert result.motion is True
    assert result.bbox_fraction is not None
    x0, y0, x1, y1 = result.bbox_fraction

    # Tight fractional bbox should hug the top-left quadrant.  JPEG
    # compression bleeds the edge slightly so we don't pin exact values;
    # we pin the quadrant.
    assert 0.0 <= x0 < 0.1
    assert 0.0 <= y0 < 0.1
    assert 0.4 < x1 < 0.6
    assert 0.4 < y1 < 0.6


def test_motion_in_bottom_right_produces_bbox_in_bottom_right(tmp_path: Path) -> None:
    """Symmetric sanity check: bright rectangle in bottom-right → bbox in bottom-right."""
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_gray(first, (320, 180), fill=100)
    _write_with_rect(second, (320, 180), (240, 135, 320, 180), bg=100, fg=250)

    gate = MotionGate(MotionConfig(pixel_threshold=25, frame_fraction=0.005))
    gate.check(first)
    result = gate.check(second)

    assert result.bbox_fraction is not None
    x0, y0, x1, y1 = result.bbox_fraction
    assert x0 > 0.5
    assert y0 > 0.5
    assert 0.95 <= x1 <= 1.0
    assert 0.95 <= y1 <= 1.0


def test_bbox_coordinates_are_well_formed(tmp_path: Path) -> None:
    """x0 < x1, y0 < y1, all in [0, 1]."""
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_gray(first, (320, 180), fill=100)
    _write_with_rect(second, (320, 180), (100, 60, 220, 140), bg=100, fg=250)

    gate = MotionGate(MotionConfig(pixel_threshold=25, frame_fraction=0.005))
    gate.check(first)
    result = gate.check(second)

    assert result.bbox_fraction is not None
    x0, y0, x1, y1 = result.bbox_fraction
    assert 0.0 <= x0 < x1 <= 1.0
    assert 0.0 <= y0 < y1 <= 1.0


# ---------------------------------------------------------------------------
# crop_to_bbox — dimensions, padding, squaring, size clamps
# ---------------------------------------------------------------------------


def test_crop_produces_square_output_for_non_square_bbox(tmp_path: Path) -> None:
    """A wide bbox gets squared up by expanding the shorter axis."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    _write_gray(src, (1000, 1000), fill=200)

    # Wide bbox: 60% wide, 10% tall, centered vertically.
    w, h = crop_to_bbox(
        src,
        dst,
        bbox_fraction=(0.2, 0.45, 0.8, 0.55),
        padding=0.0,  # disable padding so we test squaring in isolation
        min_side_px=1,  # disable min-side so we test squaring alone
        max_side_px=10_000,
    )
    assert w == h, f"crop should be square, got {w}x{h}"


def test_crop_enforces_minimum_side(tmp_path: Path) -> None:
    """A tiny bbox gets expanded up to min_side_px."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    _write_gray(src, (1000, 1000), fill=200)

    # Bbox is 2% × 2% of a 1000×1000 = 20×20 px source bbox.
    w, h = crop_to_bbox(
        src,
        dst,
        bbox_fraction=(0.49, 0.49, 0.51, 0.51),
        padding=0.0,
        min_side_px=224,
        max_side_px=896,
    )
    assert w >= 224 and h >= 224, f"crop must be >= min_side_px, got {w}x{h}"


def test_crop_clamps_to_max_side(tmp_path: Path) -> None:
    """A full-frame bbox on a huge source is downscaled to max_side_px."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    # Simulate a 2304x1296 Horus capture.
    _write_gray(src, (2304, 1296), fill=200)

    w, h = crop_to_bbox(
        src,
        dst,
        bbox_fraction=(0.0, 0.0, 1.0, 1.0),
        padding=0.0,
        min_side_px=224,
        max_side_px=896,
    )
    assert max(w, h) <= 896, f"crop must be <= max_side_px, got {w}x{h}"


def test_crop_clamps_to_image_bounds(tmp_path: Path) -> None:
    """A bbox against the edge + padding must not push the crop rect negative."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    _write_gray(src, (1000, 1000), fill=200)

    # Corner bbox + generous padding — would go negative without clamping.
    w, h = crop_to_bbox(
        src,
        dst,
        bbox_fraction=(0.0, 0.0, 0.1, 0.1),
        padding=1.0,  # pad by 100% of longer side on each edge
        min_side_px=1,
        max_side_px=10_000,
    )
    # Expected: crop didn't crash and produced a non-empty image.
    with Image.open(dst) as saved:
        assert saved.size == (w, h)
        assert w > 0 and h > 0


def test_crop_writes_valid_jpeg(tmp_path: Path) -> None:
    """Output must be a readable JPEG, not a truncated write."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    _write_gray(src, (1000, 1000), fill=200)

    crop_to_bbox(src, dst, bbox_fraction=(0.3, 0.3, 0.7, 0.7))

    with Image.open(dst) as saved:
        saved.verify()  # raises if the JPEG is malformed
    # Re-open for actual use (verify() closes the file).
    with Image.open(dst) as saved:
        assert saved.format == "JPEG"
        assert saved.mode == "RGB"


def test_crop_accepts_source_eq_destination(tmp_path: Path) -> None:
    """In-place overwrite is supported — we load() the source before writing.

    Horus's capture loop may overwrite the ring-buffer JPEG with its crop
    to avoid a separate blob.  Pillow's lazy loading makes this subtle —
    verify we don't trip on it.
    """
    path = tmp_path / "inplace.jpg"
    _write_gray(path, (1000, 1000), fill=200)
    original_bytes = path.read_bytes()

    crop_to_bbox(path, path, bbox_fraction=(0.3, 0.3, 0.7, 0.7))

    new_bytes = path.read_bytes()
    assert new_bytes != original_bytes, "file should have been rewritten"
    with Image.open(path) as saved:
        saved.verify()


@pytest.mark.parametrize(
    "src_size,bbox",
    [
        ((2304, 1296), (0.1, 0.2, 0.5, 0.8)),
        ((1920, 1080), (0.4, 0.1, 0.9, 0.5)),
        ((640, 480), (0.0, 0.0, 0.3, 0.3)),
    ],
)
def test_crop_shape_invariants(
    tmp_path: Path,
    src_size: tuple[int, int],
    bbox: tuple[float, float, float, float],
) -> None:
    """Across different source aspect ratios, default knobs produce a valid square-ish crop."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    _write_gray(src, src_size, fill=200)

    w, h = crop_to_bbox(src, dst, bbox_fraction=bbox)

    assert w >= 1 and h >= 1
    assert w <= src_size[0] and h <= src_size[1] or (max(w, h) <= 896)
    # The crop is at most max_side_px; the exact floor depends on clamping
    # behavior when the center is near an edge, so we don't pin min strictly.
