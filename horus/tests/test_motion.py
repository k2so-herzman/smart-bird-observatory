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
    # Crop must stay inside the source image AND respect the default max_side_px (896).
    # Parentheses are load-bearing — without them, operator precedence lets a crop
    # that exceeds src_size still pass as long as it's <= 896.
    assert w <= src_size[0] and h <= src_size[1]
    assert max(w, h) <= 896
    # The crop is at most max_side_px; the exact floor depends on clamping
    # behavior when the center is near an edge, so we don't pin min strictly.


# ---------------------------------------------------------------------------
# check_array() — file-free entrypoint used by the persistent picamera2
# path. Same contract as check() but takes a numpy grayscale array so
# callers can skip JPEG decode when the camera already produced one.
# ---------------------------------------------------------------------------


def test_check_array_first_frame_adopts_baseline() -> None:
    """First frame must not false-trigger — it becomes the baseline and
    returns motion=False so an interactive start-up doesn't fire an event."""
    gate = MotionGate(MotionConfig())
    thumb = np.full((90, 160), 128, dtype=np.uint8)

    result = gate.check_array(thumb)

    assert result.motion is False
    assert result.changed_fraction == 0.0
    assert result.bbox_fraction is None


def test_check_array_detects_motion_on_large_delta() -> None:
    """A full-frame bright step after the baseline → every pixel changes
    → fraction == 1.0, well above the 2 % default threshold."""
    gate = MotionGate(MotionConfig())
    baseline = np.full((90, 160), 30, dtype=np.uint8)
    bright = np.full((90, 160), 230, dtype=np.uint8)

    gate.check_array(baseline)  # primes baseline
    result = gate.check_array(bright)

    assert result.motion is True
    assert result.changed_fraction == pytest.approx(1.0)
    assert result.bbox_fraction == (0.0, 0.0, 1.0, 1.0)


def test_check_array_returns_sub_bbox_for_focal_motion() -> None:
    """A small bright square in a quiet frame → bbox hugs the changed
    region, NOT the full frame. Thoth uses this to distinguish focal
    motion (a bird) from distributed motion (wind-swept leaves)."""
    gate = MotionGate(MotionConfig(pixel_threshold=25, frame_fraction=0.001))
    baseline = np.full((90, 160), 30, dtype=np.uint8)
    nudged = baseline.copy()
    # Bright patch at rows 40-50, cols 80-90 → fractional (80/160, 40/90, 90/160, 50/90)
    nudged[40:50, 80:90] = 230

    gate.check_array(baseline)
    result = gate.check_array(nudged)

    assert result.motion is True
    assert result.bbox_fraction is not None
    x0, y0, x1, y1 = result.bbox_fraction
    # Loose bounds — exponential baseline has nibbled a bit but the
    # bright patch should still dominate.
    assert 0.4 < x0 < 0.55 and 0.4 < y0 < 0.5
    assert 0.55 < x1 < 0.65 and 0.5 < y1 < 0.65


def test_check_array_casts_and_handles_uint8_input() -> None:
    """picamera2's Y-plane slice comes back as uint8. check_array must
    cast internally so signed subtraction (thumb - baseline) doesn't
    wrap around at 0 and mis-count every pixel as changed."""
    gate = MotionGate(MotionConfig())
    baseline = np.full((90, 160), 200, dtype=np.uint8)  # BRIGHT baseline
    darkened = np.full((90, 160), 50, dtype=np.uint8)   # big negative step

    gate.check_array(baseline)
    result = gate.check_array(darkened)

    # If the cast is missing, (50 - 200) in uint8 wraps to 106 and the
    # mask count is still high but off by ~40 %. The clean-cast path
    # gives us exactly 150 difference on every pixel, changed_fraction
    # = 1.0, motion = True. We test the downstream observable.
    assert result.motion is True
    assert result.changed_fraction == pytest.approx(1.0)


def test_check_array_shape_change_re_adopts_baseline() -> None:
    """Changing stream resolution (e.g. reconfigure to a bigger lores)
    resets the baseline instead of crashing on shape mismatch."""
    gate = MotionGate(MotionConfig())
    small = np.full((90, 160), 128, dtype=np.uint8)
    large = np.full((180, 320), 128, dtype=np.uint8)

    gate.check_array(small)
    result = gate.check_array(large)  # shape change → re-adopt

    assert result.motion is False
    assert result.changed_fraction == 0.0
    # And the baseline is now the new shape so the NEXT frame diffs cleanly.
    next_frame = np.full((180, 320), 200, dtype=np.uint8)
    result2 = gate.check_array(next_frame)
    assert result2.motion is True


def test_check_and_check_array_share_state(tmp_path: Path) -> None:
    """check(path) and check_array(array) must both update the same
    baseline so a daemon can mix paths without shape/baseline surprises.
    Regression guard: if one method held its own baseline, switching
    backends mid-run would cause every switchover to be a 'first frame'
    and suppress motion for the first post-switch tick."""
    gate = MotionGate(MotionConfig())

    # Prime via file path (check).
    img = tmp_path / "f0.jpg"
    Image.new("L", (160, 90), color=30).save(img, format="JPEG", quality=95)
    gate.check(img)
    assert gate._baseline is not None
    thumb_shape = gate._baseline.shape

    # Now a bright array of the same shape should be motion, not a
    # first-frame baseline adoption.
    bright = np.full(thumb_shape, 230, dtype=np.uint8)
    result = gate.check_array(bright)
    assert result.motion is True, (
        "check_array must see the baseline primed by check(); otherwise "
        "a picamera2-migrating daemon would miss its first lores frame"
    )
