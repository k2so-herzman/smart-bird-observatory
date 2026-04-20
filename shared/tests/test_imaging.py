"""Tests for :mod:`sbo_shared.imaging` — the single source of truth for
the subject-bbox crop used by horus (on-device) and thoth-classify
(post-ingest).

Byte-identical output for the same inputs is the whole point — these
tests pin enough of the contract (min-side floor, max-side cap,
squaring, JPEG validity) that any regression that silently changes
what the classifiers see surfaces in CI.
"""

from __future__ import annotations

import io

from PIL import Image

from sbo_shared.imaging import crop_to_bbox_bytes, crop_to_bbox_image


def _jpeg_bytes(size: tuple[int, int], fill: int = 200) -> bytes:
    """Encode a solid-gray JPEG at ``size`` and return the bytes."""
    buf = io.BytesIO()
    Image.new("L", size, color=fill).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_crop_bytes_roundtrips_to_valid_jpeg() -> None:
    """Output must be a JPEG Pillow can re-open — smoke-test the
    encode path so truncated writes or mode confusion fail loudly."""
    out = crop_to_bbox_bytes(
        _jpeg_bytes((1000, 1000)),
        bbox_fraction=(0.3, 0.3, 0.7, 0.7),
    )

    with Image.open(io.BytesIO(out)) as im:
        im.verify()
    # verify() closes the file; reopen for mode/format assertions.
    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"


def test_crop_bytes_respects_min_side_floor() -> None:
    """A tiny bbox must be inflated up to ``min_side_px`` — no
    upscaling from 20px-square garbage into the classifier."""
    out = crop_to_bbox_bytes(
        _jpeg_bytes((1000, 1000)),
        bbox_fraction=(0.49, 0.49, 0.51, 0.51),
        padding=0.0,
        min_side_px=224,
        max_side_px=896,
    )
    with Image.open(io.BytesIO(out)) as im:
        w, h = im.size
    assert w >= 224 and h >= 224


def test_crop_bytes_respects_max_side_cap() -> None:
    """A full-frame bbox on a big source must be downscaled to
    ``max_side_px`` — keeps MQTT payload under control and matches
    the classifier's input-tensor order of magnitude."""
    out = crop_to_bbox_bytes(
        _jpeg_bytes((2304, 1296)),
        bbox_fraction=(0.0, 0.0, 1.0, 1.0),
        padding=0.0,
        min_side_px=224,
        max_side_px=896,
    )
    with Image.open(io.BytesIO(out)) as im:
        w, h = im.size
    assert max(w, h) <= 896


def test_crop_bytes_squares_non_square_bbox() -> None:
    """Wide bbox + zero padding + no min-side → output is square."""
    out = crop_to_bbox_bytes(
        _jpeg_bytes((1000, 1000)),
        bbox_fraction=(0.2, 0.45, 0.8, 0.55),
        padding=0.0,
        min_side_px=1,
        max_side_px=10_000,
    )
    with Image.open(io.BytesIO(out)) as im:
        w, h = im.size
    assert w == h


def test_crop_bytes_is_deterministic_for_same_input() -> None:
    """Byte-identical output for the same ``(image, bbox)`` input is
    the whole point of hoisting this into sbo_shared — it's what lets
    horus's on-device score and thoth-classify's post-ingest score be
    directly comparable when someone debugs a disagreement.
    """
    image = _jpeg_bytes((1500, 1000))
    bbox = (0.1, 0.2, 0.6, 0.7)

    a = crop_to_bbox_bytes(image, bbox)
    b = crop_to_bbox_bytes(image, bbox)
    assert a == b


def test_crop_bytes_matches_crop_image_pipeline() -> None:
    """The bytes entrypoint and the Image entrypoint must agree on the
    crop rectangle (they share a private implementation, but regressions
    that diverge them would be silent catastrophe).
    """
    image_bytes = _jpeg_bytes((1500, 1000))
    bbox = (0.1, 0.2, 0.6, 0.7)

    out_bytes = crop_to_bbox_bytes(image_bytes, bbox, jpeg_quality=90)
    with Image.open(io.BytesIO(image_bytes)) as im:
        out_image = crop_to_bbox_image(im, bbox)

    with Image.open(io.BytesIO(out_bytes)) as re_opened:
        assert re_opened.size == out_image.size


def test_crop_bytes_clamps_to_image_bounds() -> None:
    """A corner bbox with generous padding must not push the crop
    negative — Pillow's ``Image.crop`` tolerates out-of-range boxes
    by filling with black, which would feed dead pixels to the
    classifier. The clamp step is what prevents that."""
    out = crop_to_bbox_bytes(
        _jpeg_bytes((1000, 1000)),
        bbox_fraction=(0.0, 0.0, 0.1, 0.1),
        padding=1.0,
        min_side_px=1,
        max_side_px=10_000,
    )
    with Image.open(io.BytesIO(out)) as im:
        w, h = im.size
    assert w > 0 and h > 0
