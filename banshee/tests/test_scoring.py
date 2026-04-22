"""Tests for :mod:`banshee.scoring`.

Two surfaces under test:

1. :func:`laplacian_variance` — the PIL sharpness proxy. We synthesise a
   noisy image (random pixels) and a smooth image (flat gray) and
   assert the noisy one ranks higher. Absolute magnitude isn't
   contractual; relative ordering is.
2. :func:`hero_score` — the composite. We verify the weights, the
   range clamp, and None-handling for every input.
"""

from __future__ import annotations

import io
import random

import pytest
from PIL import Image

from banshee.scoring import hero_score, laplacian_variance


def _jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _noisy_image(size: tuple[int, int] = (320, 240), seed: int = 1) -> Image.Image:
    """Uniform-random pixels — high Laplacian response."""
    rng = random.Random(seed)
    pixels = bytes(rng.randint(0, 255) for _ in range(size[0] * size[1]))
    return Image.frombytes("L", size, pixels).convert("RGB")


def _flat_image(size: tuple[int, int] = (320, 240), value: int = 128) -> Image.Image:
    """Uniform gray — near-zero Laplacian response."""
    return Image.new("RGB", size, (value, value, value))


# ---- laplacian_variance ----------------------------------------------------


def test_laplacian_variance_noisy_beats_flat() -> None:
    noisy = laplacian_variance(_jpeg_bytes(_noisy_image()))
    flat = laplacian_variance(_jpeg_bytes(_flat_image()))
    assert noisy > flat
    # Flat should be very close to zero variance; tolerance because JPEG
    # adds its own high-frequency artefacts.
    assert flat < 50.0
    # Noisy should clear a clearly-sharp threshold — random uint8 has a
    # mean Laplacian variance well above any smooth content.
    assert noisy > 500.0


def test_laplacian_variance_returns_zero_on_garbage_bytes() -> None:
    # Non-decodable bytes — PIL raises, helper swallows + returns 0.
    assert laplacian_variance(b"not a jpeg at all") == 0.0


def test_laplacian_variance_returns_zero_on_empty_bytes() -> None:
    assert laplacian_variance(b"") == 0.0


# ---- hero_score -------------------------------------------------------------


def test_hero_score_weights_match_design() -> None:
    # With every input at its cap, the composite lands at exactly the
    # sum of the weights (which must be 1.0). Guards against weight
    # drift if someone edits the module without updating the design doc.
    full = hero_score(
        bird_score=1.0,
        sharpness=10_000.0,  # >> _SHARPNESS_NORM, clamps to 1.0
        bbox_fraction=(0.0, 0.0, 1.0, 1.0),
        classifier_confidence=1.0,
    )
    assert full == pytest.approx(1.0)


def test_hero_score_zero_when_everything_missing() -> None:
    # A payload with no detector, no bbox, no sharpness, no classifier
    # output should score zero — there's literally nothing to rank on.
    assert (
        hero_score(
            bird_score=None,
            sharpness=None,
            bbox_fraction=None,
            classifier_confidence=None,
        )
        == 0.0
    )


def test_hero_score_detector_contributes_0_4() -> None:
    # Isolating the detector term: everything else zeroed out.
    score = hero_score(
        bird_score=1.0,
        sharpness=0.0,
        bbox_fraction=None,
        classifier_confidence=None,
    )
    assert score == pytest.approx(0.4)


def test_hero_score_sharpness_normalizes_against_1000() -> None:
    # Variance of 500 → 0.5 on the sharpness axis → 0.15 composite.
    score = hero_score(
        bird_score=None,
        sharpness=500.0,
        bbox_fraction=None,
        classifier_confidence=None,
    )
    assert score == pytest.approx(0.15)


def test_hero_score_bbox_area_contributes_0_2() -> None:
    # 0.5 * 0.5 = 0.25 area → 0.05 composite.
    score = hero_score(
        bird_score=None,
        sharpness=None,
        bbox_fraction=(0.1, 0.1, 0.5, 0.5),
        classifier_confidence=None,
    )
    assert score == pytest.approx(0.2 * 0.25)


def test_hero_score_classifier_contributes_0_1() -> None:
    score = hero_score(
        bird_score=None,
        sharpness=None,
        bbox_fraction=None,
        classifier_confidence=0.8,
    )
    assert score == pytest.approx(0.08)


def test_hero_score_clamps_out_of_range_inputs() -> None:
    # Adversarial inputs shouldn't let any one term break the [0, 1]
    # contract. bird_score > 1 clamps; negative values clamp to 0.
    high = hero_score(
        bird_score=5.0,
        sharpness=1_000_000.0,
        bbox_fraction=(0, 0, 10.0, 10.0),
        classifier_confidence=2.0,
    )
    assert high == pytest.approx(1.0)

    low = hero_score(
        bird_score=-5.0,
        sharpness=-1.0,
        bbox_fraction=(0, 0, -1.0, -1.0),
        classifier_confidence=-2.0,
    )
    assert low == 0.0


def test_hero_score_ranks_sharper_frame_above_blurry_peer() -> None:
    """Two frames with identical detector+bbox — sharper wins."""
    blurry = hero_score(
        bird_score=0.8,
        sharpness=100.0,
        bbox_fraction=(0.2, 0.2, 0.3, 0.3),
        classifier_confidence=None,
    )
    sharp = hero_score(
        bird_score=0.8,
        sharpness=900.0,
        bbox_fraction=(0.2, 0.2, 0.3, 0.3),
        classifier_confidence=None,
    )
    assert sharp > blurry


def test_hero_score_malformed_bbox_falls_through_as_zero() -> None:
    # Length-3 tuple, non-numeric: both should contribute 0 rather
    # than raise. A malformed bbox should not take down a whole burst's
    # ranking.
    short = hero_score(
        bird_score=0.5,
        sharpness=None,
        bbox_fraction=(0.1, 0.1, 0.5),  # type: ignore[arg-type]
        classifier_confidence=None,
    )
    assert short == pytest.approx(0.2)  # just the detector term
