"""Tests for the pluggable classifier model layer.

The ``TFLiteClassifier`` requires ``tflite-runtime`` + ``numpy`` which
are optional deps; those paths are exercised in the Thoth LXC with a
real model artifact, not here. These tests cover the dummy fallback
and the result dataclass surface.
"""

from __future__ import annotations

import pytest

from banshee.classifier.model import (
    ClassificationResult,
    Classifier,
    DummyClassifier,
)


def test_dummy_returns_sentinel_label_and_zero_confidence() -> None:
    dummy = DummyClassifier()
    result = dummy.classify(b"\xff\xd8\xff\xe0any-bytes")
    assert isinstance(result, ClassificationResult)
    assert result.species == "unclassified"
    assert result.confidence == pytest.approx(0.0)


def test_dummy_is_stateless_across_calls() -> None:
    """Successive calls return equal results — no hidden counter/randomness."""
    dummy = DummyClassifier()
    first = dummy.classify(b"a")
    second = dummy.classify(b"b" * 2048)
    assert first == second


def test_dummy_satisfies_classifier_protocol() -> None:
    """Static check: DummyClassifier is structurally a Classifier."""
    dummy: Classifier = DummyClassifier()  # mypy-style assignment
    assert callable(dummy.classify)
