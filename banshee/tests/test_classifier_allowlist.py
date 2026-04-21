"""Tests for the species allowlist / geography filter.

Keeps the assertions at the pure-function layer so the whole suite
runs without numpy or tflite-runtime — the same policy as
``test_classifier_model.py``. The end-to-end masking inside
:class:`~banshee.classifier.model.TFLiteClassifier` is covered on
the Thoth LXC with a real model artifact, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from banshee.classifier.allowlist import (
    build_mask,
    extract_binomial,
    load_allowlist,
)


# --- extract_binomial ------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Poecile gambeli (Mountain Chickadee)", "Poecile gambeli"),
        ("Cyanocitta stelleri (Steller's Jay)", "Cyanocitta stelleri"),
        # Subspecies row — we return the binomial only so operators
        # don't need to enumerate every sub-rank.
        ("Junco hyemalis caniceps (Gray-headed Junco)", "Junco hyemalis"),
        # Hyphenated species epithets are valid Latin (e.g. x-ray fish
        # is not a bird but the regex should handle it).
        ("Genus hyphen-species (Test)", "Genus hyphen-species"),
    ],
)
def test_extract_binomial_happy_path(label: str, expected: str) -> None:
    assert extract_binomial(label) == expected


@pytest.mark.parametrize(
    "label",
    [
        "background",  # sentinel class
        "",  # empty
        "   ",  # whitespace
        "lowercase Genus (Bad)",  # violates capitalization
        "123 456 (Not Latin)",
    ],
)
def test_extract_binomial_returns_none_on_non_latin(label: str) -> None:
    assert extract_binomial(label) is None


# --- load_allowlist -------------------------------------------------


def test_load_allowlist_parses_binomials(tmp_path: Path) -> None:
    f = tmp_path / "species.txt"
    f.write_text(
        "# Colorado feeder birds\n"
        "Poecile gambeli\n"
        "Cyanocitta stelleri\n"
        "\n"
        "Spinus pinus  # inline comment\n",
        encoding="utf-8",
    )
    allowed = load_allowlist(f)
    assert allowed == frozenset(
        {"Poecile gambeli", "Cyanocitta stelleri", "Spinus pinus"}
    )


def test_load_allowlist_ignores_extra_tokens(tmp_path: Path) -> None:
    """Operator pastes a full labels line — we take only the binomial."""
    f = tmp_path / "species.txt"
    f.write_text("Poecile gambeli (Mountain Chickadee)\n", encoding="utf-8")
    assert load_allowlist(f) == frozenset({"Poecile gambeli"})


def test_load_allowlist_skips_malformed_lines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = tmp_path / "species.txt"
    f.write_text(
        "Poecile gambeli\n"
        "notlatin\n"  # one token, skipped
        "lowercase genus\n"  # bad capitalization, skipped
        "Cyanocitta stelleri\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="banshee.classifier.allowlist"):
        allowed = load_allowlist(f)
    assert allowed == frozenset({"Poecile gambeli", "Cyanocitta stelleri"})
    # Both malformed lines should log a warning — operators need to
    # see the typo, not have it silently swallowed.
    assert sum(1 for r in caplog.records if "malformed" in r.getMessage()) == 2


def test_load_allowlist_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "species.txt"
    f.write_text("# only comments\n\n", encoding="utf-8")
    assert load_allowlist(f) == frozenset()


# --- build_mask ------------------------------------------------------


LABELS = [
    "Poecile gambeli (Mountain Chickadee)",
    "Cyanocitta stelleri (Steller's Jay)",
    "Hemiphaga novaeseelandiae (New Zealand Pigeon)",
    "Acridotheres tristis (Common Myna)",
    "Ardea alba (Great Egret)",
    "Junco hyemalis caniceps (Gray-headed Junco)",
    "background",
]


def test_build_mask_allows_listed_species_and_background() -> None:
    mask = build_mask(LABELS, frozenset({"Poecile gambeli", "Cyanocitta stelleri"}))
    assert mask == [
        True,  # Poecile gambeli
        True,  # Cyanocitta stelleri
        False,  # NZ Pigeon — blocked
        False,  # Common Myna — blocked
        False,  # Great Egret — blocked
        False,  # Junco hyemalis caniceps — parent binomial not listed
        True,  # background — non-Latin sentinel always passes
    ]


def test_build_mask_subspecies_passes_on_parent_binomial() -> None:
    """A subspecies row is allowed iff the base binomial is listed.

    Operators list 'Junco hyemalis' and every Junco subspecies row in
    the labels file automatically passes.
    """
    mask = build_mask(LABELS, frozenset({"Junco hyemalis"}))
    # Index 5 is "Junco hyemalis caniceps".
    assert mask[5] is True


def test_build_mask_failsafe_when_nothing_matches(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A typo in the allowlist must not silently collapse predictions.

    If zero Latin labels match, we flip the mask to all-True so the
    classifier degrades to its unfiltered behavior. A silent collapse
    would make every prediction "background" and would take a long
    time to diagnose in production.
    """
    with caplog.at_level(logging.WARNING, logger="banshee.classifier.allowlist"):
        mask = build_mask(LABELS, frozenset({"Notreal species"}))
    assert all(mask)
    assert any("disabling mask" in r.getMessage() for r in caplog.records)


def test_build_mask_empty_allowlist_allows_only_non_latin() -> None:
    """Empty allowlist is distinct from 'nothing matched'.

    An empty allowlist means the operator hasn't configured one — it
    should NOT trigger the fail-safe (which only fires when a
    non-empty list fails to match anything). Empty → mask out every
    Latin label, keep only the background sentinel. Callers pass
    ``None`` (not an empty set) to disable filtering entirely; this
    test just locks the semantics so a future refactor doesn't merge
    the two cases.
    """
    mask = build_mask(LABELS, frozenset())
    latin_indices = [i for i, lbl in enumerate(LABELS) if extract_binomial(lbl)]
    for i in latin_indices:
        assert mask[i] is False, f"label {LABELS[i]!r} should be blocked"
    # background (non-Latin) always passes
    assert mask[-1] is True


def test_build_mask_returns_plain_bool_list() -> None:
    """Contract: no numpy in the return type. Keeps this module
    importable in CI environments that skip the `classify` extra.
    """
    mask = build_mask(LABELS, frozenset({"Poecile gambeli"}))
    assert isinstance(mask, list)
    assert all(isinstance(x, bool) for x in mask)
