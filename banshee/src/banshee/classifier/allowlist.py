"""Species allowlist / geography filter for the TFLite classifier.

The iNaturalist 965-class classifier has no geography awareness — it's
a pure image → class model. On feeder-framed input it frequently
picks a tropical or cross-continent look-alike over the actual local
species (e.g. "New Zealand Pigeon" or "Common Myna" on a Colorado
scrub-jay). See the "iNat classifier defaults to tropical species on
ambiguous inputs" note in the project memory for the field evidence.

This module implements an optional post-inference mask: callers
supply a list of allowed Latin binomials (Genus species), and we zero
out every non-listed logit before argmax runs. This is blunt — it
cannot save a genuinely out-of-list species — but it eliminates the
pathological false-global labels that plague local deployments, at
zero inference cost.

Matching is by binomial only
----------------------------
The iNat labels include some subspecies rows (e.g. ``"Junco hyemalis
caniceps"``). Those are allowed iff the base binomial ``"Junco
hyemalis"`` is in the allowlist, so operators only list species and
automatically get all subspecies. Conversely, listing only a subspecies
binomial is not supported — use the species binomial.

Fail-safe behavior
------------------
If zero labels match the allowlist (typo in the file, empty file) we
log a WARNING and return an all-True mask so the classifier degrades
to its unfiltered behavior rather than silently predicting the
``background`` class for every image. A broken allowlist is an
operator mistake; hiding it as "mystery confidence collapse" is worse
than disabling the filter and logging loudly.

File format
-----------
Newline-separated; ``#`` starts a comment; blank lines are ignored::

    # Colorado Front Range feeder birds
    Poecile gambeli
    Cyanocitta stelleri
    # ...

Only the first two whitespace-separated tokens on each line are used,
so a line like ``Poecile gambeli (Mountain Chickadee)`` parses fine
— operators can paste straight from the labels file.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


# iNat labels look like "Genus species (Common Name)" or
# "Genus species subspecies (Common Name)". We want the binomial —
# the first two Latin words. The regex is strict on capitalization
# (Genus capitalized, species lower) so "background" and English
# stray labels don't accidentally parse.
_BINOMIAL_RE = re.compile(r"^\s*([A-Z][a-z]+)\s+([a-z][a-z-]+)")


def extract_binomial(label: str) -> str | None:
    """Return ``"Genus species"`` from an iNat label, or ``None``.

    ``None`` for any label that doesn't parse as a Latin binomial
    (e.g. the ``background`` sentinel, or future hybrid labels with
    unusual formatting).
    """
    m = _BINOMIAL_RE.match(label)
    if not m:
        return None
    return f"{m.group(1)} {m.group(2)}"


def load_allowlist(path: Path) -> frozenset[str]:
    """Read a species allowlist file.

    Parameters
    ----------
    path:
        Filesystem path to a UTF-8 text file. Each line should start
        with a Latin binomial; the rest of the line (common name,
        annotations) is ignored. ``#`` starts a comment; blank lines
        are skipped.

    Returns
    -------
    frozenset[str]
        A set of ``"Genus species"`` strings. Frozen so callers can
        safely share it across threads / classifier instances without
        worrying about accidental mutation.
    """
    out: set[str] = set()
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0][:1].isupper():
            # Don't crash on a typo — log and move on. Matching is
            # by exact string anyway, so a bad row just fails to
            # whitelist anything; the harm is limited.
            log.warning(
                "allowlist %s:%d: skipping malformed line %r",
                path,
                lineno,
                raw,
            )
            continue
        out.add(f"{parts[0]} {parts[1]}")
    return frozenset(out)


def build_mask(labels: list[str], allowed: frozenset[str]) -> list[bool]:
    """Return a per-label ``True``/``False`` mask aligned with ``labels``.

    ``mask[i]`` is ``True`` iff label ``i`` is allowed. A label is
    allowed when its binomial (first two Latin words) is in
    ``allowed``. Labels that don't parse as a binomial (e.g.
    ``"background"``) are always allowed — a background / sentinel
    class must still be able to win against a blurry non-bird frame.

    The returned type is a plain ``list[bool]`` so this module stays
    numpy-free; callers that need an ``np.ndarray`` should do
    ``np.asarray(mask, dtype=bool)`` at their own dep boundary
    (typically inside the TFLite classifier where numpy is already
    imported).

    Fail-safe
    ---------
    If the resulting mask has zero ``True`` entries we log a WARNING
    and flip it to all-True. A zero mask would make argmax always
    return index 0, collapsing every prediction — an operator typo
    should not silently destroy classification. Better to disable
    the filter and surface the mismatch in the journal.
    """
    mask = [False] * len(labels)
    matched = 0
    for i, label in enumerate(labels):
        binomial = extract_binomial(label)
        if binomial is None:
            # Non-Latin labels (e.g. "background") always pass — they
            # are either sentinels or model quirks that the operator
            # hasn't been asked to curate.
            mask[i] = True
            continue
        if binomial in allowed:
            mask[i] = True
            matched += 1
    if matched == 0 and allowed:
        log.warning(
            "allowlist: zero of %d Latin labels matched the allowlist "
            "(size=%d); disabling mask to avoid degenerate predictions. "
            "Check binomial spellings — e.g. 'Poecile gambeli', not "
            "'Poecile gambeli (Mountain Chickadee)'",
            len(labels),
            len(allowed),
        )
        return [True] * len(labels)
    log.info(
        "allowlist: %d / %d Latin labels enabled (%d allowlist entries)",
        matched,
        sum(1 for lbl in labels if extract_binomial(lbl) is not None),
        len(allowed),
    )
    return mask
