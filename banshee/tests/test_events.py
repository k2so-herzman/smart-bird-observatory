"""Tests for banshee.events — payload → dataclass decoding.

Focus areas:

* Backwards compatibility: payloads that predate the observability PR
  (no bbox_fraction/af keys) must still decode successfully.
* Observability fields: when present, bbox_fraction and af must round-trip
  through ``from_payload`` unchanged and with the right types.
* Shape validation: malformed optional fields degrade gracefully — the
  field is set to None and a WARNING is logged, but the event still
  decodes. Dropping the whole event over a malformed bbox would lose
  real bird imagery for a cosmetic metadata bug.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone

import pytest  # noqa: F401  # kept for potential future exception tests

from banshee.events import ImageEvent


def _base_payload() -> dict:
    """Minimum valid image-event payload — no optional fields."""
    body = b"\xff\xd8\xff\xe0fake-jpeg"
    return {
        "schema_version": 1,
        "station": "horus",
        "captured_at": datetime(2026, 4, 18, 21, 9, 0, tzinfo=timezone.utc).isoformat(),
        "camera": "imx519",
        "trigger": "motion",
        "resolution": [896, 504],
        "content_type": "image/jpeg",
        "image_b64": base64.b64encode(body).decode("ascii"),
        "size_bytes": len(body),
        "changed_fraction": 0.024,
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_from_payload_backwards_compatible_without_optional_fields() -> None:
    """Payloads from horus builds predating the observability PR have no
    bbox_fraction/af keys — decoding must succeed with both set to None."""
    ev = ImageEvent.from_payload(_base_payload())
    assert ev.bbox_fraction is None
    assert ev.af is None


def test_from_payload_preserves_bbox_fraction() -> None:
    p = _base_payload()
    p["bbox_fraction"] = [0.1, 0.2, 0.5, 0.8]
    ev = ImageEvent.from_payload(p)
    assert ev.bbox_fraction == (0.1, 0.2, 0.5, 0.8)


def test_from_payload_preserves_af_dict() -> None:
    p = _base_payload()
    p["af"] = {"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234}
    ev = ImageEvent.from_payload(p)
    assert ev.af == {"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234}


def test_from_payload_drops_bbox_with_wrong_length(caplog) -> None:
    """Malformed bbox must not drop the whole event — the image still
    classifies fine without crop metadata. We log a WARNING so ops notices
    if this starts happening systematically, then keep going with bbox=None."""
    p = _base_payload()
    p["bbox_fraction"] = [0.1, 0.2, 0.5]  # missing y1
    caplog.set_level("WARNING", logger="banshee.events")
    ev = ImageEvent.from_payload(p)
    assert ev.bbox_fraction is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "bbox" in r.message]
    assert warnings, "expected WARNING on malformed bbox"


def test_from_payload_drops_non_numeric_bbox(caplog) -> None:
    p = _base_payload()
    p["bbox_fraction"] = ["a", "b", "c", "d"]
    caplog.set_level("WARNING", logger="banshee.events")
    ev = ImageEvent.from_payload(p)
    assert ev.bbox_fraction is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "bbox" in r.message]
    assert warnings


def test_from_payload_drops_non_dict_af(caplog) -> None:
    """If af is present but isn't a dict, something upstream is broken —
    log WARNING so we notice, but don't drop the image event over it."""
    p = _base_payload()
    p["af"] = "LensPosition=3.0"  # string instead of dict
    caplog.set_level("WARNING", logger="banshee.events")
    ev = ImageEvent.from_payload(p)
    assert ev.af is None
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "af" in r.message]
    assert warnings


def test_from_payload_malformed_bbox_still_decodes_other_fields() -> None:
    """Graceful degradation: a malformed bbox must not affect any other
    field on the event. Sanity-check that the core payload round-trips
    even when the optional bbox is garbage."""
    p = _base_payload()
    p["bbox_fraction"] = "not-a-list"
    p["af"] = {"LensPosition": 3.02}
    ev = ImageEvent.from_payload(p)
    assert ev.bbox_fraction is None
    assert ev.af == {"LensPosition": 3.02}  # other optional field unaffected
    assert ev.station == "horus"
    assert ev.resolution == (896, 504)


def test_from_payload_accepts_empty_af_dict() -> None:
    """Sidecar parsed but had no AF keys → empty dict.  Meaningful (we
    tried) vs None (we didn't), so it must round-trip."""
    p = _base_payload()
    p["af"] = {}
    ev = ImageEvent.from_payload(p)
    assert ev.af == {}


def test_from_payload_af_dict_is_defensive_copy() -> None:
    """The decoded event is frozen but its ``af`` is a mutable dict.
    Mutating the original payload dict must not leak into the event."""
    p = _base_payload()
    af_in = {"LensPosition": 3.02}
    p["af"] = af_in
    ev = ImageEvent.from_payload(p)
    af_in["LensPosition"] = 99.9
    assert ev.af == {"LensPosition": 3.02}


# ---- burst metadata (PR-B) -------------------------------------------------


def test_from_payload_preserves_burst_metadata() -> None:
    """A horus payload with burst_id + burst_seq must round-trip both
    unchanged. This is the Thoth-side of the PR-A contract — without
    it, burst grouping in the event store has nothing to key on."""
    p = _base_payload()
    p["burst_id"] = "horus-1776898133321-de38"
    p["burst_seq"] = 3
    ev = ImageEvent.from_payload(p)
    assert ev.burst_id == "horus-1776898133321-de38"
    assert ev.burst_seq == 3


def test_from_payload_without_burst_fields_is_backwards_compatible() -> None:
    """Legacy (pre-burst) horus builds publish without burst_id/seq.
    Thoth must still decode them — the ingest treats them as singletons."""
    ev = ImageEvent.from_payload(_base_payload())
    assert ev.burst_id is None
    assert ev.burst_seq is None


def test_from_payload_coerces_string_burst_seq_to_int() -> None:
    """JSON encoders occasionally serialise integers as strings. As long
    as the value parses as an int, decode succeeds."""
    p = _base_payload()
    p["burst_id"] = "horus-1776898133321-de38"
    p["burst_seq"] = "5"
    ev = ImageEvent.from_payload(p)
    assert ev.burst_seq == 5


def test_from_payload_drops_empty_string_burst_id(caplog) -> None:
    """An empty burst_id is indistinguishable from "no burst" for our
    purposes — dropping it surfaces the bug via WARNING without crashing
    ingest."""
    p = _base_payload()
    p["burst_id"] = ""
    p["burst_seq"] = 1
    caplog.set_level("WARNING", logger="banshee.events")
    ev = ImageEvent.from_payload(p)
    assert ev.burst_id is None
    warnings = [r for r in caplog.records if "burst_id" in r.message]
    assert warnings


def test_from_payload_drops_zero_burst_seq(caplog) -> None:
    """horus publishes burst_seq starting at 1; zero means the publisher
    is broken. Drop with a log rather than persisting a nonsense seq."""
    p = _base_payload()
    p["burst_id"] = "horus-1776898133321-de38"
    p["burst_seq"] = 0
    caplog.set_level("WARNING", logger="banshee.events")
    ev = ImageEvent.from_payload(p)
    assert ev.burst_seq is None
    warnings = [r for r in caplog.records if "burst_seq" in r.message]
    assert warnings


def test_from_payload_drops_non_numeric_burst_seq(caplog) -> None:
    p = _base_payload()
    p["burst_id"] = "horus-1776898133321-de38"
    p["burst_seq"] = "not-a-number"
    caplog.set_level("WARNING", logger="banshee.events")
    ev = ImageEvent.from_payload(p)
    assert ev.burst_seq is None
    # burst_id must still survive — only the malformed field drops.
    assert ev.burst_id == "horus-1776898133321-de38"
