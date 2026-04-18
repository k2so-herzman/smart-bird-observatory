"""Tests for banshee.events — payload → dataclass decoding.

Focus areas:

* Backwards compatibility: payloads that predate the observability PR
  (no bbox_fraction/af keys) must still decode successfully.
* Observability fields: when present, bbox_fraction and af must round-trip
  through ``from_payload`` unchanged and with the right types.
* Shape validation: malformed optional fields raise :class:`EventError`
  rather than silently producing a bogus dataclass.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone

import pytest

from banshee.events import EventError, ImageEvent


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


def test_from_payload_rejects_bbox_with_wrong_length() -> None:
    p = _base_payload()
    p["bbox_fraction"] = [0.1, 0.2, 0.5]  # missing y1
    with pytest.raises(EventError, match="bbox_fraction"):
        ImageEvent.from_payload(p)


def test_from_payload_rejects_non_numeric_bbox() -> None:
    p = _base_payload()
    p["bbox_fraction"] = ["a", "b", "c", "d"]
    with pytest.raises(EventError, match="bbox_fraction"):
        ImageEvent.from_payload(p)


def test_from_payload_rejects_non_dict_af() -> None:
    """If af is present but isn't a dict, something upstream is broken —
    fail loud instead of silently dropping."""
    p = _base_payload()
    p["af"] = "LensPosition=3.0"  # string instead of dict
    with pytest.raises(EventError, match="af"):
        ImageEvent.from_payload(p)


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
