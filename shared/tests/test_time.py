"""Tests for sbo_shared.time.

Pins the two invariants every SBO service depends on:
UTC timezone offset + second precision (no microseconds).
"""
from __future__ import annotations

import re

from sbo_shared.time import sbo_now_iso


def test_sbo_now_iso_is_utc():
    ts = sbo_now_iso()
    assert ts.endswith("+00:00"), f"expected UTC offset, got {ts!r}"


def test_sbo_now_iso_has_second_precision():
    ts = sbo_now_iso()
    # YYYY-MM-DDTHH:MM:SS+00:00 — no fractional component.
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"
    assert re.match(pattern, ts), f"unexpected format: {ts!r}"


def test_sbo_now_iso_roundtrips_through_fromisoformat():
    """Downstream services parse these strings with fromisoformat — make
    sure what we emit is accepted by the stdlib parser."""
    from datetime import datetime

    ts = sbo_now_iso()
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.microsecond == 0
