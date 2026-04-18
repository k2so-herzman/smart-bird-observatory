"""Canonical timestamp format for every SBO event payload.

Every MQTT event Horus publishes and every row Banshee records goes
through :func:`sbo_now_iso`. Keeping a single implementation means:

- The two sides agree on precision (seconds) and timezone (UTC).
- If we ever need to switch to millisecond precision or a different
  encoding, it's a one-line change in one place.
"""

from __future__ import annotations

from datetime import datetime, timezone


def sbo_now_iso() -> str:
    """Return the current UTC time in ISO-8601 with second precision.

    Example output: ``"2026-04-18T00:32:22+00:00"``.

    The two properties callers rely on:

    1. **UTC**: ingestion services compare timestamps across stations
       that may be in different local zones, so every payload is UTC.
    2. **Second precision**: JPEG events fire at most a few times per
       minute, microsecond resolution is noise. Trimming it keeps
       payloads diffable and InfluxDB queries predictable.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
