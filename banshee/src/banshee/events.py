"""Event type definitions for Banshee.

Mirrors the wire contract documented in shared/schema.md. We keep this
lightweight — just enough to validate structure and decode the image.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger(__name__)


class EventError(ValueError):
    """Raised when an event payload fails validation."""


@dataclass(frozen=True)
class ImageEvent:
    schema_version: int
    station: str
    captured_at: datetime
    camera: str
    trigger: str
    resolution: tuple[int, int]
    content_type: str
    image_bytes: bytes
    size_bytes: int
    changed_fraction: float
    sha256: str
    # Optional diagnostic fields — present on events from horus builds
    # with the observability PR, absent on older payloads.  Both default
    # to None so the schema stays backwards-compatible with archived
    # MQTT traffic and older clients.
    bbox_fraction: tuple[float, float, float, float] | None = None
    af: dict | None = None
    # Horus-side bird classifier output, when the station has
    # on-device classification enabled. Passed through unchanged so
    # downstream tooling can compare to Thoth's post-ingest score.
    bird_score: float | None = None
    bird_label: str | None = None
    # Burst-session metadata — populated by horus's session-based
    # capture gate (see horus/src/horus/main.py::_publish_flow).
    # Frames from the same visit share a ``burst_id`` with a
    # monotonically increasing ``burst_seq`` starting at 1. Absent on
    # payloads from stations running with ``burst.enabled = false`` or
    # older horus builds that pre-date the burst schema; Thoth treats a
    # missing burst_id as "this frame is its own singleton burst" so
    # legacy traffic still renders in the UI.
    burst_id: str | None = None
    burst_seq: int | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> "ImageEvent":
        try:
            schema_version = int(payload["schema_version"])
            station = str(payload["station"])
            captured_at = datetime.fromisoformat(payload["captured_at"])
            camera = str(payload["camera"])
            trigger = str(payload["trigger"])
            res = payload["resolution"]
            content_type = str(payload.get("content_type", "image/jpeg"))
            image_b64 = payload["image_b64"]
            size_bytes = int(payload["size_bytes"])
            changed_fraction = float(payload["changed_fraction"])
            claimed_sha = str(payload["sha256"])
        except (KeyError, ValueError, TypeError) as exc:
            raise EventError(f"invalid image event payload: {exc}") from exc

        if schema_version != 1:
            raise EventError(f"unsupported schema_version: {schema_version}")

        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except Exception as exc:
            raise EventError(f"image_b64 decode failed: {exc}") from exc

        if len(image_bytes) != size_bytes:
            raise EventError(
                f"size_bytes mismatch: claimed {size_bytes}, decoded {len(image_bytes)}"
            )

        actual_sha = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha != claimed_sha:
            raise EventError(
                f"sha256 mismatch: claimed {claimed_sha}, computed {actual_sha}"
            )

        # Optional diagnostic fields.  Validate shape loosely: a malformed
        # bbox (wrong length, non-numeric) is not worth dropping a real
        # event over — we drop the field, log at WARNING, and keep going.
        # The image still classifies; we just lose the crop metadata for
        # that one event.
        bbox_raw = payload.get("bbox_fraction")
        bbox: tuple[float, float, float, float] | None
        if bbox_raw is None:
            bbox = None
        else:
            try:
                if len(bbox_raw) != 4:
                    raise ValueError(f"expected 4 elements, got {len(bbox_raw)}")
                bbox = (
                    float(bbox_raw[0]),
                    float(bbox_raw[1]),
                    float(bbox_raw[2]),
                    float(bbox_raw[3]),
                )
            except (TypeError, ValueError) as exc:
                log.warning("dropping malformed bbox_fraction: %s", exc)
                bbox = None

        af_raw = payload.get("af")
        af: dict | None
        if af_raw is None:
            af = None
        elif isinstance(af_raw, dict):
            af = dict(af_raw)  # defensive copy — dataclass is frozen but dict isn't
        else:
            log.warning(
                "dropping malformed af (expected dict, got %s)",
                type(af_raw).__name__,
            )
            af = None

        # Horus on-device bird classifier output. Both fields optional,
        # malformed values degrade gracefully (log + None) so a garbage
        # score never blocks ingest of an otherwise valid image.
        bird_score_raw = payload.get("bird_score")
        bird_score: float | None
        if bird_score_raw is None:
            bird_score = None
        else:
            try:
                bird_score = float(bird_score_raw)
            except (TypeError, ValueError) as exc:
                log.warning("dropping malformed bird_score: %s", exc)
                bird_score = None

        bird_label_raw = payload.get("bird_label")
        bird_label: str | None
        if bird_label_raw is None:
            bird_label = None
        elif isinstance(bird_label_raw, str):
            bird_label = bird_label_raw
        else:
            log.warning(
                "dropping non-string bird_label (got %s)",
                type(bird_label_raw).__name__,
            )
            bird_label = None

        # Burst metadata — missing on singleton/legacy payloads; degrade
        # to None rather than raise so Thoth can still ingest the frame
        # as a one-shot event.
        burst_id_raw = payload.get("burst_id")
        burst_id: str | None
        if burst_id_raw is None:
            burst_id = None
        elif isinstance(burst_id_raw, str) and burst_id_raw:
            burst_id = burst_id_raw
        else:
            log.warning(
                "dropping malformed burst_id (expected non-empty str, got %r)",
                burst_id_raw,
            )
            burst_id = None

        burst_seq_raw = payload.get("burst_seq")
        burst_seq: int | None
        if burst_seq_raw is None:
            burst_seq = None
        else:
            try:
                burst_seq = int(burst_seq_raw)
                if burst_seq < 1:
                    raise ValueError(f"burst_seq must be >= 1, got {burst_seq}")
            except (TypeError, ValueError) as exc:
                log.warning("dropping malformed burst_seq: %s", exc)
                burst_seq = None

        return cls(
            schema_version=schema_version,
            station=station,
            captured_at=captured_at,
            camera=camera,
            trigger=trigger,
            resolution=(int(res[0]), int(res[1])),
            content_type=content_type,
            image_bytes=image_bytes,
            size_bytes=size_bytes,
            changed_fraction=changed_fraction,
            sha256=actual_sha,
            bbox_fraction=bbox,
            af=af,
            bird_score=bird_score,
            bird_label=bird_label,
            burst_id=burst_id,
            burst_seq=burst_seq,
        )


@dataclass(frozen=True)
class StatusEvent:
    schema_version: int
    station: str
    ts: datetime
    raw: dict

    @classmethod
    def from_payload(cls, payload: dict) -> "StatusEvent":
        try:
            return cls(
                schema_version=int(payload["schema_version"]),
                station=str(payload["station"]),
                ts=datetime.fromisoformat(payload["ts"]),
                raw=payload,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise EventError(f"invalid status payload: {exc}") from exc
