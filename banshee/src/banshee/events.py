"""Event type definitions for Banshee.

Mirrors the wire contract documented in shared/schema.md. We keep this
lightweight — just enough to validate structure and decode the image.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime


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
