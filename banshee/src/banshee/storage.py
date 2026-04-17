"""On-disk image persistence for Banshee."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from .config import StorageConfig
from .events import ImageEvent

log = logging.getLogger(__name__)


def save_image(event: ImageEvent, cfg: StorageConfig) -> Path:
    """Write an event's JPEG to the image_dir.

    Path layout:  <image_dir>/<station>/YYYY-MM-DD/HHMMSS_<sha8>.jpg
    """
    day = event.captured_at.strftime("%Y-%m-%d")
    hms = event.captured_at.strftime("%H%M%S")
    sha_prefix = event.sha256[:8]
    station_dir = cfg.image_dir / event.station / day
    station_dir.mkdir(parents=True, exist_ok=True)

    path = station_dir / f"{hms}_{sha_prefix}.jpg"
    path.write_bytes(event.image_bytes)
    log.debug("saved %s (%d bytes)", path, len(event.image_bytes))
    return path


def prune_old(cfg: StorageConfig) -> int:
    """Delete images older than retention_days. Returns count deleted."""
    if cfg.retention_days <= 0:
        return 0
    if not cfg.image_dir.exists():
        return 0

    cutoff = datetime.now().timestamp() - cfg.retention_days * 86400
    deleted = 0
    for path in cfg.image_dir.rglob("*.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError as exc:
            log.warning("could not prune %s: %s", path, exc)
    return deleted
