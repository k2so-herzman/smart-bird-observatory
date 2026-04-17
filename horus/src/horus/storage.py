"""Local ring-buffer storage for captured frames."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .config import StorageConfig

log = logging.getLogger(__name__)


def prune(cfg: StorageConfig) -> None:
    """Delete oldest files until local_dir is under max_local_mb."""
    root = cfg.local_dir
    if not root.exists():
        return
    files = sorted(
        (p for p in root.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    while total_mb > cfg.max_local_mb and files:
        victim = files.pop(0)
        size_mb = victim.stat().st_size / (1024 * 1024)
        try:
            victim.unlink()
            total_mb -= size_mb
            log.debug("pruned %s (%.1f MB)", victim, size_mb)
        except OSError as exc:
            log.warning("could not prune %s: %s", victim, exc)


def next_capture_path(cfg: StorageConfig) -> Path:
    """Return a timestamped path inside local_dir for the next capture."""
    day = time.strftime("%Y-%m-%d")
    name = time.strftime("%H%M%S") + f"_{os.getpid()}.jpg"
    path = cfg.local_dir / day / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
