"""Local ring-buffer storage for captured frames."""

from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import date, datetime, timedelta, timezone
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


# Characters that are safe to keep verbatim in a gated-archive filename.
# Anything else (spaces, punctuation, parentheses from species labels like
# "Ardea alba (Great Egret)") collapses to a dash so the filename is both
# shell-safe and human-readable.
_SAFE_LABEL_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def _sanitize_label(label: str | None) -> str:
    if not label:
        return "unknown"
    cleaned = "".join(c if c in _SAFE_LABEL_CHARS else "-" for c in label)
    # Collapse runs of dashes and strip leading/trailing dashes so the
    # filename stays tidy even on labels like "Ardea alba (Great Egret)".
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    return (cleaned or "unknown")[:48]


def save_gated_sample(
    archive_dir: Path,
    source_path: Path,
    *,
    score: float,
    label: str | None,
    timestamp: datetime | None = None,
) -> Path:
    """Copy a gated (classifier-dropped) image into the review archive.

    The filename encodes the score and label so a human reviewer can
    scan the directory and spot near-threshold misses without opening
    each file. Day-level subdirectories make :func:`prune_gated` cheap.

    Parameters
    ----------
    archive_dir:
        Root of the gated archive (e.g. ``/var/lib/horus/gated``).
        Created if it doesn't exist.
    source_path:
        The image the classifier scored and dropped. Copied, not moved
        — the capture-local ring buffer deletes the original on its own.
    score:
        Confidence the classifier returned. Included in the filename so
        the reviewer can sort / filter by near-threshold candidates.
    label:
        Top-1 species label. Sanitized to be filename-safe. ``None``
        becomes ``"unknown"`` so filenames are uniform.
    timestamp:
        Override for the event time (used in tests). Defaults to UTC now.
    """
    timestamp = timestamp or datetime.now(timezone.utc)
    day_dir = archive_dir / timestamp.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{timestamp.strftime('%Y%m%dT%H%M%S')}"
        f"_score-{score:.3f}"
        f"_{_sanitize_label(label)}"
    )
    dest = day_dir / f"{stem}{source_path.suffix}"
    shutil.copy2(source_path, dest)
    return dest


def prune_gated(archive_dir: Path, max_age_days: int) -> int:
    """Delete gated-archive day-directories older than ``max_age_days``.

    Returns the number of directories removed so callers can log a
    useful one-liner. Non-conforming entries (stray files, malformed
    directory names) are left alone — we only touch ``YYYY-MM-DD``
    directories so an operator-dropped README or reviewer notes file
    survives pruning.
    """
    if not archive_dir.exists():
        return 0
    cutoff: date = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).date()
    removed = 0
    for child in archive_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            dir_date = datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            # Not a date-shaped directory — leave it for the operator.
            continue
        if dir_date < cutoff:
            shutil.rmtree(child)
            removed += 1
    return removed
