"""Camera capture via rpicam-still.

Keeps the dependency footprint minimal — no picamera2, no libcamera
python bindings. Just subprocess to the system `rpicam-still` binary,
which is already installed on Debian 13 / Bookworm Pi OS.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from .config import CaptureConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when rpicam-still fails."""


# Subset of rpicam-still metadata fields we surface in INFO logs.
# Full metadata is always persisted to the sidecar JSON file regardless.
_AF_LOG_FIELDS = ("LensPosition", "AfState", "FocusFoM")


def metadata_path_for(output_path: Path) -> Path:
    """Return the sidecar metadata path for a given image path.

    ``foo/bar/123.jpg`` → ``foo/bar/123.jpg.json``.  Sidecar-with-suffix
    pattern keeps the stem coupled to the image so directory listings
    stay grouped.
    """
    return output_path.with_suffix(output_path.suffix + ".json")


def discard(image_path: Path) -> None:
    """Delete an image and its metadata sidecar if present.

    Used by the capture daemon to drop frames that failed the motion
    gate.  Missing files are silently ignored so callers don't need to
    track whether a capture actually produced output.
    """
    for p in (image_path, metadata_path_for(image_path)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _log_af_summary(metadata_path: Path) -> None:
    """Read the metadata JSON and log key AF fields at INFO level.

    Failures are logged at DEBUG and swallowed — metadata logging is a
    diagnostic aid, not a capture requirement.
    """
    try:
        with metadata_path.open() as f:
            md = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("could not parse metadata at %s: %s", metadata_path, exc)
        return

    summary = {k: md.get(k) for k in _AF_LOG_FIELDS if k in md}
    if summary:
        log.info("af metadata: %s", summary)


def capture(output_path: Path, cfg: CaptureConfig, timeout_ms: int = 2000) -> Path:
    """Capture one still to `output_path`.

    Also writes a sidecar metadata JSON to ``output_path.json`` containing
    the full libcamera per-capture metadata (LensPosition, AfState,
    FocusFoM, ExposureTime, AnalogueGain, ColourGains, etc.).  Key AF
    fields are logged at INFO level for ops visibility.

    Returns the image path on success.  The sidecar path can be derived
    by the caller via ``output_path.with_suffix(output_path.suffix + ".json")``.

    Raises CameraError on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path_for(output_path)

    cmd: list[str] = [
        "rpicam-still",
        "-n",  # no preview
        "-o",
        str(output_path),
        "--width",
        str(cfg.width),
        "--height",
        str(cfg.height),
        "--timeout",
        str(timeout_ms),
        "-q",
        str(cfg.jpeg_quality),
        "--metadata",
        str(metadata_path),
        "--metadata-format",
        "json",
    ]
    cmd.extend(cfg.rpicam_extra_args)

    log.debug("running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_ms / 1000 + 5,
    )
    if result.returncode != 0:
        raise CameraError(
            f"rpicam-still exited {result.returncode}: {result.stderr.strip()}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise CameraError(f"rpicam-still produced no output at {output_path}")

    _log_af_summary(metadata_path)

    return output_path
