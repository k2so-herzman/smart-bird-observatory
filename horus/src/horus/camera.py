"""Camera capture via rpicam-still.

Keeps the dependency footprint minimal — no picamera2, no libcamera
python bindings. Just subprocess to the system `rpicam-still` binary,
which is already installed on Debian 13 / Bookworm Pi OS.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import CaptureConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when rpicam-still fails."""


def capture(output_path: Path, cfg: CaptureConfig, timeout_ms: int = 2000) -> Path:
    """Capture one still to `output_path`.

    Returns the path on success. Raises CameraError on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    return output_path
