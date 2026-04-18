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


def read_af_fields(image_path: Path) -> dict | None:
    """Return the AF-subset of the sidecar metadata, or ``None`` on failure.

    Looks up the sidecar via :func:`metadata_path_for` and returns a
    dict filtered to ``_AF_LOG_FIELDS`` (``LensPosition``, ``AfState``,
    ``FocusFoM``).  Missing-field keys are simply absent from the
    result — callers should treat the dict as sparse.

    Log-level differentiates by *cause* — see :func:`_log_af_summary`
    for rationale.  Returns ``None`` whenever the sidecar can't be
    read or parsed; returns ``{}`` (empty but not None) when the
    sidecar exists and parses but contains none of the AF fields.
    """
    metadata_path = metadata_path_for(image_path)
    try:
        with metadata_path.open() as f:
            md = json.load(f)
    except FileNotFoundError:
        log.warning("rpicam-still did not write metadata sidecar at %s", metadata_path)
        return None
    except json.JSONDecodeError as exc:
        log.debug("metadata at %s is not valid JSON: %s", metadata_path, exc)
        return None
    except OSError as exc:
        log.warning("could not read metadata at %s: %s", metadata_path, exc)
        return None
    return {k: md[k] for k in _AF_LOG_FIELDS if k in md}


def _log_af_summary(image_path: Path) -> None:
    """Log the AF subset of the sidecar metadata at INFO level.

    Thin wrapper around :func:`read_af_fields` — exists to keep the
    capture path's logging side-effect localized and to preserve the
    log-line format (``af metadata: {...}``) that ops greps for.
    """
    summary = read_af_fields(image_path)
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

    _log_af_summary(output_path)

    return output_path
