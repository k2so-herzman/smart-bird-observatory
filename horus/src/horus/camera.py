"""Camera capture.

Two code paths coexist during the picamera2 migration spike:

1. **Legacy path** — :func:`capture` shells out to the system
   ``rpicam-still`` binary. Simple, no Python bindings, but pays a
   cold-start subprocess cost (~3-4s including sensor warm-up) on
   every frame. This is what :mod:`horus.main` still uses today.

2. **Persistent path** — :class:`Camera` holds a long-lived
   :class:`picamera2.Picamera2` session open between captures so the
   per-capture cost drops to the time needed to grab one frame from a
   running pipeline (~50-150ms on an IMX519). Wired into ``main.py``
   in a follow-up commit.

The picamera2 import is pinned behind :func:`_picamera2_factory` so
unit tests can monkeypatch a :class:`FakePicamera2` in without
needing the (ARM-only) real wheel installed, and so dev machines
without libcamera keep importing :mod:`horus.camera` cleanly.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .config import CaptureConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera (rpicam-still or picamera2) fails."""


# Subset of libcamera per-capture metadata fields we surface in INFO logs.
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


# ---------------------------------------------------------------------------
# Persistent picamera2 session (spike — not yet wired into main.py)
# ---------------------------------------------------------------------------


def _picamera2_factory() -> type:
    """Return the :class:`picamera2.Picamera2` class.

    Indirection seam so tests can ``monkeypatch.setattr(camera,
    "_picamera2_factory", lambda: FakePicamera2)`` without needing the
    real ARM-only wheel installed.  Import is lazy — :mod:`horus.camera`
    stays importable on dev hosts that have no libcamera stack.

    Raises:
        CameraError: if picamera2 isn't installed on the host.  Callers
            that want "fall back to rpicam-still" should catch this and
            degrade, not crash.
    """
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CameraError(
            "picamera2 is not installed; run the venv with "
            "--system-site-packages or `pip install picamera2`"
        ) from exc
    return Picamera2


class Camera:
    """Persistent picamera2 still-capture session.

    Opens one :class:`picamera2.Picamera2` instance on :meth:`start` and
    keeps it running until :meth:`stop`.  Each :meth:`capture` pulls a
    request off the running pipeline, saves it as JPEG, and writes the
    libcamera metadata to the sidecar path used by the rest of horus.

    The per-capture cost on an IMX519 drops from ~3-4s (subprocess
    cold start + sensor warm-up under rpicam-still) to ~50-150ms
    (single frame grab from an already-running pipeline).  That's the
    whole point of the spike.

    Thread safety: :meth:`capture` is **not** re-entrant.  The capture
    daemon's ``_tick`` is single-threaded so this is fine today; a
    future preview-stream commit will need explicit locking if we start
    calling ``capture_request`` from two threads.

    Lifecycle:
        camera = Camera(cfg)
        camera.start()
        try:
            while running:
                camera.capture(next_path)
        finally:
            camera.stop()
    """

    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self._picam2: Any | None = None

    def start(self) -> None:
        """Open the picamera2 session and start the pipeline.

        Configures a single full-resolution ``main`` stream sized to
        ``cfg.width``/``cfg.height`` in RGB888 — JPEG encoding happens
        at save time via PIL (controlled by ``picam2.options["quality"]``).

        Idempotent: calling ``start()`` while already started is a no-op.
        Raises :class:`CameraError` on configure/start failure so callers
        can degrade to the rpicam path.
        """
        if self._picam2 is not None:
            log.debug("camera.start() called while already started — no-op")
            return

        picamera2_cls = _picamera2_factory()
        picam2 = picamera2_cls()
        try:
            config = picam2.create_still_configuration(
                main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"},
            )
            picam2.configure(config)
            # JPEG quality is a picamera2-level option because the
            # default save path uses PIL.Image.save which reads it here.
            picam2.options["quality"] = self.cfg.jpeg_quality
            picam2.start()
        except Exception as exc:
            # Be defensive — half-configured Picamera2 instances leak
            # the /dev/video* fds and wedge the sensor until reboot.
            try:
                picam2.close()
            except Exception:
                log.exception("picamera2.close() failed during start() rollback")
            raise CameraError(f"picamera2 start failed: {exc}") from exc
        self._picam2 = picam2
        log.info(
            "picamera2 started: size=(%d, %d) quality=%d",
            self.cfg.width,
            self.cfg.height,
            self.cfg.jpeg_quality,
        )

    def stop(self) -> None:
        """Stop the pipeline and close the picamera2 session.

        Idempotent: safe to call on an already-stopped camera.  Swallows
        per-step exceptions and logs them — shutdown must not raise,
        or the daemon's ``finally`` blocks will mask the real error.
        """
        if self._picam2 is None:
            return
        try:
            self._picam2.stop()
        except Exception:
            log.exception("picamera2.stop() failed")
        try:
            self._picam2.close()
        except Exception:
            log.exception("picamera2.close() failed")
        self._picam2 = None

    def capture(self, output_path: Path) -> Path:
        """Capture one still to ``output_path`` and write its sidecar.

        Matches the return/side-effect contract of :func:`capture` (the
        rpicam shim): writes JPEG at ``output_path``, writes libcamera
        metadata JSON at ``output_path.json``, logs key AF fields at
        INFO, returns ``output_path``.

        Raises:
            CameraError: if the camera is not started, or the picamera2
                request/save round-trip fails.  Callers (the daemon
                ``_tick``) treat this exactly like the rpicam failure
                path — log and skip the tick.
        """
        if self._picam2 is None:
            raise CameraError("Camera.capture() called before Camera.start()")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_path_for(output_path)

        try:
            request = self._picam2.capture_request()
        except Exception as exc:
            raise CameraError(f"capture_request failed: {exc}") from exc

        try:
            try:
                request.save("main", str(output_path))
            except Exception as exc:
                raise CameraError(f"request.save failed: {exc}") from exc
            try:
                metadata = request.get_metadata()
            except Exception as exc:
                # Don't fail the whole capture over a metadata read —
                # the JPEG is already on disk and useful.
                log.warning("request.get_metadata failed: %s", exc)
                metadata = {}
        finally:
            try:
                request.release()
            except Exception:
                log.exception("request.release failed")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise CameraError(f"picamera2 produced no output at {output_path}")

        # Write the sidecar in the same shape as rpicam-still's --metadata
        # JSON dump so downstream read_af_fields / Thoth ingest don't care
        # which backend produced the frame.  default=str handles numpy /
        # Fraction / enum values that aren't natively JSON-serializable.
        try:
            metadata_path.write_text(json.dumps(metadata, default=str))
        except OSError as exc:
            log.warning("could not write metadata sidecar at %s: %s", metadata_path, exc)

        _log_af_summary(output_path)

        return output_path

    def __enter__(self) -> Camera:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
