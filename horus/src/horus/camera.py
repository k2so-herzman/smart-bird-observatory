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
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

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
        # Preview-stream state — only populated when lores is enabled.
        # The lock guards the latest-frame slot between the background
        # sampling thread and :meth:`latest_lores`.
        self._preview_thread: threading.Thread | None = None
        self._preview_stop = threading.Event()
        self._latest_lock = threading.Lock()
        self._latest_lores: tuple[float, np.ndarray] | None = None
        self._lores_frames_seen = 0
        # Populated by start() when the preview thread is launched; read
        # by _preview_loop to compute time-to-first-frame for the warmup
        # log line.  Left as None when lores is disabled.
        self._preview_started_mono: float | None = None

    @property
    def lores_enabled(self) -> bool:
        """``True`` when the configured CaptureConfig asks for a lores stream.

        Both ``lores_width`` and ``lores_height`` must be positive —
        setting only one is treated as "disabled" rather than crashing
        the daemon at start-up.  Keeping this as a property (not a
        cached attribute) means tests can mutate ``self.cfg`` and
        observe the change without reconstructing the Camera.
        """
        return self.cfg.lores_width > 0 and self.cfg.lores_height > 0

    def start(self) -> None:
        """Open the picamera2 session and start the pipeline.

        Two shapes:

        - **Still-only** (``lores_width == 0`` or ``lores_height == 0``):
          a single full-resolution ``main`` stream in RGB888, matching
          commit-1 semantics.  No preview thread is started.
        - **Dual-stream** (both lores dims positive): a *video*
          configuration with both a full-res ``main`` (RGB888) and a
          small ``lores`` (YUV420) stream.  A background thread samples
          the lores stream at ``cfg.preview_fps`` and publishes the
          most recent Y-plane (grayscale) via :meth:`latest_lores`.

        JPEG encoding happens at save time via PIL, controlled by
        ``picam2.options["quality"]`` regardless of stream shape.

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
            if self.lores_enabled:
                # Video config is what keeps both streams live between
                # captures — create_still_configuration would tear down
                # the preview each still, defeating the whole purpose.
                config = picam2.create_video_configuration(
                    main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"},
                    lores={
                        "size": (self.cfg.lores_width, self.cfg.lores_height),
                        "format": "YUV420",
                    },
                )
            else:
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

        if self.lores_enabled:
            # Clear stop-flag in case the instance was restarted after a
            # previous stop() — Event.clear() is idempotent.
            self._preview_stop.clear()
            self._latest_lores = None
            self._lores_frames_seen = 0
            self._preview_started_mono = time.monotonic()
            self._preview_thread = threading.Thread(
                target=self._preview_loop,
                name=f"horus-camera-preview-{id(self)}",
                daemon=True,
            )
            self._preview_thread.start()
            log.info(
                "picamera2 started (dual-stream): main=(%d, %d) lores=(%d, %d) quality=%d fps=%.1f",
                self.cfg.width,
                self.cfg.height,
                self.cfg.lores_width,
                self.cfg.lores_height,
                self.cfg.jpeg_quality,
                self.cfg.preview_fps,
            )
        else:
            log.info(
                "picamera2 started: size=(%d, %d) quality=%d",
                self.cfg.width,
                self.cfg.height,
                self.cfg.jpeg_quality,
            )

    def stop(self) -> None:
        """Stop the pipeline and close the picamera2 session.

        Order matters:

        1. Signal the preview thread to exit via the stop flag.
        2. Join the preview thread (bounded wait — a stuck thread is
           better than hanging shutdown).  The thread holds no
           references to the picamera2 instance beyond ``capture_array``
           calls, which return promptly once the pipeline stops.
        3. Stop + close the picamera2 session.

        Reversing steps 2 and 3 would let the preview loop call
        ``capture_array`` against a closed pipeline and spam the log
        with errors on the way out.

        Idempotent: safe to call on an already-stopped camera.  Swallows
        per-step exceptions and logs them — shutdown must not raise,
        or the daemon's ``finally`` blocks will mask the real error.
        """
        if self._picam2 is None and self._preview_thread is None:
            return

        self._preview_stop.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2.0)
            if self._preview_thread.is_alive():
                log.warning("preview thread did not exit within 2s — detaching")
            self._preview_thread = None

        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                log.exception("picamera2.stop() failed")
            try:
                self._picam2.close()
            except Exception:
                log.exception("picamera2.close() failed")
            self._picam2 = None

    def _preview_loop(self) -> None:
        """Background thread: sample the lores stream at ``preview_fps``.

        Each iteration pulls the most recent lores frame via
        ``picam2.capture_array("lores")`` and stashes the Y-plane
        (first H rows of the YUV420 byte layout) under the latest-frame
        lock.  Motion gating code reads it via :meth:`latest_lores`.

        YUV420 layout note: picamera2 returns YUV420 as a stacked 2D
        array of shape ``(H*3/2, W)`` — rows ``[0:H]`` are the Y plane
        (full-resolution luma, effectively grayscale), rows ``[H:]``
        are the interleaved U/V planes at half resolution.  Pulling
        just the Y plane gives us a grayscale thumbnail at no cost.

        Errors inside the loop are logged and the loop sleeps and
        continues — a transient capture_array failure should not kill
        the preview stream permanently.  The daemon is already
        fault-tolerant to a missing lores frame (``latest_lores()``
        returns None).
        """
        interval = 1.0 / max(self.cfg.preview_fps, 1e-3)
        lores_h = self.cfg.lores_height
        lores_w = self.cfg.lores_width
        log.debug("preview loop starting (interval=%.3fs)", interval)
        while not self._preview_stop.is_set():
            picam2 = self._picam2
            if picam2 is None:
                # Race: stop() ran between our last wait and this read.
                break
            try:
                frame = picam2.capture_array("lores")
            except Exception:
                log.exception("capture_array('lores') failed; retrying")
                # Sleep through the interval before the next attempt so
                # a hard failure doesn't hot-loop against a broken driver.
                if self._preview_stop.wait(interval):
                    break
                continue
            # Y plane = first lores_h rows, lores_w columns wide.  Copy
            # detaches our snapshot from picamera2's internal buffer so
            # consumers can't observe a half-written next frame.
            try:
                gray = np.asarray(frame[:lores_h, :lores_w], dtype=np.uint8).copy()
            except Exception:
                log.exception("failed to slice lores frame; skipping")
                if self._preview_stop.wait(interval):
                    break
                continue
            with self._latest_lock:
                self._latest_lores = (time.monotonic(), gray)
                self._lores_frames_seen += 1
                first_frame = self._lores_frames_seen == 1
            if first_frame:
                # One-shot warmup telemetry: time from start() to first
                # usable lores frame tells ops how long the motion path
                # was skipped during preview spin-up.  Ticks that ran
                # before this moment short-circuited in _tick_lores.
                started = self._preview_started_mono
                elapsed = (
                    time.monotonic() - started if started is not None else float("nan")
                )
                log.info("first lores frame ready after %.3fs", elapsed)
            # Event.wait returns True when set — propagate stop without
            # waiting out the rest of the interval.
            if self._preview_stop.wait(interval):
                break
        log.debug("preview loop exited")

    def latest_lores(self) -> tuple[float, np.ndarray] | None:
        """Return the most recent ``(timestamp, grayscale_frame)`` pair, or None.

        Returns ``None`` when the lores stream is disabled or when the
        preview thread has not yet produced a frame.  The timestamp is
        a monotonic-clock value (``time.monotonic()``) so callers can
        detect stale frames without fighting wall-clock skew.

        The returned array is the stored numpy view; callers must not
        mutate it in place.  If you need to hold a copy across ticks,
        ``arr.copy()`` it explicitly.
        """
        with self._latest_lock:
            if self._latest_lores is None:
                return None
            return self._latest_lores

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
