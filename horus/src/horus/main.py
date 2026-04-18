"""Horus capture daemon — main loop and process entrypoint.

Horus runs on the observatory Pi and owns the full capture-and-publish
pipeline:

1. ``camera.capture()`` fires ``rpicam-still`` and writes a JPEG to local
   ring-buffer storage.
2. ``MotionGate.check()`` compares the frame against a running baseline; if
   it falls below the motion threshold the file is deleted immediately.
3. Frames that pass the gate are published to the MQTT broker via
   ``EventBus.publish_image_event()``, where Banshee picks them up for
   species identification.
4. A periodic heartbeat (``publish_status``) keeps the station visible on
   the dashboard even when there is no bird activity.

``Daemon`` is the orchestrator. ``main()`` is the CLI entrypoint wired to
the ``horus`` console script defined in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from . import camera, storage
from .config import HorusConfig, load
from .events import EventBus
from .motion import MotionGate, crop_to_bbox

log = logging.getLogger("horus")


class Daemon:
    """Capture-and-publish daemon for a single observatory station.

    Orchestrates the per-frame loop: triggers a camera capture, runs the
    result through ``MotionGate`` to discard static scenes, enforces a
    post-event cooldown to avoid flooding Banshee, publishes passing frames
    to MQTT via ``EventBus``, and prunes local storage to stay within the
    configured disk budget.

    Lifecycle: call ``run()`` to start the blocking loop. Wire ``stop()``
    to SIGTERM and SIGINT (as ``main()`` does) to request a clean shutdown.
    """

    def __init__(self, cfg: HorusConfig) -> None:
        self.cfg = cfg
        self.bus = EventBus(cfg)
        self.gate = MotionGate(cfg.motion)
        self._stop = False
        self._last_event_ts = 0.0
        self._last_heartbeat_ts = 0.0

    def _heartbeat_maybe(self) -> None:
        """Publish a status heartbeat if enough time has elapsed since the last one.

        No-ops when inside the ``heartbeat_interval_s`` window. Swallows
        publish failures so a transient broker blip does not abort the loop.
        """
        now = time.monotonic()
        if now - self._last_heartbeat_ts < self.cfg.heartbeat_interval_s:
            return
        try:
            self.bus.publish_status({"camera_ok": True})
        except Exception:
            log.exception("heartbeat publish failed")
        self._last_heartbeat_ts = now

    def _tick(self) -> None:
        """Capture one frame, run the motion gate, and publish if warranted.

        Steps:
          1. Allocate a timestamped path and call ``camera.capture``.
          2. If capture fails, log and return — the tick is skipped silently.
          3. Pass the frame through ``MotionGate``; delete it and return if no
             motion is detected.
          4. Enforce the per-station cooldown (``motion.cooldown_s``); return
             without publishing if inside the cooldown window.
          5. Publish via ``EventBus.publish_image_event``. If the publish
             fails, do *not* advance the cooldown timestamp so the next tick
             retries with a fresh motion-gate evaluation.
        """
        path = storage.next_capture_path(self.cfg.storage)

        try:
            camera.capture(path, self.cfg.capture)
        except camera.CameraError:
            log.exception("capture failed")
            return

        result = self.gate.check(path)

        if not result.motion:
            # Not interesting — drop the file (and its metadata sidecar) and move on.
            camera.discard(path)
            return

        now = time.monotonic()
        if now - self._last_event_ts < self.cfg.motion.cooldown_s:
            log.debug("motion in cooldown, skipping publish")
            return

        # Prefer the bird-centered crop if motion gave us a bbox — this is
        # what Thoth's classifier sees, and a tight crop massively
        # improves confidence on feeder-framed shots. Fall back to the
        # full frame if something goes wrong (unexpected image format,
        # degenerate bbox, etc.) so we never drop a real motion event
        # just because the crop step tripped.
        publish_path = path
        publish_resolution: tuple[int, int] | None = None
        if result.bbox_fraction is not None:
            crop_path = path.with_name(path.stem + "_crop.jpg")
            try:
                publish_resolution = crop_to_bbox(
                    path,
                    crop_path,
                    result.bbox_fraction,
                    jpeg_quality=self.cfg.capture.jpeg_quality,
                )
                publish_path = crop_path
            except Exception:
                # Log with traceback but keep the full-frame fallback.
                log.exception("crop_to_bbox failed; publishing full frame")
                publish_path = path
                publish_resolution = None

        # Attach AF state (LensPosition/AfState/FocusFoM) from the rpicam
        # sidecar so Thoth can correlate "was the lens focused" with
        # downstream classifier confidence. Missing/corrupt sidecar → None,
        # and the AF block is simply omitted from the payload.
        af = camera.read_af_fields(path)

        try:
            published = self.bus.publish_image_event(
                publish_path,
                result.changed_fraction,
                resolution_override=publish_resolution,
                bbox_fraction=result.bbox_fraction,
                af=af,
            )
        except Exception:
            log.exception("publish failed")
            return

        if not published:
            # EventBus already warned with the failure mode. Don't advance
            # the cooldown — otherwise a dropped event suppresses the next
            # real one. Retrying on the next tick is the right behavior;
            # the motion gate will re-evaluate against a fresh frame.
            log.warning(
                "motion event dropped (dropped_publishes=%d); will retry next tick",
                self.bus.dropped_publishes,
            )
            return

        self._last_event_ts = now
        log.info(
            "motion event published (frac=%.3f, bbox=%s, af=%s)",
            result.changed_fraction,
            result.bbox_fraction,
            af,
        )

    def run(self) -> int:
        """Start the capture loop and block until ``stop()`` is called.

        Connects to the MQTT broker and publishes a startup status message,
        then repeatedly calls ``_tick()``, ``_heartbeat_maybe()``, and
        ``storage.prune()`` with a ``capture.interval_s`` sleep between
        iterations.

        Publishes a shutdown status message and disconnects from the broker
        before returning. Returns 0 on clean exit.
        """
        self.bus.connect()
        try:
            self.bus.publish_status({"camera_ok": True, "starting": True})
            while not self._stop:
                self._tick()
                self._heartbeat_maybe()
                storage.prune(self.cfg.storage)
                time.sleep(self.cfg.capture.interval_s)
        finally:
            self.bus.publish_status({"camera_ok": True, "stopping": True})
            self.bus.disconnect()
        return 0

    def stop(self, *_: object) -> None:
        """Request a graceful shutdown on the next loop iteration.

        Designed for use as a ``signal.signal`` handler — accepts and ignores
        the signal number and stack-frame arguments.
        """
        self._stop = True


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, load config, wire signal handlers, and run the daemon.

    ``argv`` is forwarded to ``argparse``; pass ``None`` to consume
    ``sys.argv[1:]``. Returns the daemon's exit code (0 on clean shutdown).
    """
    parser = argparse.ArgumentParser(description="Horus capture daemon")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/etc/horus/horus.yaml"),
        help="Path to station YAML config",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load(args.config)
    daemon = Daemon(cfg)

    signal.signal(signal.SIGTERM, daemon.stop)
    signal.signal(signal.SIGINT, daemon.stop)

    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
