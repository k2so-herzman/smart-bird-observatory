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


def _load_classifier(cfg: HorusConfig):
    """Instantiate the on-device bird classifier, or return None.

    Returns None (and logs) when the classifier is disabled in config
    or when construction fails — a missing model file or a broken
    tflite-runtime install is not worth crashing the capture loop
    over. The daemon degrades to "publish every motion event" (the
    pre-classifier behavior) in that case.
    """
    if not cfg.classifier.enabled:
        log.info("on-device classifier disabled in config")
        return None
    try:
        from .classifier import BirdClassifier

        return BirdClassifier(cfg.classifier.model_path, cfg.classifier.labels_path)
    except Exception:
        log.exception(
            "failed to load classifier (model=%s); publishing every event",
            cfg.classifier.model_path,
        )
        return None


def _load_detector(cfg: HorusConfig):
    """Instantiate the on-device object detector, or return None.

    Same degradation philosophy as ``_load_classifier``: a missing
    model file or a broken tflite-runtime install falls back to
    "no detector", which means the gate reverts to whatever the
    classifier does (including "publish every event" if the
    classifier is also unavailable). A working but mis-configured
    detector is worse than no detector — better to log and keep
    capturing than to crash the daemon.
    """
    if not cfg.detector.enabled:
        log.info("on-device detector disabled in config")
        return None
    try:
        from .detector import BirdDetector

        return BirdDetector(
            cfg.detector.model_path,
            cfg.detector.labels_path,
            min_score=cfg.detector.min_score,
        )
    except Exception:
        log.exception(
            "failed to load detector (model=%s); falling back to classifier gate",
            cfg.detector.model_path,
        )
        return None


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
        self.detector = _load_detector(cfg)
        self.classifier = _load_classifier(cfg)
        self._stop = False
        self._last_event_ts = 0.0
        self._last_heartbeat_ts = 0.0
        # Throttle gated-archive pruning to once an hour so we don't
        # stat the filesystem 3600 times per capture cycle.
        self._last_gated_prune_ts = 0.0

    def _prune_gated_maybe(self) -> None:
        """Prune the gated-archive day-dirs at most once per hour.

        No-op when the archive isn't configured. We throttle to avoid
        walking the directory on every capture tick — the operational
        cost is negligible but makes log output unnecessarily noisy.
        """
        archive_dir = self.cfg.classifier.gated_archive_dir
        if archive_dir is None:
            return
        now = time.monotonic()
        if now - self._last_gated_prune_ts < 3600:
            return
        try:
            removed = storage.prune_gated(
                archive_dir,
                self.cfg.classifier.gated_archive_max_age_days,
            )
            if removed:
                log.info("gated archive: pruned %d old day-dirs", removed)
        except Exception:
            log.exception("prune_gated failed")
        self._last_gated_prune_ts = now

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

        # --- on-device gates -------------------------------------------------
        # Two-stage design:
        #   1. Object detector (if enabled) is the PRIMARY gate. A COCO-trained
        #      "is there a bird?" model returns a clean zero on wind-sway, where
        #      the species classifier's top-1 threshold has to wrestle with
        #      favorite-fallback classes (Great Egret, NZ Pigeon) that come
        #      back with moderate confidence on any textured frame.
        #   2. Species classifier (if enabled) runs for the label — but only
        #      gates when the detector is NOT configured (legacy behavior).
        # Failures in either stage log and fall through — we prefer false
        # positives over silently dropping a real bird.
        detector_score: float | None = None
        detector_bbox: tuple[float, float, float, float] | None = None
        if self.detector is not None:
            try:
                det_result = self.detector.detect(publish_path.read_bytes())
                detector_score = det_result.score
                detector_bbox = det_result.bbox_fraction
            except Exception:
                log.exception("detector inference failed; skipping detector gate")
            else:
                if not det_result.has_bird:
                    log.info(
                        "detector gated (score=%.3f, bbox=%s); dropping",
                        detector_score,
                        detector_bbox,
                    )
                    archive_dir = self.cfg.classifier.gated_archive_dir
                    if archive_dir is not None:
                        try:
                            storage.save_gated_sample(
                                archive_dir,
                                publish_path,
                                score=detector_score,
                                label="detector-no-bird",
                            )
                        except Exception:
                            log.exception("save_gated_sample failed")
                    camera.discard(path)
                    if publish_path != path:
                        publish_path.unlink(missing_ok=True)
                    self._last_event_ts = time.monotonic()
                    return

        # Species classifier. Runs on the exact bytes we're about to publish
        # so scores are comparable to Thoth's post-ingest classifier.
        bird_score: float | None = None
        bird_label: str | None = None
        if self.classifier is not None:
            try:
                result_cls = self.classifier.classify(publish_path.read_bytes())
                bird_score = result_cls.confidence
                bird_label = result_cls.species
            except Exception:
                log.exception("classifier inference failed; publishing anyway")
            else:
                # Legacy classifier-gate path: only used when the detector
                # is not configured. When detector is live, the classifier
                # is informational-only (attach label, don't gate).
                if self.detector is None:
                    threshold = self.cfg.classifier.min_confidence
                    if bird_score < threshold:
                        log.info(
                            "classifier gated (score=%.3f < %.2f, top=%r, bbox=%s); dropping",
                            bird_score,
                            threshold,
                            bird_label,
                            result.bbox_fraction,
                        )
                        archive_dir = self.cfg.classifier.gated_archive_dir
                        if archive_dir is not None:
                            try:
                                storage.save_gated_sample(
                                    archive_dir,
                                    publish_path,
                                    score=bird_score,
                                    label=bird_label,
                                )
                            except Exception:
                                log.exception("save_gated_sample failed")
                        camera.discard(path)
                        if publish_path != path:
                            publish_path.unlink(missing_ok=True)
                        self._last_event_ts = time.monotonic()
                        return

        try:
            published = self.bus.publish_image_event(
                publish_path,
                result.changed_fraction,
                resolution_override=publish_resolution,
                bbox_fraction=result.bbox_fraction,
                af=af,
                bird_score=bird_score,
                bird_label=bird_label,
                detector_score=detector_score,
                detector_bbox_fraction=detector_bbox,
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
            "motion event published (frac=%.3f, bbox=%s, af=%s, det=%s, bird_score=%s, bird=%s)",
            result.changed_fraction,
            result.bbox_fraction,
            af,
            f"{detector_score:.3f}" if detector_score is not None else None,
            f"{bird_score:.3f}" if bird_score is not None else None,
            bird_label,
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
                self._prune_gated_maybe()
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
