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
import secrets
import signal
import sys
import time
from pathlib import Path

from sbo_shared.imaging import crop_to_bbox_bytes

from . import camera, storage
from .camera import Camera
from .config import HorusConfig, load
from .events import EventBus
from .motion import MotionGate

log = logging.getLogger("horus")


def _make_burst_id(station: str) -> str:
    """Generate a stable burst identifier for a new motion session.

    Format: ``{station}-{wall_ms}-{rand4}``.  Wall-clock ms (not
    monotonic) so the id reads naturally in logs and correlates
    against MQTT/DB timestamps.  A 4-hex-char random suffix prevents
    collisions across reboots (monotonic resets) and across stations
    that happen to start a burst in the same millisecond.
    """
    wall_ms = int(time.time() * 1000)
    return f"{station}-{wall_ms}-{secrets.token_hex(2)}"


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
        # Persistent picamera2 session.  Populated in :meth:`run` when
        # the lores preview stream is configured; left as ``None`` for
        # stations still on the rpicam-still legacy path so existing
        # deployments keep working with no YAML changes.
        self.camera: Camera | None = None
        self._stop = False
        self._last_event_ts = 0.0
        self._last_heartbeat_ts = 0.0
        # Throttle gated-archive pruning to once an hour so we don't
        # stat the filesystem 3600 times per capture cycle.
        self._last_gated_prune_ts = 0.0
        # How often to log a "preview frame not yet available" complaint
        # — we don't want to spam every tick while the preview thread
        # is still warming up (which it does in <100ms typically).
        self._last_lores_wait_log_ts = 0.0
        # Burst-session bookkeeping. A burst groups published frames
        # from the same motion event so Thoth can fold them into one
        # tile with a hero + alternates. State is committed only on
        # successful publish (gated drops do NOT advance the burst).
        # ``_burst_id``/``_burst_seq`` carry the last-emitted identity;
        # ``_burst_started_at`` is the monotonic start so max_duration
        # is measured from first publish, not first motion trip;
        # ``_burst_last_frame_ts`` is the monotonic anchor for the
        # idle-close check.  All zeros/None → "no active burst".
        self._burst_id: str | None = None
        self._burst_seq = 0
        self._burst_started_at = 0.0
        self._burst_last_frame_ts = 0.0

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
        """Per-loop iteration: route to the picamera2 path when a persistent
        session is active, or the legacy rpicam-still path otherwise.

        Dispatch lives here so the two implementations stay separable —
        the legacy path is the stable "works on any rpicam-enabled Pi"
        fallback, and the lores path is the low-latency motion-gated
        design from the picamera2 spike.  We branch on ``self.camera``
        rather than a config flag because the daemon constructs Camera
        lazily in :meth:`run` and can decide at runtime to degrade to
        the legacy path (e.g. Camera.start() failed).
        """
        if self.camera is not None:
            self._tick_lores()
        else:
            self._tick_legacy()

    def _tick_lores(self) -> None:
        """Picamera2 path: motion on the lores preview, capture on trip.

        Steps:
          1. Pull the most recent lores frame from the camera's preview
             thread.  If none is available yet (preview warming up or a
             hiccup after a reconfigure), skip the tick silently.
          2. Run the motion gate on the numpy array directly — no JPEG
             decode.  This is the whole point of the spike: the motion
             check costs ~1ms instead of ~100ms of rpicam + decode.
          3. On motion, trigger a still capture against the *already-
             running* picamera2 session (~50-150ms on IMX519 instead
             of the 3-4s rpicam cold start).
          4. Flow is otherwise identical to the legacy path — the
             crop → detector → classifier → publish chain is shared.
        """
        assert self.camera is not None  # narrowed by _tick dispatch

        latest = self.camera.latest_lores()
        if latest is None:
            # Preview thread hasn't produced a frame yet.  Rate-limit the
            # complaint — happens for the first few ticks after start,
            # and occasionally after a stream reconfiguration.
            now_mono = time.monotonic()
            if now_mono - self._last_lores_wait_log_ts > 5.0:
                log.debug("lores preview not ready yet; skipping tick")
                self._last_lores_wait_log_ts = now_mono
            return

        _ts, thumb = latest
        result = self.gate.check_array(thumb)

        if not result.motion:
            return

        # Legacy cooldown gate: only active when burst capture is
        # disabled.  In burst mode the publish pipeline groups frames
        # via burst_id/burst_seq instead of rate-limiting at the tick,
        # so holding off here would defeat the whole "capture while the
        # bird is there" story.
        if not self.cfg.burst.enabled:
            now = time.monotonic()
            if now - self._last_event_ts < self.cfg.motion.cooldown_s:
                log.debug("motion in cooldown, skipping publish")
                return

        # Motion confirmed — capture the full-resolution still.  The
        # picamera2 capture_request is against the already-running
        # pipeline, so latency is frame-grab time, not sensor warm-up.
        path = storage.next_capture_path(self.cfg.storage)
        try:
            self.camera.capture(path)
        except camera.CameraError:
            log.exception("camera.capture failed on motion trip")
            return

        self._publish_flow(path, result)

    def _tick_legacy(self) -> None:
        """rpicam-still path: capture every tick, motion on the JPEG.

        Preserved verbatim from pre-spike behavior so stations without
        the picamera2 dependency (or with lores disabled in YAML) run
        unchanged.  When the spike stabilizes and every station opts
        into the lores stream, this path can be retired.

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

        # Legacy cooldown gate: only active when burst capture is
        # disabled (see the matching comment in ``_tick_lores``).  In
        # burst mode the publish pipeline groups frames via burst_id /
        # burst_seq, so we intentionally skip the per-tick cooldown.
        if not self.cfg.burst.enabled:
            now = time.monotonic()
            if now - self._last_event_ts < self.cfg.motion.cooldown_s:
                log.debug("motion in cooldown, skipping publish")
                return

        self._publish_flow(path, result)

    def _publish_flow(self, path: Path, result) -> None:
        """Crop → detector → classifier → publish, shared by both _tick paths.

        Extracted so the legacy and lores paths don't drift on bug
        fixes to the publish pipeline.  ``path`` is the full-resolution
        JPEG on disk; ``result`` is the :class:`~horus.motion.MotionResult`
        from whichever gate variant produced it.  Cooldown and motion
        checks have already been satisfied by the caller.
        """

        # Compute a bird-centered crop (padded + squared around the motion
        # bbox) for the on-device detector+classifier stages — tight crops
        # massively improve confidence on feeder-framed shots.  The crop
        # is *internal* to horus: we publish the FULL frame to MQTT so
        # Thoth can show the bird-in-context in the UI, and thoth-classify
        # reproduces this same crop (via sbo_shared.imaging.crop_to_bbox_bytes)
        # before running its own inference.  Keeping the crop path internal
        # means the wire payload stays ~800KB regardless of bbox.
        #
        # IMPORTANT: the inference bytes come from the SHARED helper —
        # `crop_to_bbox_bytes` — not from re-reading a locally-written
        # file.  That's what makes horus's on-device score and thoth-
        # classify's post-ingest score byte-identical: same helper, same
        # JPEG quality (``CLASSIFIER_JPEG_QUALITY`` = 90, enforced on
        # both sides).  The on-disk crop sibling is simply the same
        # bytes dumped to a file for :func:`storage.save_gated_sample`
        # (reviewer archive) to copy.
        #
        # ``gate_bytes`` is what the detector/classifier see (always set).
        # ``gate_path`` is what :func:`save_gated_sample` copies (crop
        # if available, else the full frame).  ``crop_path`` is the
        # sibling file or ``None``.  ``path`` is always the full frame
        # and is what we publish.
        full_frame_bytes = path.read_bytes()
        gate_bytes: bytes = full_frame_bytes
        gate_path = path
        crop_path: Path | None = None
        if result.bbox_fraction is not None:
            try:
                gate_bytes = crop_to_bbox_bytes(
                    full_frame_bytes, result.bbox_fraction
                )
            except Exception:
                # Log with traceback but keep the full-frame fallback.
                # Missing the crop is worse than running the gate on the
                # whole frame — we still get a signal, just a weaker one.
                log.exception(
                    "crop_to_bbox_bytes failed; running gates on full frame"
                )
                gate_bytes = full_frame_bytes
            else:
                # Persist the same bytes to a sibling file so
                # :func:`storage.save_gated_sample` can copy the crop
                # into the reviewer archive when a gate drops the event.
                # Archive parity with the inference bytes is the point —
                # the reviewer must see what the model saw.
                crop_path = path.with_name(path.stem + "_crop.jpg")
                try:
                    crop_path.write_bytes(gate_bytes)
                    gate_path = crop_path
                except Exception:
                    log.exception(
                        "writing crop sibling to disk failed; "
                        "archive (if triggered) will fall back to full frame"
                    )
                    crop_path = None
                    gate_path = path

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
                det_result = self.detector.detect(gate_bytes)
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
                                gate_path,
                                score=detector_score,
                                label="detector-no-bird",
                            )
                        except Exception:
                            log.exception("save_gated_sample failed")
                    camera.discard(path)
                    if crop_path is not None:
                        crop_path.unlink(missing_ok=True)
                    self._last_event_ts = time.monotonic()
                    return

        # Species classifier. Runs on the cropped gate_path so on-device
        # scores are comparable to thoth-classify's post-ingest score
        # (which re-crops the published full frame with identical math).
        bird_score: float | None = None
        bird_label: str | None = None
        if self.classifier is not None:
            try:
                result_cls = self.classifier.classify(gate_bytes)
                bird_score = result_cls.confidence
                bird_label = result_cls.species
            except Exception:
                log.exception("classifier inference failed; publishing anyway")
            else:
                # Classifier floor.  Applies regardless of whether the
                # detector is upstream.  Rationale: the detector tells us
                # "bird-shaped," the classifier tells us "and I can assign
                # a species."  Non-birds (apples on the feeder, shadows,
                # leaves) pass the detector at ~0.30-0.45 but the species
                # classifier can't commit to a label — scores collapse to
                # the 0.03-0.08 band.  A low floor here (~0.10) cuts the
                # obvious garbage without AND-gating confident detections
                # against an uncertain species call.
                #
                # When `classifier.min_confidence` is 0.0 the floor is
                # effectively disabled, preserving pre-2026-04-22
                # "label-only" behavior for anyone who wants it.
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
                                gate_path,
                                score=bird_score,
                                label=bird_label,
                            )
                        except Exception:
                            log.exception("save_gated_sample failed")
                    camera.discard(path)
                    if crop_path is not None:
                        crop_path.unlink(missing_ok=True)
                    self._last_event_ts = time.monotonic()
                    return

        # Compute the burst identity for this publish BEFORE we call
        # publish_image_event, but DO NOT commit it to ``self._burst_*``
        # yet — a publish failure must leave the active burst unchanged
        # so the next successful publish gets the same id (not an
        # orphan gap).  The assignment-on-success block below is the
        # only place burst state advances.
        #
        # Continuation rules:
        #   * idle_close_s: max gap from the last published frame
        #   * max_duration_s: hard cap from the first frame of the burst
        # Both must hold, otherwise open a new burst.  When
        # ``cfg.burst.enabled`` is False, we skip burst metadata
        # entirely (legacy singleton behavior).
        next_burst_id: str | None = None
        next_burst_seq: int | None = None
        next_burst_started_at = self._burst_started_at
        if self.cfg.burst.enabled:
            now_burst = time.monotonic()
            is_continuation = (
                self._burst_id is not None
                and now_burst - self._burst_started_at
                <= self.cfg.burst.max_duration_s
                and now_burst - self._burst_last_frame_ts
                <= self.cfg.burst.idle_close_s
            )
            if is_continuation:
                next_burst_id = self._burst_id
                next_burst_seq = self._burst_seq + 1
                # next_burst_started_at unchanged — keeps max_duration
                # anchored to the first frame of the session.
            else:
                next_burst_id = _make_burst_id(self.cfg.station)
                next_burst_seq = 1
                next_burst_started_at = now_burst

        # Publish the FULL frame — Thoth's UI shows the bird in context,
        # and thoth-classify re-crops from this frame using bbox_fraction
        # before running its model.  resolution_override is left unset so
        # publish_image_event emits the configured (width, height).
        try:
            published = self.bus.publish_image_event(
                path,
                result.changed_fraction,
                bbox_fraction=result.bbox_fraction,
                af=af,
                bird_score=bird_score,
                bird_label=bird_label,
                detector_score=detector_score,
                detector_bbox_fraction=detector_bbox,
                burst_id=next_burst_id,
                burst_seq=next_burst_seq,
            )
        except Exception:
            log.exception("publish failed")
            return

        if not published:
            # EventBus already warned with the failure mode. Don't advance
            # the cooldown or burst state — otherwise a dropped event
            # suppresses the next real one AND gaps the burst seq.
            # Retrying on the next tick is the right behavior; the motion
            # gate will re-evaluate against a fresh frame, and the burst
            # stays anchored on the last *acked* frame.
            log.warning(
                "motion event dropped (dropped_publishes=%d); will retry next tick",
                self.bus.dropped_publishes,
            )
            return

        # Commit burst state only on successful publish, for the reason
        # above. This keeps seq monotonic w.r.t. frames Thoth actually
        # sees and keeps burst_id stable across transient broker blips.
        commit_ts = time.monotonic()
        self._last_event_ts = commit_ts
        if self.cfg.burst.enabled:
            self._burst_id = next_burst_id
            self._burst_seq = next_burst_seq or 0
            self._burst_started_at = next_burst_started_at
            self._burst_last_frame_ts = commit_ts
        log.info(
            "motion event published (frac=%.3f, bbox=%s, af=%s, det=%s, bird_score=%s, bird=%s, burst=%s/%s)",
            result.changed_fraction,
            result.bbox_fraction,
            af,
            f"{detector_score:.3f}" if detector_score is not None else None,
            f"{bird_score:.3f}" if bird_score is not None else None,
            bird_label,
            next_burst_id,
            next_burst_seq,
        )

    def run(self) -> int:
        """Start the capture loop and block until ``stop()`` is called.

        Connects to the MQTT broker, opens the persistent picamera2
        session if the lores preview stream is configured, publishes a
        startup status message, then repeatedly calls ``_tick()``,
        ``_heartbeat_maybe()``, and ``storage.prune()`` with a
        ``capture.interval_s`` sleep between iterations.

        Publishes a shutdown status message, stops the camera, and
        disconnects from the broker before returning.  Returns 0 on
        clean exit.

        Camera degradation: if the picamera2 session can't start
        (hardware missing, wheel missing, configure error) we log the
        failure and fall back to the legacy rpicam-still path rather
        than crashing the daemon.  The whole point of the spike is
        zero-downtime rollout — a broken picamera2 must not take the
        station offline.
        """
        self.bus.connect()
        try:
            self._maybe_start_camera()
            self.bus.publish_status({"camera_ok": True, "starting": True})
            while not self._stop:
                # Wrap the loop body so a transient bug in any one step
                # (a Pillow decode blowing up on a truncated JPEG, a
                # gated-archive write failing on a full disk, an MQTT
                # publish tripping a type error, etc.) cannot take the
                # whole daemon offline.  systemd would restart us with
                # backoff, but we lose the preview-thread warmup and —
                # more importantly — we get a gap in coverage.  Keep
                # the camera session alive, log the failure, move on.
                #
                # KeyboardInterrupt / SystemExit intentionally propagate
                # via BaseException so ctrl-C and systemd TERM still
                # unblock run() cleanly.
                try:
                    self._tick()
                    self._heartbeat_maybe()
                    storage.prune(self.cfg.storage)
                    self._prune_gated_maybe()
                except Exception:
                    log.exception("unhandled exception in capture loop; continuing")
                time.sleep(self.cfg.capture.interval_s)
        finally:
            self.bus.publish_status({"camera_ok": True, "stopping": True})
            if self.camera is not None:
                try:
                    self.camera.stop()
                except Exception:
                    log.exception("camera.stop() during shutdown raised")
            self.bus.disconnect()
        return 0

    def _maybe_start_camera(self) -> None:
        """Open the persistent picamera2 session if the config asks for it.

        No-op when ``cfg.capture.lores_width`` or ``lores_height`` is
        zero — the daemon stays on the rpicam-still legacy path and
        :attr:`camera` remains ``None``.

        On ``Camera.start()`` failure we log and leave :attr:`camera`
        as ``None``, which makes :meth:`_tick` route to the legacy
        path automatically.  This is the "picamera2 absent, rpicam
        still works" degradation story.
        """
        if self.cfg.capture.lores_width <= 0 or self.cfg.capture.lores_height <= 0:
            log.info("lores preview disabled; using rpicam-still legacy path")
            return
        try:
            cam = Camera(self.cfg.capture)
            cam.start()
        except camera.CameraError:
            log.exception(
                "picamera2 session failed to start; falling back to rpicam-still"
            )
            return
        self.camera = cam
        log.info("picamera2 session active; motion runs on lores preview")

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
