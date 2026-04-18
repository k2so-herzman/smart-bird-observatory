"""MQTT event publisher for Horus.

Image bytes ride on MQTT (base64-encoded) so Banshee doesn't need any
shared filesystem. Keep an eye on broker `message_size_limit` — a
2304x1296 JPEG at q=90 is ~600KB, which base64-expands to ~800KB.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt
from sbo_shared import (
    TOPIC_IMAGE_EVENT,
    TOPIC_STATUS,
    build_topic,
    sbo_now_iso,
)

from .config import HorusConfig

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EventBus:
    """Thin wrapper around paho-mqtt with SBO topic conventions."""

    def __init__(self, cfg: HorusConfig) -> None:
        self.cfg = cfg
        self.dropped_publishes = 0
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"horus-{cfg.station}-{int(time.time())}",
        )
        if cfg.mqtt.username:
            self._client.username_pw_set(cfg.mqtt.username, cfg.mqtt.password)

    def connect(self) -> None:
        self._client.connect(self.cfg.mqtt.host, self.cfg.mqtt.port, keepalive=60)
        self._client.loop_start()

    def disconnect(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def _topic(self, suffix: str) -> str:
        return build_topic(self.cfg.mqtt.topic_prefix, self.cfg.station, suffix)

    def _publish(self, topic: str, payload: dict[str, Any], retain: bool = False) -> bool:
        """Publish a JSON payload at QoS 1 and block until the broker acks.

        `info.rc` reflects the *enqueue* result (queue full, not connected,
        etc.) — it does NOT reflect whether the broker received the message.
        For QoS 1 delivery we have to wait for PUBACK via
        `wait_for_publish()` and then confirm with `is_published()`.

        Returns True iff the broker acknowledged. Failures are logged +
        counted (`dropped_publishes`) so callers can decide whether to
        retry, advance state, or alert.
        """
        info = self._client.publish(topic, json.dumps(payload), qos=1, retain=retain)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning("enqueue to %s failed: rc=%s", topic, info.rc)
            self.dropped_publishes += 1
            return False
        try:
            info.wait_for_publish(timeout=5)
        except (RuntimeError, ValueError) as exc:
            log.warning("publish to %s failed while awaiting ack: %s", topic, exc)
            self.dropped_publishes += 1
            return False
        # is_published() itself can raise if the loop thread updated rc to
        # an error between wait_for_publish returning and us reading state.
        # Treat that as a drop rather than crashing _tick.
        try:
            published = info.is_published()
        except (RuntimeError, ValueError) as exc:
            log.warning("publish to %s failed during ack check: %s", topic, exc)
            self.dropped_publishes += 1
            return False
        if not published:
            log.warning("publish to %s timed out waiting for PUBACK", topic)
            self.dropped_publishes += 1
            return False
        return True

    # ---- public event helpers ---------------------------------------------

    def publish_image_event(
        self,
        image_path: Path,
        changed_fraction: float,
        *,
        resolution_override: tuple[int, int] | None = None,
        bbox_fraction: tuple[float, float, float, float] | None = None,
        af: dict[str, Any] | None = None,
        bird_score: float | None = None,
        bird_label: str | None = None,
    ) -> bool:
        """Publish a motion image event. Returns True iff broker acked.

        ``resolution_override``, when provided, replaces the default
        ``(capture.width, capture.height)`` in the published payload.
        Used when the sender cropped the frame before publish (see
        :func:`horus.motion.crop_to_bbox`) — in that case the sensor-config
        resolution no longer matches the bytes, and downstream consumers
        expect ``resolution`` to describe the attached image.  When
        ``None`` (the default) behavior is unchanged: the config
        resolution is published, matching pre-crop callers.

        ``bbox_fraction``, when provided, is the motion-bbox in
        ``(x0, y0, x1, y1)`` unit-square coordinates relative to the
        *source* frame (not the crop). Thoth uses this to distinguish
        "distributed motion" (wind-sway, whole-frame fraction) from
        "focal motion" (a bird occupying a small region).  Omitted from
        the payload when ``None``.

        ``af``, when provided, is the rpicam-still per-capture
        autofocus summary — typically ``{"LensPosition": ..., "AfState":
        ..., "FocusFoM": ...}`` read from the metadata sidecar by
        :func:`horus.camera.read_af_fields`.  Omitted when ``None``.

        ``bird_score`` is the top-1 confidence from the on-device bird
        classifier when the station has classification enabled. Always
        attached when provided — even sub-threshold scores are useful
        for tuning. ``bird_label`` is the corresponding species string.
        Both omitted when ``None`` (classifier disabled or inference
        failed — dropped silently rather than blocking publish).
        """
        image_bytes = image_path.read_bytes()
        if resolution_override is not None:
            resolution = list(resolution_override)
        else:
            resolution = [self.cfg.capture.width, self.cfg.capture.height]
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "station": self.cfg.station,
            "captured_at": sbo_now_iso(),
            "camera": self.cfg.camera,
            "trigger": "motion",
            "resolution": resolution,
            "content_type": "image/jpeg",
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "size_bytes": len(image_bytes),
            "changed_fraction": changed_fraction,
            "sha256": _sha256_bytes(image_bytes),
        }
        if bbox_fraction is not None:
            payload["bbox_fraction"] = list(bbox_fraction)
        if af is not None:
            payload["af"] = af
        if bird_score is not None:
            payload["bird_score"] = float(bird_score)
        if bird_label is not None:
            payload["bird_label"] = bird_label
        log.debug(
            "publishing image event: %d bytes (%.1fKB base64)",
            len(image_bytes),
            len(payload["image_b64"]) / 1024,
        )
        return self._publish(self._topic(TOPIC_IMAGE_EVENT), payload)

    def publish_status(self, extra: dict[str, Any] | None = None) -> bool:
        """Publish a retained status heartbeat. Returns True iff broker acked."""
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "station": self.cfg.station,
            "ts": sbo_now_iso(),
            "hostname": socket.gethostname(),
        }
        if extra:
            payload.update(extra)
        return self._publish(self._topic(TOPIC_STATUS), payload, retain=True)
