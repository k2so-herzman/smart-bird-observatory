"""MQTT event publisher for Horus."""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt

from .config import HorusConfig

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventBus:
    """Thin wrapper around paho-mqtt with SBO topic conventions."""

    def __init__(self, cfg: HorusConfig) -> None:
        self.cfg = cfg
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
        return f"{self.cfg.mqtt.topic_prefix}/{self.cfg.station}/{suffix}"

    def _publish(self, topic: str, payload: dict[str, Any], retain: bool = False) -> None:
        info = self._client.publish(topic, json.dumps(payload), qos=1, retain=retain)
        info.wait_for_publish(timeout=5)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.warning("publish to %s failed: rc=%s", topic, info.rc)

    # ---- public event helpers ---------------------------------------------

    def publish_image_event(
        self,
        image_path: Path,
        changed_fraction: float,
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "station": self.cfg.station,
            "captured_at": _now_iso(),
            "camera": self.cfg.camera,
            "trigger": "motion",
            "resolution": [self.cfg.capture.width, self.cfg.capture.height],
            "file": str(image_path),
            "changed_fraction": changed_fraction,
            "sha256": _sha256(image_path),
        }
        self._publish(self._topic("image/event"), payload)

    def publish_status(self, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "station": self.cfg.station,
            "ts": _now_iso(),
            "hostname": socket.gethostname(),
        }
        if extra:
            payload.update(extra)
        self._publish(self._topic("status"), payload, retain=True)
