"""MQTT subscriber for Banshee.

Subscribes to SBO topics, decodes events, hands them to the pipeline.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt

from .config import BansheeConfig
from .events import EventError, ImageEvent, StatusEvent

log = logging.getLogger(__name__)


ImageHandler = Callable[[ImageEvent], None]
StatusHandler = Callable[[StatusEvent], None]


class Subscriber:
    def __init__(
        self,
        cfg: BansheeConfig,
        on_image: ImageHandler,
        on_status: StatusHandler,
    ) -> None:
        self.cfg = cfg
        self._on_image = on_image
        self._on_status = on_status
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="banshee-bird-brain",
        )
        if cfg.mqtt.username:
            self._client.username_pw_set(cfg.mqtt.username, cfg.mqtt.password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _image_topic(self) -> str:
        return f"{self.cfg.mqtt.topic_prefix}/{self.cfg.mqtt.station_filter}/image/event"

    def _status_topic(self) -> str:
        return f"{self.cfg.mqtt.topic_prefix}/{self.cfg.mqtt.station_filter}/status"

    def _on_connect(self, client: mqtt.Client, _userdata: Any, _flags: Any, rc: int, _props: Any = None) -> None:
        if rc != 0:
            log.error("MQTT connect failed: rc=%s", rc)
            return
        log.info("MQTT connected to %s:%s", self.cfg.mqtt.host, self.cfg.mqtt.port)
        client.subscribe([(self._image_topic(), 1), (self._status_topic(), 1)])
        log.info("subscribed to %s and %s", self._image_topic(), self._status_topic())

    def _on_message(self, _client: mqtt.Client, _userdata: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError as exc:
            log.warning("non-JSON payload on %s: %s", msg.topic, exc)
            return

        if msg.topic.endswith("/image/event"):
            self._dispatch_image(payload)
        elif msg.topic.endswith("/status"):
            self._dispatch_status(payload)
        else:
            log.debug("unhandled topic: %s", msg.topic)

    def _dispatch_image(self, payload: dict) -> None:
        try:
            event = ImageEvent.from_payload(payload)
        except EventError as exc:
            log.warning("bad image event: %s", exc)
            return
        try:
            self._on_image(event)
        except Exception:
            log.exception("image handler raised")

    def _dispatch_status(self, payload: dict) -> None:
        try:
            event = StatusEvent.from_payload(payload)
        except EventError as exc:
            log.warning("bad status event: %s", exc)
            return
        try:
            self._on_status(event)
        except Exception:
            log.exception("status handler raised")

    def run_forever(self) -> None:
        self._client.connect(self.cfg.mqtt.host, self.cfg.mqtt.port, keepalive=60)
        self._client.loop_forever()

    def stop(self) -> None:
        self._client.disconnect()
