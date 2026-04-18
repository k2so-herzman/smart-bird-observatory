"""MQTT subscriber for Banshee.

Subscribes to SBO topics, decodes payloads into typed events, and
hands them to registered handlers. The on-the-wire topic layout
(``{prefix}/{station}/{suffix}``) is defined in :mod:`sbo_shared.topics`.

Design choices
--------------

* **Table-driven dispatch.** A ``{suffix: handler}`` map replaces the
  old ``if msg.topic.endswith(...)`` chain. Adding a new topic type
  means one entry in ``_DISPATCH`` and one handler method — no
  Open/Closed violation.
* **Injectable paho client.** Tests pass an ``mqtt.Client`` fake to
  exercise the full ``on_connect`` / ``on_message`` callback flow
  without a running broker.
* **Handler exceptions are contained.** A buggy image handler must
  not crash the MQTT loop — we log and continue so one bad payload
  doesn't take the ingestor down.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import paho.mqtt.client as mqtt
from sbo_shared import (
    TOPIC_IMAGE_EVENT,
    TOPIC_STATUS,
    build_topic,
    topic_suffix,
)

from .config import BansheeConfig
from .events import EventError, ImageEvent, StatusEvent

log = logging.getLogger(__name__)


ImageHandler = Callable[[ImageEvent], None]
StatusHandler = Callable[[StatusEvent], None]


class _MqttClientLike(Protocol):
    """Subset of ``paho.mqtt.client.Client`` the subscriber relies on."""

    on_connect: Any
    on_message: Any

    def username_pw_set(self, username: str, password: str | None) -> None: ...

    def connect(self, host: str, port: int, keepalive: int) -> int: ...

    def subscribe(self, topics: Any) -> Any: ...

    def loop_forever(self) -> int: ...

    def disconnect(self) -> int: ...


def _default_client_factory(client_id: str) -> mqtt.Client:
    """Build the real paho client. Pulled out for test injection."""
    return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)


class Subscriber:
    """Subscribe to SBO topics and route decoded events to handlers.

    Parameters
    ----------
    cfg:
        Full Banshee config; only ``cfg.mqtt`` fields are read here.
    on_image, on_status:
        Handler callbacks. Typically methods on the ``Pipeline``.
    client:
        Optional pre-built MQTT client. If ``None``, a paho client is
        created using ``cfg.mqtt.client_id`` or a stable default. Tests
        pass a fake to verify the dispatch flow without a broker.
    """

    DEFAULT_CLIENT_ID = "banshee-bird-brain"

    def __init__(
        self,
        cfg: BansheeConfig,
        on_image: ImageHandler,
        on_status: StatusHandler,
        client: _MqttClientLike | None = None,
    ) -> None:
        self.cfg = cfg
        self._on_image = on_image
        self._on_status = on_status

        client_id = cfg.mqtt.client_id or self.DEFAULT_CLIENT_ID
        self._client = client if client is not None else _default_client_factory(client_id)

        if cfg.mqtt.username:
            self._client.username_pw_set(cfg.mqtt.username, cfg.mqtt.password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        # Table-driven dispatch by topic suffix. To add a new topic
        # type: register a handler method and add one entry here.
        self._dispatch: Mapping[str, Callable[[dict], None]] = {
            TOPIC_IMAGE_EVENT: self._dispatch_image,
            TOPIC_STATUS: self._dispatch_status,
        }

    # ---- topic builders ----------------------------------------------------

    def _image_topic(self) -> str:
        return build_topic(
            self.cfg.mqtt.topic_prefix,
            self.cfg.mqtt.station_filter,
            TOPIC_IMAGE_EVENT,
        )

    def _status_topic(self) -> str:
        return build_topic(
            self.cfg.mqtt.topic_prefix,
            self.cfg.mqtt.station_filter,
            TOPIC_STATUS,
        )

    # ---- paho callbacks ----------------------------------------------------

    def _on_connect(
        self,
        client: _MqttClientLike,
        _userdata: Any,
        _flags: Any,
        rc: int,
        _props: Any = None,
    ) -> None:
        """Subscribe to both topics once the broker accepts the connection."""
        if rc != 0:
            log.error("MQTT connect failed: rc=%s", rc)
            return
        log.info("MQTT connected to %s:%s", self.cfg.mqtt.host, self.cfg.mqtt.port)
        client.subscribe([(self._image_topic(), 1), (self._status_topic(), 1)])
        log.info(
            "subscribed to %s and %s", self._image_topic(), self._status_topic()
        )

    def _on_message(
        self,
        _client: _MqttClientLike,
        _userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        """Parse JSON payload and route by topic suffix."""
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError as exc:
            log.warning("non-JSON payload on %s: %s", msg.topic, exc)
            return

        suffix = topic_suffix(msg.topic)
        handler = self._dispatch.get(suffix) if suffix else None
        if handler is None:
            log.debug("unhandled topic: %s", msg.topic)
            return
        handler(payload)

    # ---- per-suffix decoders -----------------------------------------------

    def _dispatch_image(self, payload: dict) -> None:
        try:
            event = ImageEvent.from_payload(payload)
        except EventError as exc:
            log.warning("bad image event: %s", exc)
            return
        try:
            self._on_image(event)
        except Exception:
            # A bug in the handler must not take down the MQTT loop.
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

    # ---- lifecycle ---------------------------------------------------------

    def run_forever(self) -> None:
        """Connect and block in the paho event loop until ``stop()``."""
        self._client.connect(self.cfg.mqtt.host, self.cfg.mqtt.port, keepalive=60)
        self._client.loop_forever()

    def stop(self) -> None:
        """Disconnect cleanly. Safe to call from a signal handler."""
        self._client.disconnect()
