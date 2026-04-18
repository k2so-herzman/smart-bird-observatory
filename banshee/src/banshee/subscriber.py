"""MQTT subscriber for Banshee.

``Subscriber`` is the inbound boundary of the Banshee pipeline.  It
maintains a single paho MQTT connection, decodes raw JSON payloads into
typed :mod:`~banshee.events` objects, and forwards them to handler
callbacks — normally the ``on_image`` / ``on_status`` methods of a
:class:`~banshee.pipeline.Pipeline` instance.

Relationship to Pipeline
------------------------
``Subscriber`` knows nothing about storage or enrichment.  Its only
job is *transport* and *deserialization*:

1. Connect to the broker and subscribe to configured topics.
2. Validate and decode every arriving message into a typed event.
3. Hand the event to the callback the caller registered at construction
   time.

The callbacks are supplied as plain callables so that tests can inject
stubs and the caller (``Pipeline``) can keep its own state without
coupling to MQTT internals.

Design choices
--------------

* **Table-driven dispatch.** A ``{suffix: handler}`` map replaces the
  old ``if msg.topic.endswith(...)`` chain. Adding a new topic type
  means one entry in ``_dispatch`` and one handler method — no
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

    Lifecycle
    ---------
    1. **Construct** — pass a :class:`~banshee.config.BansheeConfig`,
       handler callbacks, and an optional pre-built MQTT client::

           sub = Subscriber(cfg, pipeline.on_image, pipeline.on_status)

    2. **Run** — hand control to paho's blocking I/O loop::

           sub.run_forever()   # blocks; runs until stop() is called

    3. **Stop** — call from a signal handler or a separate thread to
       initiate a clean disconnect and unblock ``run_forever``::

           sub.stop()

    Dispatch model
    --------------
    Incoming messages are routed by *topic suffix* (the part of the
    topic after ``{prefix}/{station}/``).  The routing table
    ``_dispatch`` maps each known suffix constant to a private handler
    method.  ``_on_message`` calls :func:`sbo_shared.topics.topic_suffix`
    to extract the suffix from the full topic string, looks it up in the
    table, and delegates — no ``if/elif`` chains, no ``endswith`` calls
    scattered around the class.

    To add a new topic type:

    a. Add a ``TOPIC_*`` constant to :mod:`sbo_shared.topics` and keep
       :func:`~sbo_shared.topics.topic_suffix` in sync.
    b. Add one entry to ``_dispatch`` in :meth:`__init__`.
    c. Add a ``_dispatch_<x>`` method that accepts a raw ``dict``
       payload, constructs the appropriate typed event, and calls the
       registered callback.

    Parameters
    ----------
    cfg:
        Full Banshee config; only ``cfg.mqtt`` fields are read here.
    on_image:
        Called with a validated :class:`~banshee.events.ImageEvent` for
        every ``TOPIC_IMAGE_EVENT`` message.  Invoked on the paho network
        thread — **must not block for long** or MQTT keepalives will
        time out.  Exceptions are caught and logged so they cannot crash
        the loop.
    on_status:
        Called with a validated :class:`~banshee.events.StatusEvent` for
        every ``TOPIC_STATUS`` message.  Same threading caveats as
        ``on_image``.
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
        """Wire up paho callbacks and build the topic-suffix dispatch table.

        Parameters
        ----------
        cfg:
            Full Banshee config.  ``cfg.mqtt.host``, ``cfg.mqtt.port``,
            ``cfg.mqtt.topic_prefix``, ``cfg.mqtt.station_filter``,
            ``cfg.mqtt.client_id``, ``cfg.mqtt.username``, and
            ``cfg.mqtt.password`` are consumed here.
        on_image:
            Callback invoked (on the paho network thread) with each
            successfully decoded :class:`~banshee.events.ImageEvent`.
            **Threading caveat:** paho calls ``on_message`` — and
            therefore this callback — from its own internal thread
            spawned by ``loop_forever``.  If the handler accesses shared
            state it must synchronise externally (e.g. a ``queue.Queue``
            or a thread-safe pipeline stage).
        on_status:
            Same threading caveats as ``on_image``.  Called for each
            decoded :class:`~banshee.events.StatusEvent`.
        client:
            Inject a pre-built (or fake) MQTT client.  ``None`` creates
            a real paho client using ``cfg.mqtt.client_id``.
        """
        self.cfg = cfg
        self._on_image = on_image
        self._on_status = on_status

        client_id = cfg.mqtt.client_id or self.DEFAULT_CLIENT_ID
        self._client = client if client is not None else _default_client_factory(client_id)

        if cfg.mqtt.username:
            self._client.username_pw_set(cfg.mqtt.username, cfg.mqtt.password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        # Table-driven dispatch: key = topic suffix as returned by
        # sbo_shared.topics.topic_suffix(), value = method that accepts
        # a raw dict payload and forwards a typed event to the registered
        # callback.  To add a new topic: (a) add a TOPIC_* constant to
        # sbo_shared.topics and update topic_suffix() there, (b) add one
        # entry here, (c) add a _dispatch_<x> method below.
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
        """Paho ``on_connect`` callback — subscribe to topics on success.

        Paho invokes this on the network thread immediately after the
        broker sends a ``CONNACK``.  ``rc == 0`` means the connection was
        accepted; any other value is a paho error code (see
        ``paho.mqtt.client.CONNACK_ERRORS``).

        On success both topics (image-event and status) are subscribed at
        QoS 1, so every message is delivered at least once.  Subscribing
        inside ``on_connect`` rather than in :meth:`run_forever` ensures
        subscriptions are automatically reinstated if paho reconnects
        after a broker restart.

        Parameters
        ----------
        client:
            The paho client instance (same object as ``self._client``).
        rc:
            Return code from the broker.  ``0`` = success.
        """
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
        """Paho ``on_message`` callback — decode JSON and dispatch by topic suffix.

        Called by paho on the network thread for every message that
        arrives on a subscribed topic.

        Processing steps:

        1. Attempt ``json.loads`` on ``msg.payload``.  Non-JSON bytes are
           logged at WARNING and dropped — they indicate a misconfigured
           publisher, not a transient error.
        2. Call :func:`sbo_shared.topics.topic_suffix` to strip the
           ``{prefix}/{station}/`` prefix from the full topic string and
           return the canonical suffix (e.g. ``"image/event"``).
        3. Look the suffix up in ``_dispatch``.  An unrecognised suffix
           (``None`` from ``topic_suffix``, or a suffix not in the table)
           is logged at DEBUG and skipped — future topic types arrive
           here before a handler is wired up and should not be noisy.
        4. Delegate to the matched handler method, which validates,
           decodes, and forwards the event to the registered callback.

        Parameters
        ----------
        msg:
            Paho message object.  ``msg.topic`` is the full wire topic;
            ``msg.payload`` is the raw bytes.
        """
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
        """Decode a raw image-event payload and call the ``on_image`` handler.

        Constructs an :class:`~banshee.events.ImageEvent` via
        :meth:`~banshee.events.ImageEvent.from_payload`, which validates
        required fields, base64-decodes the image bytes, verifies the
        ``sha256`` checksum, and checks ``size_bytes``.  Any validation
        failure raises :class:`~banshee.events.EventError`, which is
        caught here and logged at WARNING so a single corrupt message
        cannot stall the loop.

        If decoding succeeds the validated event is forwarded to the
        ``on_image`` callback registered at construction.  Exceptions
        from that callback are also caught and logged (at ERROR with
        traceback) so a bug in downstream processing cannot crash the
        MQTT loop.

        Parameters
        ----------
        payload:
            Pre-parsed JSON dict from ``_on_message``.
        """
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
        """Decode a raw status payload and call the ``on_status`` handler.

        Constructs a :class:`~banshee.events.StatusEvent` via
        :meth:`~banshee.events.StatusEvent.from_payload`, which validates
        the required ``schema_version``, ``station``, and ``ts`` fields.
        The full raw dict is also preserved on the event so callers can
        inspect fields that are not yet promoted to typed attributes.

        Validation failures (:class:`~banshee.events.EventError`) are
        logged at WARNING and dropped.  Exceptions from the ``on_status``
        callback are caught and logged at ERROR so they cannot crash the
        MQTT loop.

        Parameters
        ----------
        payload:
            Pre-parsed JSON dict from ``_on_message``.
        """
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
        """Connect to the broker and block in the paho event loop.

        Calls ``paho.Client.connect`` (synchronous TCP handshake + MQTT
        ``CONNECT``) then hands control to ``loop_forever``, which
        manages keepalives, reconnection, and callback dispatch on its
        own internal thread.

        This method blocks until :meth:`stop` is called (typically from a
        ``SIGTERM`` / ``SIGINT`` handler in the process entry-point).
        """
        self._client.connect(self.cfg.mqtt.host, self.cfg.mqtt.port, keepalive=60)
        self._client.loop_forever()

    def stop(self) -> None:
        """Disconnect cleanly and unblock ``run_forever``.

        Sends an MQTT ``DISCONNECT`` packet to the broker, closes the TCP
        socket, and causes ``loop_forever`` to return.  Safe to call from
        a signal handler or any thread — paho serialises the disconnect
        internally.

        After ``stop`` returns, the ``Subscriber`` instance should be
        considered finished; do not call :meth:`run_forever` again on the
        same instance.
        """
        self._client.disconnect()
