"""Shared MQTT configuration primitives.

``MqttBaseConfig`` is the common field set every SBO service needs:
broker endpoint, credentials, topic prefix. Packages extend it (via
subclass or by composition) with the fields specific to their role —
e.g. banshee adds ``station_filter`` and ``client_id`` for the
subscriber side.

Having the base here keeps the two sides from drifting: if we add a
TLS field tomorrow, both horus and banshee get it automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

MQTT_DEFAULT_PORT = 1883
"""Default MQTT port for every SBO service.

Single source of truth — horus, banshee, and any future node type
resolve the default here rather than hardcoding ``1883`` inline.
"""

_MQTT_DEFAULT_TOPIC_PREFIX = "sbo"


@dataclass(frozen=True)
class MqttBaseConfig:
    """Broker connection + authentication + topic prefix.

    Fields
    ------
    host:
        Broker hostname or IP. Required; no default to avoid pointing
        at something unintentional.
    port:
        Broker TCP port. Defaults to :data:`MQTT_DEFAULT_PORT`.
    username, password:
        Optional credentials. Leave ``None`` for anonymous brokers.
    topic_prefix:
        Top-level namespace for every SBO topic. Changing this across
        the whole fleet would isolate a test broker from prod traffic
        without any code changes.
    """

    host: str
    port: int = MQTT_DEFAULT_PORT
    username: str | None = None
    password: str | None = None
    topic_prefix: str = _MQTT_DEFAULT_TOPIC_PREFIX

    @classmethod
    def from_env(
        cls,
        host_var: str = "MQTT_HOST",
        port_var: str = "MQTT_PORT",
        username_var: str = "MQTT_USERNAME",
        password_var: str = "MQTT_PASSWORD",
        topic_prefix_var: str = "MQTT_TOPIC_PREFIX",
    ) -> "MqttBaseConfig":
        """Construct from process environment.

        Raises :class:`ValueError` if ``host_var`` is unset — the broker
        address is the one field with no sensible default.
        """
        host = os.environ.get(host_var)
        if not host:
            raise ValueError(f"required env var {host_var} is unset")
        return cls(
            host=host,
            port=int(os.environ.get(port_var, str(MQTT_DEFAULT_PORT))),
            username=os.environ.get(username_var) or None,
            password=os.environ.get(password_var) or None,
            topic_prefix=os.environ.get(
                topic_prefix_var, _MQTT_DEFAULT_TOPIC_PREFIX
            ),
        )
