"""Shared primitives for the Smart Bird Observatory services.

This package holds the *only canonical* definitions of things that
must agree across every SBO node:

- :mod:`sbo_shared.time` — the ISO-8601 timestamp format every event
  payload uses.
- :mod:`sbo_shared.topics` — MQTT topic suffix constants and the
  ``build_topic`` / ``topic_suffix`` helpers. If a service hardcodes
  a topic string, this is the bug.
- :mod:`sbo_shared.mqtt_config` — the ``MqttBaseConfig`` dataclass
  and ``MQTT_DEFAULT_PORT`` constant that horus (publisher) and
  banshee (subscriber) both extend.

Intentionally dependency-free: anything that would force a runtime
dependency belongs in the consuming package.
"""

from .mqtt_config import MQTT_DEFAULT_PORT, MqttBaseConfig
from .time import sbo_now_iso
from .topics import (
    TOPIC_AUDIO_DETECTION,
    TOPIC_IMAGE_EVENT,
    TOPIC_STATUS,
    build_topic,
    topic_suffix,
)

__all__ = [
    "MQTT_DEFAULT_PORT",
    "MqttBaseConfig",
    "TOPIC_AUDIO_DETECTION",
    "TOPIC_IMAGE_EVENT",
    "TOPIC_STATUS",
    "build_topic",
    "sbo_now_iso",
    "topic_suffix",
]
