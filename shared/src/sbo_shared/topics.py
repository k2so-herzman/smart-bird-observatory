"""Canonical MQTT topic suffixes + helpers.

The wire-level convention (see ``shared/schema.md``) is::

    {topic_prefix}/{station}/{suffix}

Example: ``sbo/horus/image/event``.

Publishers (horus) and subscribers (banshee) must agree on the
suffix strings. Any code that hardcodes a literal like
``"image/event"`` outside this module is a latent refactor hazard —
rename the topic and the compiler won't help you. Always import the
constants below instead.
"""

from __future__ import annotations

TOPIC_IMAGE_EVENT = "image/event"
"""Motion-triggered image capture. Payload: full image event schema.

Subscribed to by banshee at ``{prefix}/{station_filter}/image/event``.
"""

TOPIC_AUDIO_DETECTION = "audio/detection"
"""BirdNET-Go audio detection. Payload: species + confidence.

Not produced yet (planned for station-level BirdNET-Go integration),
but the suffix is reserved here so subscribers can wire a handler
without a repo-wide grep when it lands.
"""

TOPIC_STATUS = "status"
"""Retained station heartbeat. Payload: camera_ok, hostname, ts.

Retained so a late-joining subscriber (or Home Assistant) immediately
learns whether each station is alive.
"""


def build_topic(topic_prefix: str, station: str, suffix: str) -> str:
    """Assemble a full topic string.

    Parameters
    ----------
    topic_prefix:
        Top-level namespace, conventionally ``"sbo"``.
    station:
        Station identifier. Use ``"+"`` for an MQTT single-level
        wildcard when subscribing to every station.
    suffix:
        One of the ``TOPIC_*`` constants in this module.

    Examples
    --------
    >>> build_topic("sbo", "horus", TOPIC_IMAGE_EVENT)
    'sbo/horus/image/event'
    >>> build_topic("sbo", "+", TOPIC_STATUS)
    'sbo/+/status'
    """
    return f"{topic_prefix}/{station}/{suffix}"


def topic_suffix(topic: str) -> str | None:
    """Return the suffix for a concrete topic, or ``None`` if unrecognized.

    Used by dispatch tables so subscribers don't have to grep
    ``endswith()`` calls for every new suffix.

    Examples
    --------
    >>> topic_suffix("sbo/horus/image/event")
    'image/event'
    >>> topic_suffix("sbo/horus/status")
    'status'
    >>> topic_suffix("sbo/horus/nope") is None
    True
    """
    for suffix in (TOPIC_IMAGE_EVENT, TOPIC_AUDIO_DETECTION, TOPIC_STATUS):
        if topic.endswith("/" + suffix) or topic == suffix:
            return suffix
    return None
