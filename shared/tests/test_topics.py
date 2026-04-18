"""Tests for sbo_shared.topics.

These aren't just format checks — they pin the on-the-wire contract
between horus and banshee. Changing a constant here is a breaking
protocol change and both sides must be updated together.
"""
from __future__ import annotations

import pytest

from sbo_shared.topics import (
    TOPIC_AUDIO_DETECTION,
    TOPIC_IMAGE_EVENT,
    TOPIC_STATUS,
    build_topic,
    topic_suffix,
)


def test_topic_constants_match_schema_doc():
    """If you change these, ``shared/schema.md`` must change too."""
    assert TOPIC_IMAGE_EVENT == "image/event"
    assert TOPIC_STATUS == "status"
    assert TOPIC_AUDIO_DETECTION == "audio/detection"


def test_build_topic_concrete_station():
    assert build_topic("sbo", "horus", TOPIC_IMAGE_EVENT) == "sbo/horus/image/event"


def test_build_topic_wildcard_station():
    """``"+"`` is the MQTT single-level wildcard. Banshee uses it to
    subscribe to every station at once."""
    assert build_topic("sbo", "+", TOPIC_STATUS) == "sbo/+/status"


@pytest.mark.parametrize(
    "topic,expected",
    [
        ("sbo/horus/image/event", TOPIC_IMAGE_EVENT),
        ("sbo/kali/status", TOPIC_STATUS),
        ("sbo/owl/audio/detection", TOPIC_AUDIO_DETECTION),
        ("sbo/horus/unknown", None),
        ("garbage", None),
    ],
)
def test_topic_suffix_extraction(topic: str, expected: str | None):
    assert topic_suffix(topic) == expected
