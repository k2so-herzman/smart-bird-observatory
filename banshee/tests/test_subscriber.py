"""Tests for banshee.subscriber — table-driven dispatch + injection.

Covers the goal of issue #10: a new topic means one entry in
``_dispatch`` and one handler method, not an ``elif`` chain edit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from banshee.config import (
    BansheeConfig,
    InfluxConfig,
    MqttConfig,
    NotifyConfig,
    ThothStorageConfig,
)
from banshee.events import ImageEvent, StatusEvent
from banshee.minio_store import MinioConfig
from banshee.subscriber import Subscriber


def _cfg(tmp_path: Path, station_filter: str = "+") -> BansheeConfig:
    return BansheeConfig(
        mqtt=MqttConfig(host="unused", station_filter=station_filter),
        storage=ThothStorageConfig(
            db_path=tmp_path / "events.db",
            minio=MinioConfig(endpoint="x:9000", access_key="k", secret_key="s"),
        ),
        influx=InfluxConfig(),
        notify=NotifyConfig(),
    )


def _image_payload() -> dict:
    body = b"x"
    return {
        "schema_version": 1,
        "station": "horus",
        "captured_at": datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc).isoformat(),
        "camera": "imx519",
        "trigger": "motion",
        "resolution": [100, 100],
        "content_type": "image/jpeg",
        "image_b64": "eA==",  # "x"
        "size_bytes": len(body),
        "changed_fraction": 0.01,
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def _status_payload() -> dict:
    return {
        "schema_version": 1,
        "station": "horus",
        "ts": datetime(2026, 4, 17, 22, 0, tzinfo=timezone.utc).isoformat(),
        "camera_ok": True,
    }


@pytest.fixture
def subscriber(tmp_path: Path):
    on_image = MagicMock()
    on_status = MagicMock()
    client = MagicMock()
    sub = Subscriber(
        _cfg(tmp_path),
        on_image=on_image,
        on_status=on_status,
        client=client,
    )
    return sub, on_image, on_status, client


def test_subscribe_topics_use_shared_builder(subscriber, tmp_path):
    sub, _on_image, _on_status, _client = subscriber
    assert sub._image_topic() == "sbo/+/image/event"
    assert sub._status_topic() == "sbo/+/status"


def test_concrete_station_filter_narrows_subscriptions(tmp_path):
    sub = Subscriber(
        _cfg(tmp_path, station_filter="horus"),
        on_image=MagicMock(),
        on_status=MagicMock(),
        client=MagicMock(),
    )
    assert sub._image_topic() == "sbo/horus/image/event"


def test_dispatch_table_has_both_handlers(subscriber):
    """If someone accidentally drops a dispatch entry this test catches it."""
    sub, *_ = subscriber
    from sbo_shared import TOPIC_IMAGE_EVENT, TOPIC_STATUS
    assert TOPIC_IMAGE_EVENT in sub._dispatch
    assert TOPIC_STATUS in sub._dispatch


def _msg(topic: str, payload: dict) -> MagicMock:
    m = MagicMock()
    m.topic = topic
    m.payload = json.dumps(payload).encode()
    return m


def test_image_message_routes_to_image_handler(subscriber):
    sub, on_image, on_status, _client = subscriber
    sub._on_message(None, None, _msg("sbo/horus/image/event", _image_payload()))
    on_image.assert_called_once()
    assert isinstance(on_image.call_args.args[0], ImageEvent)
    on_status.assert_not_called()


def test_status_message_routes_to_status_handler(subscriber):
    sub, on_image, on_status, _client = subscriber
    sub._on_message(None, None, _msg("sbo/horus/status", _status_payload()))
    on_status.assert_called_once()
    assert isinstance(on_status.call_args.args[0], StatusEvent)
    on_image.assert_not_called()


def test_unknown_topic_is_ignored(subscriber):
    sub, on_image, on_status, _client = subscriber
    sub._on_message(None, None, _msg("sbo/horus/not-a-real-topic", {}))
    on_image.assert_not_called()
    on_status.assert_not_called()


def test_bad_json_payload_does_not_crash(subscriber):
    sub, on_image, on_status, _client = subscriber
    bad = MagicMock()
    bad.topic = "sbo/horus/image/event"
    bad.payload = b"{not-json"
    sub._on_message(None, None, bad)
    on_image.assert_not_called()


def test_image_handler_exception_is_logged_not_raised(subscriber, caplog):
    """Handler bugs must not take down the paho loop thread."""
    sub, on_image, _on_status, _client = subscriber
    on_image.side_effect = RuntimeError("boom")
    sub._on_message(None, None, _msg("sbo/horus/image/event", _image_payload()))
    # Did not propagate; logged.
    assert any("image handler raised" in r.message for r in caplog.records)
