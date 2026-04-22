"""Tests for horus.events — PUBACK-aware publish path.

The bug being regressed: ``EventBus._publish`` used to check
``info.rc`` *after* ``wait_for_publish``. ``rc`` is the enqueue result,
not the broker ack, so a silently-dropped message (e.g. broker timeout
without PUBACK) would log nothing. The current contract:

* ``rc != MQTT_ERR_SUCCESS`` at enqueue → warn + return.
* ``wait_for_publish`` raises → warn + return.
* ``wait_for_publish`` returns but ``is_published()`` is False
  (timeout) → warn.

Connection-observability contract (added later):

* ``__init__`` wires ``on_connect``/``on_disconnect`` and caps
  reconnect backoff via ``reconnect_delay_set``.
* ``connect()`` registers an LWT (``online: false``) on the status
  topic *before* calling broker ``connect()``, so the broker honors it.
* ``publish_status`` stamps ``online: true`` so the retained topic
  pairs with the LWT.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest

from horus.config import (
    CaptureConfig,
    HorusConfig,
    MotionConfig,
    MqttConfig,
    StorageConfig,
)
from horus.events import EventBus


@pytest.fixture
def cfg(tmp_path: Path) -> HorusConfig:
    return HorusConfig(
        station="horus-test",
        camera="imx519",
        mqtt=MqttConfig(host="localhost"),
        capture=CaptureConfig(),
        motion=MotionConfig(),
        storage=StorageConfig(local_dir=tmp_path),
    )


def _make_bus(cfg: HorusConfig, info: MagicMock) -> EventBus:
    """Build an EventBus whose paho client returns ``info`` from publish()."""
    bus = EventBus(cfg)
    bus._client = MagicMock()
    bus._client.publish.return_value = info
    return bus


def _ack_info() -> MagicMock:
    info = MagicMock()
    info.rc = mqtt.MQTT_ERR_SUCCESS
    info.wait_for_publish.return_value = None
    info.is_published.return_value = True
    return info


def test_publish_success_returns_true_and_is_silent(cfg, caplog):
    info = _ack_info()
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is True
    assert bus.dropped_publishes == 0
    assert caplog.records == []
    info.wait_for_publish.assert_called_once()


def test_publish_enqueue_failure_short_circuits(cfg, caplog):
    """If the enqueue fails we must NOT wait for a PUBACK that never comes."""
    info = _ack_info()
    info.rc = mqtt.MQTT_ERR_NO_CONN
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    info.wait_for_publish.assert_not_called()
    assert any("enqueue" in r.message for r in caplog.records)


def test_publish_wait_raises_returns_false(cfg, caplog):
    info = _ack_info()
    info.wait_for_publish.side_effect = RuntimeError("disconnected mid-publish")
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    assert any("awaiting ack" in r.message for r in caplog.records)


def test_publish_no_puback_within_timeout_returns_false(cfg, caplog):
    """This is the regression: wait_for_publish returns (no raise) but the
    broker never acked. ``rc`` still shows ENQUEUE success (0), so the old
    check passed silently. We now require is_published() to be True and
    return False so callers can skip success side-effects."""
    info = _ack_info()
    info.is_published.return_value = False  # no PUBACK
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    assert any("timed out" in r.message for r in caplog.records)


def test_publish_is_published_raises_returns_false(cfg, caplog):
    """Guard against the race where the paho loop thread flips rc to an
    error between wait_for_publish returning and is_published() being
    called. We treat this as a drop rather than letting the exception
    propagate into the capture loop."""
    info = _ack_info()
    info.is_published.side_effect = ValueError("publish not complete")
    bus = _make_bus(cfg, info)
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        ok = bus._publish("sbo/horus-test/status", {"ok": True})
    assert ok is False
    assert bus.dropped_publishes == 1
    assert any("ack check" in r.message for r in caplog.records)


def test_dropped_publishes_counter_accumulates(cfg):
    info = _ack_info()
    info.is_published.return_value = False
    bus = _make_bus(cfg, info)
    bus._publish("sbo/horus-test/status", {"ok": True})
    bus._publish("sbo/horus-test/status", {"ok": True})
    bus._publish("sbo/horus-test/status", {"ok": True})
    assert bus.dropped_publishes == 3


def test_status_payload_carries_camera_label(cfg):
    info = _ack_info()
    bus = _make_bus(cfg, info)
    bus.publish_status({"camera_ok": True})
    # Assert the status topic + retain flag were used and payload is JSON.
    args, kwargs = bus._client.publish.call_args
    topic, payload = args[0], args[1]
    assert topic == "sbo/horus-test/status"
    assert kwargs.get("retain") is True
    assert '"station": "horus-test"' in payload


def test_image_event_uses_cfg_camera_label(cfg, tmp_path):
    """The event payload's ``camera`` field must reflect cfg — that's the
    only place downstream services learn which sensor produced the frame."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-ish bytes
    ok = bus.publish_image_event(img, changed_fraction=0.03)
    assert ok is True
    payload = bus._client.publish.call_args.args[1]
    assert '"camera": "imx519"' in payload


def test_image_event_returns_false_on_drop(cfg, tmp_path):
    """Callers (main._tick) rely on this return to skip advancing the
    cooldown on a failed publish."""
    info = _ack_info()
    info.is_published.return_value = False
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    ok = bus.publish_image_event(img, changed_fraction=0.03)
    assert ok is False
    assert bus.dropped_publishes == 1


def test_image_event_uses_cfg_resolution_by_default(cfg, tmp_path):
    """No resolution_override → published ``resolution`` matches capture cfg."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03)
    payload = bus._client.publish.call_args.args[1]
    # CaptureConfig default is 2304 x 1296.
    assert '"resolution": [2304, 1296]' in payload


def test_image_event_resolution_override_is_used(cfg, tmp_path):
    """When the caller passes resolution_override (crop path) the published
    ``resolution`` reflects the actual image bytes, not the sensor config."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03, resolution_override=(640, 640))
    payload = bus._client.publish.call_args.args[1]
    assert '"resolution": [640, 640]' in payload


def test_image_event_includes_bbox_when_provided(cfg, tmp_path):
    """bbox_fraction lets Thoth distinguish focal motion (bird) from
    distributed motion (wind sway). When passed, it must ride on the payload."""
    import json

    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(
        img,
        changed_fraction=0.03,
        bbox_fraction=(0.1, 0.2, 0.5, 0.8),
    )
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert payload["bbox_fraction"] == [0.1, 0.2, 0.5, 0.8]


def test_image_event_omits_bbox_when_absent(cfg, tmp_path):
    """Full-frame fallback publishes — no bbox → no key in payload."""
    import json

    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03)
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert "bbox_fraction" not in payload


def test_image_event_includes_af_when_provided(cfg, tmp_path):
    """AF summary (LensPosition/AfState/FocusFoM) piggybacks on the event so
    Thoth can correlate focus state with classifier confidence."""
    import json

    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(
        img,
        changed_fraction=0.03,
        af={"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234},
    )
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert payload["af"] == {"LensPosition": 3.02, "AfState": 2, "FocusFoM": 1234}


def test_image_event_omits_af_when_absent(cfg, tmp_path):
    """Missing sidecar → caller passes af=None → key absent from payload."""
    import json

    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03)
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert "af" not in payload


def test_image_event_preserves_empty_af_dict(cfg, tmp_path):
    """Empty AF dict (sidecar parsed but no AF keys) — we still pass it
    through; callers decide whether to pass None vs {}.  An empty dict is
    meaningfully different from ``None`` (we tried and got nothing) so it
    is preserved in the payload rather than silently dropped."""
    import json

    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03, af={})
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert payload["af"] == {}


def test_image_event_includes_bird_score_when_provided(cfg, tmp_path):
    """On-device bird classifier score rides on the payload so Thoth can
    surface it in the API and downstream tooling can compare it against
    Thoth's post-ingest classifier confidence."""
    import json

    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(
        img,
        changed_fraction=0.03,
        bird_score=0.55,
        bird_label="carolina chickadee",
    )
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert payload["bird_score"] == pytest.approx(0.55)
    assert payload["bird_label"] == "carolina chickadee"


def test_image_event_omits_bird_score_when_absent(cfg, tmp_path):
    """Classifier disabled → no bird_score/bird_label keys in payload.
    Older Thoth builds read schema_version=1 strictly, so absent-key
    must stay absent rather than being set to null."""
    import json

    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03)
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert "bird_score" not in payload
    assert "bird_label" not in payload


def test_image_event_includes_burst_metadata_when_provided(cfg, tmp_path):
    """Burst grouping (PR-A): frames from the same motion session share a
    burst_id and carry a monotonic burst_seq so Thoth can fold them into
    a single tile with a hero frame plus alternates.  When the caller
    supplies both fields, they must ride on the payload unchanged."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(
        img,
        changed_fraction=0.03,
        burst_id="horus-test-1700000000000-abcd",
        burst_seq=3,
    )
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert payload["burst_id"] == "horus-test-1700000000000-abcd"
    assert payload["burst_seq"] == 3


def test_image_event_omits_burst_when_absent(cfg, tmp_path):
    """Legacy singleton publishes (burst.enabled=False) omit burst keys
    entirely.  Absent-key is the signal to Thoth: treat as a singleton
    rather than a one-frame burst, which matters for the UI grouping
    rules on older schema_version=1 consumers."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(img, changed_fraction=0.03)
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert "burst_id" not in payload
    assert "burst_seq" not in payload


def test_image_event_coerces_burst_seq_to_int(cfg, tmp_path):
    """Defensive: even though callers pass ints today, coerce so a
    stray numpy int or other non-plain-int value round-trips as a
    clean JSON integer rather than surfacing as a typed scalar that
    trips downstream parsers."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    bus.publish_image_event(
        img,
        changed_fraction=0.03,
        burst_id="horus-test-1700000000000-abcd",
        burst_seq=True,  # worst-case non-int-int: bool is a subclass of int
    )
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert payload["burst_seq"] == 1
    assert isinstance(payload["burst_seq"], int)


# --- connection-observability tests ------------------------------------


def test_init_configures_reconnect_delay(cfg):
    """Paho's default max reconnect delay is 120s — way too long for a
    60s heartbeat cadence. Confirm we cap it."""
    with patch("horus.events.mqtt.Client") as client_cls:
        EventBus(cfg)
    client = client_cls.return_value
    client.reconnect_delay_set.assert_called_once_with(min_delay=1, max_delay=30)


def test_init_registers_connect_disconnect_handlers(cfg):
    """Without these callbacks, broker flaps are invisible in capture.log
    and publish failures can't be correlated to disconnect events.

    Compare bound methods by equality (not ``is``) — Python recreates
    bound-method objects on each attribute access so identity doesn't
    hold even for the same underlying function.
    """
    with patch("horus.events.mqtt.Client") as client_cls:
        bus = EventBus(cfg)
    client = client_cls.return_value
    assert client.on_connect == bus._on_connect
    assert client.on_disconnect == bus._on_disconnect


def test_connect_registers_lwt_before_broker_connect(cfg):
    """LWT must be declared before CONNECT — the broker only honors
    will_set calls that happened pre-connect. Published payload must
    be ``online: false`` on the retained status topic."""
    with patch("horus.events.mqtt.Client") as client_cls:
        bus = EventBus(cfg)
        bus.connect()
    client = client_cls.return_value
    # Order-of-operations: will_set must be called before connect().
    call_order = [c[0] for c in client.method_calls]
    assert call_order.index("will_set") < call_order.index("connect")
    will_call = client.will_set.call_args
    topic = will_call.args[0]
    payload = json.loads(will_call.args[1])
    assert topic == "sbo/horus-test/status"
    assert will_call.kwargs.get("qos") == 1
    assert will_call.kwargs.get("retain") is True
    assert payload["station"] == "horus-test"
    assert payload["online"] is False


def test_connect_starts_network_loop(cfg):
    """Regression: loop_start() must still be called post-connect; the
    LWT change shouldn't break the thread that drives PINGREQ + auto-
    reconnect."""
    with patch("horus.events.mqtt.Client") as client_cls:
        bus = EventBus(cfg)
        bus.connect()
    client = client_cls.return_value
    client.loop_start.assert_called_once()


def test_on_connect_logs_info_on_success(cfg, caplog):
    bus = EventBus(cfg)
    success_rc = MagicMock()
    success_rc.is_failure = False
    with caplog.at_level(logging.INFO, logger="horus.events"):
        bus._on_connect(MagicMock(), None, {}, success_rc)
    assert any("connected" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_on_connect_logs_warning_on_failure(cfg, caplog):
    bus = EventBus(cfg)
    fail_rc = MagicMock()
    fail_rc.is_failure = True
    fail_rc.__str__ = lambda self: "Not authorized"
    with caplog.at_level(logging.WARNING, logger="horus.events"):
        bus._on_connect(MagicMock(), None, {}, fail_rc)
    assert any("connect" in r.message.lower() and "failed" in r.message.lower() for r in caplog.records)


def test_on_disconnect_logs_warning_on_unexpected_drop(cfg, caplog):
    """An unexpected disconnect log line is the anchor for diagnosing
    ``rc=4`` drops. Must fire at WARNING so it shows up in a grep that
    skips INFO, and the message must flag that reconnect is coming."""
    bus = EventBus(cfg)
    fail_rc = MagicMock()
    fail_rc.is_failure = True
    with caplog.at_level(logging.INFO, logger="horus.events"):
        bus._on_disconnect(MagicMock(), None, {}, fail_rc)
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("disconnected unexpectedly" in r.message.lower() for r in warn_records)
    assert any("auto-reconnect" in r.message.lower() for r in warn_records)


def test_on_disconnect_logs_info_on_clean_shutdown(cfg, caplog):
    """Clean disconnects (our own ``disconnect()``) must NOT emit a
    warning — paho doesn't auto-reconnect in this case and the misleading
    "auto-reconnect scheduled" noise would burn operator attention
    every shutdown."""
    bus = EventBus(cfg)
    clean_rc = MagicMock()
    clean_rc.is_failure = False
    with caplog.at_level(logging.INFO, logger="horus.events"):
        bus._on_disconnect(MagicMock(), None, {}, clean_rc)
    # No WARNING records at all on clean path.
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    # And the INFO message should say "cleanly" so it's greppable.
    assert any("cleanly" in r.message.lower() for r in caplog.records)


def test_publish_status_stamps_online_true(cfg):
    """Pairs with the LWT: retained status topic should carry
    ``online: true`` while the client is alive, so consumers can trust
    a single topic as the source of liveness truth."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    bus.publish_status({"camera_ok": True})
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert payload["online"] is True
    assert payload["camera_ok"] is True


def test_disconnect_publishes_offline_status_then_stops_loop(cfg):
    """Graceful shutdown should write ``online: false`` to the retained
    status topic before tearing down — otherwise the last heartbeat
    (``online: true``) stays retained until the next boot, and any
    consumer polling during the gap sees stale liveness."""
    info = _ack_info()
    bus = _make_bus(cfg, info)
    bus.disconnect()
    # Last publish() call on _client should have been the offline status.
    topic = bus._client.publish.call_args.args[0]
    payload = json.loads(bus._client.publish.call_args.args[1])
    assert topic == "sbo/horus-test/status"
    assert payload["online"] is False
    bus._client.loop_stop.assert_called_once()
    bus._client.disconnect.assert_called_once()


def test_disconnect_is_best_effort_on_publish_failure(cfg):
    """If the offline publish itself fails (e.g. broker already gone),
    we still need to tear down loop_start() and disconnect the socket —
    otherwise the process hangs on shutdown waiting for the network
    thread. LWT will carry the signal to the broker on the broker's
    side when the socket FINs."""
    info = _ack_info()
    info.rc = mqtt.MQTT_ERR_NO_CONN  # enqueue fails → _publish returns False
    bus = _make_bus(cfg, info)
    bus.disconnect()  # must not raise
    bus._client.loop_stop.assert_called_once()
    bus._client.disconnect.assert_called_once()
