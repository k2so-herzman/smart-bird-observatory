"""Tests for sbo_shared.mqtt_config."""
from __future__ import annotations

import pytest

from sbo_shared.mqtt_config import MQTT_DEFAULT_PORT, MqttBaseConfig


def test_default_port_constant():
    assert MQTT_DEFAULT_PORT == 1883


def test_base_config_defaults():
    cfg = MqttBaseConfig(host="broker.local")
    assert cfg.host == "broker.local"
    assert cfg.port == MQTT_DEFAULT_PORT
    assert cfg.username is None
    assert cfg.password is None
    assert cfg.topic_prefix == "sbo"


def test_base_config_from_env(monkeypatch):
    monkeypatch.setenv("MQTT_HOST", "192.168.1.73")
    monkeypatch.setenv("MQTT_PORT", "8883")
    monkeypatch.setenv("MQTT_USERNAME", "u")
    monkeypatch.setenv("MQTT_PASSWORD", "p")
    monkeypatch.setenv("MQTT_TOPIC_PREFIX", "sbo-test")
    cfg = MqttBaseConfig.from_env()
    assert cfg.host == "192.168.1.73"
    assert cfg.port == 8883
    assert cfg.username == "u"
    assert cfg.password == "p"
    assert cfg.topic_prefix == "sbo-test"


def test_base_config_from_env_missing_host_raises(monkeypatch):
    monkeypatch.delenv("MQTT_HOST", raising=False)
    with pytest.raises(ValueError, match="MQTT_HOST"):
        MqttBaseConfig.from_env()


def test_base_config_from_env_custom_var_names(monkeypatch):
    """Callers can remap variable names — handy for multi-broker setups
    where one service talks to two MQTT brokers."""
    monkeypatch.setenv("ALT_HOST", "alt.broker")
    cfg = MqttBaseConfig.from_env(host_var="ALT_HOST")
    assert cfg.host == "alt.broker"
    assert cfg.port == MQTT_DEFAULT_PORT


def test_base_config_is_frozen():
    """Config objects are shared across threads once built — mutation
    would be a concurrency hazard."""
    cfg = MqttBaseConfig(host="b")
    with pytest.raises((AttributeError, TypeError)):
        cfg.host = "other"  # type: ignore[misc]
