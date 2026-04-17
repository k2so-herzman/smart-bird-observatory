"""Tests for env-driven and YAML config loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from banshee.config import BansheeConfig, ConfigError, MqttConfig, load


REQUIRED_ENV = {
    "MQTT_HOST": "192.168.1.73",
    "MINIO_ENDPOINT": "http://192.168.1.65:9000",
    "MINIO_ACCESS_KEY": "ak",
    "MINIO_SECRET_KEY": "sk",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    # Scrub any env vars this suite cares about so the test env is sterile.
    for name in (
        "MQTT_HOST",
        "MQTT_PORT",
        "MQTT_USERNAME",
        "MQTT_PASSWORD",
        "MQTT_TOPIC_PREFIX",
        "MQTT_STATION_FILTER",
        "MQTT_CLIENT_ID",
        "THOTH_DB_PATH",
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_BUCKET",
        "MINIO_SECURE",
        "INFLUX_URL",
        "INFLUX_TOKEN",
        "INFLUX_ORG",
        "INFLUX_BUCKET",
        "TELEGRAM_MIN_CONFIDENCE",
        "HA_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_from_env_minimal(clean_env: pytest.MonkeyPatch) -> None:
    for k, v in REQUIRED_ENV.items():
        clean_env.setenv(k, v)

    cfg = BansheeConfig.from_env()

    assert cfg.mqtt.host == "192.168.1.73"
    assert cfg.mqtt.port == 1883
    assert cfg.mqtt.topic_prefix == "sbo"
    assert cfg.mqtt.station_filter == "+"

    assert cfg.storage.db_path == Path("/var/lib/thoth/events.db")
    assert cfg.storage.minio.endpoint == "http://192.168.1.65:9000"
    assert cfg.storage.minio.bucket == "thoth"
    assert cfg.storage.minio.secure is False

    assert cfg.influx.url == "http://192.168.1.24:8086"
    assert cfg.influx.org == "herzman"
    assert cfg.influx.bucket == "sbo"


def test_from_env_missing_required_raises(clean_env: pytest.MonkeyPatch) -> None:
    # MQTT_HOST is deliberately unset.
    clean_env.setenv("MINIO_ENDPOINT", "http://mini:9000")
    clean_env.setenv("MINIO_ACCESS_KEY", "ak")
    clean_env.setenv("MINIO_SECRET_KEY", "sk")

    with pytest.raises(ConfigError, match="MQTT_HOST"):
        BansheeConfig.from_env()


def test_from_env_overrides(clean_env: pytest.MonkeyPatch) -> None:
    for k, v in REQUIRED_ENV.items():
        clean_env.setenv(k, v)
    clean_env.setenv("MQTT_PORT", "8883")
    clean_env.setenv("MQTT_STATION_FILTER", "horus")
    clean_env.setenv("MINIO_SECURE", "TRUE")
    clean_env.setenv("THOTH_DB_PATH", "/tmp/test.db")
    clean_env.setenv("INFLUX_URL", "http://influx:8086")
    clean_env.setenv("HA_ENABLED", "false")

    cfg = BansheeConfig.from_env()

    assert cfg.mqtt.port == 8883
    assert cfg.mqtt.station_filter == "horus"
    assert cfg.storage.minio.secure is True
    assert cfg.storage.db_path == Path("/tmp/test.db")
    assert cfg.influx.url == "http://influx:8086"
    assert cfg.notify.ha_enabled is False


def test_mqtt_empty_username_treated_as_none(clean_env: pytest.MonkeyPatch) -> None:
    for k, v in REQUIRED_ENV.items():
        clean_env.setenv(k, v)
    clean_env.setenv("MQTT_USERNAME", "")

    cfg = MqttConfig.from_env()
    assert cfg.username is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("off", False),
        ("0", False),
        ("", False),
    ],
)
def test_minio_secure_truthy_variants(
    clean_env: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    for k, v in REQUIRED_ENV.items():
        clean_env.setenv(k, v)
    clean_env.setenv("MINIO_SECURE", raw)

    cfg = BansheeConfig.from_env()
    assert cfg.storage.minio.secure is expected


def test_minio_secure_invalid_raises(clean_env: pytest.MonkeyPatch) -> None:
    for k, v in REQUIRED_ENV.items():
        clean_env.setenv(k, v)
    clean_env.setenv("MINIO_SECURE", "maybe")

    with pytest.raises(ConfigError, match="MINIO_SECURE"):
        BansheeConfig.from_env()


def test_yaml_loader_roundtrip(tmp_path: Path) -> None:
    yaml_text = """
mqtt:
  host: 192.168.1.73
  port: 1883
  topic_prefix: sbo
  station_filter: "+"

storage:
  db_path: /var/lib/thoth/events.db
  minio:
    endpoint: http://192.168.1.65:9000
    access_key: ak
    secret_key: sk
    bucket: thoth
    secure: false

influx:
  url: http://192.168.1.24:8086
  token: ""
  org: herzman
  bucket: sbo

notify:
  telegram_min_confidence: 0.8
  ha_enabled: true
"""
    path = tmp_path / "banshee.yaml"
    path.write_text(yaml_text)

    cfg = load(path)

    assert cfg.mqtt.host == "192.168.1.73"
    assert cfg.storage.db_path == Path("/var/lib/thoth/events.db")
    assert cfg.storage.minio.access_key == "ak"
    assert cfg.notify.telegram_min_confidence == 0.8
