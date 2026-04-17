"""Thoth / Banshee configuration.

Two loaders live here:

- `from_env()` — reads `/etc/thoth/env` style environment variables,
  the production path. systemd's `EnvironmentFile=/etc/thoth/env`
  populates these before the service starts.
- `load(path)` — reads a YAML file. Kept for local dev and tests.

Both produce the same `BansheeConfig`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .minio_store import MinioConfig


class ConfigError(ValueError):
    pass


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise ConfigError(f"required env var {name} is unset")
    return val


def _optional(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int = 1883
    username: str | None = None
    password: str | None = None
    topic_prefix: str = "sbo"
    # Which stations to subscribe to. "+" = all.
    station_filter: str = "+"
    client_id: str | None = None

    @classmethod
    def from_env(cls) -> "MqttConfig":
        return cls(
            host=_require("MQTT_HOST"),
            port=int(_optional("MQTT_PORT", "1883")),
            username=os.environ.get("MQTT_USERNAME") or None,
            password=os.environ.get("MQTT_PASSWORD") or None,
            topic_prefix=_optional("MQTT_TOPIC_PREFIX", "sbo"),
            station_filter=_optional("MQTT_STATION_FILTER", "+"),
            client_id=os.environ.get("MQTT_CLIENT_ID") or None,
        )


@dataclass(frozen=True)
class ThothStorageConfig:
    """SQLite + MinIO storage settings for the Thoth ingest service."""

    db_path: Path
    minio: MinioConfig

    @classmethod
    def from_env(cls) -> "ThothStorageConfig":
        return cls(
            db_path=Path(_optional("THOTH_DB_PATH", "/var/lib/thoth/events.db")),
            minio=MinioConfig(
                endpoint=_require("MINIO_ENDPOINT"),
                access_key=_require("MINIO_ACCESS_KEY"),
                secret_key=_require("MINIO_SECRET_KEY"),
                bucket=_optional("MINIO_BUCKET", "thoth"),
                secure=_optional("MINIO_SECURE", "false").lower() == "true",
            ),
        )


@dataclass(frozen=True)
class InfluxConfig:
    url: str = "http://192.168.1.24:8086"
    token: str = ""
    org: str = "herzman"
    bucket: str = "sbo"

    @classmethod
    def from_env(cls) -> "InfluxConfig":
        return cls(
            url=_optional("INFLUX_URL", "http://192.168.1.24:8086"),
            token=_optional("INFLUX_TOKEN", ""),
            org=_optional("INFLUX_ORG", "herzman"),
            bucket=_optional("INFLUX_BUCKET", "sbo"),
        )


@dataclass(frozen=True)
class NotifyConfig:
    telegram_min_confidence: float = 0.75
    ha_enabled: bool = True

    @classmethod
    def from_env(cls) -> "NotifyConfig":
        return cls(
            telegram_min_confidence=float(
                _optional("TELEGRAM_MIN_CONFIDENCE", "0.75")
            ),
            ha_enabled=_optional("HA_ENABLED", "true").lower() == "true",
        )


@dataclass(frozen=True)
class BansheeConfig:
    mqtt: MqttConfig
    storage: ThothStorageConfig
    influx: InfluxConfig
    notify: NotifyConfig = field(default_factory=NotifyConfig)

    @classmethod
    def from_env(cls) -> "BansheeConfig":
        return cls(
            mqtt=MqttConfig.from_env(),
            storage=ThothStorageConfig.from_env(),
            influx=InfluxConfig.from_env(),
            notify=NotifyConfig.from_env(),
        )


def load(path: Path | str) -> BansheeConfig:
    """Load from a YAML file (dev / test path)."""
    data = yaml.safe_load(Path(path).read_text())

    mqtt = MqttConfig(**data["mqtt"])

    storage_data = dict(data["storage"])
    storage_data["db_path"] = Path(storage_data["db_path"])
    minio_data = storage_data.pop("minio")
    storage = ThothStorageConfig(
        db_path=storage_data["db_path"],
        minio=MinioConfig(**minio_data),
    )

    influx = InfluxConfig(**data.get("influx", {}))
    notify = NotifyConfig(**data.get("notify", {}))

    return BansheeConfig(mqtt=mqtt, storage=storage, influx=influx, notify=notify)
