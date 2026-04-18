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
from sbo_shared import MQTT_DEFAULT_PORT, MqttBaseConfig

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


_TRUTHY = frozenset({"true", "yes", "on", "1"})
_FALSY = frozenset({"false", "no", "off", "0", ""})


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var accepting common truthy/falsy forms."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    norm = raw.strip().lower()
    if norm in _TRUTHY:
        return True
    if norm in _FALSY:
        return False
    raise ConfigError(
        f"env var {name}={raw!r} is not a valid boolean "
        f"(expected one of: {sorted(_TRUTHY | _FALSY)})"
    )


@dataclass(frozen=True)
class MqttConfig(MqttBaseConfig):
    """Subscriber-side MQTT config.

    Extends :class:`sbo_shared.MqttBaseConfig` with the two fields only
    a subscriber needs:

    * ``station_filter`` — MQTT single-level wildcard pattern. ``"+"``
      means "every station"; a concrete name like ``"horus"`` narrows
      the subscription to one station.
    * ``client_id`` — paho ``client_id``. Useful to set explicitly when
      running multiple Banshee processes against the same broker.
    """

    # Which stations to subscribe to. "+" = all.
    station_filter: str = "+"
    client_id: str | None = None

    @classmethod
    def from_env(cls) -> "MqttConfig":
        """Build from process env (``/etc/thoth/env`` in production).

        Translates the shared base's :class:`ValueError` into banshee's
        :class:`ConfigError` so callers can catch a single, repo-local
        exception type regardless of which field triggered the error.
        """
        try:
            base = MqttBaseConfig.from_env()
        except ValueError as exc:
            # ConfigError subclasses ValueError — existing catchers
            # keep working; banshee-specific catchers still see
            # ConfigError.
            raise ConfigError(str(exc)) from exc
        return cls(
            host=base.host,
            port=base.port,
            username=base.username,
            password=base.password,
            topic_prefix=base.topic_prefix,
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
                secure=_bool_env("MINIO_SECURE", False),
            ),
        )


# Homelab InfluxDB endpoint. Lives here rather than inline in default
# args so a grep for the IP finds exactly one hit and operators have
# a named symbol to override in tests.
DEFAULT_INFLUX_URL = "http://192.168.1.24:8086"


@dataclass(frozen=True)
class InfluxConfig:
    """InfluxDB connection settings.

    An empty ``token`` is treated by :class:`banshee.influx.InfluxWriter`
    as "writes disabled" — a convenience for local dev where you may
    not want to run a real Influx instance. In production, systemd's
    EnvironmentFile supplies ``INFLUX_TOKEN``.
    """

    url: str = DEFAULT_INFLUX_URL
    token: str = ""
    org: str = "herzman"
    bucket: str = "sbo"

    @classmethod
    def from_env(cls) -> "InfluxConfig":
        return cls(
            url=_optional("INFLUX_URL", DEFAULT_INFLUX_URL),
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
            ha_enabled=_bool_env("HA_ENABLED", True),
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
