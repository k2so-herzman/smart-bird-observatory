"""Banshee configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int = 1883
    username: str | None = None
    password: str | None = None
    topic_prefix: str = "sbo"
    # Which stations to subscribe to. "+" = all.
    station_filter: str = "+"


@dataclass(frozen=True)
class StorageConfig:
    # Where to persist the JPEGs we receive over MQTT.
    image_dir: Path = Path("/var/lib/banshee/images")
    # Max retention days. Anything older gets pruned by a separate job.
    retention_days: int = 30


@dataclass(frozen=True)
class InfluxConfig:
    url: str = "http://192.168.1.87:8086"
    token: str = ""
    org: str = "herzman"
    bucket: str = "sbo"


@dataclass(frozen=True)
class NotifyConfig:
    # Minimum classifier confidence (0..1) before we post to Telegram.
    telegram_min_confidence: float = 0.75
    # Home Assistant REST hook (or MQTT topic) gets every detection.
    ha_enabled: bool = True


@dataclass(frozen=True)
class BansheeConfig:
    mqtt: MqttConfig
    storage: StorageConfig
    influx: InfluxConfig
    notify: NotifyConfig = field(default_factory=NotifyConfig)


def load(path: Path | str) -> BansheeConfig:
    data = yaml.safe_load(Path(path).read_text())

    mqtt = MqttConfig(**data["mqtt"])

    storage_data = dict(data.get("storage", {}))
    if "image_dir" in storage_data:
        storage_data["image_dir"] = Path(storage_data["image_dir"])
    storage = StorageConfig(**storage_data)

    influx = InfluxConfig(**data.get("influx", {}))
    notify = NotifyConfig(**data.get("notify", {}))

    return BansheeConfig(
        mqtt=mqtt,
        storage=storage,
        influx=influx,
        notify=notify,
    )
