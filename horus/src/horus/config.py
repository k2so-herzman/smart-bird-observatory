"""Station configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int = 1883
    username: str | None = None
    password: str | None = None
    topic_prefix: str = "sbo"


@dataclass(frozen=True)
class CaptureConfig:
    width: int = 2304
    height: int = 1296
    jpeg_quality: int = 90
    # Seconds between capture attempts (motion gate runs at this cadence)
    interval_s: float = 1.0
    # Sensor tuning
    rpicam_extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class MotionConfig:
    # Fraction of pixels that must differ above `pixel_threshold`
    # for a frame to register as motion.
    pixel_threshold: int = 25
    frame_fraction: float = 0.02
    # Cooldown between published events, seconds.
    cooldown_s: float = 5.0


@dataclass(frozen=True)
class StorageConfig:
    # Local ring-buffer directory. Old files get pruned.
    local_dir: Path = Path("/var/lib/horus/captures")
    # Max local disk MB before pruning.
    max_local_mb: int = 512
    # Shared NFS mount where Banshee reads from. If None, MQTT carries a
    # reference to the local_dir only.
    nfs_dir: Path | None = None


@dataclass(frozen=True)
class HorusConfig:
    station: str
    camera: str
    mqtt: MqttConfig
    capture: CaptureConfig
    motion: MotionConfig
    storage: StorageConfig
    heartbeat_interval_s: int = 60


def load(path: Path | str) -> HorusConfig:
    """Load a station YAML config into a HorusConfig."""
    data = yaml.safe_load(Path(path).read_text())

    mqtt = MqttConfig(**data["mqtt"])
    capture = CaptureConfig(
        **{
            k: (tuple(v) if k == "rpicam_extra_args" else v)
            for k, v in data.get("capture", {}).items()
        }
    )
    motion = MotionConfig(**data.get("motion", {}))
    storage_data = dict(data.get("storage", {}))
    if "local_dir" in storage_data:
        storage_data["local_dir"] = Path(storage_data["local_dir"])
    if "nfs_dir" in storage_data and storage_data["nfs_dir"]:
        storage_data["nfs_dir"] = Path(storage_data["nfs_dir"])
    storage = StorageConfig(**storage_data)

    return HorusConfig(
        station=data["station"],
        camera=data["camera"],
        mqtt=mqtt,
        capture=capture,
        motion=motion,
        storage=storage,
        heartbeat_interval_s=data.get("heartbeat_interval_s", 60),
    )
