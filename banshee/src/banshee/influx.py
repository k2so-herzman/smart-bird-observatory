"""InfluxDB writer for Banshee.

Writes one point per event into the configured bucket. Classification
fields are added in a later PR.
"""

from __future__ import annotations

import logging

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from .config import InfluxConfig
from .events import ImageEvent, StatusEvent

log = logging.getLogger(__name__)


class InfluxWriter:
    def __init__(self, cfg: InfluxConfig) -> None:
        self.cfg = cfg
        self._client: InfluxDBClient | None = None
        self._write_api = None

    def connect(self) -> None:
        if not self.cfg.token:
            log.warning("InfluxDB token not set — writes will be skipped")
            return
        self._client = InfluxDBClient(
            url=self.cfg.url,
            token=self.cfg.token,
            org=self.cfg.org,
        )
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def write_image_event(self, event: ImageEvent, image_path: str) -> None:
        if self._write_api is None:
            return
        point = (
            Point("image_event")
            .tag("station", event.station)
            .tag("camera", event.camera)
            .tag("trigger", event.trigger)
            .field("size_bytes", event.size_bytes)
            .field("changed_fraction", event.changed_fraction)
            .field("width", event.resolution[0])
            .field("height", event.resolution[1])
            .field("image_path", image_path)
            .time(event.captured_at, WritePrecision.S)
        )
        self._write_api.write(bucket=self.cfg.bucket, record=point)

    def write_status(self, event: StatusEvent) -> None:
        if self._write_api is None:
            return
        point = (
            Point("status")
            .tag("station", event.station)
            .time(event.ts, WritePrecision.S)
        )
        # Pull any numeric / boolean fields out of raw
        for key, value in event.raw.items():
            if key in {"schema_version", "station", "ts"}:
                continue
            if isinstance(value, (int, float, bool)):
                point = point.field(key, value)
        self._write_api.write(bucket=self.cfg.bucket, record=point)
