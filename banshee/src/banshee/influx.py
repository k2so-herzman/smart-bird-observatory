"""InfluxDB writer for Thoth.

Responsible for emitting one measurement point per ingested event.
Images become ``sbo_image`` points tagged by station/camera/trigger.
Status heartbeats become ``sbo_status`` points; any numeric/bool
extras on the heartbeat payload become fields. Schema contract lives
in ``docs/thoth-design.md``; changes there should land in lockstep
with changes here.

Design notes
------------

* Classification fields (``species``, ``confidence``) are intentionally
  out of scope for this module. They land in a follow-up PR once the
  classifier is wired up.
* Writes are synchronous — we'd rather know immediately that a write
  failed than queue losses silently. The caller (``Pipeline``) is
  already tolerant of best-effort behavior, so synchronous is safe.
* Tests inject a fake client via the ``client_factory`` parameter, so
  the real InfluxDB is never reached. A missing ``token`` disables
  writes entirely — convenient for local runs without an Influx
  instance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from .config import InfluxConfig
from .events import ImageEvent, StatusEvent

log = logging.getLogger(__name__)


class _WriteApiLike(Protocol):
    """Subset of ``influxdb_client.WriteApi`` the writer uses.

    Exists so tests can hand in a fake with a single ``write`` method
    without having to construct a real ``WriteApi``.
    """

    def write(self, bucket: str, record: object) -> object: ...


class _InfluxClientLike(Protocol):
    """Subset of ``influxdb_client.InfluxDBClient`` the writer uses."""

    def write_api(self, write_options: object) -> _WriteApiLike: ...

    def close(self) -> None: ...


ClientFactory = Callable[[InfluxConfig], _InfluxClientLike]
"""Factory that builds an Influx client from config. Pluggable for tests."""


def _default_client_factory(cfg: InfluxConfig) -> InfluxDBClient:
    """Build the real ``InfluxDBClient`` from config.

    Pulled out as a named function so the construction path exists in
    exactly one place and tests can skip it with their own factory.
    """
    return InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org)


class InfluxWriter:
    """Emits measurement points for image + status events.

    Lifecycle::

        writer = InfluxWriter(cfg)
        writer.connect()           # opens the client
        writer.write_image_event(...)
        writer.write_status(...)
        writer.close()             # releases resources

    An empty ``cfg.token`` short-circuits ``connect()`` — writes then
    become no-ops. That is an intentional convenience for dev setups
    without an InfluxDB instance; production systemd always supplies
    a token.
    """

    def __init__(
        self,
        cfg: InfluxConfig,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.cfg = cfg
        self._client_factory = client_factory or _default_client_factory
        self._client: _InfluxClientLike | None = None
        self._write_api: _WriteApiLike | None = None

    def connect(self) -> None:
        """Open the Influx client and write API.

        Becomes a no-op when ``cfg.token`` is empty — logs a warning so
        operators notice if the dev fallback is active in production.
        """
        if not self.cfg.token:
            log.warning("InfluxDB token not set — writes will be skipped")
            return
        self._client = self._client_factory(self.cfg)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    def close(self) -> None:
        """Release the underlying client. Safe to call when never connected."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._write_api = None

    def write_image_event(
        self,
        event: ImageEvent,
        event_id: str,
        media_key: str,
    ) -> None:
        """Emit an ``sbo_image`` measurement for a captured frame.

        The ``media_key`` is stored as a field (not a tag) on purpose —
        high-cardinality values like object keys would explode Influx's
        series index if tagged.
        """
        if self._write_api is None:
            return
        point = (
            Point("sbo_image")
            .tag("station", event.station)
            .tag("camera", event.camera)
            .tag("trigger", event.trigger)
            .field("event_id", event_id)
            .field("size_bytes", event.size_bytes)
            .field("changed_fraction", event.changed_fraction)
            .field("width", event.resolution[0])
            .field("height", event.resolution[1])
            .field("media_key", media_key)
            .time(event.captured_at, WritePrecision.S)
        )
        self._write_api.write(bucket=self.cfg.bucket, record=point)

    def write_status(self, event: StatusEvent) -> None:
        """Emit an ``sbo_status`` point for a station heartbeat.

        Any numeric / boolean key present in ``event.raw`` (besides the
        meta fields ``schema_version``, ``station``, ``ts``) is turned
        into an Influx field. Non-scalar fields are dropped — Influx
        rejects arrays and nested objects on a point.
        """
        if self._write_api is None:
            return
        point = (
            Point("sbo_status")
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
