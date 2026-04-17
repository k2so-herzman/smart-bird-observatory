"""Thoth ingest daemon entrypoint.

Wires the MQTT subscriber to MinIO (media), SQLite (event index), and
InfluxDB (metrics). Classification + notifications land in follow-up PRs.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import uuid
from pathlib import Path

from .config import BansheeConfig, load
from .eventstore import EventStore
from .events import ImageEvent, StatusEvent
from .influx import InfluxWriter
from .minio_store import MinioStore
from .subscriber import Subscriber

log = logging.getLogger("thoth.ingest")


class Pipeline:
    """MQTT → MinIO + SQLite + InfluxDB.

    Each image event gets a UUID up front so the same id threads through
    all three stores. MinIO write is first — if the blob upload fails we
    refuse to index the event, so the SQLite row never points at a key
    that doesn't exist.
    """

    def __init__(self, cfg: BansheeConfig) -> None:
        self.cfg = cfg
        self.eventstore = EventStore(cfg.storage.db_path)
        self.minio = MinioStore(cfg.storage.minio)
        self.influx = InfluxWriter(cfg.influx)
        self.subscriber = Subscriber(
            cfg,
            on_image=self._handle_image,
            on_status=self._handle_status,
        )

    def _handle_image(self, event: ImageEvent) -> None:
        event_id = str(uuid.uuid4())
        try:
            media_key = self.minio.put_image(event, event_id)
        except Exception:
            # MinIO is the source of truth for media; if it fails we drop
            # the event rather than index a dangling row.
            log.exception("MinIO upload failed for %s; dropping event", event_id)
            return

        self.eventstore.record_image(event, media_key, event_id=event_id)
        self.influx.write_image_event(event, event_id, media_key)

        log.info(
            "image from %s: %dx%d, %d bytes, frac=%.3f → %s",
            event.station,
            event.resolution[0],
            event.resolution[1],
            event.size_bytes,
            event.changed_fraction,
            media_key,
        )

    def _handle_status(self, event: StatusEvent) -> None:
        self.influx.write_status(event)
        log.debug("status from %s at %s", event.station, event.ts.isoformat())

    def run(self) -> int:
        self.eventstore.init()
        self.minio.ensure_bucket()
        self.influx.connect()
        try:
            self.subscriber.run_forever()
        finally:
            self.influx.close()
            self.eventstore.close()
        return 0

    def stop(self, *_: object) -> None:
        self.subscriber.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thoth SBO ingest service")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path. When unset, configuration is read "
        "from the process environment (systemd EnvironmentFile=/etc/thoth/env).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load(args.config) if args.config else BansheeConfig.from_env()
    pipeline = Pipeline(cfg)

    signal.signal(signal.SIGTERM, pipeline.stop)
    signal.signal(signal.SIGINT, pipeline.stop)

    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
