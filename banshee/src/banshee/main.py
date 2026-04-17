"""Banshee daemon entrypoint."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from . import storage
from .config import BansheeConfig, load
from .events import ImageEvent, StatusEvent
from .influx import InfluxWriter
from .subscriber import Subscriber

log = logging.getLogger("banshee")


class Pipeline:
    """Ties together the subscriber, storage, and Influx writer.

    Classification + notifications land in follow-up PRs. For now the
    pipeline persists the image, writes a bare InfluxDB point, and logs
    what it saw.
    """

    def __init__(self, cfg: BansheeConfig) -> None:
        self.cfg = cfg
        self.influx = InfluxWriter(cfg.influx)
        self.subscriber = Subscriber(
            cfg,
            on_image=self._handle_image,
            on_status=self._handle_status,
        )

    def _handle_image(self, event: ImageEvent) -> None:
        path = storage.save_image(event, self.cfg.storage)
        self.influx.write_image_event(event, str(path))
        log.info(
            "image from %s: %dx%d, %d bytes, frac=%.3f → %s",
            event.station,
            event.resolution[0],
            event.resolution[1],
            event.size_bytes,
            event.changed_fraction,
            path.name,
        )

    def _handle_status(self, event: StatusEvent) -> None:
        self.influx.write_status(event)
        log.debug("status from %s at %s", event.station, event.ts.isoformat())

    def run(self) -> int:
        self.influx.connect()
        try:
            self.subscriber.run_forever()
        finally:
            self.influx.close()
        return 0

    def stop(self, *_: object) -> None:
        self.subscriber.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Banshee SBO aggregator")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/etc/banshee/banshee.yaml"),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load(args.config)
    pipeline = Pipeline(cfg)

    signal.signal(signal.SIGTERM, pipeline.stop)
    signal.signal(signal.SIGINT, pipeline.stop)

    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
