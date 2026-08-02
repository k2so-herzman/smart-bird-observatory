"""Thoth ingest daemon entrypoint.

Wires the MQTT subscriber to the blob store (media — local filesystem
by default, MinIO if configured), SQLite (event index), and InfluxDB
(metrics).

Process model
-------------
One :class:`Pipeline` instance owns all three storage sinks and one
:class:`~banshee.subscriber.Subscriber`. ``pipeline.run()`` blocks in the
paho MQTT event loop until a SIGTERM or SIGINT arrives; the signal handler
calls ``pipeline.stop()``, which tells the subscriber to break out of its
loop, then ``run()`` tears down InfluxDB and SQLite connections in its
``finally`` block.

Deployment
----------
Runs as **systemd unit** ``thoth-ingest.service`` on the ingest host.
Environment variables are injected via
``EnvironmentFile=/etc/thoth/env`` (see :func:`~banshee.config.BansheeConfig.from_env`).
A YAML file can be supplied with ``--config`` for local dev runs.

Exit codes
----------
``0`` — clean shutdown (SIGTERM / SIGINT received and handled).
Non-zero — unhandled exception propagated from ``run()``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import uuid
from pathlib import Path

from .blobstore import BlobStore, build_store
from .config import BansheeConfig, load
from .eventstore import EventStore
from .events import ImageEvent, StatusEvent
from .influx import InfluxWriter
from .scoring import laplacian_variance
from .subscriber import Subscriber

log = logging.getLogger("thoth.ingest")


class Pipeline:
    """MQTT → blob store + SQLite + InfluxDB.

    Each image event gets a UUID up front so the same id threads through
    all three stores. The blob write is first — if it fails we refuse to
    index the event, so the SQLite row never points at a key that
    doesn't exist.
    """

    def __init__(
        self,
        cfg: BansheeConfig,
        eventstore: EventStore | None = None,
        store: BlobStore | None = None,
        influx: InfluxWriter | None = None,
        subscriber: Subscriber | None = None,
    ) -> None:
        """Wire the pipeline.

        Every storage sink is injectable so tests can pass fakes
        without monkey-patching module-level symbols. Production
        callers pass ``cfg`` and let the defaults build real clients.
        """
        self.cfg = cfg
        self.eventstore = eventstore if eventstore is not None else EventStore(cfg.storage.db_path)
        self.store = store if store is not None else build_store(cfg.storage)
        self.influx = influx if influx is not None else InfluxWriter(cfg.influx)
        self.subscriber = subscriber if subscriber is not None else Subscriber(
            cfg,
            on_image=self._handle_image,
            on_status=self._handle_status,
        )

    def _handle_image(self, event: ImageEvent) -> None:
        """Persist a single image event to all three sinks.

        Called from the paho MQTT callback thread whenever the subscriber
        decodes an ``ImageEvent``.  The write order is intentional:

        1. **Blob store** — media write first.  If this fails, the event is
           dropped entirely so no SQLite row ever references a missing key.
        2. **SQLite** — authoritative index row.  If this fails, the blob
           is removed to avoid orphaned storage, then the event is
           dropped.
        3. **InfluxDB** — best-effort metrics write.  Failure is logged but
           does not drop the event; metrics are recoverable from SQLite.

        Args:
            event: Decoded image event from the MQTT payload.
        """
        event_id = str(uuid.uuid4())
        try:
            media_key = self.store.put_image(event, event_id)
        except Exception:
            # The blob store is the source of truth for media; if it fails
            # we drop the event rather than index a dangling row.
            log.exception("blob write failed for %s; dropping event", event_id)
            return

        # Compute sharpness *before* the SQLite insert so it can be
        # persisted alongside the row. ``laplacian_variance`` swallows
        # its own decode/filter errors and returns 0.0 on failure, so
        # this call never raises — worst case we log a warning inside
        # the helper and the frame just ranks poorly on the hero axis.
        sharpness = laplacian_variance(event.image_bytes)

        # SQLite is the authoritative index. If the insert fails the
        # blob is an orphan — no row points at it, so no reader will
        # ever find it. Clean it up so we don't leak storage.
        try:
            self.eventstore.record_image(
                event,
                media_key,
                event_id=event_id,
                sharpness=sharpness,
            )
        except Exception:
            log.exception(
                "eventstore insert failed for %s; removing orphan blob %s",
                event_id,
                media_key,
            )
            self.store.remove_object(media_key)
            return

        # Influx is best-effort — metrics are derivable from SQLite +
        # the blob store, so a transient write failure shouldn't drop the event.
        # Log loudly so we notice if it's persistent.
        try:
            self.influx.write_image_event(event, event_id, media_key)
        except Exception:
            log.exception(
                "Influx write failed for %s; event %s is indexed but not metered",
                event_id,
                media_key,
            )

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
        """Forward a station status event to InfluxDB.

        Called from the paho MQTT callback thread whenever the subscriber
        decodes a ``StatusEvent``.  InfluxDB is the only sink for status
        events; failure is logged but does not raise so the MQTT loop
        continues.

        Args:
            event: Decoded status event from the MQTT payload.
        """
        try:
            self.influx.write_status(event)
        except Exception:
            log.exception("Influx status write failed for %s", event.station)
        log.debug("status from %s at %s", event.station, event.ts.isoformat())

    def run(self) -> int:
        """Start the ingest pipeline and block until shutdown.

        Initialises all storage sinks in dependency order (SQLite schema,
        blob-store root/bucket, InfluxDB connection), then hands control to the MQTT
        subscriber's blocking event loop.  On exit — whether clean or via
        an exception — InfluxDB and SQLite connections are closed in the
        ``finally`` block.

        Returns:
            ``0`` on clean shutdown.  Any exception from the subscriber
            loop propagates to the caller unchanged.

        Side effects:
            Creates the SQLite database file and WAL journal if they do not
            exist.  Creates the local storage root (or MinIO bucket) if
            absent.  Opens a persistent InfluxDB write client.
        """
        self.eventstore.init()
        self.store.ensure_ready()
        self.influx.connect()
        try:
            self.subscriber.run_forever()
        finally:
            self.influx.close()
            self.eventstore.close()
        return 0

    def stop(self, *_: object) -> None:
        """Signal the pipeline to shut down gracefully.

        Registered as the handler for ``SIGTERM`` and ``SIGINT`` by
        :func:`main`.  Calls :meth:`~banshee.subscriber.Subscriber.stop`
        which sets a flag that causes the paho MQTT loop to return on its
        next iteration, allowing :meth:`run` to proceed to its ``finally``
        block.

        Args:
            *_: Accepts the signal number and frame arguments passed by
                :mod:`signal` but ignores them so the method can also be
                called directly in tests.
        """
        self.subscriber.stop()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Thoth ingest service.

    Parses arguments, configures logging, builds a :class:`Pipeline` from
    the resolved :class:`~banshee.config.BansheeConfig`, installs signal
    handlers, then delegates to :meth:`Pipeline.run`.

    Args:
        argv: Argument list to parse.  ``None`` falls through to
            ``sys.argv[1:]`` (standard :mod:`argparse` behaviour).  Pass an
            explicit list in tests to avoid touching the real ``sys.argv``.

    Returns:
        ``0`` on clean shutdown; non-zero if :meth:`Pipeline.run` raises.

    CLI flags:
        ``--config PATH``
            Optional YAML config file.  When omitted, configuration is read
            from the process environment via
            :meth:`~banshee.config.BansheeConfig.from_env`.

        ``--log-level LEVEL``
            Standard :mod:`logging` level name (default ``INFO``).

    Environment variables (when ``--config`` is not supplied):
        Consumed by :meth:`~banshee.config.BansheeConfig.from_env` — see
        ``banshee/config.py`` for the full list (``MQTT_*``, ``MINIO_*``,
        ``INFLUX_*``, ``DB_PATH``, etc.).

    Signals:
        ``SIGTERM`` and ``SIGINT`` are wired to :meth:`Pipeline.stop` so
        that systemd and Ctrl-C both trigger a clean shutdown.
    """
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
