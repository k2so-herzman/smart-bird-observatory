"""Thoth classifier daemon entrypoint.

Wires an :class:`~.worker.ClassifierWorker` to the real
:class:`~banshee.eventstore.EventStore`, :class:`~banshee.minio_store.MinioStore`,
and the configured :class:`~.model.Classifier` (TFLite if
``THOTH_MODEL_PATH`` is set, otherwise the dummy).

Deployment
----------
Runs as **systemd unit** ``thoth-classify.service`` on the Thoth LXC.
Shares ``/etc/thoth/env`` with ``thoth-ingest`` so MQTT / MinIO /
SQLite configuration is DRY across services.

Signals
-------
``SIGTERM`` and ``SIGINT`` call :meth:`ClassifierWorker.stop`, which
exits the poll loop after the current tick completes (so we never
leave a half-classified row).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from ..config import BansheeConfig, load
from ..eventstore import EventStore
from ..minio_store import MinioStore
from .model import Classifier, DummyClassifier, TFLiteClassifier
from .worker import ClassifierWorker

log = logging.getLogger("thoth.classify")


def _build_classifier(cfg: BansheeConfig) -> Classifier:
    """Pick the concrete classifier based on config.

    * ``classifier.model_path`` set and the file exists → TFLite.
    * Otherwise → :class:`DummyClassifier`, with a WARNING so it's
      obvious in the journal that no real model is running.
    """
    c = cfg.classifier
    if c.model_path and Path(c.model_path).is_file():
        if not c.labels_path or not Path(c.labels_path).is_file():
            log.error(
                "THOTH_MODEL_PATH set but labels file missing (%s); "
                "falling back to DummyClassifier",
                c.labels_path,
            )
            return DummyClassifier()
        return TFLiteClassifier(Path(c.model_path), Path(c.labels_path))

    log.warning(
        "no model configured (THOTH_MODEL_PATH unset or missing); "
        "using DummyClassifier — rows will be marked classified but "
        "species will be 'unclassified'"
    )
    return DummyClassifier()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the classifier service.

    Parses flags, loads config (YAML if ``--config`` supplied, env
    otherwise), constructs the worker, installs signal handlers,
    and blocks until shutdown.
    """
    parser = argparse.ArgumentParser(description="Thoth SBO classifier service")
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

    eventstore = EventStore(cfg.storage.db_path)
    eventstore.init()  # idempotent — ingest creates it first but this is safe
    minio = MinioStore(cfg.storage.minio)
    classifier = _build_classifier(cfg)

    worker = ClassifierWorker(
        eventstore=eventstore,
        minio=minio,
        classifier=classifier,
        poll_interval_seconds=cfg.classifier.poll_interval_seconds,
        batch_size=cfg.classifier.batch_size,
    )

    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)

    try:
        worker.run_forever()
    finally:
        eventstore.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
