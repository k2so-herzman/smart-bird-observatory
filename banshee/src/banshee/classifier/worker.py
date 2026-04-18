"""Classifier worker loop.

Polls the SQLite event store for image events that haven't been
classified yet, fetches each image from MinIO, runs the configured
:class:`~.model.Classifier`, and writes the result back to the row.

Designed as a plain ``while not stopped`` loop rather than an MQTT
subscriber for Phase 1 — see the package docstring for the reasoning.
The loop is self-healing on crashes: an unhandled exception is logged
and the next tick just tries again, so a corrupt image or transient
MinIO hiccup can't hang the service.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ..eventstore import EventStore, PendingClassification
from ..minio_store import MinioStore
from .model import Classifier

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    """Tuning knobs for :class:`ClassifierWorker`.

    Attributes
    ----------
    poll_interval_seconds:
        Sleep between polls when the queue is empty.  Short enough
        that classifications feel ~live, long enough that an idle
        service doesn't thrash the DB.  Default 2s.
    batch_size:
        Max rows fetched per tick.  Cap prevents a long-idle service
        from monopolising the event loop on restart if thousands of
        rows accumulated.  Default 8.
    """

    poll_interval_seconds: float = 2.0
    batch_size: int = 8


class ClassifierWorker:
    """Pull pending classifications from SQLite, run the model, write back.

    The worker owns no IO of its own — :class:`EventStore`,
    :class:`MinioStore`, and the :class:`Classifier` are all injected
    so tests can substitute fakes without monkey-patching.

    Lifecycle
    ---------
    1. **Construct** with the three collaborators + an optional
       :class:`WorkerConfig`.
    2. **Run** via :meth:`run_forever` — blocks in a poll loop until
       :meth:`stop` is called (typically from a SIGTERM handler).
    3. **Stop** any time; the loop exits after its current tick.

    Poll tick
    ---------
    Each tick:

    * Calls ``eventstore.fetch_pending_classification(limit=batch_size)``.
    * If empty, sleeps ``poll_interval_seconds``.
    * Otherwise iterates the batch, fetching bytes from MinIO and
      handing them to the classifier.  A per-row failure is logged
      and the row is skipped (``classified_at`` stays NULL so it's
      retried on the next restart-or-backfill).  Logging is at
      ``exception`` so the traceback makes it to the journal.
    """

    def __init__(
        self,
        eventstore: EventStore,
        minio: MinioStore,
        classifier: Classifier,
        cfg: WorkerConfig | None = None,
    ) -> None:
        self.eventstore = eventstore
        self.minio = minio
        self.classifier = classifier
        self.cfg = cfg or WorkerConfig()
        self._stop = threading.Event()

    def run_forever(self) -> None:
        """Block in the poll loop until :meth:`stop` is called."""
        log.info(
            "classifier worker starting: poll=%.1fs batch=%d model=%s",
            self.cfg.poll_interval_seconds,
            self.cfg.batch_size,
            type(self.classifier).__name__,
        )
        while not self._stop.is_set():
            processed = self.tick()
            if processed == 0:
                # Event-waited sleep so stop() interrupts immediately.
                self._stop.wait(self.cfg.poll_interval_seconds)
        log.info("classifier worker stopped")

    def stop(self, *_: object) -> None:
        """Signal the poll loop to exit. Safe from any thread or signal handler."""
        self._stop.set()

    def tick(self) -> int:
        """Process one batch; return the number of rows classified.

        Zero means "queue is empty, caller should sleep"; a positive
        return means the caller should loop immediately in case more
        work arrived while this batch was running.
        """
        try:
            pending = self.eventstore.fetch_pending_classification(
                limit=self.cfg.batch_size
            )
        except Exception:
            # A DB error here is serious but not fatal — log and back off.
            log.exception("fetch_pending_classification failed")
            return 0

        for row in pending:
            self._classify_one(row)
        return len(pending)

    def _classify_one(self, row: PendingClassification) -> None:
        """Fetch image bytes, run the model, persist the result.

        All failure modes are caught so one bad row never poisons
        the loop.  A row that fails to classify keeps ``classified_at
        IS NULL`` and will be retried on the next worker restart.
        """
        try:
            image_bytes = self._fetch_image(row.media_key)
        except Exception:
            log.exception(
                "failed to fetch image for event %s (media_key=%s)",
                row.event_id,
                row.media_key,
            )
            return

        try:
            result = self.classifier.classify(image_bytes)
        except Exception:
            log.exception("classifier.classify raised for event %s", row.event_id)
            return

        try:
            self.eventstore.record_classification(
                row.event_id,
                species=result.species,
                confidence=result.confidence,
            )
        except Exception:
            log.exception(
                "failed to persist classification for event %s (%s, %.3f)",
                row.event_id,
                result.species,
                result.confidence,
            )
            return

        log.info(
            "classified %s: %s (%.3f)",
            row.event_id,
            result.species,
            result.confidence,
        )

    def _fetch_image(self, media_key: str) -> bytes:
        """Read the full blob from MinIO into memory.

        Bird crops are small (a few hundred KB), so a full read is
        fine.  Streaming would add complexity the classifier can't
        use — TFLite wants a fully-decoded array in memory anyway.
        """
        stream, _length = self.minio.get_object_stream(media_key)
        return b"".join(stream)
