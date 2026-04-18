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

import contextlib
import logging
import threading

from ..eventstore import EventStore, PendingClassification
from ..minio_store import MinioStore
from .model import Classifier

log = logging.getLogger(__name__)


# Defaults mirror :class:`banshee.config.ClassifierConfig`. Kept as module
# constants (not a second dataclass) so there's exactly one place in the
# codebase to change them — see the note in
# :class:`banshee.config.ClassifierConfig` docstring.
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_BATCH_SIZE = 8


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
        *,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.eventstore = eventstore
        self.minio = minio
        self.classifier = classifier
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_size = batch_size
        self._stop = threading.Event()

    def run_forever(self) -> None:
        """Block in the poll loop until :meth:`stop` is called."""
        log.info(
            "classifier worker starting: poll=%.1fs batch=%d model=%s",
            self.poll_interval_seconds,
            self.batch_size,
            type(self.classifier).__name__,
        )
        while not self._stop.is_set():
            processed = self.tick()
            if processed == 0:
                # Event-waited sleep so stop() interrupts immediately.
                self._stop.wait(self.poll_interval_seconds)
        log.info("classifier worker stopped")

    def stop(self, *_: object) -> None:
        """Signal the poll loop to exit. Safe from any thread or signal handler."""
        self._stop.set()

    def tick(self) -> int:
        """Process one batch; return the number of rows **successfully** classified.

        Returning only successes (not ``len(pending)``) matters when the
        whole batch fails — e.g. MinIO is down. With ``len(pending)`` the
        outer loop would re-tick immediately with no sleep, hammering
        the DB + MinIO during an outage. With successes-only the loop
        falls into the ``poll_interval_seconds`` sleep on zero and backs
        off naturally.
        """
        try:
            pending = self.eventstore.fetch_pending_classification(
                limit=self.batch_size
            )
        except Exception:
            # A DB error here is serious but not fatal — log and back off.
            log.exception("fetch_pending_classification failed")
            return 0

        classified = 0
        for row in pending:
            if self._classify_one(row):
                classified += 1
        return classified

    def _classify_one(self, row: PendingClassification) -> bool:
        """Fetch image bytes, run the model, persist the result.

        Returns ``True`` on a full success (row written), ``False`` on
        any failure — fetch, classify, or persist. Callers use the
        return value to decide whether real progress was made (see
        :meth:`tick`'s note on hot-loop avoidance).

        All failure modes are caught so one bad row never poisons the
        loop. A row that fails to classify keeps ``classified_at IS
        NULL`` and will be retried on the next tick or worker restart.
        """
        try:
            image_bytes = self._fetch_image(row.media_key)
        except Exception:
            log.exception(
                "failed to fetch image for event %s (media_key=%s)",
                row.event_id,
                row.media_key,
            )
            return False

        try:
            result = self.classifier.classify(image_bytes)
        except Exception:
            log.exception("classifier.classify raised for event %s", row.event_id)
            return False

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
            return False

        log.info(
            "classified %s: %s (%.3f)",
            row.event_id,
            result.species,
            result.confidence,
        )
        return True

    def _fetch_image(self, media_key: str) -> bytes:
        """Read the full blob from MinIO into memory.

        Bird crops are small (a few hundred KB), so a full read is
        fine.  Streaming would add complexity the classifier can't
        use — TFLite wants a fully-decoded array in memory anyway.

        Wrapped in :func:`contextlib.closing` so the underlying HTTP
        response is released even if ``b"".join()`` raises mid-stream
        (truncated read, malformed chunk, etc.). Without this, the
        ``finally: response.close()`` inside MinIO's generator only
        fires on exhaustion or GC — not a guarantee we can rely on.
        """
        stream, _length = self.minio.get_object_stream(media_key)
        with contextlib.closing(stream):
            return b"".join(stream)
