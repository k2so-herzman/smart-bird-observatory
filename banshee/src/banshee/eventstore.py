"""SQLite event store for Thoth.

The event store is the authoritative index of everything Thoth has
ingested. Media payloads live in MinIO; this table holds the metadata,
the MinIO keys, and the classifier results once they land.

Schema intentionally matches `docs/thoth-design.md` § "Storage model".
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sbo_shared import sbo_now_iso

from .events import ImageEvent

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingClassification:
    """A single image-event row awaiting classifier processing.

    Returned by :meth:`EventStore.fetch_pending_classification`. Holds
    only the fields the classifier needs — the rest of the row stays
    in the DB to keep the fetch cheap and the serialization narrow.

    ``bbox_fraction`` is the motion bbox in ``[0.0, 1.0]`` full-frame
    coords as published by horus. When present, the classifier worker
    crops the fetched full-frame image with the shared
    ``sbo_shared.imaging.crop_to_bbox_bytes`` helper before running
    inference — byte-identical to the crop horus already scored
    on-device. ``None`` on legacy events (pre-bbox schema) or when
    motion produced no bbox; the worker classifies the full frame as
    a fallback.
    """

    event_id: str
    station: str
    media_key: str
    captured_at: datetime
    bbox_fraction: tuple[float, float, float, float] | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id             TEXT PRIMARY KEY,
  station        TEXT NOT NULL,
  event_type     TEXT NOT NULL,
  captured_at    TEXT NOT NULL,
  received_at    TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  payload_json   TEXT NOT NULL,
  media_key      TEXT,
  thumb_key      TEXT,
  species        TEXT,
  confidence     REAL,
  classified_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_captured ON events(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_station ON events(station, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_species ON events(species, captured_at DESC);
"""


def _bbox_from_payload(
    payload_json: str | None,
) -> tuple[float, float, float, float] | None:
    """Decode ``bbox_fraction`` from the serialized event payload, or None.

    ``record_image`` stores the bbox under ``payload["bbox_fraction"]``
    when horus publishes one.  Older events pre-dating the bbox schema
    won't have it — and a malformed value (length != 4, non-numeric)
    is worth logging and treating as absent rather than crashing the
    classifier worker on a single bad row.
    """
    if not payload_json:
        return None
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        # Include the traceback + a truncated snippet of the offending
        # payload so field corruption is diagnosable from the journal
        # without having to reproduce the row.
        snippet = payload_json[:200] if isinstance(payload_json, str) else "<non-str>"
        log.warning(
            "payload_json decode failed; skipping bbox (first 200 chars: %r)",
            snippet,
            exc_info=True,
        )
        return None
    raw = payload.get("bbox_fraction") if isinstance(payload, dict) else None
    if raw is None:
        return None
    try:
        if len(raw) != 4:
            raise ValueError(f"expected 4 elements, got {len(raw)}")
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError) as exc:
        log.warning(
            "dropping malformed bbox_fraction from payload (raw=%r): %s",
            raw,
            exc,
        )
        return None


ConnectionFactory = Callable[[Path], sqlite3.Connection]
"""Callable that opens a SQLite connection for a given db path.

Pluggable so tests can swap in an in-memory connection without
touching the module-level ``sqlite3`` symbol.
"""


def _default_connect(db_path: Path) -> sqlite3.Connection:
    """Default connection factory used in production.

    Enables autocommit + WAL + cross-thread access. Tests can pass a
    ``lambda _path: sqlite3.connect(":memory:", check_same_thread=False)``
    and skip the file system entirely.
    """
    conn = sqlite3.connect(
        db_path,
        # isolation_level=None → autocommit mode. Every execute()
        # commits itself, so callers do NOT need conn.commit().
        isolation_level=None,
        timeout=30.0,
        # The ingest pipeline is single-threaded today (MQTT
        # callbacks run on the paho loop thread), but signal
        # handlers and the eventual classifier thread both need
        # to touch this connection on shutdown. Allowing
        # cross-thread use is safe here because every path goes
        # through this serialized connection with autocommit —
        # no shared transaction state to corrupt.
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    # WAL makes concurrent reads from the API service cheap.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class EventStore:
    """Thin wrapper over a SQLite connection.

    One instance per process. ``init()`` is idempotent — safe to call
    on every start. The connection is opened lazily on first write so
    construction is side-effect free (good for tests).

    Inject ``connection_factory`` to swap the real ``sqlite3.connect``
    call for a test double (e.g. an in-memory connection).
    """

    def __init__(
        self,
        db_path: Path,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        self.db_path = db_path
        self._connect_factory = connection_factory or _default_connect
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        """Create parent dir + tables. Idempotent."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield the shared connection, creating it (via the injected
        factory) if needed."""
        if self._conn is None:
            self._conn = self._connect_factory(self.db_path)
        yield self._conn

    def record_image(
        self,
        event: ImageEvent,
        media_key: str,
        event_id: str | None = None,
    ) -> str:
        """Insert an image event row. Returns the event id."""
        event_id = event_id or str(uuid.uuid4())
        payload = {
            "station": event.station,
            "camera": event.camera,
            "trigger": event.trigger,
            "resolution": list(event.resolution),
            "content_type": event.content_type,
            "size_bytes": event.size_bytes,
            "changed_fraction": event.changed_fraction,
            "sha256": event.sha256,
        }
        # Preserve diagnostic fields when present.  Absent on payloads
        # from older horus builds — don't insert a null key for those.
        if event.bbox_fraction is not None:
            payload["bbox_fraction"] = list(event.bbox_fraction)
        if event.af is not None:
            payload["af"] = event.af
        if event.bird_score is not None:
            payload["bird_score"] = float(event.bird_score)
        if event.bird_label is not None:
            payload["bird_label"] = event.bird_label
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (
                  id, station, event_type, captured_at, received_at,
                  schema_version, payload_json, media_key
                )
                VALUES (?, ?, 'image', ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event.station,
                    event.captured_at.isoformat(),
                    sbo_now_iso(),
                    event.schema_version,
                    json.dumps(payload, separators=(",", ":")),
                    media_key,
                ),
            )
        return event_id

    def fetch_pending_classification(
        self, limit: int = 8
    ) -> list[PendingClassification]:
        """Return image rows that still need classifying, oldest first.

        A row is "pending" when:

        * ``event_type = 'image'`` — audio/status events are classified
          on a different path (future work).
        * ``media_key IS NOT NULL`` — can't classify without bytes to
          fetch from MinIO.
        * ``classified_at IS NULL`` — worker sets this on every
          successful :meth:`record_classification` call so a row is
          never processed twice.

        Ordering is FIFO by ``captured_at`` so a backlog after a
        classifier outage is worked off in the order events arrived.

        Parameters
        ----------
        limit:
            Max rows to return in one call. Keep this modest (single
            digits) so the worker's "tick" stays bounded and shutdown
            latency stays low.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, station, media_key, captured_at, payload_json
                FROM events
                WHERE event_type = 'image'
                  AND media_key IS NOT NULL
                  AND classified_at IS NULL
                ORDER BY captured_at ASC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()
        return [
            PendingClassification(
                event_id=row["id"],
                station=row["station"],
                media_key=row["media_key"],
                captured_at=datetime.fromisoformat(row["captured_at"]),
                bbox_fraction=_bbox_from_payload(row["payload_json"]),
            )
            for row in rows
        ]

    def record_classification(
        self,
        event_id: str,
        *,
        species: str,
        confidence: float,
    ) -> None:
        """Persist classifier output onto an existing event row.

        Sets ``species``, ``confidence``, and ``classified_at`` (the
        latter to the current wall-clock time). Expected to run
        exactly once per event; calling a second time silently
        overwrites the earlier values, which is fine for re-classify
        workflows driven by a later model upgrade.

        Parameters
        ----------
        event_id:
            Primary key of the row in ``events``.
        species:
            Predicted label. Must be non-empty — NULL species means
            "not yet classified" in this schema, so we require the
            caller to supply a sentinel (e.g. ``"unclassified"``)
            rather than pass an empty string.
        confidence:
            Model confidence in ``[0.0, 1.0]``. Not clamped here;
            the worker trusts the model's output so that e.g. a
            downstream audit can spot a mis-calibrated model that
            returns 1.7.
        """
        if not species:
            raise ValueError("record_classification requires a non-empty species")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE events
                SET species = ?, confidence = ?, classified_at = ?
                WHERE id = ?
                """,
                (species, float(confidence), sbo_now_iso(), event_id),
            )
            # A zero rowcount means the caller passed an event_id that
            # isn't in the table. Silently succeeding here would mask a
            # caller bug (e.g. the worker handing us the wrong id), so
            # raise. Use LookupError — callers that legitimately race
            # with row deletion can catch it narrowly.
            if cursor.rowcount == 0:
                raise LookupError(
                    f"record_classification: no event row with id={event_id!r}"
                )
