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
from .scoring import hero_score as compute_hero_score

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


# Schema additions that landed after the initial table definition.
# SQLite doesn't support ``ADD COLUMN IF NOT EXISTS``, so we probe the
# existing column set in :func:`_migrate` and issue plain ``ALTER TABLE
# ... ADD COLUMN`` statements only when the column is missing. Idempotent
# by construction — running on a fresh DB and on a live-deployed DB both
# converge to the same schema.
#
# Added for PR-B (burst grouping + hero selection):
#   burst_id    — shared session identifier from horus (``{station}-{ms}-{rand}``).
#                 NULL for singleton frames and legacy pre-burst traffic.
#   burst_seq   — 1-based monotonic index within the burst. NULL when burst_id is.
#   sharpness   — Laplacian variance from :func:`scoring.laplacian_variance`,
#                 computed once at ingest. Persisted so the classifier
#                 recompute path (which doesn't see raw bytes) can rebuild
#                 ``hero_score`` without re-decoding the image.
#   hero_score  — composite rank from :func:`scoring.hero_score`. The
#                 ``?group=burst`` API uses ``MAX(hero_score)`` per burst
#                 to pick the canonical frame.
_COLUMN_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("burst_id", "ALTER TABLE events ADD COLUMN burst_id TEXT"),
    ("burst_seq", "ALTER TABLE events ADD COLUMN burst_seq INTEGER"),
    ("sharpness", "ALTER TABLE events ADD COLUMN sharpness REAL"),
    ("hero_score", "ALTER TABLE events ADD COLUMN hero_score REAL"),
)


# Indexes that depend on the migrated columns. Split from ``SCHEMA``
# because they can't be part of the initial ``CREATE TABLE`` block —
# SQLite rejects indexes referencing columns that were just added in the
# same executescript on some versions. Kept idempotent via ``IF NOT EXISTS``.
_POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_events_burst
  ON events(burst_id, burst_seq);

-- Hero-picking index: for a given burst, the row with the highest
-- hero_score is the canonical frame. DESC ordering on the scored column
-- means the ``MAX(hero_score)`` sub-query in the API traverses the
-- index forwards from the leading edge.
CREATE INDEX IF NOT EXISTS idx_events_burst_hero
  ON events(burst_id, hero_score DESC);
"""


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return the set of column names currently on ``table``.

    Wraps the ``PRAGMA table_info`` round-trip so the migration logic
    reads cleanly. Returns an empty set for a missing table — the
    caller then creates it via ``SCHEMA`` before invoking the column
    migration.
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply pending column additions. Idempotent.

    Called from :meth:`EventStore.init` after the base ``SCHEMA`` has
    been executed. Reads the current column set once, then issues an
    ``ALTER TABLE ADD COLUMN`` for each entry in
    :data:`_COLUMN_MIGRATIONS` that isn't already present. Post-migration
    indexes are created last so they can reference the new columns.
    """
    existing = _existing_columns(conn, "events")
    for column, ddl in _COLUMN_MIGRATIONS:
        if column not in existing:
            conn.execute(ddl)
    conn.executescript(_POST_MIGRATION_INDEXES)


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


def _bird_score_from_payload(payload_json: str | None) -> float | None:
    """Extract ``bird_score`` from a stored payload, or None.

    Used by :meth:`EventStore.record_classification` to rebuild the
    composite hero score without asking the caller to re-read the image
    or pass the detector output a second time. Malformed payloads fall
    through as ``None`` — the composite treats that as "no detector
    signal", same as a payload that never had the field.
    """
    if not payload_json:
        return None
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        return None
    raw = payload.get("bird_score") if isinstance(payload, dict) else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _recompute_hero_score(
    *,
    payload_json: str | None,
    sharpness: float | None,
    classifier_confidence: float,
) -> float:
    """Rebuild ``hero_score`` from a stored row after classification.

    ``record_image`` fires the composite with classifier=0 because the
    worker hasn't run yet. Once it has, this helper reassembles the
    inputs that still live on the row (``bird_score`` in payload_json,
    ``bbox_fraction`` in payload_json, ``sharpness`` in its own column)
    and folds in the final confidence. Keeping this in one place
    prevents record_classification from duplicating the composite
    math.
    """
    bbox = _bbox_from_payload(payload_json)
    bird = _bird_score_from_payload(payload_json)
    return compute_hero_score(
        bird_score=bird,
        sharpness=sharpness,
        bbox_fraction=bbox,
        classifier_confidence=classifier_confidence,
    )


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
        """Create parent dir + tables + run pending column migrations.

        Idempotent — safe on every start. Order:

        1. ``SCHEMA`` creates the base table and the original indexes
           (all ``IF NOT EXISTS``).
        2. :func:`_migrate` adds any columns and indexes introduced
           after the initial schema. Each step probes the current
           column set, so running against an up-to-date DB is a no-op.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            _migrate(conn)

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
        *,
        sharpness: float | None = None,
    ) -> str:
        """Insert an image event row. Returns the event id.

        Parameters
        ----------
        event:
            The decoded :class:`ImageEvent`. ``event.burst_id`` and
            ``event.burst_seq`` (when set by horus) are persisted into
            the dedicated columns so grouped queries can index them
            directly without touching ``payload_json``.
        media_key:
            MinIO object key where the full-frame blob was stored.
        event_id:
            Pre-assigned UUID. Callers that need the id before inserting
            (e.g. the ingest pipeline, which builds the MinIO key from
            it) pass their own; otherwise a fresh UUIDv4 is generated.
        sharpness:
            Laplacian variance as returned by
            :func:`scoring.laplacian_variance`. When provided (and when
            the other scoring inputs are available), ``hero_score`` is
            computed from ``bird_score + sharpness + bbox_area`` at
            insert time. The classifier term is zero here — the
            composite is recomputed on
            :meth:`record_classification` once the species confidence
            lands.  ``None`` leaves both ``sharpness`` and ``hero_score``
            NULL, which is legitimate for an ingest path that can't
            score (e.g. a future non-image event).
        """
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

        # Hero score is computed eagerly at ingest from the inputs we
        # have right now. The classifier term is 0.0 because the worker
        # hasn't run yet; record_classification will recompute the
        # composite with the final confidence once it does.
        hero = compute_hero_score(
            bird_score=event.bird_score,
            sharpness=sharpness,
            bbox_fraction=event.bbox_fraction,
            classifier_confidence=None,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (
                  id, station, event_type, captured_at, received_at,
                  schema_version, payload_json, media_key,
                  burst_id, burst_seq, sharpness, hero_score
                )
                VALUES (?, ?, 'image', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event.station,
                    event.captured_at.isoformat(),
                    sbo_now_iso(),
                    event.schema_version,
                    json.dumps(payload, separators=(",", ":")),
                    media_key,
                    event.burst_id,
                    event.burst_seq,
                    float(sharpness) if sharpness is not None else None,
                    float(hero),
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
        conf = float(confidence)
        with self._connect() as conn:
            # Pull the inputs we need to recompute hero_score on the same
            # connection so this stays race-free with a concurrent
            # re-classify. payload_json carries bird_score + bbox_fraction;
            # sharpness sits in its own column as of PR-B.
            row = conn.execute(
                "SELECT payload_json, sharpness FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise LookupError(
                    f"record_classification: no event row with id={event_id!r}"
                )

            hero = _recompute_hero_score(
                payload_json=row["payload_json"],
                sharpness=row["sharpness"],
                classifier_confidence=conf,
            )

            cursor = conn.execute(
                """
                UPDATE events
                SET species = ?,
                    confidence = ?,
                    classified_at = ?,
                    hero_score = ?
                WHERE id = ?
                """,
                (species, conf, sbo_now_iso(), hero, event_id),
            )
            # A zero rowcount means the row vanished between the SELECT
            # and the UPDATE — unlikely but possible if a separate
            # admin path is pruning events. Treat as the same
            # ``LookupError`` for caller symmetry.
            if cursor.rowcount == 0:
                raise LookupError(
                    f"record_classification: no event row with id={event_id!r}"
                )
