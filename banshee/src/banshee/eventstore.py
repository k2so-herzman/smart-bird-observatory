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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .events import ImageEvent

log = logging.getLogger(__name__)


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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class EventStore:
    """Thin wrapper over a SQLite connection.

    One instance per process. `init()` is idempotent — safe to call on
    every start. The connection is opened lazily on first write so
    construction is side-effect free (good for tests).
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        """Create parent dir + tables. Idempotent."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield the shared connection, creating it if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                isolation_level=None,  # autocommit off — we commit explicitly
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            # WAL makes concurrent reads from the API service cheap.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
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
                    _utcnow_iso(),
                    event.schema_version,
                    json.dumps(payload, separators=(",", ":")),
                    media_key,
                ),
            )
            conn.commit()
        return event_id
