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
from pathlib import Path
from typing import Iterator

from sbo_shared import sbo_now_iso

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
