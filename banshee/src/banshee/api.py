"""Thoth read API — FastAPI app backing the photo browser UI.

Exposes the SQLite event index and the MinIO image store as a small
REST surface. Runs as ``thoth-api.service`` on the Thoth LXC; Caddy
sits in front.

Endpoints
---------

* ``GET /health`` — liveness + DB reachability probe.
* ``GET /events`` — paginated list with station / species / since filters.
* ``GET /events/{event_id}`` — one event row.
* ``GET /images/{event_id}`` — streams the JPEG from MinIO.

The API is read-only. Writes go through ``thoth-ingest`` only.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from .config import BansheeConfig
from .minio_store import MinioStore

log = logging.getLogger(__name__)

# Default floor below which a *classified* event is hidden from the
# default ``/events`` listing. Calibrated to match the horus-side
# classifier floor so the Thoth dashboard shows the same pool of
# "plausible" events regardless of whether classification happened at
# the station or here. Override with ``THOTH_API_MIN_CONFIDENCE`` or
# by passing ``min_confidence`` to :func:`create_app`.
DEFAULT_API_MIN_CONFIDENCE = 0.10


# ---- connection helpers ----------------------------------------------------


def _read_only_connection(db_path: str) -> sqlite3.Connection:
    """Open a read-only SQLite connection using a file: URI.

    Writers are the ingest service; the API must never mutate the DB.
    WAL mode on the writer side lets us read concurrently without
    blocking the producer.
    """
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False,
        timeout=5.0,
    )
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite Row from the events table to a JSON-safe dict."""
    return {
        "id": row["id"],
        "station": row["station"],
        "event_type": row["event_type"],
        "captured_at": row["captured_at"],
        "received_at": row["received_at"],
        "schema_version": row["schema_version"],
        "payload": json.loads(row["payload_json"]),
        "media_key": row["media_key"],
        "thumb_key": row["thumb_key"],
        "species": row["species"],
        "confidence": row["confidence"],
        "classified_at": row["classified_at"],
    }


# ---- app factory -----------------------------------------------------------


def create_app(
    cfg: BansheeConfig | None = None,
    *,
    db_connection: sqlite3.Connection | None = None,
    minio_store: MinioStore | None = None,
    min_confidence: float = DEFAULT_API_MIN_CONFIDENCE,
) -> FastAPI:
    """Build a FastAPI instance.

    Args:
        cfg: Runtime config. Defaults to ``BansheeConfig.from_env()``.
        db_connection: Optional pre-built sqlite3 connection. When set,
            every request borrows this shared connection (and it is
            NOT closed by request teardown). Tests pass an in-memory
            connection here. When ``None`` (production), each request
            opens + closes its own read-only connection via
            :func:`_read_only_connection` — this is what keeps the API
            thread-safe under FastAPI's threadpool dispatch.
        minio_store: Optional :class:`MinioStore`. Tests inject a fake
            so the suite never touches the network.
        min_confidence: Default floor applied to the ``/events`` listing.
            Events with ``confidence < min_confidence`` are hidden;
            unclassified events (``confidence IS NULL``) are always
            returned so the pipeline's in-flight rows remain visible.
            Clients can override per-request via ``?min_confidence=``
            (pass ``0`` to disable). Defaults to
            :data:`DEFAULT_API_MIN_CONFIDENCE`.
    """
    cfg = cfg or BansheeConfig.from_env()

    # Captured once at app-build time. ``_minio`` is lazy-initialised on
    # first use so a missing MinIO at import time doesn't break health checks.
    _shared_db = db_connection
    _minio_holder: dict[str, MinioStore | None] = {"store": minio_store}
    _default_min_confidence = float(min_confidence)

    def get_db() -> Iterator[sqlite3.Connection]:
        """FastAPI dependency: yield a sqlite connection for one request.

        When ``db_connection`` was passed to :func:`create_app` (tests),
        we yield it without closing so request-to-request state is
        preserved. In production we open a fresh read-only connection
        per request, which is cheap under WAL and avoids shared-state
        threading hazards.
        """
        if _shared_db is not None:
            yield _shared_db
            return
        conn = _read_only_connection(str(cfg.storage.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def get_minio() -> MinioStore:
        """FastAPI dependency: lazily build the MinioStore once per process."""
        if _minio_holder["store"] is None:
            _minio_holder["store"] = MinioStore(cfg.storage.minio)
        assert _minio_holder["store"] is not None  # narrow for type checker
        return _minio_holder["store"]

    app = FastAPI(
        title="Thoth API",
        version="0.1.0",
        description="Read API for Smart Bird Observatory events.",
    )

    # ---- routes ------------------------------------------------------------

    @app.get("/health")
    def health(conn: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()
            return {"status": "ok", "db": "ok", "event_count": int(row["n"])}
        except sqlite3.Error as exc:
            raise HTTPException(status_code=503, detail=f"db error: {exc}") from exc

    @app.get("/events")
    def list_events(
        station: str | None = Query(None, description="Filter by station name."),
        species: str | None = Query(None, description="Filter by classified species."),
        since: str | None = Query(
            None,
            description="ISO-8601 timestamp; returns events captured at or after this.",
        ),
        min_confidence: float | None = Query(
            None,
            ge=0.0,
            le=1.0,
            description=(
                "Minimum classifier confidence [0.0, 1.0]. Events below the "
                "floor are hidden; unclassified events (NULL confidence) "
                "always pass through. Pass 0 to disable the floor. Omit to "
                "use the server default (THOTH_API_MIN_CONFIDENCE)."
            ),
        ),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        effective_floor = (
            _default_min_confidence if min_confidence is None else float(min_confidence)
        )

        clauses: list[str] = []
        params: list[Any] = []
        if station:
            clauses.append("station = ?")
            params.append(station)
        if species:
            clauses.append("species = ?")
            params.append(species)
        if since:
            clauses.append("captured_at >= ?")
            params.append(since)
        if effective_floor > 0.0:
            # NULL confidence = "classifier hasn't run yet"; keep those
            # so in-flight events remain visible in the dashboard. Only
            # hide rows the classifier actively judged low-confidence.
            clauses.append("(confidence IS NULL OR confidence >= ?)")
            params.append(effective_floor)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, station, event_type, captured_at, received_at, "
            "schema_version, payload_json, media_key, thumb_key, "
            "species, confidence, classified_at "
            f"FROM events {where} "
            "ORDER BY captured_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        return {
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "events": [_row_to_event(r) for r in rows],
        }

    @app.get("/events/{event_id}")
    def get_event(
        event_id: str,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        row = conn.execute(
            "SELECT id, station, event_type, captured_at, received_at, "
            "schema_version, payload_json, media_key, thumb_key, "
            "species, confidence, classified_at "
            "FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        return _row_to_event(row)

    @app.get("/images/{event_id}")
    def get_image(
        event_id: str,
        conn: sqlite3.Connection = Depends(get_db),
        minio: MinioStore = Depends(get_minio),
    ) -> StreamingResponse:
        row = conn.execute(
            "SELECT media_key, payload_json FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        if not row["media_key"]:
            raise HTTPException(status_code=404, detail="event has no media")

        payload = json.loads(row["payload_json"])
        content_type = payload.get("content_type", "image/jpeg")

        try:
            body, length = minio.get_object_stream(row["media_key"])
        except Exception as exc:  # noqa: BLE001 — surface any fetch failure
            log.exception("failed to fetch media %s", row["media_key"])
            raise HTTPException(
                status_code=502, detail=f"media fetch failed: {exc}"
            ) from exc

        headers = {"Content-Length": str(length)} if length else {}
        return StreamingResponse(body, media_type=content_type, headers=headers)

    return app


# ---- uvicorn entrypoint -----------------------------------------------------


def _min_confidence_from_env() -> float:
    """Parse ``THOTH_API_MIN_CONFIDENCE`` from the environment.

    Falls back to :data:`DEFAULT_API_MIN_CONFIDENCE` when unset or
    unparseable. We log (not raise) on bad input so a typo in
    ``/etc/thoth/env`` doesn't prevent the service from booting —
    operators are more likely to notice a dashboard full of noise than
    a fresh-config startup crash.
    """
    raw = os.environ.get("THOTH_API_MIN_CONFIDENCE")
    if raw is None or raw.strip() == "":
        return DEFAULT_API_MIN_CONFIDENCE
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "ignoring invalid THOTH_API_MIN_CONFIDENCE=%r; using default %s",
            raw,
            DEFAULT_API_MIN_CONFIDENCE,
        )
        return DEFAULT_API_MIN_CONFIDENCE
    if not 0.0 <= value <= 1.0:
        log.warning(
            "THOTH_API_MIN_CONFIDENCE=%s out of range [0, 1]; using default %s",
            value,
            DEFAULT_API_MIN_CONFIDENCE,
        )
        return DEFAULT_API_MIN_CONFIDENCE
    return value


def main() -> None:
    """Production entrypoint — called by ``thoth-api.service``.

    Builds the app from the process environment (``/etc/thoth/env`` in
    production) and hands it to uvicorn. Binding to 127.0.0.1 — Caddy
    is the only public face.
    """
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    min_confidence = _min_confidence_from_env()
    log.info("thoth-api listing floor: min_confidence=%.2f", min_confidence)

    uvicorn.run(
        create_app(min_confidence=min_confidence),
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
