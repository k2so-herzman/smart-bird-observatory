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
import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from .config import BansheeConfig
from .eventstore import EventStore
from .minio_store import MinioStore

log = logging.getLogger(__name__)


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


# ---- app factory -----------------------------------------------------------


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


def create_app(
    cfg: BansheeConfig | None = None,
    *,
    connection_factory=None,
    minio_store: MinioStore | None = None,
) -> FastAPI:
    """Build a FastAPI instance.

    Args:
        cfg: Runtime config. Defaults to ``BansheeConfig.from_env()``
            for production.
        connection_factory: Optional ``(db_path: str) -> sqlite3.Connection``
            factory. Tests use this to inject an in-memory DB.
        minio_store: Optional :class:`MinioStore`. Tests inject a fake
            so the suite never touches the network.
    """
    cfg = cfg or BansheeConfig.from_env()
    connect = connection_factory or _read_only_connection

    # Shared per-process connection; opened on first request.
    state: dict[str, Any] = {"db": None, "minio": minio_store}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:  # noqa: ARG001 — FastAPI signature
        state["db"] = connect(str(cfg.storage.db_path))
        if state["minio"] is None:
            state["minio"] = MinioStore(cfg.storage.minio)
        try:
            yield
        finally:
            if state["db"] is not None:
                state["db"].close()
                state["db"] = None

    app = FastAPI(
        title="Thoth API",
        version="0.1.0",
        description="Read API for Smart Bird Observatory events.",
        lifespan=lifespan,
    )

    # ---- routes ------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        conn: sqlite3.Connection = state["db"]
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
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
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

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, station, event_type, captured_at, received_at, "
            "schema_version, payload_json, media_key, thumb_key, "
            "species, confidence, classified_at "
            f"FROM events {where} "
            "ORDER BY captured_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        conn: sqlite3.Connection = state["db"]
        rows = conn.execute(sql, params).fetchall()
        return {
            "count": len(rows),
            "limit": limit,
            "offset": offset,
            "events": [_row_to_event(r) for r in rows],
        }

    @app.get("/events/{event_id}")
    def get_event(event_id: str) -> dict[str, Any]:
        conn: sqlite3.Connection = state["db"]
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
    def get_image(event_id: str) -> StreamingResponse:
        conn: sqlite3.Connection = state["db"]
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

        minio: MinioStore = state["minio"]
        try:
            body, length = minio.get_object_stream(row["media_key"])
        except Exception as exc:  # noqa: BLE001 — surface any fetch failure
            log.exception("failed to fetch media %s", row["media_key"])
            raise HTTPException(status_code=502, detail=f"media fetch failed: {exc}") from exc

        headers = {"Content-Length": str(length)} if length else {}
        return StreamingResponse(body, media_type=content_type, headers=headers)

    return app


# ---- uvicorn entrypoint -----------------------------------------------------


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

    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
