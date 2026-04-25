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
    """Convert a sqlite Row from the events table to a JSON-safe dict.

    Burst metadata (``burst_id``, ``burst_seq``, ``hero_score``) and
    ``sharpness`` are included when the columns exist — they were added
    in PR-B and are NULL on pre-migration rows, which serialises as
    JSON ``null``.
    """
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
        "burst_id": _row_get(row, "burst_id"),
        "burst_seq": _row_get(row, "burst_seq"),
        "sharpness": _row_get(row, "sharpness"),
        "hero_score": _row_get(row, "hero_score"),
    }


def _row_get(row: sqlite3.Row, key: str) -> Any:
    """Return ``row[key]`` or ``None`` when the column is absent.

    Tests occasionally build in-memory DBs from a stale schema snapshot.
    The production ``EventStore.init`` always runs the migration, but
    being defensive here keeps one more bug-class out of the API layer.
    """
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


# Columns selected by /events list + detail endpoints. Kept as a module
# constant so the SELECT lists don't drift between routes and the
# burst-grouped query below reuses the same projection.
_EVENT_COLUMNS = (
    "id, station, event_type, captured_at, received_at, "
    "schema_version, payload_json, media_key, thumb_key, "
    "species, confidence, classified_at, "
    "burst_id, burst_seq, sharpness, hero_score"
)


# ---- query helpers ---------------------------------------------------------


def _list_events_flat(
    conn: sqlite3.Connection,
    *,
    where: str,
    params: list[Any],
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Return raw event rows in captured_at DESC order (pre-PR-B behaviour).

    Used when the caller explicitly asks for ``?group=none`` to see
    every burst frame individually. The query is a straight paginated
    SELECT — identical shape to what the API returned before burst
    grouping landed.
    """
    sql = (
        f"SELECT {_EVENT_COLUMNS} "
        f"FROM events {where} "
        "ORDER BY captured_at DESC LIMIT ? OFFSET ?"
    )
    rows = conn.execute(sql, [*params, limit, offset]).fetchall()
    return [_row_to_event(r) for r in rows]


def _list_events_grouped_by_burst(
    conn: sqlite3.Connection,
    *,
    where: str,
    params: list[Any],
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Collapse each burst to one hero row, preserving the alternates.

    Strategy
    --------
    1. Pull a window of candidate rows sized to cover
       ``(limit + offset) * _MAX_BURST_FANOUT`` matching the caller's
       filters.  That window is sized to cover a realistic worst case
       of big bursts (feeder under constant attack) so paging through
       it still advances by ``limit`` events per request, even at
       deep offsets.
    2. Walk the window in captured_at DESC order, grouping rows whose
       ``burst_id`` matches.  Within a burst the hero is the row with
       the highest ``hero_score`` (ties broken by the lower
       ``burst_seq`` for stability — the earliest frame with the same
       score wins).
    3. Singleton rows (``burst_id IS NULL``) act as their own burst —
       they render with no alternates.
    4. Slice the resulting group list to ``[offset, offset + limit]``
       so the caller's pagination window is applied to *groups*, not
       raw rows.

    Why group in Python instead of a window function: the grouping
    logic needs to pick the hero *and* materialize the alternate ids
    *and* preserve burst order across the collapsed list. SQLite's
    window/aggregate support can do the hero pick, but composing all
    three cleanly in SQL obscures the intent. A Python pass over a
    pre-filtered row window is easier to test and understand; the
    indexes on (burst_id, hero_score DESC) and (captured_at DESC)
    keep the scan cost acceptable for the window sizes this API
    returns.
    """
    # Size the candidate window so ``(limit + offset)`` collapsed groups
    # can always be resolved, even in the worst case where every group
    # runs the full burst cap (:data:`_MAX_BURST_FANOUT` frames).
    #
    # There is deliberately no hard ceiling here: FastAPI already caps
    # ``limit`` at 500 via the :class:`Query` validator, and callers
    # paginating deep into the archive pay the scan cost explicitly via
    # ``offset``. An earlier implementation clamped the window to 500
    # rows, which silently truncated any request where
    # ``(limit + offset) * fanout > 500`` — e.g. ``limit=10 offset=20``
    # needed ~900 rows but got 500, returning a partial page with no
    # error. Better to scan than to lie.
    window = (limit + offset) * _MAX_BURST_FANOUT
    sql = (
        f"SELECT {_EVENT_COLUMNS} "
        f"FROM events {where} "
        "ORDER BY captured_at DESC LIMIT ?"
    )
    rows = conn.execute(sql, [*params, window]).fetchall()

    # Group while preserving insertion order so the first row in a
    # burst determines the burst's position in the listing.  ``dict``
    # has been insertion-ordered since 3.7 — relying on that instead
    # of ``OrderedDict`` for readability.
    groups: dict[str, list[sqlite3.Row]] = {}
    # A single pass over ``rows`` appends each row to its group bucket.
    # ``burst_id IS NULL`` rows go into a per-row bucket keyed on the
    # event id so they render as singletons with zero alternates.
    for row in rows:
        bid = _row_get(row, "burst_id")
        key = bid if bid else f"__singleton__:{row['id']}"
        groups.setdefault(key, []).append(row)

    collapsed: list[dict[str, Any]] = []
    for frames in groups.values():
        # Cap-watch: if a group is exactly _MAX_BURST_FANOUT frames wide,
        # we may be missing tail frames that fell outside the candidate
        # window. We can't tell from inside the query alone, but the cap
        # signal is enough to warn an operator that horus is producing
        # bursts at the configured ceiling and the constant should be
        # re-evaluated. Logging at WARNING (not raising) so a hot feeder
        # day doesn't wedge the API.
        if len(frames) >= _MAX_BURST_FANOUT:
            burst_id = _row_get(frames[0], "burst_id")
            log.warning(
                "burst fanout hit cap: burst_id=%r frames=%d cap=%d "
                "(some frames may be excluded; consider raising _MAX_BURST_FANOUT)",
                burst_id,
                len(frames),
                _MAX_BURST_FANOUT,
            )
        hero_row, alternate_ids = _pick_hero_and_alternates(frames)
        event = _row_to_event(hero_row)
        event["alternate_ids"] = alternate_ids
        event["alternate_count"] = len(alternate_ids)
        collapsed.append(event)

    # Page the grouped view — offset/limit now act on bursts, which is
    # the contract clients need for "show 24 birds" to mean "24 bursts"
    # regardless of how many frames horus shipped per burst.
    return collapsed[offset : offset + limit]


# A single burst maxes out at horus's ``burst.max_duration_s`` /
# ``interval_s`` = 30 / 1 = ~30 frames under current config. Pull
# extra headroom so that a page of 100 bursts still survives a burst
# that overruns the cap; we'd rather over-fetch once than miss events.
_MAX_BURST_FANOUT = 30


def _pick_hero_and_alternates(
    frames: list[sqlite3.Row],
) -> tuple[sqlite3.Row, list[str]]:
    """Return ``(hero, alternate_ids)`` from a list of burst frames.

    Hero is the frame with the highest ``hero_score`` (NULL scores
    treated as ``-inf`` so they never beat a computed score). Ties are
    broken by the lower ``burst_seq`` so the *earliest* frame with the
    winning score wins — stable across re-fetches and avoids flapping
    the UI when two frames tie after a classifier rerun.

    ``alternate_ids`` is every other frame in the group, ordered by
    ``burst_seq`` ascending so expander UIs can render them in capture
    order regardless of which frame won the hero pick.
    """
    def _score(row: sqlite3.Row) -> float:
        raw = _row_get(row, "hero_score")
        if raw is None:
            return float("-inf")
        return float(raw)

    def _seq(row: sqlite3.Row) -> int:
        raw = _row_get(row, "burst_seq")
        # Singletons (no burst_seq) get a huge seq so stable sort puts
        # them last when sequence is the tie-breaker.
        return int(raw) if raw is not None else 1 << 30

    hero = max(frames, key=lambda r: (_score(r), -_seq(r)))
    alternates = [row for row in frames if row["id"] != hero["id"]]
    alternates.sort(key=_seq)
    return hero, [row["id"] for row in alternates]


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
        group: str = Query(
            "burst",
            pattern="^(burst|none)$",
            description=(
                "Grouping mode. 'burst' (default) collapses frames sharing a "
                "burst_id to one row — the frame with the highest hero_score "
                "— and attaches the others as alternate_ids. 'none' returns "
                "every frame individually."
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

        if group == "burst":
            events = _list_events_grouped_by_burst(
                conn, where=where, params=params, limit=limit, offset=offset
            )
        else:
            events = _list_events_flat(
                conn, where=where, params=params, limit=limit, offset=offset
            )

        return {
            "count": len(events),
            "limit": limit,
            "offset": offset,
            "group": group,
            "events": events,
        }

    @app.get("/events/{event_id}")
    def get_event(
        event_id: str,
        conn: sqlite3.Connection = Depends(get_db),
    ) -> dict[str, Any]:
        row = conn.execute(
            f"SELECT {_EVENT_COLUMNS} FROM events WHERE id = ?",
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
