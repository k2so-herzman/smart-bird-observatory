# Thoth — Aggregator Design

Thoth is the LXC (CT 113) on Banshee that subscribes to MQTT events
from station nodes, classifies images, stores media, writes time-series,
and serves the web UI. Named after the Egyptian ibis god of knowledge —
fits the pantheon (Horus) and matches the role (aggregating + indexing).

## Provisioning

```bash
pct create 113 local:vztmpl/debian-12-standard_*.tar.zst \
  --hostname thoth \
  --cores 4 \
  --memory 4096 \
  --swap 512 \
  --rootfs local-lvm:32 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --onboot 1 \
  --unprivileged 1 \
  --features keyctl=1

pct start 113
```

No Docker inside — services run as systemd units directly. Nested
containers buy nothing here and complicate networking and permissions.

## External dependencies

| Service    | Host / endpoint                            | Use                              |
|------------|--------------------------------------------|----------------------------------|
| MQTT       | `192.168.1.73:1883` (HA Mosquitto add-on)  | Event transport                  |
| MinIO      | `192.168.1.65:9000` (absu)                 | Media blob store                 |
| imgproxy   | `https://images.chickenmilkbomb.com`       | On-the-fly resize + format delivery |
| InfluxDB   | `192.168.1.24:8086`                        | Time-series metrics              |
| HA         | existing                                   | Dashboards + notifications       |
| Telegram   | via k2so                                   | Detection alerts                 |

Credentials live in `/etc/thoth/env` (chmod 600), loaded by systemd via
`EnvironmentFile=`. Never committed to git.

## Services (systemd units)

| Unit                        | Role                                              |
|-----------------------------|---------------------------------------------------|
| `thoth-ingest.service`      | MQTT subscriber → event store + MinIO + Influx    |
| `thoth-classify.service`    | TFLite image classifier, consumes classify queue  |
| `thoth-api.service`         | FastAPI (uvicorn) on :8000 — read API for UI      |
| `caddy.service`             | TLS + reverse proxy + static frontend             |

Deferred to later PRs:
- `thoth-notify.service` — HA + Telegram posts (may live inside ingest for Phase 1)

## Storage model

**Event metadata** — SQLite at `/var/lib/thoth/events.db`. Schema roughly:

```sql
CREATE TABLE events (
  id           TEXT PRIMARY KEY,       -- UUID
  station      TEXT NOT NULL,
  event_type   TEXT NOT NULL,          -- 'image' | 'audio' | 'status'
  captured_at  TIMESTAMP NOT NULL,
  received_at  TIMESTAMP NOT NULL,
  schema_version INTEGER NOT NULL,
  payload_json TEXT NOT NULL,          -- full event metadata
  media_key    TEXT,                   -- MinIO object key (if any)
  thumb_key    TEXT,                   -- MinIO thumb key (if any)
  species      TEXT,                   -- classifier output
  confidence   REAL,
  classified_at TIMESTAMP,
  -- Burst grouping + hero selection (added in PR-B; see shared/schema.md)
  burst_id     TEXT,                   -- shared session id from horus, NULL for singletons
  burst_seq    INTEGER,                -- 1-based monotonic index within the burst
  sharpness    REAL,                   -- Laplacian variance computed once at ingest
  hero_score   REAL                    -- composite [0, 1]; NULL when no scoring inputs available
);
CREATE INDEX idx_events_captured ON events(captured_at DESC);
CREATE INDEX idx_events_station ON events(station, captured_at DESC);
CREATE INDEX idx_events_species ON events(species, captured_at DESC);
CREATE INDEX idx_events_burst ON events(burst_id, burst_seq)
  WHERE burst_id IS NOT NULL;
CREATE INDEX idx_events_burst_hero ON events(burst_id, hero_score DESC)
  WHERE burst_id IS NOT NULL;
```

**Media** — MinIO bucket `thoth` (single bucket, prefixed keys):

```
{station}/image/{YYYY}/{MM}/{DD}/{event_id}.jpg
{station}/audio/{YYYY}/{MM}/{DD}/{event_id}.wav
```

No thumbnail pre-generation. Image variants are served through imgproxy
on request (see below). Ingest stores the full-res JPEG once and lets
imgproxy handle every size/format the UI needs.

Bucket is created idempotently on ingest service startup if missing.

### Image delivery via imgproxy

All image reads go through the existing household imgproxy at
`https://images.chickenmilkbomb.com`, which reads from MinIO directly.
The Thoth API emits imgproxy URLs instead of presigning MinIO.

Example:

```
https://images.chickenmilkbomb.com/<signature>/resize:fit:250:250/format:webp/plain/s3://thoth/horus/image/2026/04/17/<event_id>.jpg
```

Benefits:
- No thumb generation step at ingest
- Callers pick their own size + format (AVIF/WebP on modern browsers)
- Caching / CDN story handled by imgproxy, not Thoth

The API exposes the full-res MinIO key in the event JSON; the frontend
composes imgproxy URLs client-side. A helper endpoint
`GET /events/{id}/image?w=&h=&fmt=` can also emit a signed imgproxy URL
server-side for convenience.

**Time-series** — InfluxDB bucket `sbo` (matches existing config):
- `sbo_image` measurement — per-event with tags `station`, `camera`, `species`
- `sbo_audio` measurement — per-detection with tags `station`, `species`
- `sbo_status` measurement — per-heartbeat

## Ingest pipeline

```
MQTT message
   │
   ▼
validate (schema_version, sha256, size caps)
   │
   ├─► write MinIO blob (media_key only — no thumb; imgproxy handles resize)
   │
   ├─► insert event row (SQLite)
   │
   ├─► write InfluxDB point
   │
   └─► if image: enqueue for classifier
```

Classify queue is a simple SQLite table (`classify_queue`) polled by
the classifier. No Redis, no external broker. Rationale: single-node,
low-volume, don't add a daemon for the sake of it.

## Read API (FastAPI)

```
GET  /events?station=&type=&species=&from=&to=&limit=&group=
GET  /events/{id}
GET  /events/{id}/media         → 302 to presigned MinIO URL (originals)
GET  /events/{id}/image?w=&h=&fmt=  → 302 to signed imgproxy URL (variants)
GET  /stations                  → last_seen + status per station
GET  /species?window=7d         → species counts
GET  /stats/activity?window=24h → hourly event counts
GET  /healthz
```

Read-only. Writes happen via MQTT → ingest service. Keep the API
stateless so horizontal scaling is an option later (though unlikely
to be needed).

### `/events` query parameters

| Param            | Default  | Meaning                                                                                                                                                                                                                                                |
| ---------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `station`        | —        | Filter by station name (e.g. `horus`).                                                                                                                                                                                                                  |
| `species`        | —        | Filter by classified species label (exact match).                                                                                                                                                                                                       |
| `since`          | —        | ISO-8601 timestamp; returns events captured at or after this.                                                                                                                                                                                            |
| `min_confidence` | server default (`THOTH_API_MIN_CONFIDENCE`, default `0.10`) | Floor on classifier confidence. Events below the floor are hidden; unclassified events (NULL confidence) always pass through. Pass `0` to disable.                                                  |
| `group`          | `burst`  | Grouping mode. `burst` (default) collapses frames sharing a `burst_id` to one row — the frame with the highest `hero_score` — and attaches the others as `alternate_ids`. `none` returns every frame individually (legacy flat shape).                  |
| `limit`          | `100`    | Page size, `[1, 500]`. With `group=burst` this counts *bursts*, not frames.                                                                                                                                                                              |
| `offset`         | `0`      | Page offset. With `group=burst` this offsets *bursts*.                                                                                                                                                                                                   |

#### Response shape

```json
{
  "count": 3,
  "limit": 100,
  "offset": 0,
  "group": "burst",
  "events": [
    {
      "id": "b-2",
      "station": "horus",
      "event_type": "image",
      "captured_at": "2026-04-18T12:00:01+00:00",
      "...": "…",
      "burst_id": "burst-A",
      "burst_seq": 2,
      "sharpness": 900.0,
      "hero_score": 0.75,
      "alternate_ids": ["b-1", "b-3"],
      "alternate_count": 2
    }
  ]
}
```

`alternate_ids` and `alternate_count` are present only on `group=burst`
responses. Singletons (no `burst_id`) carry an empty `alternate_ids`
and `alternate_count: 0`. Under `group=none` neither field is emitted
and every frame in a burst is returned as its own row (pre-PR-B
behaviour, preserved for callers that need every frame — e.g. the
classifier dashboard).

The default flipped from flat to `group=burst` in PR-B. UIs paginating
by event count want one tile per bird visit, not one per shutter
release; the `group=none` opt-out covers tooling that still needs the
raw stream.

## Web UI

- Next.js, static export served by Caddy
- Phase 1: event timeline, station health, photo grid by day, species list
- Phase 2: per-bird pages, audio clip playback, classifier feedback
  (confirm/reject to build training set)
- No auth in Phase 1 — LAN-only via Caddy. Add HA OIDC later if exposed.

## Audio integration (BirdNET-Go → Thoth)

BirdNET-Go on Horus has native MQTT output but its payload shape does
not match `sbo/<station>/audio/detection`. Options:

1. **Native BirdNET-Go MQTT, reshape in Thoth.** Subscribe to BirdNET-Go's
   topic directly, translate to our event shape at ingest. Simplest.
2. **Webhook → Horus shim → our MQTT schema.** Config BirdNET-Go to POST
   detections; a tiny shim on Horus republishes in our shape with audio
   clip attached. Most control but more moving parts.

Recommend (1) for Phase 1. Revisit if we need pre-enqueue filtering.

## Unified outdoor events (future)

Thoth, Skywatch, and Skylapse all emit the same conceptual shape:
`{timestamp, source, type, location, media[], metadata}`. A future
unified viewer (working name: **Vantage**) consumes their APIs and
renders a single timeline. Thoth's API is designed with this in mind:
generic event model, presigned media URLs, no UI assumptions baked
into the backend.

## Naming

The aggregator Python package is still `banshee` (named after the host
it runs on). The LXC is `thoth`. Long-term this should be reconciled —
proposal: rename `banshee/src/banshee/` → `banshee/src/thoth/`, rename
the systemd unit, bump the config path to `/etc/thoth/thoth.yaml`.
Defer to a dedicated PR after the ingest pipeline is stable.

## Open questions

- **Caddy vs. existing reverse proxy** — if there's already a household
  ingress (Traefik on k2?), Thoth can publish a plain HTTP API and let
  the external proxy handle TLS. Needs a call.
- **imgproxy signing** — does imgproxy have `IMGPROXY_KEY` / `IMGPROXY_SALT`
  set (signed URLs required) or is it unsigned for LAN trust? Thoth
  needs the secrets in `/etc/thoth/env` if signing is required.
- **imgproxy MinIO access** — does imgproxy already have read perms on
  the `thoth` bucket (either shared creds or bucket policy), or does
  that need to be configured at bucket-create time?
- **Retention policy** — how long do we keep raw audio + full-res
  images? Propose 30 days full-res, keep event metadata forever.
  Configurable.
- **Species model** — MobileNet general vs. iNaturalist bird head vs.
  BirdNET image model (if it exists). Benchmark in Phase 1.
- **Classifier backpressure** — if classify queue grows unbounded during
  a migratory flurry, do we drop, batch, or block ingest? Probably
  drop-oldest with a bounded queue.

## Acceptance criteria — Phase 1 (Thoth online)

- [ ] LXC CT 113 provisioned, `thoth` hostname, on main VLAN
- [ ] `thoth-ingest.service` running, MinIO bucket auto-created
- [ ] Horus image event end-to-end: MQTT → MinIO blob + event row + Influx point
- [ ] Horus BirdNET-Go detection end-to-end: MQTT → MinIO audio clip + event row
- [ ] `thoth-api.service` serves `/events` and `/healthz`
- [ ] Caddy serving placeholder UI at `thoth.local` (or whatever hostname)
- [ ] HA notification fires on classifier confidence ≥ threshold
