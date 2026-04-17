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
| MQTT       | `192.168.1.87:1883` (HA)                   | Event transport                  |
| MinIO      | `192.168.1.65:9000` (absu)                 | Media blob store                 |
| imgproxy   | `https://images.chickenmilkbomb.com`       | On-the-fly resize + format delivery |
| InfluxDB   | `banshee` CT 111                           | Time-series metrics              |
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
  classified_at TIMESTAMP
);
CREATE INDEX idx_events_captured ON events(captured_at DESC);
CREATE INDEX idx_events_station ON events(station, captured_at DESC);
CREATE INDEX idx_events_species ON events(species, captured_at DESC);
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
GET  /events?station=&type=&species=&from=&to=&limit=
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
