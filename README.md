# Smart Bird Observatory

Multi-station bird feedercam + audio ID system for the Herzman Mesa.

## Architecture

```
Station nodes (Horus, ...)           Banshee host (LXC: thoth, CT 113)
  camera + mic                         MQTT subscriber (sbo/+/#)
  motion gate (frame diff)             TFLite image classifier
  BirdNET-Go (audio ID)                Event store (SQLite) + media (MinIO)
  MQTT publish          ───────►       InfluxDB writes
                                       FastAPI read API + web UI
                                       Home Assistant + Telegram posts
```

Stations are dumb sensors. All heavy inference and storage happens in
the Thoth LXC on Banshee.

## Stations

| Name   | Host              | Hardware                                    | Role                  |
|--------|-------------------|---------------------------------------------|-----------------------|
| Horus  | 192.168.1.251     | Pi 3 B+ (905MB), IMX519 16MP + USB mic      | Seed feeder (primary) |

## Repo layout

- `horus/` — capture node code (runs on each Pi station)
  - `src/horus/` — camera, motion gate, MQTT publisher
  - `config/` — per-station YAML
- `banshee/` — LXC aggregator service (runs in the `thoth` LXC)
  - `src/banshee/` — MQTT subscriber, storage, InfluxDB writer
  - `config/` — aggregator YAML
- `shared/` — MQTT schema + common types (`schema.md`)
- `systemd/` — service units for both sides
- `docs/` — design notes, BOM, runbooks
  - `thoth-design.md` — aggregator service architecture
  - `STATUS.md` — phase tracking

> **Naming note.** The aggregator package is still called `banshee`
> (host name), but the LXC it runs in is `thoth`. A rename is planned
> (see `docs/thoth-design.md` §naming) but deferred to avoid churn.

## Status

Phase 0 (bootstrap) and Phase 1 (first end-to-end image event) stubs are
landed. Audio via BirdNET-Go is installed on Horus but not yet wired to
MQTT. See `docs/STATUS.md` for detail and `docs/thoth-design.md` for the
aggregator design.
