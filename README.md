# Smart Bird Observatory

Multi-station bird feedercam + audio ID system for the Herzman Mesa.

## Architecture

```
Station nodes (Horus, ...)           Banshee (LXC: bird-brain)
  camera + mic                         MQTT subscriber
  motion gate (frame diff)             TFLite image classifier
  BirdNET-Go (audio ID)                BirdNET post-processing
  MQTT publish          ───────►       InfluxDB writes
                                       Home Assistant + Telegram posts
                                       Photo browser
```

Stations are dumb sensors. All heavy inference happens on Banshee.

## Stations

| Name   | Host              | Hardware                         | Role                  |
|--------|-------------------|----------------------------------|-----------------------|
| Horus  | 192.168.1.173     | Pi 3 B+ (905MB), IMX708 wide     | Seed feeder (primary) |

## Repo layout

- `horus/` — capture node code (runs on Pi)
  - `capture/` — libcamera / rpicam wrappers
  - `events/` — MQTT publisher
  - `config/` — per-station YAML
- `banshee/` — classifier + aggregator (runs in LXC)
  - `classifier/` — TFLite image inference
  - `mqtt/` — subscriber
  - `storage/` — InfluxDB client
  - `notify/` — Home Assistant + Telegram hooks
- `shared/` — MQTT schema, common types
- `systemd/` — service units for both sides
- `docs/` — design notes, BOM, runbooks

## Status

Scaffold only. See `docs/STATUS.md` for phase tracking.
