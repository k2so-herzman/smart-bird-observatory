# MQTT Event Schema

Broker: (TBD — use HA's Mosquitto at 192.168.1.87 or dedicated?)

## Topics

```
sbo/<station>/image/event     # motion-triggered image capture
sbo/<station>/audio/detection # BirdNET-Go detection
sbo/<station>/status          # heartbeat + camera/mic health
```

## Payloads

### `sbo/<station>/image/event`

```json
{
  "schema_version": 1,
  "station": "horus",
  "captured_at": "2026-04-17T16:57:00Z",
  "camera": "imx708_wide",
  "trigger": "motion",
  "resolution": [2304, 1296],
  "file": "nfs://banshee/sbo/horus/2026-04-17/1657_00_motion.jpg",
  "motion_region": [x, y, w, h],
  "sha256": "..."
}
```

Image itself lives on shared NFS. Message references path — MQTT is for signalling only, not image payload.

### `sbo/<station>/audio/detection`

BirdNET-Go native JSON output, plus a `station` field. Clip path points to NFS.

### `sbo/<station>/status`

```json
{
  "schema_version": 1,
  "station": "horus",
  "ts": "2026-04-17T16:57:00Z",
  "camera_ok": true,
  "mic_ok": true,
  "load_1m": 0.12,
  "mem_free_mb": 340,
  "uptime_s": 12345
}
```

Published every 60s. Banshee tracks last-seen per station.
