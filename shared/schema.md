# MQTT Event Schema

**Broker**: Home Assistant Mosquitto at `192.168.1.73:1883` (the HA host's Mosquitto add-on).

**Transport model**: MQTT carries everything. Image bytes ride inline as
base64 on the `image/event` topic — no NFS, no shared filesystem, no
separate HTTP fetch. Banshee is a pure MQTT consumer.

Consequence: mind the broker's `message_size_limit`. A 2304×1296
JPEG at q=90 is ~600KB raw, ~800KB base64-encoded. Mosquitto defaults
are generous (256MB in recent versions), but verify on first deploy.

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
  "content_type": "image/jpeg",
  "image_b64": "<base64-encoded JPEG>",
  "size_bytes": 612345,
  "changed_fraction": 0.043,
  "sha256": "...",
  "bbox_fraction": [0.18, 0.42, 0.31, 0.55],
  "bird_score": 0.82,
  "bird_label": "house finch"
}
```

Image payload is inline (base64). QoS 1, no retain. Subscribers decode
`image_b64` with `base64.b64decode()` to get the raw JPEG.

**Image is always the full frame.** horus applies its on-device gate
(object detector + species classifier) against a bird-centered crop,
but publishes the full sensor frame so downstream UIs (Thoth) can show
the whole scene. Subscribers that want the tight crop the model scored
on must re-apply the crop using `bbox_fraction` and the shared
`sbo_shared.imaging.crop_to_bbox_bytes` helper — byte-identical math
guarantees horus's on-device score and any post-ingest score are
directly comparable.

Optional fields:

| Field            | Type              | Meaning |
| ---------------- | ----------------- | ------- |
| `bbox_fraction`  | `[x0,y0,x1,y1]` in `[0,1]` full-frame coords, or absent | Motion-bbox of the subject. Absent on pre-bbox builds or when motion produced no bbox — subscribers should classify the full frame as a fallback. |
| `bird_score`     | `float` in `[0,1]`, or absent | horus's on-device species-classifier confidence for the top label, already computed against the crop. |
| `bird_label`     | `str`, or absent  | horus's on-device species label at `bird_score`. |
| `af`             | object, or absent | Autofocus diagnostic snapshot (lens position, mode). |

### `sbo/<station>/audio/detection`

BirdNET-Go native JSON output, plus a `station` field. Audio clip (if any)
rides inline as base64 under `audio_b64` with `content_type: audio/wav`.

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
