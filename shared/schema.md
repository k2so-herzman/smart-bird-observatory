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

## Events table — burst grouping columns (PR-B)

The Thoth `events` table grew four columns in PR-B to support
burst grouping and hero-frame selection. These are populated at
ingest from horus's burst session metadata + a Laplacian-variance
sharpness pass on the inbound JPEG. Pre-PR-B rows leave them NULL
and the API treats NULL as "not part of a burst" / "no score".

| Column        | Type      | Meaning |
| ------------- | --------- | ------- |
| `burst_id`    | `TEXT`    | Shared burst session identifier published by horus (`{station}-{ms}-{rand}`). All frames from one feeder visit share this id. NULL for singleton frames and legacy pre-burst traffic. |
| `burst_seq`   | `INTEGER` | 1-based monotonic index within the burst (frame 1, 2, 3, …). NULL whenever `burst_id` is NULL. Used as a tie-breaker when two frames share the top `hero_score` — earliest wins. |
| `sharpness`   | `REAL`    | Laplacian variance of a 640-wide grayscale thumbnail (`scoring.laplacian_variance`). Higher is sharper. Stored on its own column (rather than only inside `hero_score`) so the post-classify recompute path can rebuild the composite without re-decoding the image. |
| `hero_score`  | `REAL`    | Tier-1 composite hero rank, `[0, 1]`. Composition: `0.4*bird_score + 0.3*sharpness_norm + 0.2*bbox_area + 0.1*classifier_confidence`. The `?group=burst` API picks `MAX(hero_score)` per burst as the canonical frame. NULL when none of the four inputs were available at insert time (e.g. ingest paths that can't score); rows with at least one input get a numeric score. |

Indexes added alongside these columns:

- `idx_events_burst` on `(burst_id, burst_seq) WHERE burst_id IS NOT NULL` — burst-frame lookups.
- `idx_events_burst_hero` on `(burst_id, hero_score DESC) WHERE burst_id IS NOT NULL` — hero pick.

Both are partial on `burst_id IS NOT NULL` so legacy / singleton rows
don't bloat the index. See `banshee/src/banshee/eventstore.py` for the
migration code; it's idempotent so applying against an already-migrated
DB is a no-op.
