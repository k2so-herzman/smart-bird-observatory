# Banshee — Thoth ingest service

The Python package that runs as `thoth-ingest.service` inside the Thoth
LXC. Subscribes to MQTT, writes images to local disk (NVMe), records
events in SQLite, and emits metrics to InfluxDB.

The package is still named `banshee` for history; see `docs/thoth-design.md`
§ "Naming" for the rename plan (deferred).

## Responsibilities

- Subscribe to `sbo/+/image/event` and `sbo/+/status`
- Validate payloads (schema version, sha256 integrity)
- Write JPEGs atomically to the local media root (default
  `/var/lib/thoth/media`) under
  `{station}/image/{YYYY}/{MM}/{DD}/{event_id}.jpg`
- Index each event in SQLite at `/var/lib/thoth/events.db`
- Emit `sbo_image` + `sbo_status` points to InfluxDB bucket `sbo`
- Auto-create the media root on first start

MinIO object storage is still available as a legacy backend
(`THOTH_STORAGE_BACKEND=minio` + `MINIO_*` vars) but is not required —
the MinIO host was decommissioned and local NVMe storage is the
default.

## Out of scope (follow-up PRs)

- TFLite image classifier (`thoth-classify.service`)
- Reconciling BirdNET audio confidence with image class
- FastAPI read API (`thoth-api.service`)
- Home Assistant + Telegram notifications
- Photo browser UI
- Retention / prune

## Deploy to Thoth LXC

See `docs/thoth-ingest-deploy.md` for the SSH-driven deploy runbook.
Short version: clone to `/opt/thoth/src`, `pip install` into
`/opt/thoth/venv`, fill `/etc/thoth/env`, `systemctl enable --now
thoth-ingest.service`.

## Configuration

Production: env-driven via `/etc/thoth/env` (loaded by systemd as
`EnvironmentFile=`). Template at `banshee/config/thoth.env.example`.

Local dev: run `banshee --config ./banshee.yaml` against a YAML file.
Example at `banshee/config/banshee.example.yaml`.

## Layout

```
banshee/
  pyproject.toml
  config/
    banshee.example.yaml   # dev-only YAML config
    thoth.env.example      # production env template (mirrors /etc/thoth/env)
  src/banshee/
    __init__.py
    main.py                # daemon entrypoint + pipeline
    config.py              # env + YAML loaders
    subscriber.py          # MQTT subscriber + dispatch
    events.py              # payload validation (ImageEvent, StatusEvent)
    blobstore.py           # backend protocol + shared key scheme + factory
    localfs_store.py       # local-filesystem media backend (default)
    minio_store.py         # legacy MinIO backend
    eventstore.py          # SQLite schema + event insert
    influx.py              # InfluxDB writer
  tests/
    test_config.py
    test_eventstore.py
    test_blobstore.py
    test_localfs_store.py
    test_minio_store.py
```

## Tests

```bash
cd banshee
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest tests/ -v
```

All tests are hermetic — no MinIO or MQTT required.
