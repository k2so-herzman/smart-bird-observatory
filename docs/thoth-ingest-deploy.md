# Thoth Ingest — First Code Deploy

Deploys the `thoth-ingest` service onto the Thoth LXC and flips the
`thoth-ingest.service` systemd unit from scaffold to live.

Precondition: the host is provisioned per `docs/thoth-provisioning.md`
(user, dirs, env template, stub units in place).

All commands assume you are SSH'd to the k2so host and operating on
`root@192.168.1.95` (Thoth).

---

## Phase E — Install the ingest code

### E.1 — Clone the repo to `/opt/thoth/src`

```bash
ssh root@192.168.1.95 'bash -s' <<'EOF'
set -e
apt-get install -y -qq git python3-venv
install -d -o root -g root -m 0755 /opt/thoth
test -d /opt/thoth/src || \
  git clone https://github.com/k2so-herzman/smart-bird-observatory.git /opt/thoth/src
cd /opt/thoth/src && git fetch --quiet && git checkout main && git pull --ff-only --quiet
EOF
```

Re-run this block to update. It's idempotent: clone once, fast-forward
every subsequent deploy.

### E.2 — Build the venv

```bash
ssh root@192.168.1.95 'bash -s' <<'EOF'
set -e
python3 -m venv /opt/thoth/venv
/opt/thoth/venv/bin/pip install --quiet --upgrade pip
/opt/thoth/venv/bin/pip install --quiet /opt/thoth/src/banshee
EOF
```

The venv lives at `/opt/thoth/venv` and owns itself under `root:root`.
The `thoth` service user reads it via `/opt/thoth/venv/bin/thoth-ingest`.
No write access is needed at runtime — all state goes to
`/var/lib/thoth` and `/var/log/thoth`.

### E.3 — Fill in `/etc/thoth/env`

First deploy only. The env template is already on the host from
provisioning:

```bash
ssh root@192.168.1.95 '
  test -f /etc/thoth/env || \
    install -m 0640 -o root -g thoth /etc/thoth/env.example /etc/thoth/env
  ls -la /etc/thoth/env
'
```

Edit in place and fill real values for:

- `MQTT_USERNAME`, `MQTT_PASSWORD` — from HA Mosquitto add-on config
- `THOTH_STORAGE_BACKEND=local` and `THOTH_STORAGE_ROOT` — media lands
  on thoth's local NVMe (default `/var/lib/thoth/media`). Remove any
  stale `MINIO_*` lines; MinIO is decommissioned.
- `INFLUX_TOKEN` — from the `sbo` bucket's token on
  `192.168.1.24:8086`

```bash
ssh root@192.168.1.95 'nano /etc/thoth/env'
```

Never commit the real file. Only `thoth.env.example` is tracked.

### E.4 — Install the updated systemd unit

The unit ships in the repo at `systemd/thoth-ingest.service` and was
first placed by the provisioning runbook as a scaffold. Replace it
with the current version and reload:

```bash
ssh root@192.168.1.95 '
  install -m 0644 /opt/thoth/src/systemd/thoth-ingest.service /etc/systemd/system/thoth-ingest.service
  systemctl daemon-reload
'
```

### E.5 — Start + enable the service

```bash
ssh root@192.168.1.95 '
  systemctl enable --now thoth-ingest.service
  systemctl status thoth-ingest.service --no-pager -n 20
'
```

Expected steady state:

```
Active: active (running)
...
INFO  thoth.ingest MQTT connected to 192.168.1.73:1883
INFO  thoth.ingest subscribed to sbo/+/image/event and sbo/+/status
```

If you see `local blob store ready at /var/lib/thoth/media`, the media
root exists (it is auto-created on first run).

### E.6 — Verify end-to-end from the k2so host

```bash
# 1. Tail the journal for real-time ingest events
ssh root@192.168.1.95 'journalctl -u thoth-ingest.service -f'

# 2. From another shell, publish a fake status event to the broker.
#    Replace <PASS> with MQTT_PASSWORD from /etc/thoth/env.
mosquitto_pub -h 192.168.1.73 -u homeassistant-mqtt -P '<PASS>' \
  -t 'sbo/horus/status' \
  -m '{"schema_version":1,"station":"horus","ts":"2026-04-17T22:00:00Z","camera_ok":true}'

# 3. Expect in the journal:
#    DEBUG thoth.ingest status from horus at 2026-04-17T22:00:00+00:00
```

A real image event from Horus will show:

- A `DEBUG` line from `localfs_store` with the key under `horus/image/YYYY/MM/DD/<uuid>.jpg`
- An `INFO thoth.ingest image from horus: ...` line
- A new row in `/var/lib/thoth/events.db`
- A `sbo_image` point in InfluxDB bucket `sbo`

### E.7 — Quick smoke checks

```bash
# SQLite event count
ssh root@192.168.1.95 'sqlite3 /var/lib/thoth/events.db "SELECT count(*) FROM events;"'

# Media blobs on thoth's local storage
ssh root@192.168.1.95 'find /var/lib/thoth/media -type f | tail -5'

# Most recent event row
ssh root@192.168.1.95 'sqlite3 /var/lib/thoth/events.db \
  "SELECT id, station, captured_at, media_key FROM events ORDER BY captured_at DESC LIMIT 1;"'
```

## Rollback

```bash
ssh root@192.168.1.95 '
  systemctl disable --now thoth-ingest.service
  # Re-drop the scaffold stub:
  cp /opt/thoth/src/systemd/thoth-ingest.service /etc/systemd/system/thoth-ingest.service
  systemctl daemon-reload
'
```

(Or revert to a previous `/opt/thoth/src` commit and re-run Phase E.2.)

## Troubleshooting

### `required env var MQTT_HOST is unset`

The service started without `/etc/thoth/env` loaded, or the file is
missing `MQTT_HOST`. Check:

```bash
systemctl cat thoth-ingest.service | grep EnvironmentFile
ls -la /etc/thoth/env
grep -E '^(MQTT_HOST|THOTH_STORAGE_BACKEND|THOTH_STORAGE_ROOT)=' /etc/thoth/env
```

### `blob write failed ... dropping event`

The service logs the write failure and skips the event — the SQLite
row is *not* created, so indexes never point at a dangling key. The
process keeps running; fix the storage side and the next event will
land cleanly. Causes to check:

- Disk full on the volume behind `THOTH_STORAGE_ROOT` (`df -h`)
- Permissions: the `thoth` user must own/write the root, and the
  systemd unit's `ReadWritePaths=` must cover it (the stock unit
  covers `/var/lib/thoth`; add the path if you point the root
  elsewhere, e.g. a dedicated NVMe mount)
- For the legacy MinIO backend: bad credentials, bucket policy, or
  endpoint scheme mismatch

### `InfluxDB token not set — writes will be skipped`

Benign log line if you deliberately deferred Influx. To enable, set
`INFLUX_TOKEN` in `/etc/thoth/env` and `systemctl restart thoth-ingest.service`.
