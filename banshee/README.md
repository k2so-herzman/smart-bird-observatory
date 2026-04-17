# Banshee — Bird Brain

Runs in an LXC container on Banshee. Subscribes to MQTT, persists
frames, writes to InfluxDB. Classification + notifications arrive in
follow-up PRs.

## Responsibilities (this PR)

- Subscribe to `sbo/+/image/event` and `sbo/+/status`
- Validate payloads (schema version, sha256 integrity)
- Persist JPEGs under `<image_dir>/<station>/YYYY-MM-DD/`
- Write event metadata to InfluxDB (`sbo` bucket)

## Responsibilities (future PRs)

- TFLite image classifier on saved crops
- Reconcile BirdNET audio confidence with image class
- Post to Home Assistant
- Post high-confidence finds to Telegram
- Serve a photo browser

## Non-responsibilities

- No camera or mic hardware. Banshee never touches sensors.
- No NFS. Images come over MQTT inline (base64).

## Install (LXC)

```bash
sudo mkdir -p /opt/banshee /etc/banshee /var/lib/banshee/images
sudo useradd -r -s /usr/sbin/nologin -d /opt/banshee banshee
sudo chown -R banshee:banshee /opt/banshee /var/lib/banshee

cd /opt/banshee
sudo -u banshee git clone https://github.com/k2so-herzman/smart-bird-observatory.git .
sudo -u banshee python3 -m venv .venv
sudo -u banshee .venv/bin/pip install -e banshee

sudo cp banshee/config/banshee.example.yaml /etc/banshee/banshee.yaml
# edit — set influx.token

sudo cp systemd/banshee.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now banshee
journalctl -u banshee -f
```

## Layout

```
banshee/
  pyproject.toml
  config/banshee.example.yaml
  src/banshee/
    __init__.py
    main.py          # daemon entrypoint + pipeline
    config.py        # YAML → dataclass loader
    subscriber.py    # MQTT subscriber + dispatch
    events.py        # payload validation (ImageEvent, StatusEvent)
    storage.py       # on-disk JPEG persistence
    influx.py        # InfluxDB writer
```
