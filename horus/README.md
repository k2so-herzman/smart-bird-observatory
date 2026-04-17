# Horus — Capture Node

Runs on Raspberry Pi at 192.168.1.173.

## What it does

1. Captures a JPEG from `imx708_wide` every `interval_s` (default 1s).
2. Runs a cheap frame-diff motion gate.
3. On motion → publishes an event to `sbo/<station>/image/event`.
4. On idle → deletes the frame (no flooding).
5. Heartbeat on `sbo/<station>/status` every 60s.

## What it doesn't do

- No image classification.
- No bird species ID.
- No long-term storage (ring-buffered locally, Banshee keeps history).

## Install on the Pi

```bash
sudo mkdir -p /opt/horus /etc/horus /var/lib/horus/captures
sudo chown -R k2:k2 /opt/horus /etc/horus /var/lib/horus
cd /opt/horus
git clone https://github.com/k2so-herzman/smart-bird-observatory.git .
python3 -m venv .venv
.venv/bin/pip install -e horus

sudo cp horus/config/horus.example.yaml /etc/horus/horus.yaml
# edit MQTT host etc.

sudo cp systemd/horus-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now horus-capture
journalctl -u horus-capture -f
```

## Local dev

Nothing Pi-specific about the Python — you can run `horus/src/horus/motion.py`
and `events.py` locally for unit tests. The only Pi-only piece is
`camera.py`, which shells out to `rpicam-still`.

## Layout

```
horus/
  pyproject.toml
  config/horus.example.yaml
  src/horus/
    __init__.py
    main.py        # daemon entrypoint
    config.py      # YAML → dataclass loader
    camera.py      # rpicam-still wrapper
    motion.py      # frame-diff gate
    events.py      # MQTT publisher
    storage.py     # ring-buffer + paths
```
