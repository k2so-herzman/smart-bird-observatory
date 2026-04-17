# Horus — Capture Node

Runs on Raspberry Pi at 192.168.1.173.

## Install (eventual)

```
git clone https://github.com/k2so-herzman/smart-bird-observatory.git
cd smart-bird-observatory/horus
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo cp ../systemd/horus-capture.service /etc/systemd/system/
sudo systemctl enable --now horus-capture
```

## Responsibilities

- Capture stills from `imx708_wide` via `rpicam-still`
- Motion gate (frame diff) to avoid flooding Banshee
- Publish events to MQTT
- Publish heartbeat every 60s
- Run BirdNET-Go as a sibling service

## Non-responsibilities

- No image classification
- No bird species ID
- No storage (other than a short local ring buffer)
