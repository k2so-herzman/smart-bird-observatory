# Banshee — Bird Brain

Runs in an LXC container on Banshee. Subscribes to MQTT, classifies, notifies.

## Responsibilities

- Subscribe to `sbo/+/image/event` and `sbo/+/audio/detection`
- Run TFLite image classifier on motion crops
- Reconcile BirdNET audio confidence with image class
- Write to InfluxDB (`sbo` bucket)
- Post to Home Assistant
- Post high-confidence finds to Telegram
- Serve a photo browser

## Non-responsibilities

- No camera or mic hardware. Banshee never touches sensors.
