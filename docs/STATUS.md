# Project Status

## Stations + hosts

| Name   | Host              | Hardware                                    | Role                  |
|--------|-------------------|---------------------------------------------|-----------------------|
| Horus  | 192.168.1.251     | Pi 3 B+ (905MB), IMX519 16MP + USB Lavalier | Seed feeder (primary) |
| Thoth  | 192.168.1.95      | Proxmox LXC CT 113 on Banshee (Debian 12)   | Aggregator / API / UI |

## Phase 0 — Bootstrap (done)

- [x] IMX519 16MP sensor attached and detected on Horus (tested at 2328×1748)
- [x] Test frame captured
- [x] SSH key auth from k2so → k2@horus
- [x] Repo scaffold
- [x] MQTT schema draft (`shared/schema.md`)
- [x] Horus capture daemon stub
- [x] systemd unit for capture daemon
- [x] Banshee MQTT subscriber stub (local-disk storage; MinIO migration pending in Phase 1)
- [x] Thoth LXC (CT 113) provisioned on Banshee — see `docs/thoth-provisioning.md`
- [x] Caddy installed on Thoth (site config deferred)
- [x] `thoth-ingest.service` systemd unit (scaffold on host; impl deferred)
- [x] `thoth-api.service` systemd unit (scaffold on host; FastAPI impl deferred)
- [x] `thoth-classify.service` systemd unit (scaffold on host; classifier impl deferred)

## Phase 1 — Thoth online + first end-to-end event

- [ ] MinIO bucket `thoth` auto-created by ingest service
- [ ] Storage backend swapped from local disk → MinIO
- [ ] `thoth-ingest.service` real ExecStart (replaces scaffold)
- [ ] `thoth-api.service` real ExecStart — FastAPI read API
- [ ] Caddy serving placeholder UI
- [ ] Horus: motion-gated still → MQTT publish
- [ ] Thoth ingest: subscribe → MinIO write + SQLite event + InfluxDB point
- [ ] Thoth: TFLite classifier service (MobileNet or iNaturalist bird head)
- [ ] HA notification on high-confidence detection
- [ ] Telegram posts crop + species + confidence

## Phase 2 — Audio via BirdNET-Go

- [x] BirdNET-Go installed on Horus (systemd service, USB Lavalier @ 48kHz mono)
- [ ] BirdNET-Go native MQTT output enabled
- [ ] Thoth ingest reshapes BirdNET-Go events into `sbo/<station>/audio/detection`
- [ ] Audio clips saved to MinIO (`{station}/audio/...`)
- [ ] Audio + image fusion (same bird, both modalities)

## Phase 3 — Multi-station

- [ ] Station 2 hardware + deploy recipe
- [ ] Station 3 (hummingbird, IMX462 @ 60fps)
- [ ] Station 4 (IMX462 night duty)
- [ ] Owlsight/Hawkeye wide-scene camera

## Phase 4 — Unified outdoor viewer

- [ ] "Vantage" (working name) aggregates Thoth + Skywatch + Skylapse
  into a single timeline UI. Each backend exposes a generic `/events`
  API; Vantage is a thin frontend over all of them. See
  `docs/thoth-design.md` § "Unified outdoor events".

## Open engineering decisions

1. **TPU**: Coral USB vs. on-CPU TFLite. Banshee/Thoth CPU is probably enough.
2. **Mic placement**: on-camera vs. standalone. DCMT Lavalier is currently USB-attached to Horus (indoor bench).
3. **Photo browser**: custom (decided — lives in Thoth UI).
4. **VLAN**: separate the bird observatory nodes or not. Current plan: main LAN, revisit.
5. **PoE**: power runs for Station 2/3 outdoor placement.
6. **Multi-cam reconciliation**: the Notion doc has both single-cam and multi-cam architecture sections that need to agree.
7. **Pi 3 B+ vs. Pi 4/5**: Horus is Pi 3 B+ with 905MB. Running BirdNET-Go + picamera2 + motion gate close to the limit. Upgrade TBD.
