# Project Status

## Phase 0 — Bootstrap (current)

- [x] IMX708 wide sensor attached and detected on Horus
- [x] Test frame captured (2304×1296 + 1280×720 proofs)
- [x] SSH key auth from k2so → k2@horus
- [x] Repo scaffold
- [ ] MQTT schema draft (`shared/schema.md`)
- [ ] Horus capture daemon stub
- [ ] systemd unit for capture daemon
- [ ] Banshee LXC container provisioned
- [ ] Banshee MQTT subscriber stub

## Phase 1 — First end-to-end event

- [ ] Horus: motion-gated still → MQTT publish
- [ ] Banshee: subscribe → InfluxDB write
- [ ] Banshee: TFLite classifier (MobileNet or iNaturalist bird head)
- [ ] HA notification on high-confidence detection
- [ ] Telegram posts crop + species + confidence

## Phase 2 — Audio

- [ ] BirdNET-Go installed on Horus
- [ ] Detection events → MQTT
- [ ] Audio clips saved to NFS on Banshee
- [ ] Audio + image fusion (same bird, both modalities)

## Phase 3 — Multi-station

- [ ] Station 2 hardware + deploy recipe
- [ ] Station 3 (hummingbird, IMX462 @ 60fps)

## Open engineering decisions

1. **TPU**: Coral USB vs. on-CPU TFLite. Banshee CPU is probably enough.
2. **Mic placement**: on-camera vs. standalone. DCMT Lavalier is currently USB-attached to Horus.
3. **Photo browser**: PhotoPrism? Immich? Custom?
4. **VLAN**: separate the bird observatory nodes or not.
5. **PoE**: power runs for Station 2/3 outdoor placement.
6. **Multi-cam reconciliation**: the Notion doc has both single-cam and multi-cam architecture sections that need to agree.
7. **Pi 3 B+ vs. Pi 4/5**: below BOM spec. Decide whether to upgrade or commit to the slim architecture.
