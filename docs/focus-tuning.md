# Focus tuning (IMX519 manual LP)

The picamera2 path on Horus locks the lens to a fixed `LensPosition`
(diopters, `1 / distance_in_meters`). The rpicam `--autofocus-*` flags
in `rpicam_extra_args` are **ignored** on the picamera2 path — focus is
set via `picam2.set_controls({'AfMode': Manual, 'LensPosition': …})`.

Subject distance varies per station (feeder geometry, mount height,
perch position), so the correct LP must be measured per install.
This doc describes how.

## Procedure

1. **Place a printed test card** (high-contrast, small text — a
   black-on-white eye chart or similar) at the exact distance the
   birds will be when they're the subject. The card must stand up
   in the frame, not lie flat on the feeder roof.

2. **SSH to the station** and run the sweep:

   ```sh
   # Default: 2.4–3.6 step 0.1, ROI centered on feeder perch.
   sudo ./focus_sweep.sh

   # Wider range, custom ROI over the card position.
   ROI_X=1500 ROI_Y=400 ROI_W=1200 ROI_H=1000 \
     ./focus_sweep.sh 2.0 7.0 0.5
   ```

   The script stops `horus-capture` (sudoers drop-in required —
   see `horus/deploy/` notes), sweeps manual LP across the range,
   crops a subject ROI from each frame, and scores sharpness via
   Laplacian variance. It also writes `_roi_preview.jpg` showing
   the ROI box drawn on the first frame — **always check this**.

3. **Verify the ROI actually covers the card.** If the red box in
   `_roi_preview.jpg` is below/above/beside the card, the ranked
   scores are measuring background texture, not subject focus.
   Re-run with corrected `ROI_X/Y/W/H`.

4. **Narrow in.** After the wide sweep picks a peak, run a tight
   step (0.1) around it to pick the exact best LP.

5. **Eyeball the winning crop.** Laplacian variance rewards edge
   energy; a metallic feeder rim can outscore slightly-soft text.
   Open the top 2–3 crops and pick the one where small text is
   cleanly readable. Metric is a ranking aid, not the final word.

6. **Commit to config.** Edit `/etc/horus/horus.yaml`:

   ```yaml
   capture:
     lens_position: 5.3   # your measured value
   ```

   Restart `horus-capture`:

   ```sh
   sudo systemctl restart horus-capture
   ```

## Outputs

The sweep writes to `/tmp/focus_sweep_<timestamp>/`:

- `lpX.XX.jpg` — full-frame capture at each LP
- `lpX.XX_crop.jpg` — ROI crop used for scoring
- `lpX.XX.json` — libcamera capture metadata (ExposureTime,
  LensPosition, FocusFoM)
- `_roi_preview.jpg` — first frame with the ROI box drawn
- `ranking.csv` — one row per LP for later plotting

The script prints a ranked table (highest Laplacian variance first)
when it finishes.

## Common pitfalls

**ROI placement matters more than sweep range.** A correct sweep
with the ROI on the wrong thing returns confidently-wrong answers.
The whole feeder roof and rim are high-contrast edges that remain
"sharp" across a wide LP range; only the card text discriminates.

**`FocusFoM` is whole-frame.** libcamera's built-in sharpness metric
reads the entire sensor, so background twigs and feeder texture can
win over a soft subject. The script scores the cropped ROI instead,
which is why it disagrees with `FocusFoM` when focus is far off.

**Fallen cards lie.** If the card tips forward onto the feeder roof
between runs, its surface is now near-perpendicular to the optical
axis instead of facing the camera. LP optimum shifts to whatever is
actually vertical in the frame (usually the back wall). Check the
`_roi_preview.jpg` before trusting results.

**Lens re-homing.** `rpicam-still --autofocus-mode manual
--lens-position X` re-seats the VCM actuator on every call. That's
fine for a sweep (we want each shot independent) but means results
from back-to-back single shots can be more consistent than results
from interleaved sweeps. If scores look noisy, re-run.

## AF instead of manual LP?

IMX519 supports PDAF+CDAF autofocus. libcamera's `AfRange` control
exposes three presets only — `Normal`, `Macro`, `Full` — not an
arbitrary `[min, max]` diopter clamp. Options if AF is worth trying:

1. **AF with a tight `AfWindow`** over just the card/perch region
   — cheapest, lets AF pick the range naturally.
2. **Run AF once at startup, then `AfMode=Manual` lock** to the
   converged `LensPosition` — one-shot calibration.
3. **Continuous AF + clamp in code** — read `LensPosition` each
   frame, `set_controls({'LensPosition': clamp(lp, lo, hi)})` if
   it drifts outside the acceptable range.

Manual LP is the simplest deployment and is what ships today.
