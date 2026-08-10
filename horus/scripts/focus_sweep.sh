#!/bin/bash
# focus_sweep.sh — Sweep IMX519 lens positions, score subject-ROI sharpness.
#
# Usage:
#   focus_sweep.sh [min] [max] [step] [outdir]
#
# ROI env vars (pixels, full-frame 3840x2160). Default = center-lower region
# where the feeder perch sits. Tune these before running, and sanity-check
# with the _roi_preview.jpg produced on the first frame.
#   ROI_X, ROI_Y, ROI_W, ROI_H
#
# Examples:
#   # Quick test sweep around current LP 2.9
#   ./focus_sweep.sh 2.7 3.1 0.1
#
#   # Full sweep, wider ROI
#   ROI_X=800 ROI_Y=700 ROI_W=2000 ROI_H=900 ./focus_sweep.sh 2.0 4.0 0.1
#
# Stops horus-capture while running, restarts on exit.

set -euo pipefail

MIN="${1:-2.4}"
MAX="${2:-3.6}"
STEP="${3:-0.1}"
OUTDIR="${4:-/tmp/focus_sweep_$(date +%Y%m%d_%H%M%S)}"

# Subject ROI (full-frame 3840x2160 pixel coords).
# Defaults target the feeder perch area — TUNE THESE if the subject moves.
ROI_X="${ROI_X:-1200}"
ROI_Y="${ROI_Y:-900}"
ROI_W="${ROI_W:-1440}"
ROI_H="${ROI_H:-720}"

WIDTH=3840
HEIGHT=2160

mkdir -p "$OUTDIR"

echo "=== focus sweep ==="
echo "range:  $MIN -> $MAX step $STEP"
echo "ROI:    x=$ROI_X y=$ROI_Y  ${ROI_W}x${ROI_H}  (of ${WIDTH}x${HEIGHT})"
echo "outdir: $OUTDIR"
echo

# --- free the camera -------------------------------------------------------
SERVICE_WAS_ACTIVE=0
if systemctl is-active --quiet horus-capture; then
  SERVICE_WAS_ACTIVE=1
  echo "stopping horus-capture..."
  sudo systemctl stop horus-capture
  sleep 2
fi

cleanup() {
  if [[ $SERVICE_WAS_ACTIVE -eq 1 ]]; then
    echo "restarting horus-capture..."
    sudo systemctl start horus-capture
  fi
}
trap cleanup EXIT

# --- sweep -----------------------------------------------------------------
FIRST=1
LP="$MIN"
while awk "BEGIN{exit !($LP <= $MAX + 1e-9)}"; do
  LPTAG=$(printf "lp%.2f" "$LP")
  FRAME="$OUTDIR/${LPTAG}.jpg"
  CROP="$OUTDIR/${LPTAG}_crop.jpg"
  META="$OUTDIR/${LPTAG}.json"

  echo -n "  $LPTAG ... "

  # rpicam-still: manual AF, set lens position, short warm-up, full-res.
  # --metadata writes capture metadata (ExposureTime, LensPosition, etc.)
  if ! rpicam-still \
      --timeout 1500 \
      --autofocus-mode manual \
      --lens-position "$LP" \
      --width "$WIDTH" --height "$HEIGHT" \
      --quality 95 \
      --metadata "$META" \
      --nopreview \
      -o "$FRAME" 2>/dev/null; then
    echo "FAILED"
    LP=$(awk "BEGIN{print $LP + $STEP}")
    continue
  fi

  # Crop subject ROI.
  ffmpeg -y -loglevel error \
    -i "$FRAME" \
    -vf "crop=${ROI_W}:${ROI_H}:${ROI_X}:${ROI_Y}" \
    -q:v 2 "$CROP"

  # Draw ROI preview on first frame so we can sanity-check box placement.
  if [[ $FIRST -eq 1 ]]; then
    ffmpeg -y -loglevel error \
      -i "$FRAME" \
      -vf "drawbox=x=${ROI_X}:y=${ROI_Y}:w=${ROI_W}:h=${ROI_H}:color=red@0.9:t=6" \
      -q:v 3 "$OUTDIR/_roi_preview.jpg"
    FIRST=0
  fi

  echo "ok"
  LP=$(awk "BEGIN{print $LP + $STEP}")
done

echo
echo "=== scoring (Laplacian variance on subject crops) ==="

# --- score + rank ----------------------------------------------------------
OUTDIR="$OUTDIR" python3 <<'PY'
import os, glob, json, re
import numpy as np
from PIL import Image

outdir = os.environ["OUTDIR"]

# 3x3 discrete Laplacian — pure numpy, no scipy/cv2.
K = np.array([[0, 1, 0],
              [1,-4, 1],
              [0, 1, 0]], dtype=np.float32)

def conv2d(img, k):
    # Reflect-pad, sum of 9 shifted views.
    p = np.pad(img, 1, mode="reflect")
    h, w = img.shape
    out = np.zeros_like(img)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            kv = k[dy+1, dx+1]
            if kv == 0: continue
            out += kv * p[1+dy:1+dy+h, 1+dx:1+dx+w]
    return out

def lap_var(path):
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    lap = conv2d(img, K)
    return float(lap.var())

rows = []
for crop in sorted(glob.glob(os.path.join(outdir, "lp*_crop.jpg"))):
    m = re.search(r"lp([0-9.]+)_crop\.jpg$", crop)
    if not m: continue
    lp = float(m.group(1))
    try:
        score = lap_var(crop)
    except Exception as e:
        score = float("nan")

    # Pull ExposureTime / FocusFoM from sidecar if present.
    meta_path = crop.replace("_crop.jpg", ".json")
    et = fom = None
    if os.path.exists(meta_path):
        try:
            md = json.load(open(meta_path))
            et = md.get("ExposureTime")
            fom = md.get("FocusFoM")
        except Exception:
            pass
    rows.append((lp, score, et, fom, crop))

# Ranked print (best first).
rows_sorted = sorted(rows, key=lambda r: (float("-inf") if np.isnan(r[1]) else -r[1]))
print(f"{'LP':>5}  {'ROI-Laplacian':>13}  {'Exp µs':>7}  {'FocusFoM':>8}")
for lp, s, et, fom, _ in rows_sorted:
    print(f"{lp:5.2f}  {s:13.1f}  {str(et or '-'):>7}  {str(fom or '-'):>8}")

# CSV for later plotting.
csv_path = os.path.join(outdir, "ranking.csv")
with open(csv_path, "w") as f:
    f.write("lp,roi_laplacian_var,exposure_us,focus_fom,crop_path\n")
    for lp, s, et, fom, p in sorted(rows):
        f.write(f"{lp},{s},{et or ''},{fom or ''},{p}\n")
print(f"\nCSV: {csv_path}")
print(f"ROI preview: {outdir}/_roi_preview.jpg")

if rows_sorted:
    best_lp, best_s, *_ = rows_sorted[0]
    print(f"\nBest LP by subject-ROI sharpness: {best_lp:.2f}  (score {best_s:.1f})")
PY

echo
echo "Done. Review: $OUTDIR"
