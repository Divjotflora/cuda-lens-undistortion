#!/usr/bin/env bash
# End-to-end demo: build, generate data, run every back end, collect proof.
#
# Environment overrides:
#   COUNT=200 WIDTH=800 HEIGHT=600     small-image dataset
#   LARGE=1 LARGE_COUNT=20 LARGE_WIDTH=2560 LARGE_HEIGHT=1440
#   STREAMS=4                          CUDA streams for the main run
#   ARCH="-arch=sm_75"                 forwarded to make
#
# Results:
#   output/corrected/        corrected images (not committed, large)
#   output/logs/*.log        full console logs of each run
#   output/metrics.csv       per-image timings and PSNR
#   output/samples/*.png     before/after comparisons
#   artifacts.tar.gz         everything above except the corrected images

set -uo pipefail
cd "$(dirname "$0")"

COUNT=${COUNT:-200}
WIDTH=${WIDTH:-800}
HEIGHT=${HEIGHT:-600}
LARGE=${LARGE:-1}
LARGE_COUNT=${LARGE_COUNT:-20}
LARGE_WIDTH=${LARGE_WIDTH:-2560}
LARGE_HEIGHT=${LARGE_HEIGHT:-1440}
STREAMS=${STREAMS:-4}

BIN=./bin/lens_undistort
LOGS=output/logs
FAILED=0

banner() { printf '\n######## %s ########\n' "$*"; }

# Runs the binary, tees the log, and records (not aborts on) failures so the
# remaining evidence is still collected.
run_step() {
  local log="$1"; shift
  banner "$log: $BIN $*"
  if ! "$BIN" "$@" 2>&1 | tee "$LOGS/$log"; then
    echo "!! step $log failed" | tee -a "$LOGS/$log"
    FAILED=1
  fi
}

banner "build"
if [[ -n "${ARCH:-}" ]]; then make ARCH="$ARCH"; else make; fi || exit 1

mkdir -p "$LOGS"
{
  echo "date: $(date -u)"
  echo "host: $(hostname)"
  nvcc --version | tail -n 2
  nvidia-smi || echo "nvidia-smi unavailable"
} > "$LOGS/environment.log" 2>&1
cat "$LOGS/environment.log"

banner "dataset: $COUNT images ${WIDTH}x${HEIGHT}"
n_existing=$(ls data/distorted/*.ppm 2>/dev/null | wc -l)
if [[ "$n_existing" -ne "$COUNT" ]]; then
  rm -rf data
  python3 scripts/make_dataset.py --out data --count "$COUNT" \
      --width "$WIDTH" --height "$HEIGHT" || exit 1
else
  echo "reusing existing data/ ($n_existing images)"
fi

COMMON=(--input_dir data/distorted --camera data/camera.txt
        --gt_dir data/ground_truth)

# 1. Main run: custom map kernel, writes every corrected image + metrics.
run_step kernel_map.log "${COMMON[@]}" --backend kernel_map \
    --streams "$STREAMS" --verify 16 --output_dir output/corrected \
    --csv output/metrics.csv

# 2. Other GPU back ends on the same data (benchmark, no image writes).
run_step kernel_fused.log "${COMMON[@]}" --backend kernel_fused \
    --streams "$STREAMS" --verify 16 --no_write
run_step npp.log "${COMMON[@]}" --backend npp \
    --streams "$STREAMS" --verify 16 --no_write

# 3. Stream-count sweep: shows the effect of overlapping copies and compute.
for s in 1 2 4 8; do
  run_step "streams_$s.log" "${COMMON[@]}" --backend kernel_map \
      --streams "$s" --verify 0 --no_write --quiet
done

# 4. Single-threaded CPU reference on a subset (baseline).
run_step cpu_reference.log "${COMMON[@]}" --backend cpu --limit 20 --no_write

# 5. A few tens of large frames.
if [[ "$LARGE" == "1" ]]; then
  banner "large dataset: $LARGE_COUNT images ${LARGE_WIDTH}x${LARGE_HEIGHT}"
  n_large=$(ls data_large/distorted/*.ppm 2>/dev/null | wc -l)
  if [[ "$n_large" -ne "$LARGE_COUNT" ]]; then
    rm -rf data_large
    python3 scripts/make_dataset.py --out data_large --count "$LARGE_COUNT" \
        --width "$LARGE_WIDTH" --height "$LARGE_HEIGHT" --prefix large \
        --no_ground_truth || exit 1
  fi
  run_step large_kernel_map.log --input_dir data_large/distorted \
      --camera data_large/camera.txt --backend kernel_map \
      --streams "$STREAMS" --verify 4 --output_dir output/corrected_large
fi

banner "before/after samples"
python3 scripts/make_contact_sheet.py --input_dir data/distorted \
    --output_dir output/corrected --gt_dir data/ground_truth \
    --out output/samples --samples 6
if [[ "$LARGE" == "1" ]]; then
  python3 scripts/make_contact_sheet.py --input_dir data_large/distorted \
      --output_dir output/corrected_large --out output/samples_large \
      --samples 2 --tile_width 640
fi

banner "summary"
grep -H -E "^(backend +:|images processed|wall time|compute time|H2D|PSNR|verify|speed-up|result)" \
    "$LOGS"/*.log | tee "$LOGS/summary.txt"

bash scripts/package_artifacts.sh

if [[ "$FAILED" -ne 0 ]]; then
  echo "One or more steps failed - see $LOGS/"
  exit 1
fi
echo "All steps finished. Proof of execution: artifacts.tar.gz"
