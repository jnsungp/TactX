#!/bin/bash
# Run temporal analysis on the three 20260408 checkpoints (B32, B64, B128)

set -e

cd "$(dirname "$0")"
export PYTHONPATH=.

CHECKPOINT_ROOT=~/cross_modal_tactile/checkpoints
FRAME_GAPS="10 20 30 40"
DEVICE="cuda:0"

RUNS=(
  "dataset_cross_modal_run_20260408_213021_B128"
  "dataset_cross_modal_run_20260408_223408_B64"
  "dataset_cross_modal_run_20260408_224726_B32"
)

for run in "${RUNS[@]}"; do
  echo ""
  echo "############################################################"
  echo "# $run"
  echo "############################################################"
  python scripts/temporal_analysis.py \
    --run_name "$run" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --frame_gaps $FRAME_GAPS \
    --device "$DEVICE"
done

echo ""
echo "All done."
