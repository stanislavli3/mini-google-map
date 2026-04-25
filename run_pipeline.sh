#!/bin/bash
# run_pipeline.sh — end-to-end orchestration for Stage 1 → 3.
# All output goes to stdout; callers should redirect to a log file.

set -euo pipefail

cd "$(dirname "$0")"

# Force Python to flush print() immediately so pipeline.log shows live
# progress instead of dumping each subprocess's output only at exit.
export PYTHONUNBUFFERED=1

PY=/Users/stanislav/mini-google-map/.venv/bin/python

stage() {
  printf '\n================================================================\n'
  printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$1"
  printf '================================================================\n'
}

finish() {
  printf '\n[%s] %s\n' "$(date +'%H:%M:%S')" "$1"
}

START_TS=$(date +%s)

stage "1/8  Build sf_road_network.graphml (no-op if exists)"
"$PY" src/stage1_graph/build_graphml.py

stage "2/8  Baseline metrics on 20 trajectory files"
"$PY" src/stage2_matching/baseline_metrics.py --files 20 --out baseline_metrics.txt

stage "3/8  Calibrate sigma_z and beta on 50 files"
"$PY" src/stage2_matching/calibrate_params.py --files 50

stage "4/8  Stage 2 Phase A+B  (map matching → complete_speeds.pkl)"
# 4 workers, clear any stale per-file match cache so the calibrated
# sigma_z / beta actually take effect.
"$PY" src/stage2_matching/run_stage2_full.py --workers 4 --clear-match-cache

stage "5/8  Process test cases  (→ vehicle_features.pkl)"
"$PY" src/stage2_matching/process_test_cases.py

stage "6/8  Stage 3 quick_diagnostics (smoke test)"
"$PY" -c "
import sys, os
sys.path.insert(0, 'src/stage3_eta')
sys.path.insert(0, 'src')
import _path_bootstrap  # noqa
from stage3_eta_prediction import quick_diagnostics
quick_diagnostics()
"

stage "7/8  Stage 3 full training + test prediction"
"$PY" src/stage3_eta/stage3_eta_prediction.py

stage "8/8  Verify submission.csv"
if [ ! -f submission.csv ]; then
  echo "ERROR: submission.csv not produced"
  exit 1
fi
ROWS=$(wc -l < submission.csv)
echo "submission.csv: $ROWS lines (expected 385 = 384 rows + header)"
head -3 submission.csv
echo "..."
tail -3 submission.csv

ELAPSED=$(( $(date +%s) - START_TS ))
finish "PIPELINE COMPLETE — elapsed $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
