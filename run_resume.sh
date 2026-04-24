#!/bin/bash
# run_resume.sh — resume the pipeline from Stage 5 after Phase A+B finished.
# Assumes complete_speeds.pkl and matching_params.pkl already exist on disk.

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1

PY=/Users/stanislav/mini-google-map/.venv/bin/python

stage() {
  printf '\n================================================================\n'
  printf '[%s] %s\n' "$(date +'%H:%M:%S')" "$1"
  printf '================================================================\n'
}

START_TS=$(date +%s)

# Sanity-check the inputs we need
for f in complete_speeds.pkl sf_road_network.graphml; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f missing — can't resume"; exit 1
  fi
done

stage "5/8  Process test cases  (→ vehicle_features.pkl)  [MAX_POINTS=400]"
"$PY" process_test_cases.py

stage "6/8  Stage 3 quick_diagnostics (smoke test)"
"$PY" -c "from stage3_eta_prediction import quick_diagnostics; quick_diagnostics()"

stage "7/8  Stage 3 full training + test prediction"
"$PY" stage3_eta_prediction.py

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
printf '\n[%s] RESUME COMPLETE — elapsed %dm %ds\n' "$(date +'%H:%M:%S')" \
  "$(( ELAPSED / 60 ))" "$(( ELAPSED % 60 ))"
