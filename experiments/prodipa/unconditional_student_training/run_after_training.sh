#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/unconditional/protgpt2_multiteacher_student}"
EVALUATION_GPUS="${EVALUATION_GPUS:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STATUS="$OUTPUT_DIR/POST_TRAINING_STATUS.json"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.post_training_pipeline.lock"
flock -n 9 || { echo "Another post-training pipeline is active" >&2; exit 2; }

write_status() {
  local state="$1"
  local stage="$2"
  local message="${3:-}"
  STATE="$state" STAGE="$stage" MESSAGE="$message" STATUS="$STATUS" "$PY" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

path = Path(os.environ["STATUS"])
temporary = path.with_suffix(".tmp")
temporary.write_text(
    json.dumps(
        {
            "status": os.environ["STATE"],
            "stage": os.environ["STAGE"],
            "message": os.environ["MESSAGE"],
            "updated_at": datetime.now().astimezone().isoformat(),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
}

fail() {
  local exit_code=$?
  write_status "FAILED" "${CURRENT_STAGE:-unknown}" "post-training pipeline exited with code $exit_code"
  exit "$exit_code"
}
trap fail ERR INT TERM

CURRENT_STAGE="waiting_for_training"
write_status "WAITING" "$CURRENT_STAGE" "waiting for a successful DONE.json"
while [[ ! -s "$OUTPUT_DIR/DONE.json" ]]; do
  if [[ -s "$OUTPUT_DIR/resume_training.pid" ]]; then
    training_pid="$(tr -dc '0-9' < "$OUTPUT_DIR/resume_training.pid")"
    if [[ -n "$training_pid" ]] && ! kill -0 "$training_pid" 2>/dev/null; then
      echo "Training process $training_pid exited without producing DONE.json" >&2
      exit 1
    fi
  fi
  sleep "$POLL_SECONDS"
done

# DONE.json is written just before the training process exits. Wait until its
# lock is released so the canonical pipeline can verify training idempotently.
while ! flock -n "$OUTPUT_DIR/.training.lock" true; do
  sleep 5
done

CURRENT_STAGE="generation_and_evaluation"
write_status "RUNNING" "$CURRENT_STAGE" "starting generation, formal scoring, and canonical publication"
cd "$ROOT"
OUTPUT_DIR="$OUTPUT_DIR" EVALUATION_GPUS="$EVALUATION_GPUS" \
  bash experiments/prodipa/unconditional_student_training/run_full_pipeline.sh

CURRENT_STAGE="verification"
test -s "$OUTPUT_DIR/PIPELINE_STATUS.json"
test -s "$ROOT/outputs/evaluation/unconditional/methods/protgpt2_multiteacher_student/METHOD_EVALUATION_COMPLETE.json"
MAIN_TABLE="$ROOT/outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv"
test -s "$MAIN_TABLE"
MAIN_TABLE="$MAIN_TABLE" "$PY" - <<'PY'
import csv
import os

with open(os.environ["MAIN_TABLE"], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
models = {
    next((value.strip() for key, value in row.items() if key.strip().lower() == "model"), "")
    for row in rows
}
if "protgpt2_multiteacher_student" not in models:
    raise SystemExit("published main table is missing protgpt2_multiteacher_student")
PY

CURRENT_STAGE="complete"
write_status "COMPLETE" "$CURRENT_STAGE" "generation, scoring, audit, and main-table publication complete"
trap - ERR INT TERM
echo "Unconditional student post-training pipeline complete"
