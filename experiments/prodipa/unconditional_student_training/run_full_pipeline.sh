#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/unconditional/protgpt2_multiteacher_student}"
EVAL_DIR="${EVAL_DIR:-$ROOT/outputs/evaluation/unconditional/methods/protgpt2_multiteacher_student}"
BASE_MODEL="${PROTGPT2_MODEL:-__PRODIPA_MODELS__/ProtGPT2}"
TRAINING_GPUS="${TRAINING_GPUS:-0,1}"
EVALUATION_GPUS="${EVALUATION_GPUS:-0,1,2,3}"
GENERATOR="$ROOT/experiments/prodipa/conditional_evaluation/generate_model.py"
MERGER="$ROOT/experiments/prodipa/conditional_multiteacher_student/merge_generation_shards.py"
SHARD_DIR="$OUTPUT_DIR/generation_shards"
LOG_DIR="$OUTPUT_DIR/logs"
STATUS="$OUTPUT_DIR/PIPELINE_STATUS.json"

mkdir -p "$OUTPUT_DIR" "$SHARD_DIR" "$LOG_DIR"
exec 9>"$OUTPUT_DIR/.full_pipeline.lock"
flock -n 9 || { echo "Another unconditional student pipeline is active" >&2; exit 2; }

write_status() {
  local state="$1"
  local stage="$2"
  local message="${3:-}"
  STATE="$state" STAGE="$stage" MESSAGE="$message" STATUS="$STATUS" "$PY" - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

Path(os.environ["STATUS"]).write_text(
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
PY
}

on_error() {
  local exit_code=$?
  write_status "FAILED" "${CURRENT_STAGE:-unknown}" "pipeline exited with code $exit_code"
  exit "$exit_code"
}
trap on_error ERR INT TERM

cd "$ROOT"

CURRENT_STAGE="training"
write_status "RUNNING" "$CURRENT_STAGE" "distilling three frozen Prefix teachers"
OUTPUT_DIR="$OUTPUT_DIR" TRAINING_GPUS="$TRAINING_GPUS" \
  bash experiments/prodipa/unconditional_student_training/run_training.sh
test -s "$OUTPUT_DIR/DONE.json"

CURRENT_STAGE="generation"
write_status "RUNNING" "$CURRENT_STAGE" "four fixed-seed generation shards"
IFS=',' read -r -a generation_gpus <<< "$EVALUATION_GPUS"
if [[ "${#generation_gpus[@]}" -ne 4 ]]; then
  echo "EVALUATION_GPUS must contain exactly four physical GPU indices" >&2
  exit 2
fi

pids=()
for shard in 0 1 2 3; do
  start=$((shard * 128))
  stop=$(((shard + 1) * 128))
  gpu="${generation_gpus[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" "$GENERATOR" \
      --paradigm unconditional \
      --model-name protgpt2_multiteacher_student \
      --base-model "$BASE_MODEL" \
      --adapter "$OUTPUT_DIR" \
      --out "$SHARD_DIR/shard_${shard}.tsv" \
      --num-sequences 512 \
      --sample-id-start "$start" \
      --sample-id-stop "$stop" \
      --base-seed 42 \
      --temperature 0.7 \
      --top-p 0.9 \
      --top-k 200 \
      --repetition-penalty 1.2 \
      --max-new-tokens 512 \
      --sequence-format legacy \
      --generation-max-aa 512 \
      --device cuda \
      > "$LOG_DIR/generation_shard_${shard}.log" 2>&1
  ) &
  pids+=("$!")
done

generation_failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || generation_failed=1
done
if ((generation_failed)); then
  echo "At least one generation shard failed" >&2
  exit 1
fi

"$PY" "$MERGER" \
  --input-dir "$SHARD_DIR" \
  --output "$OUTPUT_DIR/generation.tsv" \
  --num-sequences 512 \
  --num-shards 4

CURRENT_STAGE="evaluation"
write_status "RUNNING" "$CURRENT_STAGE" "formal scoring and pooled cohort publication"
PARALLEL_GPUS="$EVALUATION_GPUS" \
  bash experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh \
    --model-id protgpt2_multiteacher_student \
    --generation-tsv "$OUTPUT_DIR/generation.tsv" \
    --output-root "$EVAL_DIR" \
    --gpus "$EVALUATION_GPUS" \
    --publish

test -s "$EVAL_DIR/METHOD_EVALUATION_COMPLETE.json"
test -s "$ROOT/outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv"
CURRENT_STAGE="complete"
write_status "COMPLETE" "$CURRENT_STAGE" "training, generation, scoring and pooled publication complete"
trap - ERR INT TERM
echo "Unconditional multi-teacher student pipeline complete"
