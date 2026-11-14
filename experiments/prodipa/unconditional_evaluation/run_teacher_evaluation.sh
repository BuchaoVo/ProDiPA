#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/conditional_evaluation"
OUT="${EVALUATION_OUTPUT_ROOT:-$ROOT/outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers}"
GPUS="${PARALLEL_GPUS:-0,1,2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-28000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-100}"
STABLE_POLLS="${STABLE_POLLS:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVALUATE_BASE="${EVALUATE_BASE:-1}"

IFS=',' read -r -a GPU_IDS <<<"$GPUS"
(( ${#GPU_IDS[@]} >= 1 )) || { echo "PARALLEL_GPUS is empty" >&2; exit 2; }

mkdir -p "$OUT/logs" "$OUT/.done"
exec 7>"$OUT/.teacher_evaluation.lock"
flock -n 7 || { echo "another unconditional teacher evaluation is active" >&2; exit 1; }
exec > >(tee -a "$OUT/teacher_evaluation.log") 2>&1

status() {
  printf '{"state":"%s","stage":"%s","gpus":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$GPUS" "$(date --iso-8601=seconds)" > "$OUT/TEACHER_EVALUATION_STATUS.json"
}

declare -a CHILDREN=()
stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${CHILDREN[@]:-}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
}
on_stop() { status STOPPED generation; stop_children; exit 143; }
on_error() { local rc=$?; status FAILED pipeline; stop_children; exit "$rc"; }
trap on_stop INT TERM
trap on_error ERR

tasks=(
  generation_unconditional_fold_teacher
  generation_unconditional_sol_teacher
  generation_unconditional_thermo_teacher
)
if [[ "$EVALUATE_BASE" == "1" ]]; then
  tasks=(generation_unconditional_base "${tasks[@]}")
elif [[ "$EVALUATE_BASE" != "0" ]]; then
  echo "EVALUATE_BASE must be 0 or 1, got: $EVALUATE_BASE" >&2
  exit 2
fi

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4","$5}'
}

claim_task() {
  local task="" temporary
  exec 6>"$OUT/.generation_claim.lock"; flock 6
  if [[ -s "$OUT/.generation_pending" ]]; then
    task="$(head -n1 "$OUT/.generation_pending")"
    temporary="$OUT/.generation_pending.tmp.$$"
    tail -n +2 "$OUT/.generation_pending" > "$temporary"
    mv "$temporary" "$OUT/.generation_pending"
  fi
  flock -u 6; exec 6>&-
  printf '%s' "$task"
}

generation_worker() {
  local gpu="$1" row index uuid total free util stable=0 task rc
  while [[ -s "$OUT/.generation_pending" ]]; do
    row="$(gpu_row "$gpu")"
    if [[ -z "$row" ]]; then sleep "$POLL_SECONDS"; continue; fi
    IFS=',' read -r index uuid total free util <<<"$row"
    if (( free >= MIN_FREE_MIB && util <= MAX_UTIL_PCT )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    printf '{"state":"WAITING","gpu":%s,"free_mib":%s,"util_pct":%s,"stable":%s,"required":%s,"updated_at":"%s"}\n' \
      "$gpu" "$free" "$util" "$stable" "$STABLE_POLLS" "$(date --iso-8601=seconds)" \
      > "$OUT/generation_gpu_${gpu}.json"
    if (( stable < STABLE_POLLS )); then sleep "$POLL_SECONDS"; continue; fi

    exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
    if ! flock -n 9; then stable=0; sleep "$POLL_SECONDS"; continue; fi
    row="$(gpu_row "$gpu")"; IFS=',' read -r index uuid total free util <<<"$row"
    if (( free < MIN_FREE_MIB || util > MAX_UTIL_PCT )); then
      flock -u 9; exec 9>&-; stable=0; sleep "$POLL_SECONDS"; continue
    fi
    task="$(claim_task)"
    if [[ -z "$task" ]]; then flock -u 9; exec 9>&-; return 0; fi
    echo "[$(date --iso-8601=seconds)] gpu=$gpu free=$free starting $task"
    set +e
    CUDA_VISIBLE_DEVICES="$uuid" EVALUATION_OUTPUT_ROOT="$OUT" \
      bash "$HERE/run_one_task.sh" "$task" \
      > >(tee -a "$OUT/logs/$task.log") 2>&1
    rc=$?
    set -e
    flock -u 9; exec 9>&-
    if (( rc != 0 )); then
      printf '%s\n' "$task" >> "$OUT/.generation_failed"
      return "$rc"
    fi
    echo "[$(date --iso-8601=seconds)] gpu=$gpu finished $task"
  done
}

: > "$OUT/.generation_pending"
for task in "${tasks[@]}"; do
  [[ -s "$OUT/.done/$task" ]] || printf '%s\n' "$task" >> "$OUT/.generation_pending"
done
status WAITING generation
failed=0
for gpu in "${GPU_IDS[@]}"; do generation_worker "$gpu" & CHILDREN+=("$!"); done
for pid in "${CHILDREN[@]:-}"; do if ! wait "$pid"; then failed=1; fi; done
CHILDREN=()
(( failed == 0 )) || { status FAILED generation; exit 1; }

status RUNNING scoring
export GENERATION_ROOT="$OUT/generation"
export PARADIGM_EVALUATION_OUTPUT_ROOT="$OUT"
export PARALLEL_GPUS="$GPUS"
export MIN_FREE_MIB MAX_UTIL_PCT STABLE_POLLS POLL_SECONDS
bash "$HERE/run_separate_paradigm_scoring_queue.sh" unconditional

status COMPLETE complete
printf '{"status":"COMPLETE","completed_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$OUT/TEACHER_EVALUATION_COMPLETE.json"
