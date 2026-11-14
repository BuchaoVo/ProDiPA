#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/conditional_evaluation"
OUT="${EVALUATION_OUTPUT_ROOT:-$ROOT/outputs/evaluation/teacher_generation_evaluation}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
PROTEINOPD="${PROTEINOPD_ROOT:-__PRODIPA_ROOT__/third_party/ProteinOPD}"
PARALLEL_GPUS="${PARALLEL_GPUS:-0,1,2,3}"
MIN_TOTAL_MIB="${MIN_TOTAL_MIB:-70000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-15}"
STABLE_POLLS="${STABLE_POLLS:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SHARDS="${ESMFOLD_SHARDS:-16}"
HANDOFF_PID="${HANDOFF_PID:-}"
HANDOFF_TASK="${HANDOFF_TASK:-}"

IFS=',' read -r -a GPU_IDS <<<"$PARALLEL_GPUS"
(( ${#GPU_IDS[@]} > 0 )) || { echo "PARALLEL_GPUS is empty" >&2; exit 2; }

mkdir -p "$OUT/logs" "$OUT/.done" "$OUT/parallel_status"
exec 7>"$OUT/.parallel_queue.lock"
flock -n 7 || { echo "another four-GPU scheduler is active" >&2; exit 1; }
exec > >(tee -a "$OUT/parallel_gpu_queue.log") 2>&1

CURRENT_STAGE="handoff"
declare -a ACTIVE_CHILDREN=()

write_status() {
  printf '{\n  "state": "%s",\n  "stage": "%s",\n  "gpus": "%s",\n  "updated_at": "%s"\n}\n' \
    "$1" "$2" "$PARALLEL_GPUS" "$(date --iso-8601=seconds)" \
    > "$OUT/PARALLEL_QUEUE_STATUS.json"
}

stop_children() {
  local pid
  for pid in "${ACTIVE_CHILDREN[@]:-}"; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${ACTIVE_CHILDREN[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
}

on_stop() { write_status STOPPED "$CURRENT_STAGE"; stop_children; exit 143; }
on_error() {
  local rc=$?
  write_status FAILED "$CURRENT_STAGE"
  stop_children
  echo "four-GPU evaluation failed stage=$CURRENT_STAGE rc=$rc"
  exit "$rc"
}
trap on_stop INT TERM
trap on_error ERR

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" '
    {for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4","$5}'
}

wait_for_gpu_lock() {
  local gpu="$1" task_hint="$2" stable=0 row index uuid total free util
  while true; do
    row="$(gpu_row "$gpu")"
    if [[ -z "$row" ]]; then
      stable=0
      sleep "$POLL_SECONDS"
      continue
    fi
    IFS=',' read -r index uuid total free util <<<"$row"
    if (( total >= MIN_TOTAL_MIB && free >= MIN_FREE_MIB && util <= MAX_UTIL_PCT )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    printf '{"state":"WAITING_FOR_IDLE_GPU","gpu":%s,"task_hint":"%s","free_mib":%s,"util_pct":%s,"stable":%s,"required":%s,"updated_at":"%s"}\n' \
      "$gpu" "$task_hint" "$free" "$util" "$stable" "$STABLE_POLLS" \
      "$(date --iso-8601=seconds)" > "$OUT/parallel_status/gpu_${gpu}.json"
    echo "[$(date --iso-8601=seconds)] GPU=$gpu task_hint=$task_hint free=$free util=$util stable=$stable/$STABLE_POLLS"
    if (( stable < STABLE_POLLS )); then sleep "$POLL_SECONDS"; continue; fi

    exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
    if ! flock -n 9; then
      stable=0
      sleep "$POLL_SECONDS"
      continue
    fi
    row="$(gpu_row "$gpu")"
    IFS=',' read -r index uuid total free util <<<"$row"
    if (( total < MIN_TOTAL_MIB || free < MIN_FREE_MIB || util > MAX_UTIL_PCT )); then
      flock -u 9; exec 9>&-
      stable=0
      sleep "$POLL_SECONDS"
      continue
    fi
    SELECTED_UUID="$uuid"
    return 0
  done
}

release_gpu_lock() { flock -u 9 2>/dev/null || true; exec 9>&-; }

claim_task() {
  local queue_file="$1" claim_lock="$2" task="" tmp
  exec 6>"$claim_lock"
  flock 6
  if [[ -s "$queue_file" ]]; then
    task="$(head -n 1 "$queue_file")"
    tmp="${queue_file}.tmp.$$"
    tail -n +2 "$queue_file" > "$tmp"
    mv "$tmp" "$queue_file"
  fi
  flock -u 6; exec 6>&-
  printf '%s' "$task"
}

gpu_worker() {
  local gpu="$1" queue_file="$2" claim_lock="$3" stage="$4" task rc
  while [[ -s "$queue_file" ]]; do
    wait_for_gpu_lock "$gpu" "$stage"
    task="$(claim_task "$queue_file" "$claim_lock")"
    if [[ -z "$task" ]]; then release_gpu_lock; return 0; fi
    printf '{"state":"RUNNING","gpu":%s,"task":"%s","uuid":"%s","updated_at":"%s"}\n' \
      "$gpu" "$task" "$SELECTED_UUID" "$(date --iso-8601=seconds)" \
      > "$OUT/parallel_status/gpu_${gpu}.json"
    echo "[$(date --iso-8601=seconds)] GPU=$gpu uuid=$SELECTED_UUID starting task=$task"
    set +e
    CUDA_VISIBLE_DEVICES="$SELECTED_UUID" bash "$HERE/run_one_task.sh" "$task" \
      > >(tee -a "$OUT/logs/$task.log") 2>&1
    rc=$?
    set -e
    release_gpu_lock
    if (( rc != 0 )); then
      printf '{"state":"FAILED","gpu":%s,"task":"%s","rc":%s,"updated_at":"%s"}\n' \
        "$gpu" "$task" "$rc" "$(date --iso-8601=seconds)" \
        > "$OUT/parallel_status/gpu_${gpu}.json"
      return "$rc"
    fi
    printf '{"state":"TASK_COMPLETE","gpu":%s,"task":"%s","updated_at":"%s"}\n' \
      "$gpu" "$task" "$(date --iso-8601=seconds)" > "$OUT/parallel_status/gpu_${gpu}.json"
    echo "[$(date --iso-8601=seconds)] GPU=$gpu finished task=$task"
  done
}

write_pending_queue() {
  local queue_file="$1" task; shift
  : > "$queue_file"
  for task in "$@"; do
    [[ -s "$OUT/.done/$task" ]] || printf '%s\n' "$task" >> "$queue_file"
  done
}

run_gpu_stage() {
  local stage="$1" queue_file claim_lock gpu pid failed=0; shift
  queue_file="$OUT/.${stage}_pending"
  claim_lock="$OUT/.${stage}_claim.lock"
  write_pending_queue "$queue_file" "$@"
  if [[ ! -s "$queue_file" ]]; then echo "stage=$stage already complete"; return 0; fi
  CURRENT_STAGE="$stage"
  write_status RUNNING "$stage"
  ACTIVE_CHILDREN=()
  for gpu in "${GPU_IDS[@]}"; do
    gpu_worker "$gpu" "$queue_file" "$claim_lock" "$stage" &
    ACTIVE_CHILDREN+=("$!")
  done
  for pid in "${ACTIVE_CHILDREN[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  ACTIVE_CHILDREN=()
  (( failed == 0 ))
}

run_cpu_task() {
  local task="$1"
  [[ -s "$OUT/.done/$task" ]] && return 0
  bash "$HERE/run_one_task.sh" "$task" > >(tee -a "$OUT/logs/$task.log") 2>&1
}

write_status WAITING_FOR_HANDOFF "$CURRENT_STAGE"
if [[ -n "$HANDOFF_PID" ]] && kill -0 "$HANDOFF_PID" 2>/dev/null; then
  if [[ -n "$HANDOFF_TASK" ]]; then
    echo "waiting for active task boundary: task=$HANDOFF_TASK scheduler_pid=$HANDOFF_PID"
    while kill -0 "$HANDOFF_PID" 2>/dev/null && [[ ! -s "$OUT/.done/$HANDOFF_TASK" ]]; do sleep 15; done
  fi
  if kill -0 "$HANDOFF_PID" 2>/dev/null; then
    echo "stopping old single-GPU scheduler pid=$HANDOFF_PID"
    kill -TERM "$HANDOFF_PID"
    for _ in $(seq 1 30); do kill -0 "$HANDOFF_PID" 2>/dev/null || break; sleep 1; done
  fi
fi

# Share the original scheduler lock so task assignment cannot overlap.
exec 8>"$OUT/.queue.lock"
while ! flock -n 8; do echo "waiting for old scheduler lock"; sleep 5; done

cd "$ROOT"
CURRENT_STAGE="preflight"
write_status RUNNING "$CURRENT_STAGE"
preflight_args=(--project-root "$ROOT" --proteinopd-root "$PROTEINOPD" --out "$OUT/preflight.json")
[[ -n "${UNIPROTKB_FASTA:-}" ]] && preflight_args+=(--uniprotkb-fasta "$UNIPROTKB_FASTA")
[[ -n "${PROTREK_ROOT:-}" ]] && preflight_args+=(--protrek-root "$PROTREK_ROOT")
[[ -n "${PROTREK_MODEL:-}" ]] && preflight_args+=(--protrek-model "$PROTREK_MODEL")
"$PY" "$HERE/preflight.py" "${preflight_args[@]}"

run_gpu_stage generation \
  generation_unconditional_base generation_unconditional_fold_teacher \
  generation_unconditional_sol_teacher generation_unconditional_thermo_teacher \
  generation_conditional_base generation_conditional_fold_teacher \
  generation_conditional_sol_teacher generation_conditional_thermo_teacher

CURRENT_STAGE="prepare"; write_status RUNNING "$CURRENT_STAGE"
run_cpu_task prepare
CURRENT_STAGE="cpu_scoring"; write_status RUNNING "$CURRENT_STAGE"
run_cpu_task protein_sol
run_cpu_task novelty_t_unconditional
run_cpu_task novelty_t_conditional

gpu_score_tasks=(protgpt2_ppl temberture)
for ((i=0; i<SHARDS; i++)); do gpu_score_tasks+=("esmfold_$i"); done
run_gpu_stage gpu_scoring "${gpu_score_tasks[@]}"

CURRENT_STAGE="merge_structure"; write_status RUNNING "$CURRENT_STAGE"
run_cpu_task merge_structure
if [[ -n "${UNIPROTKB_FASTA:-}" && -n "${UNIPROTKB_RELEASE:-}" ]]; then
  CURRENT_STAGE="novelty_u"; write_status RUNNING "$CURRENT_STAGE"; run_cpu_task novelty_u
fi
CURRENT_STAGE="summarize"; write_status RUNNING "$CURRENT_STAGE"
run_cpu_task summarize

write_status AVAILABLE_STAGES_COMPLETE complete
printf '{"status":"AVAILABLE_STAGES_COMPLETE","completed_at":"%s","gpus":"%s","pending":["ProTrek unless configured","Novelty-U and pooled HV unless full UniProtKB is configured"]}\n' \
  "$(date --iso-8601=seconds)" "$PARALLEL_GPUS" > "$OUT/AVAILABLE_STAGES_DONE.json"
echo "all currently executable evaluation stages completed"
