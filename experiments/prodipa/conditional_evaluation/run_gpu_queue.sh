#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/conditional_evaluation"
OUT="${EVALUATION_OUTPUT_ROOT:-$ROOT/outputs/evaluation/teacher_generation_evaluation}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
PROTEINOPD="${PROTEINOPD_ROOT:-__PRODIPA_ROOT__/third_party/ProteinOPD}"
MIN_TOTAL_MIB="${MIN_TOTAL_MIB:-70000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-15}"
STABLE_POLLS="${STABLE_POLLS:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ALLOWED_GPUS="${ALLOWED_GPUS:-}"
SHARDS="${ESMFOLD_SHARDS:-16}"

mkdir -p "$OUT/logs" "$OUT/.done"
exec 8>"$OUT/.queue.lock"
if ! flock -n 8; then echo "another teacher evaluation queue is active" >&2; exit 1; fi
exec > >(tee -a "$OUT/gpu_queue.log") 2>&1

write_status() {
  printf '{\n  "state": "%s",\n  "task": "%s",\n  "updated_at": "%s"\n}\n' \
    "$1" "$2" "$(date --iso-8601=seconds)" > "$OUT/QUEUE_STATUS.json"
}

CURRENT_TASK=preflight
CHILD_PID=""
stop_queue() {
  write_status STOPPED "$CURRENT_TASK"
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  exit 143
}
on_error() {
  local rc=$?
  write_status FAILED "$CURRENT_TASK"
  echo "failed task=$CURRENT_TASK rc=$rc"
  exit "$rc"
}
trap stop_queue INT TERM
trap on_error ERR

cd "$ROOT"
preflight_args=(
  --project-root "$ROOT" --proteinopd-root "$PROTEINOPD"
  --out "$OUT/preflight.json"
)
[[ -n "${UNIPROTKB_FASTA:-}" ]] && preflight_args+=(--uniprotkb-fasta "$UNIPROTKB_FASTA")
[[ -n "${PROTREK_ROOT:-}" ]] && preflight_args+=(--protrek-root "$PROTREK_ROOT")
[[ -n "${PROTREK_MODEL:-}" ]] && preflight_args+=(--protrek-model "$PROTREK_MODEL")
"$PY" "$HERE/preflight.py" "${preflight_args[@]}"

gpu_tasks=(
  generation_unconditional_base
  generation_unconditional_fold_teacher
  generation_unconditional_sol_teacher
  generation_unconditional_thermo_teacher
  generation_conditional_base
  generation_conditional_fold_teacher
  generation_conditional_sol_teacher
  generation_conditional_thermo_teacher
  protgpt2_ppl
  temberture
)
for ((i=0; i<SHARDS; i++)); do gpu_tasks+=("esmfold_$i"); done

next_task() {
  local task
  for task in \
    generation_unconditional_base generation_unconditional_fold_teacher \
    generation_unconditional_sol_teacher generation_unconditional_thermo_teacher \
    generation_conditional_base generation_conditional_fold_teacher \
    generation_conditional_sol_teacher generation_conditional_thermo_teacher \
    prepare protein_sol novelty_t_unconditional novelty_t_conditional \
    protgpt2_ppl temberture; do
    [[ -s "$OUT/.done/$task" ]] || { echo "$task"; return; }
  done
  for ((i=0; i<SHARDS; i++)); do
    task="esmfold_$i"
    [[ -s "$OUT/.done/$task" ]] || { echo "$task"; return; }
  done
  [[ -s "$OUT/.done/merge_structure" ]] || { echo merge_structure; return; }
  if [[ -n "${UNIPROTKB_FASTA:-}" && -n "${UNIPROTKB_RELEASE:-}" ]]; then
    [[ -s "$OUT/.done/novelty_u" ]] || { echo novelty_u; return; }
  fi
  [[ -s "$OUT/.done/summarize" ]] || { echo summarize; return; }
  echo complete
}

is_gpu_task() {
  local wanted="$1" task
  for task in "${gpu_tasks[@]}"; do [[ "$task" == "$wanted" ]] && return 0; done
  return 1
}

run_task() {
  local task="$1"
  echo "[$(date --iso-8601=seconds)] starting task=$task"
  bash "$HERE/run_one_task.sh" "$task" > >(tee -a "$OUT/logs/$task.log") 2>&1 &
  CHILD_PID=$!
  wait "$CHILD_PID"
  CHILD_PID=""
  echo "[$(date --iso-8601=seconds)] finished task=$task"
}

while true; do
  CURRENT_TASK="$(next_task)"
  if [[ "$CURRENT_TASK" == complete ]]; then
    write_status AVAILABLE_STAGES_COMPLETE complete
    printf '{"status":"AVAILABLE_STAGES_COMPLETE","completed_at":"%s","pending":["ProTrek unless configured","Novelty-U and pooled HV unless full UniProtKB is configured"]}\n' \
      "$(date --iso-8601=seconds)" > "$OUT/AVAILABLE_STAGES_DONE.json"
    echo "all currently executable evaluation stages completed"
    exit 0
  fi

  if ! is_gpu_task "$CURRENT_TASK"; then
    write_status RUNNING "$CURRENT_TASK"
    run_task "$CURRENT_TASK"
    continue
  fi

  write_status WAITING_FOR_GPU "$CURRENT_TASK"
  stable_uuid=""; stable_count=0
  while true; do
    best_line="$({ nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu \
      --format=csv,noheader,nounits || true; } | awk -F',' \
      -v total="$MIN_TOTAL_MIB" -v free="$MIN_FREE_MIB" -v util="$MAX_UTIL_PCT" -v allowed="$ALLOWED_GPUS" '
      {
        for(i=1;i<=NF;i++) gsub(/^ +| +$/, "", $i)
        permitted=(allowed == "")
        if (!permitted) {
          n=split(allowed, allowed_ids, ",")
          for (j=1;j<=n;j++) if ($1 == allowed_ids[j]) permitted=1
        }
        if(permitted && $3+0>=total && $4+0>=free && $5+0<=util) print $1 "," $2 "," $3 "," $4 "," $5
      }' | sort -t',' -k4,4nr | head -n1)"
    if [[ -z "$best_line" ]]; then
      stable_uuid=""; stable_count=0
      write_status WAITING_FOR_GPU "$CURRENT_TASK"
      echo "[$(date --iso-8601=seconds)] task=$CURRENT_TASK no eligible GPU"
      sleep "$POLL_SECONDS"
      continue
    fi
    IFS=',' read -r index uuid total free util <<<"$best_line"
    if [[ "$uuid" == "$stable_uuid" ]]; then stable_count=$((stable_count+1)); else stable_uuid="$uuid"; stable_count=1; fi
    write_status WAITING_FOR_STABLE_GPU "$CURRENT_TASK"
    echo "[$(date --iso-8601=seconds)] task=$CURRENT_TASK GPU=$index uuid=$uuid free=$free util=$util stable=$stable_count/$STABLE_POLLS"
    if (( stable_count < STABLE_POLLS )); then sleep "$POLL_SECONDS"; continue; fi
    exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
    if ! flock -n 9; then stable_uuid=""; stable_count=0; sleep "$POLL_SECONDS"; continue; fi
    current="$(nvidia-smi --query-gpu=uuid,memory.total,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | awk -F',' -v uuid="$uuid" \
      '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==uuid) print $1 "," $2 "," $3 "," $4}')"
    IFS=',' read -r now_uuid now_total now_free now_util <<<"$current"
    if (( now_total < MIN_TOTAL_MIB || now_free < MIN_FREE_MIB || now_util > MAX_UTIL_PCT )); then
      flock -u 9; exec 9>&-; stable_uuid=""; stable_count=0; sleep "$POLL_SECONDS"; continue
    fi
    write_status RUNNING "$CURRENT_TASK"
    echo "[$(date --iso-8601=seconds)] selected physical GPU=$index uuid=$uuid free=$free util=$util"
    CUDA_VISIBLE_DEVICES="$uuid" run_task "$CURRENT_TASK"
    flock -u 9; exec 9>&-
    break
  done
done
