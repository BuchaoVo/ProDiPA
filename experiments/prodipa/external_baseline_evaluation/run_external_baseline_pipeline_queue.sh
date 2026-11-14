#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PROJECT_ROOT:-$(cd "$HERE/../../.." && pwd)}"
SOURCE="${TEACHER_EVALUATION_ROOT:-$ROOT/outputs/evaluation/teacher_generation_evaluation}"
UNCONDITIONAL_TEACHER_EVALUATION="${UNCONDITIONAL_TEACHER_EVALUATION_ROOT:-$ROOT/outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers}"
OUT="${EXTERNAL_BASELINE_OUTPUT_ROOT:-$ROOT/outputs/evaluation/external_baselines}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
SCORER="$ROOT/experiments/prodipa/conditional_evaluation/run_separate_paradigm_scoring_queue.sh"
RUNTIME_ENV="${BASELINE_RUNTIME_ENV_FILE:-$ROOT/configs/baselines/runtime_paths.env}"
ALLOWED_GPUS="${BASELINE_GPUS:-0,1,2,3}"
BASELINE_SCOPE="${BASELINE_SCOPE:-both}"
MIN_TOTAL_MIB="${MIN_TOTAL_MIB:-70000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-15}"
STABLE_POLLS="${STABLE_POLLS:-3}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
RESOURCE_POLL_SECONDS="${RESOURCE_POLL_SECONDS:-300}"
TEACHER_POLL_SECONDS="${TEACHER_POLL_SECONDS:-120}"

mkdir -p "$OUT"/{preflight,audits,generation/unconditional,generation/conditional,logs,.done}
case "$BASELINE_SCOPE" in
  unconditional|conditional|both) ;;
  *) echo "BASELINE_SCOPE must be unconditional, conditional, or both" >&2; exit 64 ;;
esac
exec 7>"$OUT/.pipeline_queue.lock"
flock -n 7 || { echo "another external baseline pipeline queue is active" >&2; exit 1; }
exec > >(tee -a "$OUT/pipeline_queue.log") 2>&1

ACTIVE_PIDS=()
CURRENT_STAGE="startup"
write_status() {
  local state="$1" stage="$2"
  "$PY" - "$OUT/PIPELINE_STATUS.json" "$state" "$stage" "$ALLOWED_GPUS" "$BASELINE_SCOPE" <<'PY'
import json, os, sys, tempfile
from datetime import datetime
from pathlib import Path
target, state, stage, gpus, scope = map(str, sys.argv[1:6])
path = Path(target)
value = {
    "state": state, "stage": stage, "allowed_gpus": gpus, "scope": scope,
    "updated_at": datetime.now().astimezone().isoformat(),
}
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, prefix=f".{path.name}.") as h:
    json.dump(value, h, indent=2); h.write("\n"); temporary = h.name
os.replace(temporary, path)
PY
}
on_stop() {
  write_status STOPPED "$CURRENT_STAGE"
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  exit 143
}
on_error() { local rc=$?; write_status FAILED "$CURRENT_STAGE"; exit "$rc"; }
trap on_stop INT TERM
trap on_error ERR

json_state() {
  "$PY" - "$1" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", "MISSING"))
except Exception:
    print("MISSING")
PY
}

CURRENT_STAGE="wait_for_teacher_evaluation"
while true; do
  unconditional_state="$(json_state "$UNCONDITIONAL_TEACHER_EVALUATION/SCORING_STATUS.json")"
  conditional_state="$(json_state "$SOURCE/conditional_prollama_evaluation/SCORING_STATUS.json")"
  if [[ "$BASELINE_SCOPE" == unconditional && "$unconditional_state" == COMPLETE ]]; then break; fi
  if [[ "$BASELINE_SCOPE" == conditional && "$conditional_state" == COMPLETE ]]; then break; fi
  if [[ "$BASELINE_SCOPE" == both && "$unconditional_state" == COMPLETE && "$conditional_state" == COMPLETE ]]; then break; fi
  write_status WAITING_TEACHER_EVALUATION "$CURRENT_STAGE"
  echo "[$(date --iso-8601=seconds)] teacher evaluation: unconditional=$unconditional_state conditional=$conditional_state"
  sleep "$TEACHER_POLL_SECONDS"
done
echo "[$(date --iso-8601=seconds)] prerequisite teacher evaluation is complete for scope=$BASELINE_SCOPE"

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4","$5}'
}

wait_for_best_gpu() {
  local best_gpu="" best_free=-1 row index uuid total free util gpu stable=0
  IFS=',' read -r -a gpu_ids <<<"$ALLOWED_GPUS"
  while true; do
    best_gpu=""; best_free=-1
    for gpu in "${gpu_ids[@]}"; do
      row="$(gpu_row "$gpu")"; [[ -n "$row" ]] || continue
      IFS=',' read -r index uuid total free util <<<"$row"
      if (( total >= MIN_TOTAL_MIB && free >= MIN_FREE_MIB && util <= MAX_UTIL_PCT && free > best_free )); then
        best_gpu="$gpu"; best_free="$free"
      fi
    done
    if [[ -z "$best_gpu" ]]; then stable=0; write_status WAITING_GPU "$CURRENT_STAGE"; sleep "$GPU_POLL_SECONDS"; continue; fi
    stable=$((stable + 1))
    if (( stable < STABLE_POLLS )); then sleep "$GPU_POLL_SECONDS"; continue; fi
    row="$(gpu_row "$best_gpu")"; IFS=',' read -r index uuid total free util <<<"$row"
    exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
    if ! flock -n 9; then stable=0; exec 9>&-; sleep "$GPU_POLL_SECONDS"; continue; fi
    row="$(gpu_row "$best_gpu")"; IFS=',' read -r index uuid total free util <<<"$row"
    if (( free < MIN_FREE_MIB || util > MAX_UTIL_PCT )); then
      flock -u 9; exec 9>&-; stable=0; sleep "$GPU_POLL_SECONDS"; continue
    fi
    SELECTED_GPU="$best_gpu"; SELECTED_UUID="$uuid"; SELECTED_FREE="$free"
    return 0
  done
}

load_runtime_env() {
  if [[ -f "$RUNTIME_ENV" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
    set +a
  fi
  PROGEN2_PYTHON="${PROGEN2_PYTHON:-$PY}"
  DPLM_PYTHON="${DPLM_PYTHON:-$PY}"
  ASPO_PYTHON="${ASPO_PYTHON:-$PY}"
}

preflight_method() {
  local method="$1" report="$OUT/preflight/$method.json"
  case "$method" in
    progen2_base) "$PROGEN2_PYTHON" "$ROOT/src/prodipa/baselines/run_progen2.py" preflight --project-root "$ROOT" --report "$report" ;;
    dplm_650m) "$DPLM_PYTHON" "$ROOT/src/prodipa/baselines/run_dplm.py" preflight --project-root "$ROOT" --report "$report" ;;
    aspo_tpr) "$ASPO_PYTHON" "$ROOT/src/prodipa/baselines/run_aspo_tpr.py" preflight --project-root "$ROOT" --report "$report" ;;
    prodva_tpr) "$PY" "$ROOT/src/prodipa/baselines/run_prodva_tpr.py" preflight --project-root "$ROOT" --report "$report" ;;
    *) return 64 ;;
  esac
}

run_method_command() {
  local method="$1"
  case "$method" in
    progen2_base)
      "$PROGEN2_PYTHON" "$ROOT/src/prodipa/baselines/run_progen2.py" run --project-root "$ROOT" --device cuda:0 ;;
    dplm_650m)
      "$DPLM_PYTHON" "$ROOT/src/prodipa/baselines/run_dplm.py" run --project-root "$ROOT" --device cuda:0 ;;
    aspo_tpr)
      "$ASPO_PYTHON" "$ROOT/src/prodipa/baselines/run_aspo_tpr.py" run --project-root "$ROOT" \
        --report "$OUT/preflight/aspo_tpr.json" --device cuda:0 ;;
    prodva_tpr)
      "$PY" "$ROOT/src/prodipa/baselines/run_prodva_tpr.py" run --project-root "$ROOT"
      "$PY" "$ROOT/src/prodipa/baselines/run_prodva_tpr.py" normalize --project-root "$ROOT" \
        --results "$ROOT/outputs/conditional/tpr_external_baselines/prodva/official_eval_results.json" \
        --output "$ROOT/outputs/conditional/tpr_external_baselines/prodva/generations.jsonl" ;;
  esac
}

generation_path() {
  case "$1" in
    progen2_base) echo "$ROOT/outputs/unconditional/external_baselines/progen2_base/generations.jsonl" ;;
    dplm_650m) echo "$ROOT/outputs/unconditional/external_baselines/dplm_650m/generations.jsonl" ;;
    aspo_tpr) echo "$ROOT/outputs/conditional/tpr_external_baselines/aspo_multi/generations.jsonl" ;;
    prodva_tpr) echo "$ROOT/outputs/conditional/tpr_external_baselines/prodva/generations.jsonl" ;;
  esac
}

audit_and_convert() {
  local method="$1" input benchmark source_model paradigm table_model request_prefix
  input="$(generation_path "$method")"
  case "$method" in
    progen2_base) benchmark=unconditional; source_model=progen2_base; paradigm=unconditional; table_model=progen2_base; request_prefix=progen2_base_ ;;
    dplm_650m) benchmark=unconditional; source_model=dplm_650m; paradigm=unconditional; table_model=dplm_650m; request_prefix=dplm_650m_ ;;
    aspo_tpr) benchmark=conditional_tpr; source_model=aspo_tpr_multi_attribute; paradigm=conditional; table_model=aspo_tpr; request_prefix=aspo_tpr_ ;;
    prodva_tpr) benchmark=conditional_tpr; source_model=prodva_molinst_swissprotclap; paradigm=conditional; table_model=prodva_tpr; request_prefix=prodva_tpr_ ;;
  esac
  "$PY" "$ROOT/src/prodipa/baselines/audit_generation.py" \
    --input "$input" --output "$OUT/audits/$method.json" --expected-requests 512 \
    --benchmark "$benchmark" --model "$source_model" --request-prefix "$request_prefix"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$HERE/convert_generations.py" \
    --input "$input" --output "$OUT/generation/$paradigm/$table_model.tsv" \
    --paradigm "$paradigm" --model "$table_model" --expected-requests 512
}

case "$BASELINE_SCOPE" in
  unconditional) methods=(progen2_base dplm_650m) ;;
  conditional) methods=(aspo_tpr prodva_tpr) ;;
  both) methods=(progen2_base dplm_650m aspo_tpr prodva_tpr) ;;
esac
CURRENT_STAGE="external_generation"
while true; do
  all_done=true; made_progress=false; blocked=()
  load_runtime_env
  for method in "${methods[@]}"; do
    if [[ -s "$OUT/.done/generation_$method" ]]; then continue; fi
    all_done=false
    if preflight_method "$method" >>"$OUT/logs/${method}_preflight.log" 2>&1; then
      rc=0
    else
      rc=$?
    fi
    if (( rc == 2 )); then blocked+=("$method"); continue; fi
    if (( rc != 0 )); then echo "unexpected preflight failure: $method rc=$rc" >&2; exit "$rc"; fi

    write_status WAITING_GPU "generation_$method"
    wait_for_best_gpu
    echo "[$(date --iso-8601=seconds)] start $method on physical_gpu=$SELECTED_GPU uuid=$SELECTED_UUID free_mib=$SELECTED_FREE"
    write_status RUNNING "generation_$method"
    (export CUDA_VISIBLE_DEVICES="$SELECTED_UUID"; run_method_command "$method") \
      > >(tee -a "$OUT/logs/${method}_generation.log") 2>&1 &
    generation_pid=$!
    ACTIVE_PIDS=("$generation_pid")
    if wait "$generation_pid"; then rc=0; else rc=$?; fi
    ACTIVE_PIDS=()
    flock -u 9; exec 9>&-
    if (( rc != 0 )); then echo "generation failed: $method rc=$rc" >&2; exit "$rc"; fi
    audit_and_convert "$method" >>"$OUT/logs/${method}_generation.log" 2>&1
    date --iso-8601=seconds > "$OUT/.done/generation_$method"
    made_progress=true
    echo "[$(date --iso-8601=seconds)] complete generation $method"
  done
  $all_done && break
  if ! $made_progress; then
    write_status WAITING_RESOURCES "blocked:${blocked[*]}"
    echo "[$(date --iso-8601=seconds)] waiting for baseline resources: ${blocked[*]}"
    sleep "$RESOURCE_POLL_SECONDS"
  fi
done

CURRENT_STAGE="baseline_scoring"
write_status RUNNING "$CURRENT_STAGE"
UNCOND_OUT="$OUT/unconditional_protgpt2_evaluation"
COND_OUT="$OUT/conditional_prollama_evaluation"
COMMON_ENV=(
  "PROJECT_ROOT=$ROOT" "GENERATION_ROOT=$OUT/generation" "PREPARE_SCRIPT=$HERE/prepare_sequences.py"
  "MIN_TOTAL_MIB=$MIN_TOTAL_MIB" "MIN_FREE_MIB=$MIN_FREE_MIB" "MAX_UTIL_PCT=$MAX_UTIL_PCT"
)

failed=0
scoring_pids=()
if [[ "$BASELINE_SCOPE" == unconditional || "$BASELINE_SCOPE" == both ]]; then
  env "${COMMON_ENV[@]}" PARADIGM_EVALUATION_OUTPUT_ROOT="$UNCOND_OUT" \
    EVALUATION_MODELS=progen2_base,dplm_650m PARALLEL_GPUS=0,1 \
    NOVELTY_T_REFERENCE="$UNCONDITIONAL_TEACHER_EVALUATION/prepared/references/unconditional_train_union.fasta" \
    bash "$SCORER" unconditional >"$OUT/logs/unconditional_scoring.launcher.log" 2>&1 &
  scoring_pids+=("$!")
fi
if [[ "$BASELINE_SCOPE" == conditional || "$BASELINE_SCOPE" == both ]]; then
  env "${COMMON_ENV[@]}" PARADIGM_EVALUATION_OUTPUT_ROOT="$COND_OUT" \
    EVALUATION_MODELS=aspo_tpr,prodva_tpr PARALLEL_GPUS=2,3 \
    NOVELTY_T_REFERENCE="$SOURCE/conditional_prollama_evaluation/prepared/references/conditional_train_union.fasta" \
    bash "$SCORER" conditional >"$OUT/logs/conditional_scoring.launcher.log" 2>&1 &
  scoring_pids+=("$!")
fi
ACTIVE_PIDS=("${scoring_pids[@]}")
for scoring_pid in "${scoring_pids[@]}"; do
  wait "$scoring_pid" || failed=1
done
ACTIVE_PIDS=()
(( failed == 0 )) || exit 1

CURRENT_STAGE="merge_main_tables"
write_status RUNNING "$CURRENT_STAGE"
if [[ "$BASELINE_SCOPE" == unconditional || "$BASELINE_SCOPE" == both ]]; then
  "$PY" "$HERE/merge_main_tables.py" \
    --existing "$UNCONDITIONAL_TEACHER_EVALUATION/results/unconditional_prodipa_main.tsv" \
    --additional "$UNCOND_OUT/results/unconditional_prodipa_main.tsv" \
    --expected-models progen2_base,dplm_650m
fi
if [[ "$BASELINE_SCOPE" == conditional || "$BASELINE_SCOPE" == both ]]; then
  "$PY" "$HERE/merge_main_tables.py" \
    --existing "$SOURCE/conditional_prollama_evaluation/results/conditional_prodipa_main.tsv" \
    --additional "$COND_OUT/results/conditional_prodipa_main.tsv" \
    --expected-models aspo_tpr,prodva_tpr
fi

date --iso-8601=seconds > "$OUT/.done/${BASELINE_SCOPE}_pipeline_complete"
write_status COMPLETE complete
echo "[$(date --iso-8601=seconds)] external baseline generation, scoring, and table merge complete for scope=$BASELINE_SCOPE"
