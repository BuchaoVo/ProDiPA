#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
CORE_PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
METHOD="${1:?usage: run_unconditional_baseline_pipeline.sh METHOD}"
GENERATION_GPUS="${BASELINE_GENERATION_GPUS:-0,1,2,3}"
EVALUATION_GPUS="${PARALLEL_GPUS:-0,1,2,3}"
POLL_SECONDS="${BASELINE_POLL_SECONDS:-300}"
GPU_POLL_SECONDS="${BASELINE_GPU_POLL_SECONDS:-60}"
MIN_FREE_MIB="${BASELINE_MIN_FREE_MIB:-30000}"
MAX_UTIL_PCT="${BASELINE_MAX_UTIL_PCT:-10}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"
SHARED_MAX_UTIL_PCT="${GPU_SHARED_MAX_UTIL_PCT:-80}"

case "$METHOD" in
  evodiff_oa_dm_640m)
    CONFIG="$ROOT/configs/baselines/evodiff.yaml"
    RUNNER="$ROOT/src/prodipa/baselines/run_evodiff.py"
    RUN_PY="__PRODIPA_MODELS__/external_envs/evodiff/bin/python"
    ;;
  dplm2_650m)
    CONFIG="$ROOT/configs/baselines/dplm2.yaml"
    RUNNER="$ROOT/src/prodipa/baselines/run_dplm2.py"
    RUN_PY="__PRODIPA_MODELS__/external_envs/dplm_650m/bin/python"
    ;;
  progen3_762m)
    CONFIG="$ROOT/configs/baselines/progen3_762m.yaml"
    RUNNER="$ROOT/src/prodipa/baselines/run_progen3.py"
    RUN_PY="__PRODIPA_MODELS__/external_envs/progen3/bin/python"
    ;;
  *) echo "unsupported method: $METHOD" >&2; exit 2 ;;
esac

PIPELINE_ROOT="$ROOT/outputs/evaluation/unconditional/requested_baselines/$METHOD"
GENERATION_JSONL="$($CORE_PY -c 'import json,sys; from pathlib import Path; c=json.load(open(sys.argv[1])); print((Path(sys.argv[2])/c["protocol"]["output"]).resolve())' "$CONFIG" "$ROOT")"
GENERATION_TSV="$PIPELINE_ROOT/generation/unconditional/$METHOD.tsv"
mkdir -p "$PIPELINE_ROOT"/{generation/unconditional,logs,preflight,.done}
printf '%s\n' "$$" > "$PIPELINE_ROOT/pipeline.pid"
exec 7>"$PIPELINE_ROOT/.pipeline.lock"
flock -n 7 || { echo "another $METHOD pipeline is active" >&2; exit 1; }
exec > >(exec 7>&-; tee -a "$PIPELINE_ROOT/pipeline.log") 2>&1

status() {
  printf '{"state":"%s","stage":"%s","method":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$METHOD" "$(date --iso-8601=seconds)" > "$PIPELINE_ROOT/STATUS.json"
}
CURRENT_STAGE=initialization
on_error() { local rc=$?; status FAILED "$CURRENT_STAGE"; exit "$rc"; }
trap on_error ERR

wait_for_resources() {
  local rc
  while true; do
    set +e
    "$CORE_PY" "$ROOT/src/prodipa/baselines/audit_resources.py" \
      --config "$CONFIG" --project-root "$ROOT" \
      --output "$PIPELINE_ROOT/preflight/resources.json"
    rc=$?
    set -e
    (( rc == 0 )) && return 0
    status WAITING "$CURRENT_STAGE"
    echo "[$(date --iso-8601=seconds)] $METHOD resources incomplete; retrying in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

wait_for_runtime() {
  while true; do
    if [[ -x "$RUN_PY" ]] && "$RUN_PY" -m py_compile "$RUNNER"; then
      if [[ "$METHOD" != progen3_762m ]] || "$RUN_PY" -c 'import flash_attn, megablocks, progen3' >/dev/null 2>&1; then
        return 0
      fi
    fi
    status WAITING "$CURRENT_STAGE"
    echo "[$(date --iso-8601=seconds)] $METHOD runtime incomplete; retrying in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

claim_gpu() {
  local index uuid free util candidate
  while true; do
    while IFS=',' read -r index uuid free util; do
      for candidate in ${GENERATION_GPUS//,/ }; do
        [[ "$index" == "$candidate" ]] || continue
        if (( free >= MIN_FREE_MIB && util <= MAX_UTIL_PCT )); then
          exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
          if flock -n 9; then
            SELECTED_UUID="$uuid"
            SELECTED_LOCK_MODE=exclusive
            echo "[$(date --iso-8601=seconds)] claimed physical_gpu=$index uuid=$uuid mode=$SELECTED_LOCK_MODE free_mib=$free util_pct=$util"
            return 0
          fi
          exec 9>&-
        fi
        if [[ "$ALLOW_SHARED_GPU" == 1 ]] && (( free >= MIN_FREE_MIB && util <= SHARED_MAX_UTIL_PCT )); then
          exec 8>"/tmp/prodipa_gpu_${uuid}.shared_scoring.lock"
          if flock -n 8; then
            SELECTED_UUID="$uuid"
            SELECTED_LOCK_MODE=shared
            echo "[$(date --iso-8601=seconds)] claimed physical_gpu=$index uuid=$uuid mode=$SELECTED_LOCK_MODE free_mib=$free util_pct=$util"
            return 0
          fi
          exec 8>&-
        fi
      done
    done < <(nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | awk -F',' \
      '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); print $1","$2","$3","$4}')
    status WAITING "$CURRENT_STAGE"
    echo "[$(date --iso-8601=seconds)] waiting for generation GPU in {$GENERATION_GPUS}"
    sleep "$GPU_POLL_SECONDS"
  done
}

release_gpu() {
  { flock -u 9; } 2>/dev/null || true
  { exec 9>&-; } 2>/dev/null || true
  { flock -u 8; } 2>/dev/null || true
  { exec 8>&-; } 2>/dev/null || true
}

CURRENT_STAGE=resources
status WAITING "$CURRENT_STAGE"
wait_for_resources

CURRENT_STAGE=runtime
status WAITING "$CURRENT_STAGE"
wait_for_runtime

CURRENT_STAGE=generation
if [[ ! -s "$PIPELINE_ROOT/.done/generation" || ! -s "$GENERATION_JSONL" ]]; then
  status WAITING "$CURRENT_STAGE"
  claim_gpu
  status RUNNING "$CURRENT_STAGE"
  set +e
  CUDA_VISIBLE_DEVICES="$SELECTED_UUID" "$RUN_PY" "$RUNNER" run \
    --config "$CONFIG" --project-root "$ROOT" --device cuda:0 \
    > >(exec 7>&- 8>&- 9>&-; tee -a "$PIPELINE_ROOT/logs/generation.log") 2>&1
  rc=$?
  set -e
  release_gpu
  if (( rc != 0 )); then status FAILED "$CURRENT_STAGE"; exit "$rc"; fi
  date --iso-8601=seconds > "$PIPELINE_ROOT/.done/generation"
else
  echo "[$(date --iso-8601=seconds)] reusing completed generation: $GENERATION_JSONL"
fi

CURRENT_STAGE=generation_audit
status RUNNING "$CURRENT_STAGE"
"$CORE_PY" "$ROOT/src/prodipa/baselines/audit_generation.py" \
  --input "$GENERATION_JSONL" --output "$PIPELINE_ROOT/generation_audit.json" \
  --expected-requests 512 --benchmark unconditional --model "$METHOD" \
  --request-prefix "${METHOD}_" --min-length 50 --max-length 512
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$CORE_PY" \
  "$ROOT/experiments/prodipa/external_baseline_evaluation/convert_generations.py" \
  --input "$GENERATION_JSONL" --output "$GENERATION_TSV" \
  --paradigm unconditional --model "$METHOD" --expected-requests 512

CURRENT_STAGE=evaluation
status RUNNING "$CURRENT_STAGE"
PARALLEL_GPUS="$EVALUATION_GPUS" bash \
  "$ROOT/experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh" \
  --model-id "$METHOD" --generation-tsv "$GENERATION_TSV" \
  --gpus "$EVALUATION_GPUS" --publish

CURRENT_STAGE=complete
status COMPLETE "$CURRENT_STAGE"
printf '{"status":"COMPLETE","method":"%s","main_table":"%s","completed_at":"%s"}\n' \
  "$METHOD" "$ROOT/outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv" \
  "$(date --iso-8601=seconds)" > "$PIPELINE_ROOT/COMPLETE.json"
echo "$METHOD generation, scoring, and canonical publication complete"
