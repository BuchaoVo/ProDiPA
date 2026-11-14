#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
PINAL_PY="${PINAL_PYTHON:-__PRODIPA_MODELS__/external_envs/pinal/bin/python}"
METHOD="pinal_1.2b_tpr_text"
CONFIG="$ROOT/configs/baselines/conditional_pinal.json"
PIPELINE_ROOT="$ROOT/outputs/evaluation/conditional/requested_baselines/$METHOD"
EVALUATION_ROOT="$ROOT/outputs/evaluation/conditional/methods/$METHOD"
GENERATION_JSONL="$ROOT/outputs/evaluation/conditional/baseline_generation/$METHOD.jsonl"
GENERATION_TSV="$PIPELINE_ROOT/generation/conditional/$METHOD.tsv"
EVALUATION_GPUS="${PARALLEL_GPUS:-0,1,2,3}"
GENERATION_GPUS="${PINAL_PHYSICAL_GPUS:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_MIB="${PINAL_MIN_FREE_MIB:-24000}"
MAX_UTIL_PCT="${PINAL_MAX_UTIL_PCT:-15}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"
SHARED_MAX_UTIL_PCT="${GPU_SHARED_MAX_UTIL_PCT:-80}"
PROMPT='Design a protein containing tetratricopeptide repeat (TPR) domains forming a predominantly alpha-helical repeat scaffold.'

mkdir -p "$PIPELINE_ROOT"/{generation/conditional,logs,preflight,.done}
printf '%s\n' "$$" > "$PIPELINE_ROOT/pipeline.pid"
exec 7>"$PIPELINE_ROOT/.pipeline.lock"
flock -n 7 || { echo "another Conditional Pinal pipeline is active" >&2; exit 1; }
exec > >(exec 7>&-; tee -a "$PIPELINE_ROOT/pipeline.log") 2>&1

status() {
  printf '{"state":"%s","stage":"%s","method":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$METHOD" "$(date --iso-8601=seconds)" > "$PIPELINE_ROOT/STATUS.json"
}
CURRENT_STAGE=initialization
on_error() { local rc=$?; status FAILED "$CURRENT_STAGE"; exit "$rc"; }
trap on_error ERR

claim_gpu() {
  local row index uuid free util candidate
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
    echo "[$(date --iso-8601=seconds)] waiting for an unlocked GPU in {$GENERATION_GPUS} with >=${MIN_FREE_MIB} MiB free"
    sleep "$POLL_SECONDS"
  done
}

release_gpu() {
  { flock -u 9; } 2>/dev/null || true
  { exec 9>&-; } 2>/dev/null || true
  { flock -u 8; } 2>/dev/null || true
  { exec 8>&-; } 2>/dev/null || true
}

CURRENT_STAGE=preflight
status RUNNING "$CURRENT_STAGE"
"$PY" "$ROOT/src/prodipa/baselines/audit_resources.py" \
  --config "$CONFIG" --project-root "$ROOT" \
  --output "$PIPELINE_ROOT/preflight/resources.json"
"$PINAL_PY" -m py_compile "$ROOT/src/prodipa/baselines/run_text_guided_unconditional.py"

CURRENT_STAGE=generation
if [[ ! -s "$PIPELINE_ROOT/.done/generation" || ! -s "$GENERATION_JSONL" ]]; then
  status WAITING "$CURRENT_STAGE"
  claim_gpu
  status RUNNING "$CURRENT_STAGE"
  set +e
  CUDA_VISIBLE_DEVICES="$SELECTED_UUID" "$PINAL_PY" \
    "$ROOT/src/prodipa/baselines/run_text_guided_unconditional.py" \
    --backend pinal \
    --repo __PRODIPA_MODELS__/external_repos/Denovo-Pinal \
    --weights __PRODIPA_MODELS__/external_weights/Pinal \
    --output "$GENERATION_JSONL" --requests 512 --seed 42 --batch-size 8 \
    --prompt "$PROMPT" --benchmark conditional_tpr \
    --model-id "$METHOD" --config-id pinal_1.2b_saprot_t_tpr_text_v1 \
    --generation-min-length 64 --generation-max-length 512 \
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
"$PY" "$ROOT/src/prodipa/baselines/audit_generation.py" \
  --input "$GENERATION_JSONL" --output "$PIPELINE_ROOT/generation_audit.json" \
  --expected-requests 512 --benchmark conditional_tpr \
  --model "$METHOD" --request-prefix "${METHOD}_" --min-length 50 --max-length 512
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
  "$ROOT/experiments/prodipa/external_baseline_evaluation/convert_generations.py" \
  --input "$GENERATION_JSONL" --output "$GENERATION_TSV" \
  --paradigm conditional --model "$METHOD" --expected-requests 512

CURRENT_STAGE=base_scoring
status RUNNING "$CURRENT_STAGE"
PARALLEL_GPUS="$EVALUATION_GPUS" bash \
  "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
  --model-id "$METHOD" --generation-tsv "$GENERATION_TSV" --gpus "$EVALUATION_GPUS"

CURRENT_STAGE=protrek
status WAITING "$CURRENT_STAGE"
if [[ ! -s "$EVALUATION_ROOT/scores/protrek.generated.tsv" ]]; then
  claim_gpu
  status RUNNING "$CURRENT_STAGE"
  set +e
  CUDA_VISIBLE_DEVICES="$SELECTED_UUID" "$PY" \
    "$ROOT/experiments/prodipa/conditional_evaluation/score_protrek_similarity.py" \
    --input "$EVALUATION_ROOT/prepared/unique_valid_sequences.tsv" \
    --output "$EVALUATION_ROOT/scores/protrek.generated.tsv" \
    --repo "$ROOT/third_party/ProTrek" \
    --model __PRODIPA_MODELS__/ProTrek_35M --device cuda:0 \
    > >(exec 7>&- 8>&- 9>&-; tee -a "$PIPELINE_ROOT/logs/protrek.log") 2>&1
  rc=$?
  set -e
  release_gpu
  if (( rc != 0 )); then status FAILED "$CURRENT_STAGE"; exit "$rc"; fi
fi

CURRENT_STAGE=publish
status RUNNING "$CURRENT_STAGE"
PARALLEL_GPUS="$EVALUATION_GPUS" bash \
  "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
  --model-id "$METHOD" --generation-tsv "$GENERATION_TSV" \
  --protrek-tsv "$EVALUATION_ROOT/scores/protrek.generated.tsv" \
  --gpus "$EVALUATION_GPUS" --publish

CURRENT_STAGE=complete
status COMPLETE "$CURRENT_STAGE"
printf '{"status":"COMPLETE","method":"%s","main_table":"%s","completed_at":"%s"}\n' \
  "$METHOD" "$ROOT/outputs/evaluation/conditional/results/conditional_prodipa_main.tsv" \
  "$(date --iso-8601=seconds)" > "$PIPELINE_ROOT/COMPLETE.json"
echo "Conditional Pinal generation, scoring, and canonical publication complete"
