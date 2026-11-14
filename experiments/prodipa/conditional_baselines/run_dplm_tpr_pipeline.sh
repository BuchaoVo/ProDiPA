#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
DPLM_PY="${DPLM_PYTHON:-__PRODIPA_MODELS__/external_envs/dplm_650m/bin/python}"
METHOD="dplm_650m_tpr_infilling"
CONFIG="$ROOT/configs/baselines/conditional_dplm.json"
PIPELINE_ROOT="$ROOT/outputs/evaluation/conditional/requested_baselines/$METHOD"
EVALUATION_ROOT="$ROOT/outputs/evaluation/conditional/methods/$METHOD"
GENERATION_JSONL="$ROOT/outputs/evaluation/conditional/baseline_generation/$METHOD.jsonl"
GENERATION_TSV="$PIPELINE_ROOT/generation/conditional/$METHOD.tsv"
MANIFEST="$ROOT/data/evaluation/conditional/${METHOD}_manifest.jsonl"
GENERATION_GPU="${DPLM_PHYSICAL_GPU:-2}"
EVALUATION_GPUS="${PARALLEL_GPUS:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_MIB="${DPLM_MIN_FREE_MIB:-12000}"

mkdir -p "$PIPELINE_ROOT"/{generation/conditional,logs,preflight,.done}
printf '%s\n' "$$" > "$PIPELINE_ROOT/pipeline.pid"
exec 7>"$PIPELINE_ROOT/.pipeline.lock"
flock -n 7 || { echo "another Conditional DPLM pipeline is active" >&2; exit 1; }
exec > >(tee -a "$PIPELINE_ROOT/pipeline.log") 2>&1

status() {
  printf '{"state":"%s","stage":"%s","method":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$METHOD" "$(date --iso-8601=seconds)" > "$PIPELINE_ROOT/STATUS.json"
}
CURRENT_STAGE=initialization
on_error() { local rc=$?; status FAILED "$CURRENT_STAGE"; exit "$rc"; }
trap on_error ERR

gpu_row() {
  nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$GENERATION_GPU" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4}'
}

claim_generation_gpu() {
  local row index uuid free util
  while true; do
    row="$(gpu_row)"
    if [[ -n "$row" ]]; then
      IFS=',' read -r index uuid free util <<<"$row"
      if (( free >= MIN_FREE_MIB )); then
        exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
        if flock -n 9; then
          SELECTED_UUID="$uuid"
          echo "[$(date --iso-8601=seconds)] claimed physical_gpu=$index uuid=$uuid free_mib=$free util_pct=$util"
          return 0
        fi
      fi
      echo "[$(date --iso-8601=seconds)] waiting physical_gpu=$index free_mib=$free util_pct=$util required_free_mib=$MIN_FREE_MIB"
    else
      echo "[$(date --iso-8601=seconds)] waiting: physical GPU $GENERATION_GPU is unavailable"
    fi
    sleep "$POLL_SECONDS"
  done
}

CURRENT_STAGE=condition_manifest
status RUNNING "$CURRENT_STAGE"
"$PY" "$ROOT/src/prodipa/baselines/build_conditional_dplm_manifest.py" \
  --project-root "$ROOT" --output "$MANIFEST" \
  --report "$PIPELINE_ROOT/condition_manifest_report.json"

CURRENT_STAGE=preflight
status RUNNING "$CURRENT_STAGE"
"$PY" "$ROOT/src/prodipa/baselines/run_dplm_conditional.py" preflight \
  --project-root "$ROOT" --config "$CONFIG" \
  --report "$PIPELINE_ROOT/preflight/dplm.json"

CURRENT_STAGE=generation
status WAITING "$CURRENT_STAGE"
claim_generation_gpu
status RUNNING "$CURRENT_STAGE"
set +e
CUDA_VISIBLE_DEVICES="$SELECTED_UUID" "$DPLM_PY" \
  "$ROOT/src/prodipa/baselines/run_dplm_conditional.py" run \
  --project-root "$ROOT" --config "$CONFIG" --device cuda:0 \
  > >(tee -a "$PIPELINE_ROOT/logs/generation.log") 2>&1
rc=$?
set -e
flock -u 9
exec 9>&-
if (( rc != 0 )); then
  status FAILED "$CURRENT_STAGE"
  exit "$rc"
fi
date --iso-8601=seconds > "$PIPELINE_ROOT/.done/generation"

CURRENT_STAGE=generation_audit
status RUNNING "$CURRENT_STAGE"
"$PY" "$ROOT/src/prodipa/baselines/audit_generation.py" \
  --input "$GENERATION_JSONL" --output "$PIPELINE_ROOT/generation_audit.json" \
  --expected-requests 512 --benchmark conditional_tpr \
  --model "$METHOD" --request-prefix "${METHOD}_"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
  "$ROOT/experiments/prodipa/external_baseline_evaluation/convert_generations.py" \
  --input "$GENERATION_JSONL" --output "$GENERATION_TSV" \
  --paradigm conditional --model "$METHOD" --expected-requests 512

CURRENT_STAGE=base_scoring
status RUNNING "$CURRENT_STAGE"
PARALLEL_GPUS="$EVALUATION_GPUS" bash \
  "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
  --model-id "$METHOD" --generation-tsv "$GENERATION_TSV" \
  --gpus "$EVALUATION_GPUS"

CURRENT_STAGE=protrek
status WAITING "$CURRENT_STAGE"
if [[ ! -s "$EVALUATION_ROOT/scores/protrek.generated.tsv" ]]; then
  claim_generation_gpu
  status RUNNING "$CURRENT_STAGE"
  set +e
  CUDA_VISIBLE_DEVICES="$SELECTED_UUID" "$PY" \
    "$ROOT/experiments/prodipa/conditional_evaluation/score_protrek_similarity.py" \
    --input "$EVALUATION_ROOT/prepared/unique_valid_sequences.tsv" \
    --output "$EVALUATION_ROOT/scores/protrek.generated.tsv" \
    --repo "$ROOT/third_party/ProTrek" \
    --model __PRODIPA_MODELS__/ProTrek_35M --device cuda:0 \
    > >(tee -a "$PIPELINE_ROOT/logs/protrek.log") 2>&1
  rc=$?
  set -e
  flock -u 9
  exec 9>&-
  if (( rc != 0 )); then
    status FAILED "$CURRENT_STAGE"
    exit "$rc"
  fi
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
echo "Conditional DPLM generation, scoring, and canonical publication complete"
