#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
UNCONDITIONAL="$ROOT/outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers"
LOG="$ROOT/logs/teacher_data_rebuild/tpr_resume_after_unconditional.log"
EXPECTED_ESMFOLD_SHARDS="${UNCONDITIONAL_ESMFOLD_SHARDS:-16}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

esmfold_done_count() {
  find "$UNCONDITIONAL/.done" -maxdepth 1 -type f -name 'esmfold_[0-9]*' 2>/dev/null | wc -l
}

while (( $(esmfold_done_count) < EXPECTED_ESMFOLD_SHARDS )); do
  if [[ -s "$UNCONDITIONAL/TEACHER_EVALUATION_STATUS.json" ]] && \
     grep -q '"state":"FAILED"' "$UNCONDITIONAL/TEACHER_EVALUATION_STATUS.json"; then
    echo "unconditional evaluation failed; TPR rebuild was not resumed"
    exit 1
  fi
  sleep 10
done

echo "[$(date --iso-8601=seconds)] unconditional ESMFold complete (${EXPECTED_ESMFOLD_SHARDS}/${EXPECTED_ESMFOLD_SHARDS} shards); resuming TPR rebuild"
exec env PROJECT_ROOT="$ROOT" TPR_REBUILD_GPUS=0,1,2,3 \
  TPR_LIGHT_FREE_MIB=6000 TPR_ESMFOLD_FREE_MIB=28000 \
  TPR_POLL_SECONDS=30 TPR_ESMFOLD_SHARDS=32 \
  bash "$ROOT/scripts/run_tpr_teacher_data_rebuild.sh"
