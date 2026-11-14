#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TEMB_PY="${TEMBERTURE_PYTHON:-__PRODIPA_ENVS__/temberture/bin/python}"
REBUILD="$ROOT/data/rebuild/teacher_data_threshold_85_070_060"
SCORES="$REBUILD/scores"
WORK="$REBUILD/work"
STAGING="$REBUILD/conditional_staging"
LOG_ROOT="$ROOT/logs/teacher_data_rebuild"
GPUS="${TPR_REBUILD_GPUS:-0,1,2,3}"
LIGHT_FREE_MIB="${TPR_LIGHT_FREE_MIB:-6000}"
ESMFOLD_FREE_MIB="${TPR_ESMFOLD_FREE_MIB:-28000}"
POLL_SECONDS="${TPR_POLL_SECONDS:-30}"
SHARDS="${TPR_ESMFOLD_SHARDS:-32}"

mkdir -p "$SCORES" "$WORK" "$STAGING" "$LOG_ROOT" "$REBUILD/.done"
exec 7>"$REBUILD/.rebuild.lock"
flock -n 7 || { echo "another TPR data rebuild is active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/tpr_strict_rebuild.log") 2>&1

status() {
  printf '{"state":"%s","stage":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$(date --iso-8601=seconds)" > "$REBUILD/REBUILD_STATUS.json"
}

declare -a CHILDREN=()
stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${CHILDREN[@]:-}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
}
on_stop() { status STOPPED pipeline; stop_children; exit 143; }
on_error() { local rc=$?; status FAILED pipeline; stop_children; exit "$rc"; }
trap on_stop INT TERM
trap on_error ERR

MERGED="$REBUILD/tpr_candidates_merged.tsv"
CLUSTERED="$REBUILD/tpr_candidates_clustered.tsv"

status RUNNING merge
"$PY" "$ROOT/scripts/merge_tpr_rebuild_candidates.py" \
  --legacy "$REBUILD/tpr_legacy_candidates.tsv" \
  --interpro "$REBUILD/tpr_interpro_reviewed.tsv" \
  --interpro "$REBUILD/tpr_interpro_unreviewed.tsv" \
  --min-length 50 --max-length 510 --out "$MERGED"
date --iso-8601=seconds > "$REBUILD/.done/merge"

if [[ ! -s "$REBUILD/.done/cluster" ]]; then
  (
    "$PY" "$ROOT/scripts/cluster_tpr_rebuild_candidates.py" \
      --candidates "$MERGED" --membership "$ROOT/data/provenance/split_membership.tsv.gz" \
      --mmseqs "$ROOT/tools/mmseqs/bin/mmseqs" --threads 32 \
      --work "$WORK/clustering" --out "$CLUSTERED"
    date --iso-8601=seconds > "$REBUILD/.done/cluster"
  ) > "$LOG_ROOT/tpr_clustering.log" 2>&1 &
  CHILDREN+=("$!")
fi

if [[ ! -s "$REBUILD/.done/protein_sol" ]]; then
  (
    "$PY" "$ROOT/scripts/run_protein_sol.py" --input "$MERGED" \
      --software "$ROOT/tools/protein_sol/protein-sol-sequence-prediction-software" \
      --work "$WORK/protein_sol" --out "$SCORES/tpr_sol.tsv" \
      --reuse "$SCORES/tpr_reviewed_sol.tsv"
    date --iso-8601=seconds > "$REBUILD/.done/protein_sol"
  ) > "$LOG_ROOT/tpr_all_sol.log" 2>&1 &
  CHILDREN+=("$!")
fi

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4","$5}'
}

run_when_gpu_available() {
  local required="$1" label="$2"; shift 2
  local row index uuid total free util gpu
  IFS=',' read -r -a gpu_ids <<<"$GPUS"
  while true; do
    for gpu in "${gpu_ids[@]}"; do
      row="$(gpu_row "$gpu")"; [[ -n "$row" ]] || continue
      IFS=',' read -r index uuid total free util <<<"$row"
      (( free >= required )) || continue
      exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
      if ! flock -n 9; then exec 9>&-; continue; fi
      row="$(gpu_row "$gpu")"; IFS=',' read -r index uuid total free util <<<"$row"
      if (( free < required )); then flock -u 9; exec 9>&-; continue; fi
      echo "[$(date --iso-8601=seconds)] label=$label gpu=$gpu free_mib=$free"
      set +e
      CUDA_VISIBLE_DEVICES="$uuid" "$@"
      rc=$?
      set -e
      flock -u 9; exec 9>&-
      return "$rc"
    done
    printf '{"state":"WAITING_FOR_GPU","task":"%s","required_free_mib":%s,"updated_at":"%s"}\n' \
      "$label" "$required" "$(date --iso-8601=seconds)" > "$REBUILD/GPU_QUEUE_STATUS.json"
    sleep "$POLL_SECONDS"
  done
}

if [[ ! -s "$REBUILD/.done/thermo" ]]; then
  (
    run_when_gpu_available "$LIGHT_FREE_MIB" thermo \
      "$TEMB_PY" "$ROOT/scripts/run_temBERTure_cls.py" --input "$MERGED" \
      --repo "$ROOT/tools/TemBERTure" --out "$SCORES/tpr_thermo.tsv" \
      --batch-size 8 --device cuda
    date --iso-8601=seconds > "$REBUILD/.done/thermo"
  ) > "$LOG_ROOT/tpr_all_thermo.log" 2>&1 &
  CHILDREN+=("$!")
fi

ESM_QUEUE="$REBUILD/.esmfold_pending"
: > "$ESM_QUEUE"
for ((i=0; i<SHARDS; i++)); do
  marker="$REBUILD/.done/esmfold_$(printf '%02d' "$i")"
  [[ -s "$marker" ]] || printf '%s\n' "$i" >> "$ESM_QUEUE"
done

claim_shard() {
  local shard="" temporary
  exec 6>"$REBUILD/.esmfold_claim.lock"; flock 6
  if [[ -s "$ESM_QUEUE" ]]; then
    shard="$(head -n1 "$ESM_QUEUE")"
    temporary="$ESM_QUEUE.tmp.$$"
    tail -n +2 "$ESM_QUEUE" > "$temporary"; mv "$temporary" "$ESM_QUEUE"
  fi
  flock -u 6; exec 6>&-
  printf '%s' "$shard"
}

esmfold_worker() {
  local shard output
  while true; do
    shard="$(claim_shard)"; [[ -n "$shard" ]] || return 0
    output="$SCORES/tpr_esmfold_shard_$(printf '%02d' "$shard").tsv"
    run_when_gpu_available "$ESMFOLD_FREE_MIB" "esmfold_$shard" \
      "$PY" "$ROOT/scripts/run_esmfold_properties.py" --input "$MERGED" \
      --out "$output" --model __PRODIPA_MODELS__/esmfold_v1_hf \
      --chunk-size 32 --num-recycles 4 --num-shards "$SHARDS" --shard-index "$shard"
    date --iso-8601=seconds > "$REBUILD/.done/esmfold_$(printf '%02d' "$shard")"
  done
}

IFS=',' read -r -a gpu_ids <<<"$GPUS"
for _gpu in "${gpu_ids[@]}"; do esmfold_worker & CHILDREN+=("$!"); done

status RUNNING scoring_and_clustering
failed=0
for pid in "${CHILDREN[@]:-}"; do if ! wait "$pid"; then failed=1; fi; done
CHILDREN=()
(( failed == 0 )) || exit 1

status RUNNING build_and_audit
"$PY" "$ROOT/scripts/build_tpr_teacher_datasets.py" \
  --candidates "$CLUSTERED" --score-root "$SCORES" --staging "$STAGING" \
  --base-length-reference "$ROOT/outputs/evaluation/generation/conditional/base.tsv" \
  --destination "$ROOT/data/conditional" --provenance "$ROOT/data/provenance" \
  --seed 42 --publish
date --iso-8601=seconds > "$REBUILD/.done/publish"
status COMPLETE complete
printf '{"status":"COMPLETE","completed_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$REBUILD/COMPLETE.json"
