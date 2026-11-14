#!/usr/bin/env bash
set -euo pipefail

# Resume one interrupted TPR ESMFold shard without restarting the full rebuild.
# The pending-list claim is serialized with the main rebuild workers. A failed
# shard is returned to the queue, so an OOM cannot silently lose work.

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
REBUILD="$ROOT/data/rebuild/teacher_data_threshold_85_070_060"
QUEUE="$REBUILD/.esmfold_pending"
GPU="${1:?usage: run_tpr_esmfold_resume_worker.sh PHYSICAL_GPU [SHARD]}"
PREFERRED="${2:-}"
SHARDS=32
LOG="$ROOT/logs/teacher_data_rebuild/tpr_extra_gpu${GPU}.log"

row="$(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits |
  awk -F',' -v gpu="$GPU" '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==gpu) print $1 "," $2 "," $3}')"
[[ -n "$row" ]] || { echo "physical GPU $GPU not found" >&2; exit 2; }
IFS=',' read -r index uuid free <<<"$row"

exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
flock -n 9 || { echo "project GPU lock is already held for $uuid" >&2; exit 3; }

claim_shard() {
  local shard="" tmp
  exec 8>"$REBUILD/.esmfold_claim.lock"; flock 8
  if [[ -n "$PREFERRED" ]] && grep -qx "$PREFERRED" "$QUEUE"; then
    shard="$PREFERRED"
    tmp="$QUEUE.tmp.$$"
    awk -v shard="$shard" '$0 != shard' "$QUEUE" > "$tmp"
    mv "$tmp" "$QUEUE"
  elif [[ -s "$QUEUE" ]]; then
    shard="$(head -n1 "$QUEUE")"
    tmp="$QUEUE.tmp.$$"
    tail -n +2 "$QUEUE" > "$tmp"
    mv "$tmp" "$QUEUE"
  fi
  flock -u 8; exec 8>&-
  printf '%s' "$shard"
}

return_shard() {
  local shard="$1"
  exec 8>"$REBUILD/.esmfold_claim.lock"; flock 8
  if ! grep -qx "$shard" "$QUEUE"; then printf '%s\n' "$shard" >> "$QUEUE"; fi
  flock -u 8; exec 8>&-
}

shard="$(claim_shard)"
[[ -n "$shard" ]] || { echo "no pending TPR ESMFold shard"; exit 0; }
marker="$REBUILD/.done/esmfold_$(printf '%02d' "$shard")"
output="$REBUILD/scores/tpr_esmfold_shard_$(printf '%02d' "$shard").tsv"

echo "[$(date --iso-8601=seconds)] gpu${GPU} resume worker shard=$shard free_mib=$free" | tee -a "$LOG"
set +e
CUDA_VISIBLE_DEVICES="$uuid" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" "$ROOT/scripts/run_esmfold_properties.py" \
    --input "$REBUILD/tpr_candidates_merged.tsv" --out "$output" \
    --model __PRODIPA_MODELS__/esmfold_v1_hf --chunk-size 32 \
    --num-recycles 4 --num-shards "$SHARDS" --shard-index "$shard" \
    >> "$LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  return_shard "$shard"
  echo "[$(date --iso-8601=seconds)] gpu${GPU} shard=$shard failed rc=$rc; returned to queue" | tee -a "$LOG"
  exit "$rc"
fi
date --iso-8601=seconds > "$marker"
echo "[$(date --iso-8601=seconds)] gpu${GPU} shard=$shard complete" | tee -a "$LOG"
