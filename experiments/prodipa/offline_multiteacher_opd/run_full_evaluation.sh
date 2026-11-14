#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
HERE="$ROOT/experiments/prodipa/offline_multiteacher_opd"
OUT="$ROOT/outputs/experiments/offline_multiteacher_opd_full"
EVAL="$OUT/evaluation"
GPUS="${EVAL_GPUS:-0,1,2,3}"
MIN_FREE="${GEN_MIN_FREE_MIB:-18000}"
MAX_UTIL="${GEN_MAX_UTIL_PCT:-15}"
POLL="${POLL_SECONDS:-30}"
STATUS="$OUT/EVALUATION_STATUS.json"
COND_REFERENCE="$ROOT/outputs/experiments/offline_multiteacher_opd"

mkdir -p "$OUT/logs" "$EVAL"/{generation,methods,novelty,timings}
printf '%s\n' "$$" > "$OUT/evaluation_pipeline.pid"
exec 8>"$OUT/.evaluation_pipeline.lock"
flock -n 8 || { echo "another full paired evaluation pipeline is active" >&2; exit 2; }
exec > >(tee -a "$OUT/full_evaluation.log") 2>&1

stage=wait_training
write_status() {
  "$PY" - "$STATUS" "$1" "$2" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"state":sys.argv[2],"stage":sys.argv[3],"updated_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
PY
}
on_error() { local rc=$?; write_status FAILED "$stage"; exit "$rc"; }
trap on_error ERR
write_status WAITING "$stage"
while [[ ! -e "$OUT/TRAINING_COMPLETE" ]]; do sleep "$POLL"; done

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4}'
}
declare -a CLAIM_FDS=() CLAIM_UUIDS=()
release_gpus() {
  local fd
  for fd in "${CLAIM_FDS[@]}"; do flock -u "$fd" 2>/dev/null || true; eval "exec ${fd}>&-"; done
  CLAIM_FDS=(); CLAIM_UUIDS=()
}
claim_gpus() {
  local ids="$1" required="$2" row index uuid free util fd
  IFS=',' read -r -a requested <<< "$ids"
  while true; do
    release_gpus
    for index in "${requested[@]}"; do
      row="$(gpu_row "$index")"; [[ -n "$row" ]] || continue
      IFS=',' read -r index uuid free util <<< "$row"
      (( free >= required && util <= MAX_UTIL )) || continue
      fd=$((20 + ${#CLAIM_FDS[@]})); eval "exec ${fd}>/tmp/prodipa_gpu_${uuid}.lock"
      if ! flock -n "$fd"; then eval "exec ${fd}>&-"; continue; fi
      CLAIM_FDS+=("$fd"); CLAIM_UUIDS+=("$uuid")
    done
    # Start as soon as one eligible card is available and scale up to every
    # currently eligible card.  A busy GPU must not make idle GPUs wait.
    if (( ${#CLAIM_UUIDS[@]} > 0 )); then return 0; fi
    sleep "$POLL"
  done
}

generate_model() {
  local paradigm="$1" model="$2" base="$3" adapter="$4"
  local final="$EVAL/generation/$paradigm/$model/$paradigm/$model.tsv"
  local shards="$EVAL/generation/$paradigm/$model/shards" shard start stop pid failed
  local slot batch allocation_gpu_count shard_file
  local -a batch_pids=()
  if [[ -s "$final" && $(( $(wc -l < "$final") - 1 )) -eq 512 ]]; then return 0; fi
  mkdir -p "$shards" "$OUT/logs/generation/$paradigm/$model"
  "$PY" "$ROOT/experiments/prodipa/conditional_multiteacher_student/prepare_generation_shards.py" \
    --existing "$final" --output-dir "$shards" --num-sequences 512 --num-shards 4
  claim_gpus "$GPUS" "$MIN_FREE"
  allocation_gpu_count="${#CLAIM_UUIDS[@]}"
  local allocation_started="$(date +%s)"
  failed=0
  for ((batch=0; batch<4; batch+=allocation_gpu_count)); do
    batch_pids=()
    for ((slot=0; slot<allocation_gpu_count; slot++)); do
      shard=$((batch + slot))
      (( shard < 4 )) || break
      shard_file="$shards/shard_${shard}.tsv"
      if [[ -s "$shard_file" && $(( $(wc -l < "$shard_file") - 1 )) -eq 128 ]]; then
        continue
      fi
      start=$((shard * 128)); stop=$((start + 128))
      CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[$slot]}" "$PY" \
        "$ROOT/experiments/prodipa/conditional_evaluation/generate_model.py" \
        --paradigm "$paradigm" --model-name "$model" --base-model "$base" \
        --adapter "$adapter" --out "$shard_file" \
        --num-sequences 512 --sample-id-start "$start" --sample-id-stop "$stop" \
        --base-seed 42 --temperature 0.7 --top-p 0.9 --top-k 200 \
        --repetition-penalty 1.2 --max-new-tokens 512 --device cuda \
        > "$OUT/logs/generation/$paradigm/$model/shard_${shard}.log" 2>&1 &
      batch_pids+=("$!")
    done
    for pid in "${batch_pids[@]}"; do wait "$pid" || failed=1; done
  done
  local allocation_completed="$(date +%s)"
  release_gpus
  (( failed == 0 ))
  "$PY" "$ROOT/experiments/prodipa/conditional_multiteacher_student/merge_generation_shards.py" \
    --input-dir "$shards" --output "$final" --num-sequences 512 --num-shards 4
  "$PY" - "$EVAL/timings/${paradigm}_${model}_generation.json" "$allocation_started" "$allocation_completed" "$allocation_gpu_count" <<'PY'
import json,sys
from pathlib import Path
p,start,stop,gpus=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
Path(p).write_text(json.dumps({"status":"COMPLETE","allocated_gpus":gpus,"wall_seconds":stop-start,"gpu_seconds":gpus*(stop-start)},indent=2)+"\n")
PY
}

stage=generation
write_status RUNNING "$stage"
generate_model conditional online_opd_paired __PRODIPA_MODELS__/ProLLaMA \
  "$OUT/conditional/training/online"
generate_model conditional prodipa_offline __PRODIPA_MODELS__/ProLLaMA \
  "$OUT/conditional/training/offline"
generate_model unconditional reference_theta0 __PRODIPA_MODELS__/ProtGPT2 \
  "$ROOT/outputs/unconditional/protgpt2_multiteacher_student/checkpoint-200"
generate_model unconditional online_opd_paired __PRODIPA_MODELS__/ProtGPT2 \
  "$OUT/unconditional/training/online"
generate_model unconditional prodipa_offline __PRODIPA_MODELS__/ProtGPT2 \
  "$OUT/unconditional/training/offline"

run_conditional_evaluation() {
  local model="$1"
  local generation="$EVAL/generation/conditional/$model/conditional/$model.tsv"
  local method_root="$EVAL/methods/conditional/$model"
  PARALLEL_GPUS="$GPUS" bash "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
    --model-id "$model" --generation-tsv "$generation" --output-root "$method_root" --gpus "$GPUS"
  if [[ ! -s "$method_root/scores/protrek.tsv" ]]; then
    claim_gpus "$GPUS" 8000
    local protrek_started="$(date +%s)"
    CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[0]}" "$PY" \
      "$ROOT/experiments/prodipa/conditional_evaluation/score_protrek_similarity.py" \
      --input "$method_root/prepared/unique_valid_sequences.tsv" \
      --output "$method_root/scores/protrek.tsv" \
      --repo "$ROOT/third_party/ProTrek" --model __PRODIPA_MODELS__/ProTrek_35M \
      --batch-size 32 --device cuda:0 > "$OUT/logs/${model}_protrek.log" 2>&1
    local protrek_completed="$(date +%s)"
    release_gpus
    "$PY" - "$EVAL/timings/conditional_${model}_protrek.json" "$protrek_started" "$protrek_completed" <<'PY'
import json,sys
from pathlib import Path
p,start,stop=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
Path(p).write_text(json.dumps({"status":"COMPLETE","allocated_gpus":1,"wall_seconds":stop-start,"gpu_seconds":stop-start},indent=2)+"\n")
PY
  fi
  PARALLEL_GPUS="$GPUS" bash "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
    --model-id "$model" --generation-tsv "$generation" \
    --protrek-tsv "$method_root/scores/protrek.tsv" \
    --output-root "$method_root" --gpus "$GPUS"
}

run_unconditional_evaluation() {
  local model="$1"
  local generation="$EVAL/generation/unconditional/$model/unconditional/$model.tsv"
  PARALLEL_GPUS="$GPUS" bash "$ROOT/experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh" \
    --model-id "$model" --generation-tsv "$generation" \
    --output-root "$EVAL/methods/unconditional/$model" --gpus "$GPUS"
}

stage=scoring
write_status RUNNING "$stage"
run_conditional_evaluation online_opd_paired 2>&1 | tee "$OUT/logs/conditional_online_evaluation.log"
run_conditional_evaluation prodipa_offline 2>&1 | tee "$OUT/logs/conditional_offline_evaluation.log"
for model in reference_theta0 online_opd_paired prodipa_offline; do
  run_unconditional_evaluation "$model" 2>&1 | tee "$OUT/logs/unconditional_${model}_evaluation.log"
done

stage=novelty_u
write_status RUNNING "$stage"
for paradigm in conditional unconditional; do
  if [[ "$paradigm" == conditional ]]; then
    inputs=(
      "$COND_REFERENCE/evaluation/methods/reference_theta0/prepared/unique_valid_sequences.tsv"
      "$EVAL/methods/conditional/online_opd_paired/prepared/unique_valid_sequences.tsv"
      "$EVAL/methods/conditional/prodipa_offline/prepared/unique_valid_sequences.tsv"
    )
  else
    inputs=(
      "$EVAL/methods/unconditional/reference_theta0/prepared/unique_valid_sequences.tsv"
      "$EVAL/methods/unconditional/online_opd_paired/prepared/unique_valid_sequences.tsv"
      "$EVAL/methods/unconditional/prodipa_offline/prepared/unique_valid_sequences.tsv"
    )
  fi
  "$PY" "$HERE/combine_sequences.py" --input "${inputs[@]}" \
    --output "$EVAL/novelty/${paradigm}_combined_sequences.tsv"
  "$PY" "$ROOT/experiments/prodipa/conditional_evaluation/score_mmseqs_novelty.py" \
    --input "$EVAL/novelty/${paradigm}_combined_sequences.tsv" \
    --reference-db __PRODIPA_DATABASES__/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03 \
    --out "$EVAL/novelty/${paradigm}_novelty_u.tsv" \
    --mmseqs "$ROOT/tools/mmseqs/bin/mmseqs" --threads 32 \
    --split-memory-limit 300G --tmp-dir "$EVAL/novelty/${paradigm}_mmseqs_tmp" \
    --reference-label UniProtKB_2026_03 \
    2>&1 | tee "$OUT/logs/${paradigm}_novelty_u.log"
done

stage=paper_outputs
write_status RUNNING "$stage"
"$PY" "$HERE/finalize_full_paper.py" --experiment-root "$OUT"
write_status COMPLETE complete
touch "$OUT/EVALUATION_COMPLETE"
