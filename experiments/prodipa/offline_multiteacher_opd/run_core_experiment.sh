#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TORCHRUN="${TORCHRUN_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/torchrun}"
HERE="$ROOT/experiments/prodipa/offline_multiteacher_opd"
THIRD_PARTY="$ROOT/third_party/ProteinOPD/conditional/proteinopd"
PYTHONPATH_VALUE="$ROOT/src:$THIRD_PARTY${PYTHONPATH:+:$PYTHONPATH}"
OUT="$ROOT/outputs/experiments/offline_multiteacher_opd"
CONFIG="$ROOT/experiments/prodipa/conditional_multiteacher_student/opd_config.yaml"
THETA_ZERO="$ROOT/outputs/conditional/prollama_multiteacher_student/checkpoint-192"
CACHE="$OUT/cache"
ONLINE="$OUT/training/online"
OFFLINE="$OUT/training/offline"
STATUS="$OUT/STATUS.json"
TRAIN_GPU_IDS="${TRAIN_GPUS:-0,3}"
EVAL_GPU_IDS="${EVAL_GPUS:-0,1,2,3}"
TRAIN_MIN_FREE_MIB="${TRAIN_MIN_FREE_MIB:-45000}"
GEN_MIN_FREE_MIB="${GEN_MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$OUT"/{logs,training,cache,evaluation/methods,evaluation/generation,results}
printf '%s\n' "$$" > "$OUT/pipeline.pid"
exec 7>"$OUT/.pipeline.lock"
flock -n 7 || { echo "another offline OPD core pipeline is active" >&2; exit 1; }
exec > >(tee -a "$OUT/pipeline.log") 2>&1

CURRENT_STAGE=preflight
status() {
  printf '{"state":"%s","stage":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$(date --iso-8601=seconds)" > "$STATUS"
}
on_error() { local rc=$?; status FAILED "$CURRENT_STAGE"; exit "$rc"; }
trap on_error ERR

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4}'
}

declare -a CLAIM_FDS=() CLAIM_UUIDS=()
release_gpus() {
  local fd
  for fd in "${CLAIM_FDS[@]}"; do
    flock -u "$fd" 2>/dev/null || true
    eval "exec ${fd}>&-"
  done
  CLAIM_FDS=()
  CLAIM_UUIDS=()
}

claim_gpus() {
  local ids="$1" minimum="$2" label="$3" row index uuid free util fd ok
  IFS=',' read -r -a requested <<< "$ids"
  while true; do
    release_gpus
    ok=1
    for index in "${requested[@]}"; do
      row="$(gpu_row "$index")"
      if [[ -z "$row" ]]; then ok=0; break; fi
      IFS=',' read -r index uuid free util <<< "$row"
      if (( free < minimum )); then ok=0; break; fi
      fd=$((20 + ${#CLAIM_FDS[@]}))
      eval "exec ${fd}>/tmp/prodipa_gpu_${uuid}.lock"
      if ! flock -n "$fd"; then
        eval "exec ${fd}>&-"
        ok=0
        break
      fi
      CLAIM_FDS+=("$fd")
      CLAIM_UUIDS+=("$uuid")
    done
    if (( ok == 1 && ${#CLAIM_UUIDS[@]} == ${#requested[@]} )); then
      echo "[$(date --iso-8601=seconds)] claimed $label GPUs: ${CLAIM_UUIDS[*]}"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] waiting for $label GPUs=$ids minimum_free_mib=$minimum"
    sleep "$POLL_SECONDS"
  done
}

CURRENT_STAGE=preflight
status RUNNING "$CURRENT_STAGE"
[[ -s "$CONFIG" && -s "$THETA_ZERO/adapter_config.json" && -s "$THETA_ZERO/adapter_model.bin" ]]
for attribute in fold sol thermo; do
  [[ -s "$ROOT/outputs/conditional/prollama/$attribute/final_adapter/adapter_model.safetensors" ]]
done
"$PY" - "$OUT/PREFLIGHT.json" "$CONFIG" "$THETA_ZERO" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
out,config,theta=map(Path,sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
out.write_text(json.dumps({
  "status":"PASS","checked_at":datetime.now().astimezone().isoformat(),
  "theta_zero":str(theta.resolve()),"theta_zero_adapter_sha256":sha(theta/'adapter_model.bin'),
  "opd_config":str(config.resolve()),"opd_config_sha256":sha(config),
  "online_updates":64,"offline_updates":64,"offline_cache_trajectories":512,
  "offline_reuse_epochs":4,"evaluation_requests_per_model":512,
},indent=2)+"\n")
PY

claim_gpus "$TRAIN_GPU_IDS" "$TRAIN_MIN_FREE_MIB" training
IFS=',' read -r -a TRAIN_ID_ARRAY <<< "$TRAIN_GPU_IDS"

CURRENT_STAGE=offline_cache
status RUNNING "$CURRENT_STAGE"
if [[ ! -s "$CACHE/CACHE_MANIFEST.json" ]]; then
  CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[0]}" PYTHONPATH="$PYTHONPATH_VALUE" "$PY" "$HERE/build_cache.py" \
    --config "$CONFIG" --theta-zero "$THETA_ZERO" --cache-dir "$CACHE" \
    --sample-start 0 --sample-stop 256 --sample-count 512 --batch-size 8 \
    --seed 42 --rank 0 --device cuda:0 > "$OUT/logs/cache_rank_0.log" 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[1]}" PYTHONPATH="$PYTHONPATH_VALUE" "$PY" "$HERE/build_cache.py" \
    --config "$CONFIG" --theta-zero "$THETA_ZERO" --cache-dir "$CACHE" \
    --sample-start 256 --sample-stop 512 --sample-count 512 --batch-size 8 \
    --seed 42 --rank 1 --device cuda:0 > "$OUT/logs/cache_rank_1.log" 2>&1 &
  p1=$!
  failed=0
  wait "$p0" || failed=1
  wait "$p1" || failed=1
  (( failed == 0 ))
  PYTHONPATH="$PYTHONPATH_VALUE" "$PY" "$HERE/finalize_cache.py" \
    --cache-dir "$CACHE" --config "$CONFIG" --theta-zero "$THETA_ZERO" \
    --sample-count 512 --shards 2 2>&1 | tee "$OUT/logs/cache_audit.log"
fi

CURRENT_STAGE=online_training
status RUNNING "$CURRENT_STAGE"
if [[ ! -s "$ONLINE/TRAINING_PROTOCOL.json" ]] || \
   [[ "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$ONLINE/TRAINING_PROTOCOL.json" 2>/dev/null || true)" != COMPLETE ]]; then
  CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[0]},${CLAIM_UUIDS[1]}" \
    PYTHONPATH="$PYTHONPATH_VALUE" WANDB_MODE=disabled \
    "$TORCHRUN" --nproc_per_node 2 --master_port 29641 "$HERE/train.py" \
      --mode online --config "$CONFIG" --theta-zero "$THETA_ZERO" \
      --output-dir "$ONLINE" --seed 42 2>&1 | tee "$OUT/logs/online_training.log"
fi

CURRENT_STAGE=offline_training
status RUNNING "$CURRENT_STAGE"
if [[ ! -s "$OFFLINE/TRAINING_PROTOCOL.json" ]] || \
   [[ "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$OFFLINE/TRAINING_PROTOCOL.json" 2>/dev/null || true)" != COMPLETE ]]; then
  CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[0]},${CLAIM_UUIDS[1]}" \
    PYTHONPATH="$PYTHONPATH_VALUE" WANDB_MODE=disabled \
    "$TORCHRUN" --nproc_per_node 2 --master_port 29642 "$HERE/train.py" \
      --mode offline --config "$CONFIG" --theta-zero "$THETA_ZERO" \
      --cache-dir "$CACHE" --output-dir "$OFFLINE" --seed 42 \
      2>&1 | tee "$OUT/logs/offline_training.log"
fi
release_gpus

declare -A ADAPTERS=(
  [reference_theta0]="$THETA_ZERO"
  [online_opd]="$ONLINE"
  [offline_opd]="$OFFLINE"
)

generate_model() {
  local model="$1"
  local adapter="$2"
  local final="$OUT/evaluation/generation/$model/conditional/$model.tsv"
  local shards="$OUT/evaluation/generation/$model/shards" shard start stop pid failed
  if [[ -s "$final" && $(( $(wc -l < "$final") - 1 )) -eq 512 ]]; then return 0; fi
  mkdir -p "$shards" "$OUT/logs/generation/$model"
  "$PY" "$ROOT/experiments/prodipa/conditional_multiteacher_student/prepare_generation_shards.py" \
    --existing "$final" --output-dir "$shards" --num-sequences 512 --num-shards 4
  claim_gpus "$EVAL_GPU_IDS" "$GEN_MIN_FREE_MIB" evaluation-generation
  failed=0
  for shard in 0 1 2 3; do
    start=$((shard * 128)); stop=$((start + 128))
    CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[$shard]}" "$PY" \
      "$ROOT/experiments/prodipa/conditional_evaluation/generate_model.py" \
      --paradigm conditional --model-name "$model" \
      --base-model __PRODIPA_MODELS__/ProLLaMA --adapter "$adapter" \
      --out "$shards/shard_${shard}.tsv" --num-sequences 512 \
      --sample-id-start "$start" --sample-id-stop "$stop" --base-seed 42 \
      --temperature 0.7 --top-p 0.9 --top-k 200 --repetition-penalty 1.2 \
      --max-new-tokens 512 \
      --superfamily 'Tetratricopeptide-like helical domain superfamily' --device cuda \
      > "$OUT/logs/generation/$model/shard_${shard}.log" 2>&1 &
    eval "p${shard}=\$!"
  done
  for shard in 0 1 2 3; do
    eval "pid=\$p${shard}"
    wait "$pid" || failed=1
  done
  release_gpus
  (( failed == 0 ))
  "$PY" "$ROOT/experiments/prodipa/conditional_multiteacher_student/merge_generation_shards.py" \
    --input-dir "$shards" --output "$final" --num-sequences 512 --num-shards 4
}

CURRENT_STAGE=evaluation_generation
status RUNNING "$CURRENT_STAGE"
evaluation_started_unix="$(date +%s)"
for model in reference_theta0 online_opd offline_opd; do
  generate_model "$model" "${ADAPTERS[$model]}"
done

run_evaluation() {
  local model="$1"
  local generation="$OUT/evaluation/generation/$model/conditional/$model.tsv"
  local method_root="$OUT/evaluation/methods/$model"
  PARALLEL_GPUS="$EVAL_GPU_IDS" bash \
    "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
    --model-id "$model" --generation-tsv "$generation" \
    --output-root "$method_root" --gpus "$EVAL_GPU_IDS"
  if [[ ! -s "$method_root/scores/protrek.tsv" ]]; then
    claim_gpus "${EVAL_GPU_IDS%%,*}" "$GEN_MIN_FREE_MIB" protrek
    CUDA_VISIBLE_DEVICES="${CLAIM_UUIDS[0]}" "$PY" \
      "$ROOT/experiments/prodipa/conditional_evaluation/score_protrek_similarity.py" \
      --input "$method_root/prepared/unique_valid_sequences.tsv" \
      --output "$method_root/scores/protrek.tsv" \
      --repo "$ROOT/third_party/ProTrek" --model __PRODIPA_MODELS__/ProTrek_35M \
      --batch-size 32 --device cuda:0 > "$OUT/logs/${model}_protrek.log" 2>&1
    release_gpus
  fi
  PARALLEL_GPUS="$EVAL_GPU_IDS" bash \
    "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
    --model-id "$model" --generation-tsv "$generation" \
    --protrek-tsv "$method_root/scores/protrek.tsv" \
    --output-root "$method_root" --gpus "$EVAL_GPU_IDS"
}

CURRENT_STAGE=evaluation_scoring
status RUNNING "$CURRENT_STAGE"
for model in reference_theta0 online_opd offline_opd; do
  run_evaluation "$model" 2>&1 | tee -a "$OUT/logs/${model}_evaluation.log"
done

CURRENT_STAGE=novelty_and_pooled_hv
status RUNNING "$CURRENT_STAGE"
"$PY" "$HERE/combine_sequences.py" \
  --input \
    "$OUT/evaluation/methods/reference_theta0/prepared/unique_valid_sequences.tsv" \
    "$OUT/evaluation/methods/online_opd/prepared/unique_valid_sequences.tsv" \
    "$OUT/evaluation/methods/offline_opd/prepared/unique_valid_sequences.tsv" \
  --output "$OUT/evaluation/combined_unique_valid_sequences.tsv"
"$PY" "$ROOT/experiments/prodipa/conditional_evaluation/score_mmseqs_novelty.py" \
  --input "$OUT/evaluation/combined_unique_valid_sequences.tsv" \
  --reference-db __PRODIPA_DATABASES__/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03 \
  --out "$OUT/evaluation/novelty_u.tsv" --mmseqs "$ROOT/tools/mmseqs/bin/mmseqs" \
  --threads 32 --split-memory-limit 300G --tmp-dir "$OUT/evaluation/mmseqs_tmp" \
  --reference-label UniProtKB_2026_03 2>&1 | tee "$OUT/logs/novelty_u.log"
"$PY" "$ROOT/experiments/prodipa/conditional_evaluation/complete_pooled_hv.py" \
  --sequence-scores \
    "$OUT/evaluation/methods/reference_theta0/results/sequence_scores.tsv" \
    "$OUT/evaluation/methods/online_opd/results/sequence_scores.tsv" \
    "$OUT/evaluation/methods/offline_opd/results/sequence_scores.tsv" \
  --novelty-u "$OUT/evaluation/novelty_u.tsv" \
  --output-table "$OUT/results/conditional_prodipa_main.tsv" \
  --output-sequence-scores "$OUT/results/conditional_sequence_scores.tsv" \
  --manifest "$OUT/results/conditional_hv_manifest.json" \
  --expected-models reference_theta0,online_opd,offline_opd \
  --requests-per-model 512 \
  --normalization-scope-label 'paired theta-zero, online OPD and offline OPD complete Conditional sequences'
evaluation_completed_unix="$(date +%s)"
"$PY" - "$OUT/evaluation/EVALUATION_TIMING.json" "$evaluation_started_unix" "$evaluation_completed_unix" <<'PY'
import json,sys
from pathlib import Path
out,start,stop=sys.argv[1:]
Path(out).write_text(json.dumps({
  "status":"COMPLETE","included_in_distillation_gpu_hours":False,
  "wall_seconds":int(stop)-int(start),
  "note":"Evaluation used the shared staged GPU allocator; its cost is deliberately separate from distillation GPU-hours."
},indent=2)+"\n")
PY

CURRENT_STAGE=final_report
status RUNNING "$CURRENT_STAGE"
"$PY" "$HERE/final_report.py" --experiment-root "$OUT"
status COMPLETE complete
printf '{"status":"COMPLETE","report":"%s","completed_at":"%s"}\n' \
  "$OUT/results/FINAL_REPORT.md" "$(date --iso-8601=seconds)" > "$OUT/COMPLETE.json"
echo "Offline Multi-Teacher OPD core validation complete"
