#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TORCHRUN="${TORCHRUN_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/torchrun}"
MODEL_ID=prollama_multiteacher_student
BASE="${PROLLAMA_BASE:-__PRODIPA_MODELS__/ProLLaMA}"
PROTREK_REPO="${PROTREK_REPO:-$ROOT/third_party/ProTrek}"
PROTREK_MODEL="${PROTREK_MODEL:-__PRODIPA_MODELS__/ProTrek_35M}"
HERE="$ROOT/experiments/prodipa/conditional_multiteacher_student"
SOURCE="$ROOT/third_party/ProteinOPD/conditional/proteinopd/prollama_opd_train.py"
TRAIN_OUT="$ROOT/outputs/conditional/$MODEL_ID"
EVAL_OUT="$ROOT/outputs/evaluation/conditional/methods/$MODEL_ID"
GEN_TSV="$TRAIN_OUT/generation/conditional/$MODEL_ID.tsv"
STATUS="$TRAIN_OUT/PIPELINE_STATUS.json"
GPUS="${TRAIN_GPUS:-0,1}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29631}"

mkdir -p "$TRAIN_OUT/logs" "$EVAL_OUT/logs" "$(dirname "$GEN_TSV")"
exec 8>"$TRAIN_OUT/.pipeline.lock"
flock -n 8 || { echo "another Conditional multi-teacher student pipeline is active" >&2; exit 1; }
exec > >(tee -a "$TRAIN_OUT/pipeline.log") 2>&1

write_status() {
  printf '{"state":"%s","stage":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$(date --iso-8601=seconds)" > "$STATUS"
}
CURRENT_STAGE=preflight
declare -a CHILDREN=()
MONITOR_PID=""
stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do
    [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${CHILDREN[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$MONITOR_PID" ]]; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
  fi
  CHILDREN=()
}
on_error() { local rc=$?; stop_children; write_status FAILED "$CURRENT_STAGE"; exit "$rc"; }
on_stop() { stop_children; write_status STOPPED "$CURRENT_STAGE"; exit 143; }
trap on_error ERR
trap on_stop INT TERM

cd "$ROOT"
write_status RUNNING "$CURRENT_STAGE"
[[ -s "$SOURCE" && -d "$BASE" ]]
for attribute in fold sol thermo; do
  adapter="$ROOT/outputs/conditional/prollama/$attribute/final_adapter"
  [[ -s "$adapter/adapter_config.json" && -s "$adapter/adapter_model.safetensors" ]]
done
[[ "$(awk -F',' '{print NF}' <<< "$GPUS")" == 2 ]]
[[ "$(awk -F',' '{print NF}' <<< "$EVAL_GPUS")" == 4 ]]

CURRENT_STAGE=train_student
write_status RUNNING "$CURRENT_STAGE"
if [[ ! -s "$TRAIN_OUT/TRAINING_COMPLETE.json" ]]; then
  resume_args=()
  latest_checkpoint="$(find "$TRAIN_OUT" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -n1)"
  [[ -z "$latest_checkpoint" ]] || resume_args=(--resume_from_checkpoint "$TRAIN_OUT/$latest_checkpoint")
  CUDA_VISIBLE_DEVICES="$GPUS" WANDB_MODE=disabled PYTHONPATH="$(dirname "$SOURCE")" \
    "$TORCHRUN" --nproc_per_node 2 --master_port "$MASTER_PORT" "$SOURCE" \
      --opd_config_path "$HERE/opd_config.yaml" \
      --output_dir "$TRAIN_OUT" \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 2 \
      --do_train --seed 42 --bf16 \
      --num_train_epochs 1.0 \
      --lr_scheduler_type cosine --learning_rate 2e-5 --warmup_ratio 0.1 \
      --weight_decay 0 --logging_strategy steps --logging_steps 5 \
      --report_to none --save_strategy steps --save_steps 64 --save_total_limit 2 \
      --eval_strategy no --load_in_kbits 16 --save_safetensors True \
      --ddp_find_unused_parameters False --gradient_checkpointing \
      "${resume_args[@]}" 2>&1 | tee -a "$TRAIN_OUT/logs/training.log"
  [[ -s "$TRAIN_OUT/adapter_config.json" ]]
  [[ -s "$TRAIN_OUT/adapter_model.safetensors" || -s "$TRAIN_OUT/adapter_model.bin" ]]
  "$PY" - "$TRAIN_OUT/TRAINING_COMPLETE.json" "$HERE/opd_config.yaml" <<'PY'
import hashlib, json, sys
from datetime import datetime
from pathlib import Path
out, config = map(Path, sys.argv[1:])
out.write_text(json.dumps({
    "status": "COMPLETE",
    "method": "equal-weight three-teacher JSD distillation",
    "condition": "Tetratricopeptide-like helical domain superfamily",
    "teacher_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "completed_at": datetime.now().astimezone().isoformat(),
}, indent=2) + "\n")
PY
fi

CURRENT_STAGE=generation
write_status RUNNING "$CURRENT_STAGE"
if [[ ! -s "$GEN_TSV" || $(( $(wc -l < "$GEN_TSV") - 1 )) -ne 512 ]]; then
  SHARD_DIR="$TRAIN_OUT/generation/shards"
  mkdir -p "$SHARD_DIR"
  "$PY" "$HERE/prepare_generation_shards.py" \
    --existing "$GEN_TSV" --output-dir "$SHARD_DIR" \
    --num-sequences 512 --num-shards 4
  IFS=',' read -r -a eval_gpus <<< "$EVAL_GPUS"
  CHILDREN=()
  for shard in 0 1 2 3; do
    start=$(( shard * 128 ))
    stop=$(( start + 128 ))
    shard_tsv="$SHARD_DIR/shard_${shard}.tsv"
    CUDA_VISIBLE_DEVICES="${eval_gpus[$shard]}" "$PY" \
      "$ROOT/experiments/prodipa/conditional_evaluation/generate_model.py" \
        --paradigm conditional --model-name "$MODEL_ID" --base-model "$BASE" \
        --adapter "$TRAIN_OUT" --out "$shard_tsv" --num-sequences 512 \
        --sample-id-start "$start" --sample-id-stop "$stop" --base-seed 42 \
        --temperature 0.7 --top-p 0.9 --top-k 200 --repetition-penalty 1.2 \
        --max-new-tokens 512 \
        --superfamily 'Tetratricopeptide-like helical domain superfamily' --device cuda \
        > "$TRAIN_OUT/logs/generation_shard_${shard}.log" 2>&1 &
    CHILDREN+=("$!")
  done
  "$PY" "$HERE/monitor_generation_shards.py" \
    --input-dir "$SHARD_DIR" --num-sequences 512 --num-shards 4 \
    --watch-pid "$$" --poll-seconds 10 \
    > "$TRAIN_OUT/logs/generation.log" 2>&1 &
  MONITOR_PID="$!"
  failed=0
  for pid in "${CHILDREN[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  CHILDREN=()
  if (( failed != 0 )); then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
    exit 1
  fi
  wait "$MONITOR_PID"
  MONITOR_PID=""
  "$PY" "$HERE/merge_generation_shards.py" \
    --input-dir "$SHARD_DIR" --output "$GEN_TSV" \
    --num-sequences 512 --num-shards 4
fi

CURRENT_STAGE=core_scoring
write_status RUNNING "$CURRENT_STAGE"
PARALLEL_GPUS="$EVAL_GPUS" bash "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
  --model-id "$MODEL_ID" --generation-tsv "$GEN_TSV" --output-root "$EVAL_OUT" \
  --gpus 0,1,2,3

CURRENT_STAGE=protrek
write_status RUNNING "$CURRENT_STAGE"
if [[ ! -s "$EVAL_OUT/scores/protrek.tsv" ]]; then
  CUDA_VISIBLE_DEVICES="${eval_gpus[0]:-${EVAL_GPUS%%,*}}" "$PY" \
    "$ROOT/experiments/prodipa/conditional_evaluation/score_protrek_similarity.py" \
      --input "$EVAL_OUT/prepared/unique_valid_sequences.tsv" \
      --output "$EVAL_OUT/scores/protrek.tsv" \
      --repo "$PROTREK_REPO" --model "$PROTREK_MODEL" \
      --batch-size 32 --device cuda:0 \
      2>&1 | tee -a "$EVAL_OUT/logs/protrek.log"
fi

CURRENT_STAGE=publish
write_status RUNNING "$CURRENT_STAGE"
PARALLEL_GPUS="$EVAL_GPUS" bash "$ROOT/experiments/prodipa/conditional_evaluation/run_method_evaluation.sh" \
  --model-id "$MODEL_ID" --generation-tsv "$GEN_TSV" \
  --protrek-tsv "$EVAL_OUT/scores/protrek.tsv" --output-root "$EVAL_OUT" \
  --gpus 0,1,2,3 --publish

CURRENT_STAGE=complete
write_status COMPLETE "$CURRENT_STAGE"
printf '{"status":"COMPLETE","model":"%s","completed_at":"%s"}\n' \
  "$MODEL_ID" "$(date --iso-8601=seconds)" > "$TRAIN_OUT/COMPLETE.json"
echo "Conditional multi-teacher student training and evaluation complete"
