#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/accelerate}"
BASE_MODEL="${PROTGPT2_MODEL:-__PRODIPA_MODELS__/ProtGPT2}"
TEACHER_CONFIG="${TEACHER_CONFIG:-$ROOT/src/prodipa/unconditional/teachers.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/unconditional/protgpt2_multiteacher_student}"
TRAINING_GPUS="${TRAINING_GPUS:-2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29617}"

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.training.lock"
if ! flock -n 9; then
  echo "Another unconditional student training process holds $OUTPUT_DIR/.training.lock" >&2
  exit 2
fi

if [[ -f "$OUTPUT_DIR/DONE.json" ]]; then
  echo "Training is already complete: $OUTPUT_DIR/DONE.json"
  exit 0
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$ROOT/experiments/prodipa/unconditional_student_training/preflight.py" \
  --project-root "$ROOT" \
  --teacher-config "$TEACHER_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/PREFLIGHT.json"

IFS=',' read -r -a gpu_ids <<< "$TRAINING_GPUS"
if [[ "${#gpu_ids[@]}" -ne 2 ]]; then
  echo "TRAINING_GPUS must contain exactly two physical GPU indices." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$TRAINING_GPUS"
export WANDB_MODE=disabled

resume_args=()
eos_args=()
if [[ -n "${STUDENT_LORA_ADAPTER_PATH:-}" ]]; then
  eos_args+=(--student_lora_adapter_path "$STUDENT_LORA_ADAPTER_PATH")
fi
if [[ -n "${EOS_ANCHOR_PATH:-}" ]]; then
  : "${EOS_ANCHOR_LOSS_WEIGHT:?Set an explicit full-sequence CE coefficient relative to JS; the old EOS-only default is not valid}"
  eos_args+=(--eos_anchor_path "$EOS_ANCHOR_PATH" --eos_anchor_weight "${EOS_ANCHOR_WEIGHT:-4}" --eos_anchor_loss_weight "$EOS_ANCHOR_LOSS_WEIGHT" --ddp_find_unused_parameters false)
fi
latest_checkpoint=""
while IFS= read -r candidate; do
  checkpoint="$OUTPUT_DIR/$candidate"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" \
        || ! -s "$checkpoint/optimizer.pt" \
        || ! -s "$checkpoint/scheduler.pt" \
        || ! -s "$checkpoint/trainer_state.json" ]]; then
    echo "Skipping incomplete checkpoint: $checkpoint" >&2
    continue
  fi
  if ! "$PYTHON_BIN" - "$checkpoint/optimizer.pt" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not isinstance(state, dict) or "state" not in state or "param_groups" not in state:
    raise SystemExit("optimizer checkpoint has an invalid structure")
PY
  then
    echo "Skipping corrupt checkpoint: $checkpoint" >&2
    continue
  fi
  latest_checkpoint="$candidate"
  break
done < <(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -Vr)
if [[ -n "$latest_checkpoint" ]]; then
  echo "Resuming from verified checkpoint: $OUTPUT_DIR/$latest_checkpoint"
  resume_args=(--resume_from_checkpoint "$OUTPUT_DIR/$latest_checkpoint")
fi

"$ACCELERATE_BIN" launch \
  --multi_gpu \
  --num_processes 2 \
  --mixed_precision bf16 \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --module prodipa.unconditional.train \
  --num_train_samples 8192 \
  --student_model_name_or_path "$BASE_MODEL" \
  --teacher_config_path "$TEACHER_CONFIG" \
  --prompt_mode unconditional \
  --max_new_tokens 512 \
  --max_length_aa "${MAX_LENGTH_AA:-0}" \
  --sequence_format "${SEQUENCE_FORMAT:-unwrapped}" \
  --temperature 1.0 \
  --repetition_penalty 1.2 \
  --top_p 0.95 \
  --top_k 500 \
  --beta 0.5 \
  --top_k_loss 0 \
  --useloss jsd \
  --use_entropy_rule true \
  --entropy_alpha 0.0 \
  --student_tune_mode lora \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --lora_target_modules c_attn c_proj c_fc \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 2 \
  --skip_memory_metrics false \
  --report_to none \
  --run_name protgpt2_multiteacher_student \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.1 \
  --bf16 \
  "${eos_args[@]}" \
  "${resume_args[@]}"
