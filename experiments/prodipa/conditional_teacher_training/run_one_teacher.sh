#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PYTHON_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
BASE="${PROLLAMA_MODEL:-__PRODIPA_MODELS__/ProLLaMA}"
OUT="${CONDITIONAL_TEACHER_OUTPUT_ROOT:-$ROOT/outputs/conditional/prollama}"
ATTRIBUTE="${1:?usage: run_one_teacher.sh fold|sol|thermo GPU_DEVICE}"
GPU_DEVICE="${2:?GPU UUID or physical index required}"
TRAINER="$ROOT/src/prodipa/conditional/train_teacher.py"
VERIFY="$ROOT/experiments/prodipa/conditional_teacher_training/verify_adapter.py"
DIAGNOSE="$ROOT/experiments/prodipa/conditional_teacher_training/quick_diagnosis.py"

case "$ATTRIBUTE" in
  fold)
    TRAIN="$ROOT/data/conditional/fold/train_plddt_300.json"
    TEST="$ROOT/data/conditional/fold/test_plddt_100.json"
    EPOCHS=8 ;;
  sol)
    TRAIN="$ROOT/data/conditional/sol/train_sol_300.json"
    TEST="$ROOT/data/conditional/sol/test_sol_100.json"
    EPOCHS=8 ;;
  thermo)
    TRAIN="$ROOT/data/conditional/thermo/train_thermo_300.json"
    TEST="$ROOT/data/conditional/thermo/test_thermo_100.json"
    EPOCHS=8 ;;
  *) echo "unknown attribute: $ATTRIBUTE" >&2; exit 2 ;;
esac

TEACHER_DIR="$OUT/$ATTRIBUTE"
ADAPTER="$TEACHER_DIR/final_adapter"
TEST_EVAL="$TEACHER_DIR/test_evaluation"
DONE="$TEACHER_DIR/DONE.json"
mkdir -p "$TEACHER_DIR"

if [[ -s "$DONE" ]]; then
  "$PY" - "$DONE" "$TRAIN" "$TEST" <<'PY'
import hashlib, json, sys
from pathlib import Path
done, train, test = map(Path, sys.argv[1:])
record = json.loads(done.read_text())
assert record["status"] == "COMPLETE"
assert record["close_loss_weight"] == 4.0
assert record["train_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
assert record["test_sha256"] == hashlib.sha256(test.read_bytes()).hexdigest()
PY
  echo "$ATTRIBUTE Conditional teacher already complete"
  exit 0
fi
if [[ -e "$ADAPTER" && -n "$(find "$ADAPTER" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "incomplete nonempty adapter directory refuses overwrite: $ADAPTER" >&2
  exit 1
fi

mkdir -p "$ADAPTER" "$TEST_EVAL"
echo "[$(date --iso-8601=seconds)] train attribute=$ATTRIBUTE close_loss_weight=4 epochs=$EPOCHS"
CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PY" "$TRAINER" \
  --model_name_or_path "$BASE" --tokenizer_name_or_path "$BASE" \
  --train_file "$TRAIN" --max_seq_length 512 \
  --bf16 --torch_dtype bfloat16 --load_in_kbits 16 \
  --trainable q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 --gradient_checkpointing \
  --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --weight_decay 0 --optim adamw_torch --max_grad_norm 1.0 \
  --seed 42 --data_seed 42 --report_to none \
  --logging_strategy steps --logging_steps 1 --logging_first_step true \
  --preprocessing_num_workers 1 --do_train \
  --num_train_epochs "$EPOCHS" --eval_strategy no \
  --save_strategy epoch --save_total_limit 1 \
  --close_loss_weight 4 \
  --data_cache_dir "$ADAPTER/dataset_cache" \
  --output_dir "$ADAPTER" --overwrite_output_dir \
  2>&1 | tee -a "$ADAPTER/training.log"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PY" "$VERIFY" \
  --base "$BASE" --adapter "$ADAPTER" --data "$TRAIN" \
  --out "$TEACHER_DIR/adapter_verification.json" \
  2>&1 | tee "$TEACHER_DIR/adapter_verification.log"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PY" "$DIAGNOSE" \
  --base "$BASE" --adapter "$ADAPTER" --data "$TEST" --limit 100 \
  --output "$TEACHER_DIR/diagnosis_test100.json" \
  2>&1 | tee "$TEACHER_DIR/diagnosis_test100.log"

CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$PY" "$TRAINER" \
  --model_name_or_path "$BASE" --tokenizer_name_or_path "$BASE" \
  --peft_path "$ADAPTER" --validation_file "$TEST" \
  --data_cache_dir "$TEST_EVAL/dataset_cache" --max_seq_length 512 \
  --output_dir "$TEST_EVAL" --do_eval \
  --bf16 --torch_dtype bfloat16 --load_in_kbits 16 \
  --per_device_eval_batch_size 2 --eval_strategy no --report_to none \
  --seed 42 --data_seed 42 --preprocessing_num_workers 1 \
  2>&1 | tee "$TEST_EVAL/evaluation.log"

"$PY" - "$ATTRIBUTE" "$EPOCHS" "$BASE" "$TRAIN" "$TEST" "$ADAPTER" "$TEST_EVAL" "$DONE" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

attribute, epochs, base, train, test, adapter, test_eval, done = sys.argv[1:]
train, test, adapter, test_eval, done = map(Path, (train, test, adapter, test_eval, done))
payload = {
    "status": "COMPLETE",
    "teacher": attribute,
    "method": "termination-weighted conditional LoRA",
    "close_loss_weight": 4.0,
    "seed": 42,
    "epochs": int(epochs),
    "base_model": str(Path(base).resolve()),
    "train_file": str(train.resolve()),
    "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    "test_file": str(test.resolve()),
    "test_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
    "final_adapter": str(adapter.resolve()),
    "adapter_sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
    "train_metrics": json.loads((adapter / "train_results.json").read_text()),
    "test_metrics": json.loads((test_eval / "eval_results.json").read_text()),
    "adapter_verification": json.loads((done.parent / "adapter_verification.json").read_text()),
    "termination_diagnosis": str((done.parent / "diagnosis_test100.json").resolve()),
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
done.write_text(json.dumps(payload, indent=2) + "\n")
PY

# Successful formal adapters no longer need resumable checkpoints or dataset caches.
find "$ADAPTER" -mindepth 1 -maxdepth 1 -type d \( -name 'checkpoint-*' -o -name dataset_cache \) -prune -exec find {} -depth -delete \;
find "$TEST_EVAL" -mindepth 1 -maxdepth 1 -type d -name dataset_cache -prune -exec find {} -depth -delete \;
echo "[$(date --iso-8601=seconds)] completed attribute=$ATTRIBUTE"
