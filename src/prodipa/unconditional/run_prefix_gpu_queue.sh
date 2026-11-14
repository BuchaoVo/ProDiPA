#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
: "${PROTGPT2_MODEL:?Set PROTGPT2_MODEL to your local pretrained ProtGPT2 directory}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT/outputs/unconditional/protgpt2}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/gpu_prefix_queue.py" \
  --data-dir "$PROJECT/data/unconditional" \
  --model "$PROTGPT2_MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --min-free-mib "${MIN_FREE_MIB:-51200}" \
  --max-util-pct "${MAX_UTIL_PCT:-15}" \
  --poll-seconds "${POLL_SECONDS:-60}" \
  --stable-polls "${STABLE_POLLS:-3}" \
  "$@"
