#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
OUT="${OUTPUT_ROOT:-$ROOT/outputs/experiments/unconditional_teacher_balanced_fasta60_v1}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TRAINER="$ROOT/src/prodipa/unconditional/train_teacher.py"
GPUS="${TRAINING_GPUS:-0,1,2}"

mkdir -p "$OUT/logs"
exec 8>"$OUT/.parallel_training.lock"
flock -n 8 || { echo "another balanced teacher training controller owns $OUT" >&2; exit 1; }
exec > >(tee -a "$OUT/training_controller.log") 2>&1

IFS=',' read -r -a GPU_IDS <<<"$GPUS"
if (( ${#GPU_IDS[@]} < 3 )); then
  echo "Three distinct GPU indices are required for the three independent teachers" >&2
  exit 2
fi

attributes=(fold sol thermo)
pids=()

write_status() {
  "$PY" - "$OUT/TRAINING_STATUS.json" "$1" "$2" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY
}

stop_children() {
  local pid
  for pid in "${pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
}
trap 'write_status STOPPED signal_received; stop_children; exit 143' INT TERM

write_status RUNNING "parallel Fold/Sol/Thermo training on GPU indices $GPUS"

for i in 0 1 2; do
  attribute="${attributes[$i]}"
  gpu="${GPU_IDS[$i]}"
  uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
  config="$OUT/teachers/configs/$attribute.yaml"
  [[ -s "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
  (
    exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
    flock 9
    echo "[$(date --iso-8601=seconds)] START attribute=$attribute gpu=$gpu uuid=$uuid"
    CUDA_VISIBLE_DEVICES="$uuid" CUDA_DEVICE_ORDER=PCI_BUS_ID WANDB_MODE=disabled \
      PYTHONPATH="$ROOT/src" "$PY" -u "$TRAINER" --config "$config"
    echo "[$(date --iso-8601=seconds)] COMPLETE attribute=$attribute gpu=$gpu uuid=$uuid"
  ) > >(tee -a "$OUT/logs/${attribute}_training.log") 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done

if (( failed )); then
  write_status FAILED "one or more teacher trainers failed"
  exit 1
fi

for attribute in "${attributes[@]}"; do
  [[ -s "$OUT/teachers/$attribute/DONE.json" ]]
  [[ -s "$OUT/teachers/$attribute/best_adapter/adapter_model.safetensors" ]]
done
write_status COMPLETE "all three balanced FASTA60 Prefix teachers completed"
"$PY" - "$OUT/TRAINING_COMPLETE.json" "$GPUS" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "status": "COMPLETE",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "gpu_indices": sys.argv[2],
    "teachers": ["fold", "sol", "thermo"],
}, indent=2) + "\n")
PY
