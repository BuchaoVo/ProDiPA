#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
EXP="${EXPERIMENT_ROOT:-$ROOT/outputs/experiments/unconditional_teacher_balanced_fasta60_v1}"
OUT="$EXP/evaluation"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TEMBERTURE_PY="${TEMBERTURE_PYTHON:-__PRODIPA_ENVS__/temberture/bin/python}"
BASE="${PROTGPT2_MODEL:-__PRODIPA_MODELS__/ProtGPT2}"
OLD="$ROOT/outputs/unconditional/protgpt2/sol/best_adapter"
NEW="$EXP/teachers/sol/best_adapter"
GEN="$OUT/generation/unconditional"
PREP="$OUT/prepared"
SCORES="$OUT/scores"
HIST="$ROOT/outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers/scores"
ESMFOLD="${ESMFOLD_MODEL:-__PRODIPA_MODELS__/esmfold_v1_hf}"

mkdir -p "$OUT/logs" "$GEN" "$PREP" "$SCORES" "$OUT/results"
exec 8>"$OUT/.comparison_controller.lock"
flock -n 8 || { echo "another comparison controller owns $OUT" >&2; exit 1; }
exec > >(tee -a "$OUT/evaluation.log") 2>&1

status() {
  "$PY" - "$OUT/STATUS.json" "$1" "$2" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
path=Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
payload={"state":sys.argv[2],"stage":sys.argv[3],"updated_at":datetime.now(timezone.utc).isoformat()}
with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False) as f:
    json.dump(payload,f,indent=2);f.write("\n");tmp=f.name
os.replace(tmp,path)
PY
}

run_on_gpu() {
  local gpu="$1" log="$2"; shift 2
  local uuid
  uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
  (
    exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
    flock 9
    echo "[$(date --iso-8601=seconds)] GPU=$gpu UUID=$uuid command=$*"
    CUDA_VISIBLE_DEVICES="$uuid" CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$ROOT/src" "$@"
  ) > >(tee -a "$log") 2>&1
}

status WAITING training
while [[ ! -s "$EXP/TRAINING_COMPLETE.json" ]]; do
  if [[ -s "$EXP/TRAINING_STATUS.json" ]] && grep -q '"state": "FAILED"' "$EXP/TRAINING_STATUS.json"; then
    status FAILED training
    exit 1
  fi
  sleep 30
done

status RUNNING generation_and_endpoint_eos
generate_common=(
  "$PY" -u "$ROOT/experiments/prodipa/conditional_evaluation/generate_model.py"
  --paradigm unconditional --base-model "$BASE" --num-sequences 512 --base-seed 42
  --temperature 0.7 --top-p 0.9 --top-k 200 --repetition-penalty 1.2
  --max-new-tokens 512 --generation-max-aa 512 --sequence-format legacy
)
run_on_gpu 0 "$OUT/logs/old_sol_generation.log" "${generate_common[@]}" \
  --model-name old_sol_teacher --adapter "$OLD" --out "$GEN/old_sol_teacher.tsv" &
pid_old=$!
run_on_gpu 1 "$OUT/logs/new_sol_generation.log" "${generate_common[@]}" \
  --model-name new_sol_teacher --adapter "$NEW" --out "$GEN/new_sol_teacher.tsv" &
pid_new=$!
run_on_gpu 2 "$OUT/logs/endpoint_eos.log" \
  "$PY" -u "$ROOT/experiments/prodipa/unconditional_teacher_training/measure_endpoint_eos.py" \
  --base-model "$BASE" --old-adapter "$OLD" --new-adapter "$NEW" \
  --validation "$ROOT/data/rebuild/unconditional_teacher_balanced/test_sol_100.csv" \
  --out-dir "$OUT/endpoint_eos" &
pid_eos=$!
wait "$pid_old"; wait "$pid_new"; wait "$pid_eos"

status RUNNING prepare
PYTHONPATH="$ROOT/src" "$PY" -u "$ROOT/experiments/prodipa/conditional_evaluation/prepare_sequences.py" \
  --generation-root "$OUT/generation" --out-root "$PREP" --requests-per-model 512 \
  --paradigm unconditional --models old_sol_teacher,new_sol_teacher \
  > >(tee -a "$OUT/logs/prepare.log") 2>&1

status RUNNING protein_sol
PYTHONPATH="$ROOT/src" "$PY" -u "$ROOT/scripts/run_protein_sol.py" \
  --input "$PREP/unique_valid_sequences.tsv" --work "$OUT/work/protein_sol" \
  --reuse "$HIST/protein_sol.tsv" --out "$SCORES/protein_sol.tsv" \
  > >(tee -a "$OUT/logs/protein_sol.log") 2>&1

status RUNNING temberture
run_on_gpu 0 "$OUT/logs/temberture.log" \
  "$TEMBERTURE_PY" -u "$ROOT/scripts/run_temBERTure_cls.py" \
  --input "$PREP/unique_valid_sequences.tsv" --repo "$ROOT/tools/TemBERTure" \
  --reuse "$HIST/temberture.tsv" --out "$SCORES/temberture.tsv" --batch-size 8 --device cuda

status RUNNING esmfold
PYTHONPATH="$ROOT/src" "$PY" -u "$ROOT/scripts/run_esmfold_properties.py" \
  --input "$PREP/unique_valid_sequences.tsv" --out "$SCORES/esmfold.tsv" \
  --model "$ESMFOLD" --reuse "$HIST/esmfold.tsv" --chunk-size 32 --num-recycles 4 \
  --batch-size 1 --max-tokens-per-batch 3200 --auto-gpus --max-gpus 4 \
  --min-free-memory-gib 48 --max-gpu-utilization 35 \
  > >(tee -a "$OUT/logs/esmfold.log") 2>&1

status RUNNING summarize
PYTHONPATH="$ROOT/src" "$PY" -u \
  "$ROOT/experiments/prodipa/unconditional_teacher_training/summarize_sol_teacher_comparison.py" \
  --generation-dir "$GEN" --structure "$SCORES/esmfold.tsv" \
  --sol "$SCORES/protein_sol.tsv" --thermo "$SCORES/temberture.tsv" \
  --endpoint-eos "$OUT/endpoint_eos/endpoint_eos_summary.tsv" --out-dir "$OUT/results" \
  > >(tee -a "$OUT/logs/summarize.log") 2>&1

status COMPLETE complete
date --iso-8601=seconds > "$OUT/EVALUATION_COMPLETE"
echo "Balanced Sol teacher comparison complete: $OUT/results/REPORT.md"
