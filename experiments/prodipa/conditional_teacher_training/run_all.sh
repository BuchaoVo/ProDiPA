#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PYTHON_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
BASE="${PROLLAMA_MODEL:-__PRODIPA_MODELS__/ProLLaMA}"
TRAIN_OUT="$ROOT/outputs/conditional/prollama"
EVAL_OUT="$ROOT/outputs/evaluation/conditional"
HERE="$ROOT/experiments/prodipa/conditional_teacher_training"
EVAL_CODE="$ROOT/experiments/prodipa/conditional_evaluation"
PROTREK_REPO="${PROTREK_REPO:-$ROOT/third_party/ProTrek}"
PROTREK_MODEL="${PROTREK_MODEL:-__PRODIPA_MODELS__/ProTrek_35M}"
MMSEQS="${MMSEQS_BIN:-$ROOT/tools/mmseqs/bin/mmseqs}"
UNIPROT_DB="${UNIPROT_DB:-__PRODIPA_DATABASES__/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03}"

mkdir -p "$TRAIN_OUT" "$EVAL_OUT/logs" "$EVAL_OUT/.done"
exec 8>"$TRAIN_OUT/.training_pipeline.lock"
flock -n 8 || { echo "another Conditional teacher pipeline is active" >&2; exit 1; }
exec > >(tee -a "$TRAIN_OUT/training_pipeline.log") 2>&1

write_status() {
  printf '{"state":"%s","stage":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$(date --iso-8601=seconds)" > "$TRAIN_OUT/PIPELINE_STATUS.json"
}
CURRENT_STAGE=preflight
declare -a CHILDREN=()
stop_children() {
  local pid
  for pid in "${CHILDREN[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${CHILDREN[@]:-}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
}
on_error() { local rc=$?; write_status FAILED "$CURRENT_STAGE"; stop_children; exit "$rc"; }
on_stop() { write_status STOPPED "$CURRENT_STAGE"; stop_children; exit 143; }
trap on_error ERR
trap on_stop INT TERM

cd "$ROOT"
audit="$ROOT/data/provenance/conditional_teacher_audit.json"
"$PY" - "$audit" "$TRAIN_OUT/data_preflight.json" <<'PY'
import json, shutil, sys
source, target = sys.argv[1:]
record = json.load(open(source))
if record.get("status") != "PASS" or record.get("failures"):
    raise SystemExit(f"conditional dataset audit failed: {record}")
if record.get("thresholds") != {"plddt": 85.0, "sol": 0.7, "thermo": 0.6}:
    raise SystemExit(f"unexpected thresholds: {record.get('thresholds')}")
shutil.copyfile(source, target)
PY
"$PY" - "$TRAIN_OUT/TRAINING_PROTOCOL.json" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "method": "termination-weighted conditional LoRA",
    "close_loss_weight": 4.0,
    "seed": 42,
    "epochs": {"fold": 8, "sol": 8, "thermo": 8},
    "lora": {"rank": 8, "alpha": 16, "dropout": 0.1},
    "optimizer": "AdamW",
    "learning_rate": 2e-5,
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "precision": "BF16",
    "max_sequence_tokens": 512,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
PY

gpu_uuid() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F',' -v wanted="$1" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $2}'
}

CURRENT_STAGE=train_teachers
write_status RUNNING "$CURRENT_STAGE"
CHILDREN=()
attributes=(fold sol thermo)
gpus=(1 2 3)
for index in 0 1 2; do
  attribute="${attributes[$index]}"; gpu="${gpus[$index]}"; uuid="$(gpu_uuid "$gpu")"
  [[ -n "$uuid" ]] || { echo "cannot resolve GPU $gpu" >&2; exit 1; }
  mkdir -p "$TRAIN_OUT/$attribute"
  echo "[$(date --iso-8601=seconds)] launch $attribute training on physical GPU=$gpu uuid=$uuid"
  bash "$HERE/run_one_teacher.sh" "$attribute" "$uuid" \
    > >(tee -a "$TRAIN_OUT/$attribute/launcher.log") 2>&1 &
  CHILDREN+=("$!")
done
failed=0
for pid in "${CHILDREN[@]}"; do if ! wait "$pid"; then failed=1; fi; done
CHILDREN=()
(( failed == 0 )) || exit 1
for attribute in "${attributes[@]}"; do [[ -s "$TRAIN_OUT/$attribute/DONE.json" ]] || exit 1; done
printf '{"status":"COMPLETE","method":"termination-weighted conditional LoRA","close_loss_weight":4,"completed_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$TRAIN_OUT/ALL_DONE.json"

generate_one() {
  local model="$1" gpu="$2" adapter="$3" uuid output
  uuid="$(gpu_uuid "$gpu")"
  output="$EVAL_OUT/generation/conditional/$model.tsv"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" && "$(wc -l < "$output")" -eq 513 ]]; then
    echo "reuse complete generation: $model"
    return
  fi
  adapter_args=()
  [[ -n "$adapter" ]] && adapter_args=(--adapter "$adapter")
  CUDA_VISIBLE_DEVICES="$uuid" "$PY" "$EVAL_CODE/generate_model.py" \
    --paradigm conditional --model-name "$model" --base-model "$BASE" \
    "${adapter_args[@]}" --out "$output" --num-sequences 512 --base-seed 42 \
    --temperature 0.7 --top-p 0.9 --top-k 200 --repetition-penalty 1.2 \
    --max-new-tokens 512 \
    --superfamily 'Tetratricopeptide-like helical domain superfamily' --device cuda \
    2>&1 | tee -a "$EVAL_OUT/logs/generation_${model}.log"
}

CURRENT_STAGE=generation
write_status RUNNING "$CURRENT_STAGE"
CHILDREN=()
generate_one base 0 "" & CHILDREN+=("$!")
generate_one fold_teacher 1 "$TRAIN_OUT/fold/final_adapter" & CHILDREN+=("$!")
generate_one sol_teacher 2 "$TRAIN_OUT/sol/final_adapter" & CHILDREN+=("$!")
generate_one thermo_teacher 3 "$TRAIN_OUT/thermo/final_adapter" & CHILDREN+=("$!")
failed=0
for pid in "${CHILDREN[@]}"; do if ! wait "$pid"; then failed=1; fi; done
CHILDREN=()
(( failed == 0 )) || exit 1

CURRENT_STAGE=prepare
write_status RUNNING "$CURRENT_STAGE"
"$PY" "$EVAL_CODE/prepare_sequences.py" \
  --generation-root "$EVAL_OUT/generation" --out-root "$EVAL_OUT/prepared" \
  --requests-per-model 512 --paradigm conditional \
  --models base,fold_teacher,sol_teacher,thermo_teacher \
  2>&1 | tee -a "$EVAL_OUT/logs/prepare.log"
date --iso-8601=seconds > "$EVAL_OUT/.done/prepare"
"$PY" "$EVAL_CODE/build_novelty_references.py" \
  --project-root "$ROOT" --out-root "$EVAL_OUT/prepared/references" \
  2>&1 | tee -a "$EVAL_OUT/logs/novelty_references.log"
date --iso-8601=seconds > "$EVAL_OUT/.done/novelty_references"

CURRENT_STAGE=protrek
write_status RUNNING "$CURRENT_STAGE"
CUDA_VISIBLE_DEVICES="$(gpu_uuid 1)" "$PY" "$EVAL_CODE/score_protrek_similarity.py" \
  --input "$EVAL_OUT/prepared/unique_valid_sequences.tsv" \
  --output "$EVAL_OUT/scores/protrek.tsv" --repo "$PROTREK_REPO" \
  --model "$PROTREK_MODEL" --batch-size 32 --device cuda:0 \
  2>&1 | tee -a "$EVAL_OUT/logs/protrek.log"

CURRENT_STAGE=core_scoring
write_status RUNNING "$CURRENT_STAGE"
PROJECT_ROOT="$ROOT" GENERATION_ROOT="$EVAL_OUT/generation" \
PARADIGM_EVALUATION_OUTPUT_ROOT="$EVAL_OUT" \
EVALUATION_MODELS=base,fold_teacher,sol_teacher,thermo_teacher \
PARALLEL_GPUS=0,1,2,3 MIN_FREE_MIB=28000 LIGHT_MIN_FREE_MIB=6000 \
MAX_UTIL_PCT=100 STABLE_POLLS=1 POLL_SECONDS=30 \
FOLD_THRESHOLD=85 SOL_THRESHOLD=0.7 THERMO_THRESHOLD=0.6 \
  bash "$EVAL_CODE/run_separate_paradigm_scoring_queue.sh" conditional

CURRENT_STAGE=novelty_u
write_status RUNNING "$CURRENT_STAGE"
"$PY" "$EVAL_CODE/score_mmseqs_novelty.py" \
  --input "$EVAL_OUT/prepared/unique_valid_sequences.tsv" \
  --reference-db "$UNIPROT_DB" --reference-label UniProtKB_2026_03 \
  --mmseqs "$MMSEQS" --threads 32 --split-memory-limit 300G \
  --tmp-dir "$EVAL_OUT/work/mmseqs_uniprot" --out "$EVAL_OUT/scores/novelty_u.tsv" \
  2>&1 | tee -a "$EVAL_OUT/logs/novelty_u.log"

CURRENT_STAGE=finalize
write_status RUNNING "$CURRENT_STAGE"
"$PY" "$EVAL_CODE/summarize_results.py" \
  --requests "$EVAL_OUT/prepared/all_requests.tsv" \
  --structure "$EVAL_OUT/scores/esmfold.tsv" \
  --sol "$EVAL_OUT/scores/protein_sol.tsv" \
  --thermo "$EVAL_OUT/scores/temberture.tsv" \
  --ppl "$EVAL_OUT/scores/protgpt2_ppl.tsv" \
  --novelty-t-conditional "$EVAL_OUT/scores/novelty_t.tsv" \
  --novelty-u "$EVAL_OUT/scores/novelty_u.tsv" \
  --protrek "$EVAL_OUT/scores/protrek.tsv" \
  --fold-threshold 85 --sol-threshold 0.7 --thermo-threshold 0.6 \
  --out-root "$EVAL_OUT/results" \
  2>&1 | tee -a "$EVAL_OUT/logs/finalize.log"

write_status COMPLETE complete
printf '{"status":"COMPLETE","close_loss_weight":4,"models":4,"requests_per_model":512,"completed_at":"%s"}\n' \
  "$(date --iso-8601=seconds)" > "$EVAL_OUT/COMPLETE.json"
echo "Conditional teacher training, generation, and scoring complete"
