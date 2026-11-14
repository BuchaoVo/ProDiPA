#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task name required}"
ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/conditional_evaluation"
OUT="${EVALUATION_OUTPUT_ROOT:-$ROOT/outputs/evaluation/teacher_generation_evaluation}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
PROTGPT2="${PROTGPT2_MODEL:-__PRODIPA_MODELS__/ProtGPT2}"
PROLLAMA="${PROLLAMA_MODEL:-__PRODIPA_MODELS__/ProLLaMA}"
ESMFOLD="${ESMFOLD_MODEL:-__PRODIPA_MODELS__/esmfold_v1_hf}"
MMSEQS="${MMSEQS_BIN:-$ROOT/tools/mmseqs/bin/mmseqs}"
SHARDS="${ESMFOLD_SHARDS:-16}"
UNCONDITIONAL_FOLD_ADAPTER="${UNCONDITIONAL_FOLD_ADAPTER:-$ROOT/outputs/unconditional/protgpt2/fold/best_adapter}"
UNCONDITIONAL_SOL_ADAPTER="${UNCONDITIONAL_SOL_ADAPTER:-$ROOT/outputs/unconditional/protgpt2/sol/best_adapter}"
UNCONDITIONAL_THERMO_ADAPTER="${UNCONDITIONAL_THERMO_ADAPTER:-$ROOT/outputs/unconditional/protgpt2/thermo/best_adapter}"
GEN="$OUT/generation"
PREP="$OUT/prepared"
SCORES="$OUT/scores"
mkdir -p "$GEN/unconditional" "$GEN/conditional" "$PREP" "$SCORES" "$OUT/logs" "$OUT/.done"

# Serialize duplicate invocations of the same task.  This lets the main
# evaluation handoff and manually parallelized workers share one output root
# without generating the same cohort twice or treating a partial TSV as done.
exec {TASK_LOCK_FD}>"$OUT/.task_${TASK}.lock"
flock "$TASK_LOCK_FD"
if [[ -s "$OUT/.done/$TASK" ]]; then
  echo "task already complete: $TASK"
  exit 0
fi

generate() {
  local paradigm="$1" model="$2" base="$3" adapter="$4"
  local adapter_args=()
  local protocol_args=()
  if [[ -n "$adapter" ]]; then adapter_args=(--adapter "$adapter"); fi
  if [[ "$paradigm" == unconditional ]]; then
    protocol_args=(--sequence-format legacy --generation-max-aa 512)
  fi
  "$PY" "$HERE/generate_model.py" \
    --paradigm "$paradigm" --model-name "$model" --base-model "$base" \
    "${adapter_args[@]}" --out "$GEN/$paradigm/$model.tsv" \
    --num-sequences 512 --base-seed 42 --temperature 0.7 --top-p 0.9 \
    --top-k 200 --repetition-penalty 1.2 --max-new-tokens 512 \
    "${protocol_args[@]}"
}

case "$TASK" in
  generation_unconditional_base)
    generate unconditional base "$PROTGPT2" "" ;;
  generation_unconditional_fold_teacher)
    generate unconditional fold_teacher "$PROTGPT2" "$UNCONDITIONAL_FOLD_ADAPTER" ;;
  generation_unconditional_sol_teacher)
    generate unconditional sol_teacher "$PROTGPT2" "$UNCONDITIONAL_SOL_ADAPTER" ;;
  generation_unconditional_thermo_teacher)
    generate unconditional thermo_teacher "$PROTGPT2" "$UNCONDITIONAL_THERMO_ADAPTER" ;;
  generation_conditional_base)
    generate conditional base "$PROLLAMA" "" ;;
  generation_conditional_fold_teacher)
    generate conditional fold_teacher "$PROLLAMA" "$ROOT/outputs/conditional/prollama/fold/final_adapter" ;;
  generation_conditional_sol_teacher)
    generate conditional sol_teacher "$PROLLAMA" "$ROOT/outputs/conditional/prollama/sol/final_adapter" ;;
  generation_conditional_thermo_teacher)
    generate conditional thermo_teacher "$PROLLAMA" "$ROOT/outputs/conditional/prollama/thermo/final_adapter" ;;
  prepare)
    "$PY" "$HERE/prepare_sequences.py" --generation-root "$GEN" --out-root "$PREP"
    "$PY" "$HERE/build_novelty_references.py" --project-root "$ROOT" --out-root "$PREP/references" ;;
  protein_sol)
    "$PY" "$ROOT/scripts/run_protein_sol.py" --input "$PREP/unique_valid_sequences.tsv" \
      --work "$OUT/work/protein_sol" --out "$SCORES/protein_sol.tsv" ;;
  novelty_t_unconditional)
    "$PY" "$HERE/score_mmseqs_novelty.py" --input "$PREP/unique_valid_sequences.tsv" \
      --reference "$PREP/references/unconditional_train_union.fasta" \
      --reference-label unconditional_teacher_train_union_300x3 \
      --mmseqs "$MMSEQS" --out "$SCORES/novelty_t_unconditional.tsv" ;;
  novelty_t_conditional)
    "$PY" "$HERE/score_mmseqs_novelty.py" --input "$PREP/unique_valid_sequences.tsv" \
      --reference "$PREP/references/conditional_train_union.fasta" \
      --reference-label conditional_teacher_train_union_300x3 \
      --mmseqs "$MMSEQS" --out "$SCORES/novelty_t_conditional.tsv" ;;
  novelty_u)
    : "${UNIPROTKB_FASTA:?UNIPROTKB_FASTA must point to the frozen full UniProtKB FASTA}"
    "$PY" "$HERE/score_mmseqs_novelty.py" --input "$PREP/unique_valid_sequences.tsv" \
      --reference "$UNIPROTKB_FASTA" --reference-label "${UNIPROTKB_RELEASE:?set UNIPROTKB_RELEASE}" \
      --mmseqs "$MMSEQS" --out "$SCORES/novelty_u.tsv" ;;
  protgpt2_ppl)
    "$PY" "$HERE/score_protgpt2_ppl.py" --input "$PREP/unique_valid_sequences.tsv" \
      --model "$PROTGPT2" --out "$SCORES/protgpt2_ppl.tsv" ;;
  temberture)
    "$PY" "$ROOT/scripts/run_temBERTure_cls.py" --input "$PREP/unique_valid_sequences.tsv" \
      --repo "$ROOT/tools/TemBERTure" --out "$SCORES/temberture.tsv" --batch-size 8 --device cuda ;;
  esmfold_*)
    index="${TASK#esmfold_}"
    precision_args=()
    case "${ESMFOLD_TF32:-inherit}" in
      true|1) precision_args=(--tf32) ;;
      false|0) precision_args=(--no-tf32) ;;
      inherit) ;;
      *) echo "ESMFOLD_TF32 must be inherit, true or false" >&2; exit 2 ;;
    esac
    "$PY" "$ROOT/scripts/run_esmfold_properties.py" --input "$PREP/unique_valid_sequences.tsv" \
      --out "$SCORES/esmfold_shard_$(printf '%02d' "$index").tsv" --model "$ESMFOLD" \
      --chunk-size 32 --num-recycles 4 --num-shards "$SHARDS" --shard-index "$index" \
      --batch-size "${ESMFOLD_BATCH_SIZE:-1}" --max-tokens-per-batch "${ESMFOLD_MAX_TOKENS:-3200}" \
      "${precision_args[@]}" ;;
  merge_structure)
    "$PY" "$HERE/merge_structure_shards.py" --input "$PREP/unique_valid_sequences.tsv" \
      --shards-root "$SCORES" --num-shards "$SHARDS" --out "$SCORES/esmfold.tsv" ;;
  summarize)
    extra=()
    [[ -s "$SCORES/novelty_u.tsv" ]] && extra+=(--novelty-u "$SCORES/novelty_u.tsv")
    [[ -s "$SCORES/protrek.tsv" ]] && extra+=(--protrek "$SCORES/protrek.tsv")
    "$PY" "$HERE/summarize_results.py" --requests "$PREP/all_requests.tsv" \
      --structure "$SCORES/esmfold.tsv" --sol "$SCORES/protein_sol.tsv" \
      --thermo "$SCORES/temberture.tsv" --ppl "$SCORES/protgpt2_ppl.tsv" \
      --novelty-t-unconditional "$SCORES/novelty_t_unconditional.tsv" \
      --novelty-t-conditional "$SCORES/novelty_t_conditional.tsv" \
      "${extra[@]}" --out-root "$OUT/results" ;;
  *) echo "unknown task: $TASK" >&2; exit 2 ;;
esac

date --iso-8601=seconds > "$OUT/.done/$TASK"
