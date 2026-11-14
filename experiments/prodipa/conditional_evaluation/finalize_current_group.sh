#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/conditional_evaluation"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
OUT="${CONDITIONAL_EVALUATION_OUTPUT_ROOT:-$ROOT/outputs/evaluation/conditional}"
CONFIG="${CONDITIONAL_EVALUATION_CONFIG:-$ROOT/configs/evaluation/conditional.json}"

required=(
  "$OUT/prepared/all_requests.tsv"
  "$OUT/prepared/unique_valid_sequences.tsv"
  "$OUT/scores/esmfold.tsv"
  "$OUT/scores/protein_sol.tsv"
  "$OUT/scores/temberture.tsv"
  "$OUT/scores/protgpt2_ppl.tsv"
  "$OUT/scores/novelty_t.tsv"
  "$OUT/scores/novelty_u.tsv"
  "$OUT/scores/protrek.tsv"
)
for path in "${required[@]}"; do [[ -s "$path" ]] || { echo "missing required score: $path" >&2; exit 1; }; done

"$PY" "$HERE/validate_extension_scores.py" \
  --sequences "$OUT/prepared/unique_valid_sequences.tsv" \
  --protrek "$OUT/scores/protrek.tsv"

readarray -t THRESHOLDS < <("$PY" -c '
import json,sys
t=json.load(open(sys.argv[1]))["supplementary_thresholds"]
print(t["fold_plddt_min"]); print(t["sol_min"]); print(t["thermo_min"])
' "$CONFIG")
"$PY" "$HERE/summarize_results.py" \
  --requests "$OUT/prepared/all_requests.tsv" \
  --structure "$OUT/scores/esmfold.tsv" \
  --sol "$OUT/scores/protein_sol.tsv" \
  --thermo "$OUT/scores/temberture.tsv" \
  --ppl "$OUT/scores/protgpt2_ppl.tsv" \
  --novelty-t-conditional "$OUT/scores/novelty_t.tsv" \
  --novelty-u "$OUT/scores/novelty_u.tsv" \
  --protrek "$OUT/scores/protrek.tsv" \
  --fold-threshold "${THRESHOLDS[0]}" --sol-threshold "${THRESHOLDS[1]}" \
  --thermo-threshold "${THRESHOLDS[2]}" --out-root "$OUT/results"

CONDITIONAL_EVALUATION_CONFIG="$CONFIG" bash "$HERE/run_cohort_completion_queue.sh" "$@"
