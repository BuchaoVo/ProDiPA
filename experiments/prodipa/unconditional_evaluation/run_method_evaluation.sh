#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/unconditional_evaluation"
SCORER="$ROOT/experiments/prodipa/conditional_evaluation"
GENERIC_PREPARE="$ROOT/experiments/prodipa/external_baseline_evaluation/prepare_sequences.py"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
CONFIG="$ROOT/configs/evaluation/unconditional.json"
MODEL_ID=""
GENERATION_TSV=""
OUTPUT_ROOT=""
GPUS="${PARALLEL_GPUS:-0,1,2,3}"
PUBLISH=0

usage() {
  cat <<'EOF'
Usage:
  run_method_evaluation.sh --model-id NAME --generation-tsv FILE [options]

Options:
  --output-root DIR   Method evaluation output. Default: outputs/evaluation/unconditional/methods/NAME
  --gpus IDS          Comma-separated physical GPU indices. Default: 0,1,2,3
  --config FILE       Canonical evaluation config. Default: configs/evaluation/unconditional.json
  --publish           Register the completed method and rebuild canonical Novelty-U/HV outputs

The input must contain exactly 512 standardized Unconditional generation requests.
The command is resumable and waits for eligible GPUs through the existing scoring queue.
EOF
}

while (($#)); do
  case "$1" in
    --model-id) MODEL_ID="${2:?}"; shift 2 ;;
    --generation-tsv) GENERATION_TSV="${2:?}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?}"; shift 2 ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --publish) PUBLISH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$CONFIG" = /* ]] || CONFIG="$ROOT/$CONFIG"
[[ -z "$OUTPUT_ROOT" || "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"

[[ -n "$MODEL_ID" && "$MODEL_ID" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || {
  echo "--model-id must use readable lowercase letters, digits, dot, underscore or hyphen" >&2
  exit 2
}
[[ -n "$GENERATION_TSV" && -f "$GENERATION_TSV" ]] || {
  echo "--generation-tsv is missing or not a file: $GENERATION_TSV" >&2
  exit 2
}
[[ -s "$CONFIG" ]] || { echo "evaluation config is missing: $CONFIG" >&2; exit 2; }
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$ROOT/outputs/evaluation/unconditional/methods/$MODEL_ID"
mkdir -p "$OUTPUT_ROOT/input" "$OUTPUT_ROOT/logs"

readarray -t PROTOCOL < <("$PY" -c '
import json, sys
c=json.load(open(sys.argv[1]))
g=c["generation_protocol"]
t=c["supplementary_thresholds"]
r=c["resources"]
print(c["requests_per_model"])
print(g["valid_length_aa_min"])
print(g["valid_length_aa_max"])
print(t["fold_plddt_min"])
print(t["sol_min"])
print(t["thermo_min"])
print(r["esmfold_shards"])
print(r["esmfold_chunk_size"])
print(r["esmfold_num_recycles"])
print(r["gpu_min_total_mib"])
print(r["gpu_min_free_mib"])
print(r["gpu_light_min_free_mib"])
print(r["gpu_max_util_pct"])
print(r["gpu_stable_polls"])
print(r["gpu_poll_seconds"])
print(c["references"]["ppl_model"])
print(c["scorers"]["structure"]["model"])
print(c["references"]["novelty_t_fasta"])
' "$CONFIG")
REQUESTS="${PROTOCOL[0]}"
MIN_LENGTH="${PROTOCOL[1]}"
MAX_LENGTH="${PROTOCOL[2]}"
export FOLD_THRESHOLD="${PROTOCOL[3]}"
export SOL_THRESHOLD="${PROTOCOL[4]}"
export THERMO_THRESHOLD="${PROTOCOL[5]}"
export ESMFOLD_SHARDS="${PROTOCOL[6]}"
export ESMFOLD_CHUNK_SIZE="${PROTOCOL[7]}"
export ESMFOLD_NUM_RECYCLES="${PROTOCOL[8]}"
export MIN_TOTAL_MIB="${PROTOCOL[9]}"
export MIN_FREE_MIB="${PROTOCOL[10]}"
export LIGHT_MIN_FREE_MIB="${PROTOCOL[11]}"
export MAX_UTIL_PCT="${PROTOCOL[12]}"
export STABLE_POLLS="${PROTOCOL[13]}"
export POLL_SECONDS="${PROTOCOL[14]}"
export PROTGPT2_MODEL="${PROTOCOL[15]}"
export ESMFOLD_MODEL="${PROTOCOL[16]}"
NOVELTY_T_CONFIG="${PROTOCOL[17]}"
export REQUESTS_PER_MODEL="$REQUESTS"

"$PY" "$HERE/validate_generation.py" \
  --input "$GENERATION_TSV" --model-id "$MODEL_ID" --requests "$REQUESTS" \
  --min-length "$MIN_LENGTH" --max-length "$MAX_LENGTH" \
  --report "$OUTPUT_ROOT/input/generation_validation.json"

SOURCE_SHA="$(sha256sum "$GENERATION_TSV" | awk '{print $1}')"
STAGED="$OUTPUT_ROOT/input/generation/unconditional/$MODEL_ID.tsv"
mkdir -p "$(dirname "$STAGED")"
if [[ -f "$STAGED" ]]; then
  STAGED_SHA="$(sha256sum "$STAGED" | awk '{print $1}')"
  [[ "$SOURCE_SHA" == "$STAGED_SHA" ]] || {
    echo "staged generation differs from the submitted file; use a new descriptive model id" >&2
    exit 1
  }
else
  cp -- "$GENERATION_TSV" "$STAGED"
fi
printf '%s  %s\n' "$SOURCE_SHA" "$(basename "$STAGED")" > "$OUTPUT_ROOT/input/generation.sha256"

SCORING_PROTOCOL_SHA="$("$PY" -c '
import hashlib, json, sys
c=json.load(open(sys.argv[1]))
frozen={k:c[k] for k in (
  "schema_version", "requests_per_model", "generation_protocol", "references",
  "scorers", "supplementary_thresholds", "resources"
)}
payload=json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
' "$CONFIG")"
PROTOCOL_SHA_FILE="$OUTPUT_ROOT/input/scoring_protocol.sha256"
if [[ -d "$OUTPUT_ROOT/.done" && -n "$(find "$OUTPUT_ROOT/.done" -type f -print -quit 2>/dev/null)" ]]; then
  [[ -s "$PROTOCOL_SHA_FILE" ]] || {
    echo "existing staged scores predate protocol fingerprinting; use a new descriptive model id or explicitly migrate the result" >&2
    exit 1
  }
  [[ "$(awk 'NR==1{print $1}' "$PROTOCOL_SHA_FILE")" == "$SCORING_PROTOCOL_SHA" ]] || {
    echo "evaluation protocol changed after scoring began; use a new output root/model id and re-evaluate" >&2
    exit 1
  }
fi
printf '%s\n' "$SCORING_PROTOCOL_SHA" > "$PROTOCOL_SHA_FILE"

export GENERATION_ROOT="$OUTPUT_ROOT/input/generation"
export EVALUATION_MODELS="$MODEL_ID"
export PREPARE_SCRIPT="$GENERIC_PREPARE"
export PARADIGM_EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"
export PARALLEL_GPUS="$GPUS"
if [[ "$NOVELTY_T_CONFIG" = /* ]]; then
  NOVELTY_T_PATH="$NOVELTY_T_CONFIG"
else
  NOVELTY_T_PATH="$ROOT/$NOVELTY_T_CONFIG"
fi
REFERENCE_ROOT="$(dirname "$NOVELTY_T_PATH")"
if [[ ! -s "$NOVELTY_T_PATH" ]]; then
  "$PY" "$ROOT/experiments/prodipa/conditional_evaluation/build_novelty_references.py" \
    --project-root "$ROOT" --out-root "$REFERENCE_ROOT"
fi
[[ -s "$NOVELTY_T_PATH" ]] || {
  echo "Novelty-T reference was not produced: $NOVELTY_T_PATH" >&2
  exit 1
}
export NOVELTY_T_REFERENCE="$NOVELTY_T_PATH"

bash "$SCORER/run_separate_paradigm_scoring_queue.sh" unconditional

CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
"$PY" -c '
import json, sys
from datetime import datetime
from pathlib import Path
out, model, source_sha, config, config_sha, protocol_sha = sys.argv[1:]
Path(out).write_text(json.dumps({
  "status": "COMPLETE", "model": model, "generation_sha256": source_sha,
  "evaluation_config": str(Path(config).resolve()), "evaluation_config_sha256": config_sha,
  "scoring_protocol_sha256": protocol_sha,
  "completed_at": datetime.now().astimezone().isoformat(),
}, indent=2) + "\n")
' "$OUTPUT_ROOT/METHOD_EVALUATION_COMPLETE.json" "$MODEL_ID" "$SOURCE_SHA" "$CONFIG" "$CONFIG_SHA" "$SCORING_PROTOCOL_SHA"

if (( PUBLISH )); then
  "$PY" "$HERE/register_method.py" \
    --config "$CONFIG" --project-root "$ROOT" --model-id "$MODEL_ID" \
    --evaluation-root "$OUTPUT_ROOT"
  UNCONDITIONAL_EVALUATION_CONFIG="$CONFIG" \
    bash "$HERE/run_cohort_completion_queue.sh"
fi
