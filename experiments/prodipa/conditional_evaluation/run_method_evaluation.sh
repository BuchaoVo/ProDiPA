#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/conditional_evaluation"
GENERIC_PREPARE="$ROOT/experiments/prodipa/external_baseline_evaluation/prepare_sequences.py"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
CONFIG="$ROOT/configs/evaluation/conditional.json"
MODEL_ID=""
GENERATION_TSV=""
PROTREK_TSV=""
OUTPUT_ROOT=""
GPUS="${PARALLEL_GPUS:-0,1,2,3}"
PUBLISH=0

usage() {
  cat <<'EOF'
Usage:
  run_method_evaluation.sh --model-id NAME --generation-tsv FILE [options]

Options:
  --protrek-tsv FILE      Frozen condition-to-sequence similarity scores.
  --output-root DIR       Default: outputs/evaluation/conditional/methods/NAME
  --gpus IDS              Comma-separated physical GPU indices. Default: 0,1,2,3
  --config FILE           Default: configs/evaluation/conditional.json
  --publish               Register and rebuild the canonical Conditional cohort.

The input must contain exactly 512 standardized Conditional generation requests.
Base scoring can finish without the extension TSV, but canonical publication
is fail-closed until required ProTrek evidence is present.
EOF
}

while (($#)); do
  case "$1" in
    --model-id) MODEL_ID="${2:?}"; shift 2 ;;
    --generation-tsv) GENERATION_TSV="${2:?}"; shift 2 ;;
    --protrek-tsv) PROTREK_TSV="${2:?}"; shift 2 ;;
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
[[ -z "$PROTREK_TSV" || -s "$PROTREK_TSV" ]] || { echo "invalid --protrek-tsv" >&2; exit 2; }
[[ -s "$CONFIG" ]] || { echo "evaluation config is missing: $CONFIG" >&2; exit 2; }
[[ -n "$OUTPUT_ROOT" ]] || OUTPUT_ROOT="$ROOT/outputs/evaluation/conditional/methods/$MODEL_ID"
mkdir -p "$OUTPUT_ROOT/input" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/scores"

readarray -t PROTOCOL < <("$PY" -c '
import json, sys
c=json.load(open(sys.argv[1])); g=c["generation_protocol"]; t=c["supplementary_thresholds"]; r=c["resources"]
for value in (c["requests_per_model"],g["valid_length_aa_min"],g["valid_length_aa_max"],
t["fold_plddt_min"],t["sol_min"],t["thermo_min"],r["esmfold_shards"],
r["esmfold_chunk_size"],r["esmfold_num_recycles"],r["gpu_min_total_mib"],
r["gpu_min_free_mib"],r["gpu_light_min_free_mib"],r["gpu_max_util_pct"],
r["gpu_stable_polls"],r["gpu_poll_seconds"],c["references"]["ppl_model"],
c["scorers"]["structure"]["model"],c["references"]["novelty_t_fasta"]): print(value)
' "$CONFIG")
REQUESTS="${PROTOCOL[0]}"; MIN_LENGTH="${PROTOCOL[1]}"; MAX_LENGTH="${PROTOCOL[2]}"
export FOLD_THRESHOLD="${PROTOCOL[3]}" SOL_THRESHOLD="${PROTOCOL[4]}" THERMO_THRESHOLD="${PROTOCOL[5]}"
export ESMFOLD_SHARDS="${PROTOCOL[6]}" ESMFOLD_CHUNK_SIZE="${PROTOCOL[7]}" ESMFOLD_NUM_RECYCLES="${PROTOCOL[8]}"
export MIN_TOTAL_MIB="${PROTOCOL[9]}" MIN_FREE_MIB="${PROTOCOL[10]}" LIGHT_MIN_FREE_MIB="${PROTOCOL[11]}"
export MAX_UTIL_PCT="${PROTOCOL[12]}" STABLE_POLLS="${PROTOCOL[13]}" POLL_SECONDS="${PROTOCOL[14]}"
export PROTGPT2_MODEL="${PROTOCOL[15]}" ESMFOLD_MODEL="${PROTOCOL[16]}" REQUESTS_PER_MODEL="$REQUESTS"
NOVELTY_T_CONFIG="${PROTOCOL[17]}"

"$PY" "$HERE/validate_generation.py" --input "$GENERATION_TSV" --model-id "$MODEL_ID" \
  --requests "$REQUESTS" --min-length "$MIN_LENGTH" --max-length "$MAX_LENGTH" \
  --report "$OUTPUT_ROOT/input/generation_validation.json"

SOURCE_SHA="$(sha256sum "$GENERATION_TSV" | awk '{print $1}')"
STAGED="$OUTPUT_ROOT/input/generation/conditional/$MODEL_ID.tsv"
mkdir -p "$(dirname "$STAGED")"
if [[ -f "$STAGED" ]]; then
  [[ "$SOURCE_SHA" == "$(sha256sum "$STAGED" | awk '{print $1}')" ]] || {
    echo "staged generation differs from submitted file; use a new model id" >&2; exit 1;
  }
else
  cp -- "$GENERATION_TSV" "$STAGED"
fi
printf '%s  %s\n' "$SOURCE_SHA" "$(basename "$STAGED")" > "$OUTPUT_ROOT/input/generation.sha256"

stage_extension() {
  local source="$1" target="$2" label="$3"
  [[ -n "$source" ]] || return 0
  if [[ -f "$target" ]]; then
    [[ "$(sha256sum "$source" | awk '{print $1}')" == "$(sha256sum "$target" | awk '{print $1}')" ]] || {
      echo "$label scores differ from the staged file; use a new output root" >&2; exit 1;
    }
  else
    cp -- "$source" "$target"
  fi
}
stage_extension "$PROTREK_TSV" "$OUTPUT_ROOT/scores/protrek.tsv" "ProTrek"

SCORING_PROTOCOL_SHA="$("$PY" -c '
import hashlib,json,sys
c=json.load(open(sys.argv[1])); frozen={k:c[k] for k in ("schema_version","requests_per_model","condition","generation_protocol","references","required_extensions","scorers","supplementary_thresholds","resources")}
print(hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(",", ":")).encode()).hexdigest())
' "$CONFIG")"
PROTOCOL_SHA_FILE="$OUTPUT_ROOT/input/scoring_protocol.sha256"
if [[ -d "$OUTPUT_ROOT/.done" && -n "$(find "$OUTPUT_ROOT/.done" -type f -print -quit 2>/dev/null)" ]]; then
  [[ -s "$PROTOCOL_SHA_FILE" && "$(head -n1 "$PROTOCOL_SHA_FILE")" == "$SCORING_PROTOCOL_SHA" ]] || {
    echo "evaluation protocol changed after scoring began; use a new output root" >&2; exit 1;
  }
fi
printf '%s\n' "$SCORING_PROTOCOL_SHA" > "$PROTOCOL_SHA_FILE"

export GENERATION_ROOT="$OUTPUT_ROOT/input/generation"
export EVALUATION_MODELS="$MODEL_ID"
export PREPARE_SCRIPT="$GENERIC_PREPARE"
export PARADIGM_EVALUATION_OUTPUT_ROOT="$OUTPUT_ROOT"
export PARALLEL_GPUS="$GPUS"
if [[ "$NOVELTY_T_CONFIG" = /* ]]; then NOVELTY_T_PATH="$NOVELTY_T_CONFIG"; else NOVELTY_T_PATH="$ROOT/$NOVELTY_T_CONFIG"; fi
if [[ ! -s "$NOVELTY_T_PATH" ]]; then
  "$PY" "$HERE/build_novelty_references.py" --project-root "$ROOT" --out-root "$(dirname "$NOVELTY_T_PATH")"
fi
[[ -s "$NOVELTY_T_PATH" ]] || { echo "Novelty-T reference was not produced" >&2; exit 1; }
export NOVELTY_T_REFERENCE="$NOVELTY_T_PATH"

bash "$HERE/run_separate_paradigm_scoring_queue.sh" conditional

extension_args=()
[[ -s "$OUTPUT_ROOT/scores/protrek.tsv" ]] && extension_args+=(--protrek "$OUTPUT_ROOT/scores/protrek.tsv")
if ((${#extension_args[@]})); then
  "$PY" "$HERE/validate_extension_scores.py" \
    --sequences "$OUTPUT_ROOT/prepared/unique_valid_sequences.tsv" "${extension_args[@]}"
fi

CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
"$PY" -c '
import json,sys
from datetime import datetime
from pathlib import Path
out,model,source_sha,config,config_sha,protocol_sha=sys.argv[1:]
Path(out).write_text(json.dumps({"status":"COMPLETE","model":model,"generation_sha256":source_sha,"evaluation_config":str(Path(config).resolve()),"evaluation_config_sha256":config_sha,"scoring_protocol_sha256":protocol_sha,"completed_at":datetime.now().astimezone().isoformat()},indent=2)+"\n")
' "$OUTPUT_ROOT/METHOD_EVALUATION_COMPLETE.json" "$MODEL_ID" "$SOURCE_SHA" "$CONFIG" "$CONFIG_SHA" "$SCORING_PROTOCOL_SHA"

if (( PUBLISH )); then
  "$PY" "$HERE/register_method.py" --config "$CONFIG" --project-root "$ROOT" \
    --model-id "$MODEL_ID" --evaluation-root "$OUTPUT_ROOT"
  CONDITIONAL_EVALUATION_CONFIG="$CONFIG" bash "$HERE/run_cohort_completion_queue.sh"
fi
