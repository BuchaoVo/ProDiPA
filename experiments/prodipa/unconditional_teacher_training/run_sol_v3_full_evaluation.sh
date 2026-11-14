#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
EXP="${EXPERIMENT_ROOT:-$ROOT/outputs/experiments/unconditional_teacher_sol_v3_fasta60_v1}"
EVAL="$EXP/evaluation"
METHOD="$EVAL/full_metrics"
COHORT="$EVAL/cohort"
CONFIG="$EVAL/unconditional_sol_v3_teachers.json"
GPUS="${EVALUATION_GPUS:-0,1,2,3}"
OFFICIAL_CONFIG="$ROOT/configs/evaluation/unconditional.json"
FROZEN_HV="$ROOT/outputs/evaluation/unconditional/results/unconditional_hv_manifest.json"

mkdir -p "$EVAL"
exec 8>"$EVAL/.full_evaluation.lock"
flock -n 8 || { echo "another Sol V3 teacher evaluation is active" >&2; exit 2; }
exec > >(tee -a "$EVAL/evaluation.log") 2>&1

status() {
  "$PY" - "$EVAL/EVALUATION_STATUS.json" "$1" "$2" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
value={"state":sys.argv[2], "stage":sys.argv[3],
       "updated_at":datetime.now(timezone.utc).isoformat()}
with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False) as f:
    json.dump(value, f, indent=2); f.write("\n"); temporary=f.name
os.replace(temporary, p)
PY
}

stage=preflight
on_error() { local rc=$?; status FAILED "$stage"; exit "$rc"; }
trap on_error ERR

for attribute in fold sol thermo; do
  test -s "$EXP/teachers/$attribute/DONE.json"
  test -s "$EXP/teachers/$attribute/best_adapter/adapter_model.safetensors"
done
test -s "$EXP/TRAINING_COMPLETE.json"
test -s "$OFFICIAL_CONFIG"
test -s "$FROZEN_HV"

# Isolate the new three-teacher cohort while retaining the already published
# Unconditional normalization bounds.  This makes the new HV directly
# comparable without rewriting the official table during evaluation.
"$PY" - "$OFFICIAL_CONFIG" "$CONFIG" "$METHOD" "$COHORT" "$FROZEN_HV" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
source, destination, method, cohort, frozen = map(Path, sys.argv[1:])
config=json.loads(source.read_text())
config["score_groups"]=[{
    "name":"sol_v3_retrained_prefix_teachers",
    "models":["fold_teacher","sol_teacher","thermo_teacher"],
    "evaluation_root":str(method.resolve()),
}]
config["output_root"]=str(cohort.resolve())
config["hypervolume"]["frozen_manifest"]=str(frozen.resolve())
config["hypervolume"]["normalization"]="Frozen published Unconditional min-max bounds"
config.pop("compatibility_main_table", None)
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as f:
    json.dump(config, f, indent=2); f.write("\n"); temporary=f.name
os.replace(temporary, destination)
PY

stage=generation_and_core_scoring
status RUNNING "$stage"
EVALUATE_BASE=0 \
EVALUATION_MODELS="fold_teacher,sol_teacher,thermo_teacher" \
PARALLEL_GPUS="$GPUS" \
EVALUATION_OUTPUT_ROOT="$METHOD" \
UNCONDITIONAL_FOLD_ADAPTER="$EXP/teachers/fold/best_adapter" \
UNCONDITIONAL_SOL_ADAPTER="$EXP/teachers/sol/best_adapter" \
UNCONDITIONAL_THERMO_ADAPTER="$EXP/teachers/thermo/best_adapter" \
MIN_FREE_MIB="${MIN_FREE_MIB:-28000}" MAX_UTIL_PCT="${MAX_UTIL_PCT:-100}" \
STABLE_POLLS="${STABLE_POLLS:-1}" POLL_SECONDS="${POLL_SECONDS:-30}" \
bash "$ROOT/experiments/prodipa/unconditional_evaluation/run_teacher_evaluation.sh"

stage=novelty_u_and_frozen_hv
status RUNNING "$stage"
"$PY" "$ROOT/experiments/prodipa/unconditional_evaluation/run_cohort_completion_queue.py" \
  --project-root "$ROOT" --config "$CONFIG" --poll-seconds 30

stage=complete
status COMPLETE "$stage"
"$PY" - "$EVAL/EVALUATION_COMPLETE.json" "$CONFIG" "$METHOD" "$COHORT" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
out, config, method, cohort = map(Path, sys.argv[1:])
out.write_text(json.dumps({
    "status":"COMPLETE",
    "completed_at":datetime.now(timezone.utc).isoformat(),
    "config":str(config.resolve()),
    "config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
    "core_metrics":str((method/"results/three_teacher_full_evaluation.tsv").resolve()),
    "novelty_u_hv":str((cohort/"results/unconditional_prodipa_main.tsv").resolve()),
}, indent=2)+"\n")
PY
