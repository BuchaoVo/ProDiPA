#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
HERE="$ROOT/experiments/prodipa/unconditional_evaluation"
OUT="$ROOT/outputs/evaluation/unconditional/requested_baselines"
RUNTIME_ENV="${BASELINE_RUNTIME_ENV_FILE:-$ROOT/configs/baselines/runtime_paths.env}"
ALLOWED_GPUS="${BASELINE_GPUS:-0,1,2,3}"
MIN_FREE_MIB="${BASELINE_MIN_FREE_MIB:-40000}"
MAX_UTIL_PCT="${BASELINE_MAX_UTIL_PCT:-20}"
POLL_SECONDS="${BASELINE_POLL_SECONDS:-300}"
mkdir -p "$OUT"/{preflight,logs,.done}
exec 7>"$OUT/.queue.lock"
flock -n 7 || { echo "another requested-baseline queue is active" >&2; exit 1; }
exec > >(tee -a "$OUT/queue.log") 2>&1

methods=(esm2_650m esm3_open_1.4b proteindt_t5 pinal_1.2b aspo_esm2_650m)

config_path() {
  case "$1" in
    esm2_650m) echo "$ROOT/configs/baselines/unconditional_esm2.json" ;;
    esm3_open_1.4b) echo "$ROOT/configs/baselines/unconditional_esm3.json" ;;
    proteindt_t5) echo "$ROOT/configs/baselines/unconditional_proteindt.json" ;;
    pinal_1.2b) echo "$ROOT/configs/baselines/unconditional_pinal.json" ;;
    aspo_esm2_650m) echo "$ROOT/configs/baselines/unconditional_aspo.json" ;;
  esac
}

write_status() {
  "$PY" - "$OUT/STATUS.json" "$1" "$2" "${3:-}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime
from pathlib import Path
target, state, stage, detail = sys.argv[1:]
path=Path(target)
value={"state":state,"stage":stage,"detail":detail,
       "updated_at":datetime.now().astimezone().isoformat()}
with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False,prefix=f".{path.name}.") as h:
 json.dump(value,h,indent=2); h.write("\n"); temporary=h.name
os.replace(temporary,path)
PY
}

load_runtime() {
  if [[ -f "$RUNTIME_ENV" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
    set +a
  fi
  ESM2_PYTHON="${ESM2_PYTHON:-$PY}"
  ESM3_PYTHON="${ESM3_PYTHON:-$PY}"
  PROTEINDT_PYTHON="${PROTEINDT_PYTHON:-$PY}"
  PINAL_PYTHON="${PINAL_PYTHON:-$PY}"
  ASPO_PYTHON="${ASPO_PYTHON:-$ESM2_PYTHON}"
  export ESM2_PYTHON ESM3_PYTHON PROTEINDT_PYTHON PINAL_PYTHON ASPO_PYTHON
}

preflight() {
  local method="$1" config
  config="$(config_path "$method")"
  "$PY" "$ROOT/src/prodipa/baselines/audit_resources.py" \
    --config "$config" --project-root "$ROOT" --output "$OUT/preflight/$method.json" || return $?
  case "$method" in
    esm2_650m) "$ESM2_PYTHON" -c 'import esm, torch' || return 2 ;;
    aspo_esm2_650m) "$ASPO_PYTHON" -c 'import esm, torch' || return 2 ;;
    esm3_open_1.4b) "$ESM3_PYTHON" -c 'import esm3, torch' || return 2 ;;
    proteindt_t5)
      PYTHONPATH="__PRODIPA_MODELS__/external_repos/ProteinDT" \
        "$PROTEINDT_PYTHON" -c \
        'from ProteinDT.models import T5Decoder, GaussianFacilitatorModel, ProteinTextModel; import torch, transformers' \
        || return 2 ;;
    pinal_1.2b)
      PYTHONPATH="$PINAL_REPO" "$PINAL_PYTHON" -c 'from utils.design_utils import load_pinal, PinalDesign' || return 2 ;;
  esac
}

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4","$5}'
}

wait_for_gpu() {
  local gpu row index uuid total free util best_gpu="" best_free=-1
  IFS=',' read -r -a gpu_ids <<<"$ALLOWED_GPUS"
  while true; do
    best_gpu=""; best_free=-1
    for gpu in "${gpu_ids[@]}"; do
      row="$(gpu_row "$gpu")"; [[ -n "$row" ]] || continue
      IFS=',' read -r index uuid total free util <<<"$row"
      if (( free >= MIN_FREE_MIB && util <= MAX_UTIL_PCT && free > best_free )); then
        best_gpu="$gpu"; best_free="$free"; SELECTED_UUID="$uuid"
      fi
    done
    if [[ -z "$best_gpu" ]]; then
      write_status WAITING_GPU generation "free>=${MIN_FREE_MIB}MiB util<=${MAX_UTIL_PCT}%"
      sleep 60; continue
    fi
    exec 9>"/tmp/prodipa_gpu_${SELECTED_UUID}.lock"
    if ! flock -n 9; then exec 9>&-; sleep 60; continue; fi
    row="$(gpu_row "$best_gpu")"; IFS=',' read -r index uuid total free util <<<"$row"
    if (( free < MIN_FREE_MIB || util > MAX_UTIL_PCT )); then
      flock -u 9; exec 9>&-; sleep 60; continue
    fi
    SELECTED_GPU="$best_gpu"; return 0
  done
}

run_generation() {
  local method="$1" method_root="$OUT/$1" output="$OUT/$1/generations.jsonl"
  mkdir -p "$method_root"
  case "$method" in
    esm2_650m)
      "$ESM2_PYTHON" "$ROOT/src/prodipa/baselines/run_masked_infilling.py" \
        --backend esm2 --steering-repo "$ROOT/third_party/Steering-PLMs" \
        --reference-manifest "$ROOT/data/evaluation/unconditional/masked_infilling_reference_sequences.tsv" \
        --esm2-checkpoint "$ESM2_650M_CHECKPOINT" --output "$output" ;;
    esm3_open_1.4b)
      HF_HOME="$ESM3_MODEL_DIR" "$ESM3_PYTHON" "$ROOT/src/prodipa/baselines/run_masked_infilling.py" \
        --backend esm3 --steering-repo "$ROOT/third_party/Steering-PLMs" \
        --reference-manifest "$ROOT/data/evaluation/unconditional/masked_infilling_reference_sequences.tsv" \
        --output "$output" ;;
    proteindt_t5)
      PYTHONPATH="__PRODIPA_MODELS__/external_repos/ProteinDT" \
      "$PROTEINDT_PYTHON" "$ROOT/src/prodipa/baselines/run_text_guided_unconditional.py" \
        --backend proteindt --repo "__PRODIPA_MODELS__/external_repos/ProteinDT" \
        --pretrained-folder "$PROTEINDT_PRETRAINED_FOLDER" \
        --decoder-folder "$PROTEINDT_DECODER_FOLDER" --output "$output" ;;
    pinal_1.2b)
      "$PINAL_PYTHON" "$ROOT/src/prodipa/baselines/run_text_guided_unconditional.py" \
        --backend pinal --repo "$PINAL_REPO" --weights "$PINAL_WEIGHTS" --output "$output" ;;
    aspo_esm2_650m)
      "$ASPO_PYTHON" "$ROOT/src/prodipa/baselines/run_masked_infilling.py" \
        --backend aspo --steering-repo "$ROOT/third_party/Steering-PLMs" \
        --reference-manifest "$ROOT/data/evaluation/unconditional/masked_infilling_reference_sequences.tsv" \
        --esm2-checkpoint "$ESM2_650M_CHECKPOINT" --fold-vector "$ASPO_FOLD_STEERING_VECTORS" \
        --sol-vector "$ASPO_SOL_STEERING_VECTORS" --thermo-vector "$ASPO_THERMO_STEERING_VECTORS" \
        --output "$output" ;;
  esac
}

audit_convert_evaluate() {
  local method="$1" method_root="$OUT/$1"
  "$PY" "$ROOT/src/prodipa/baselines/audit_generation.py" \
    --input "$method_root/generations.jsonl" --output "$method_root/generation_audit.json" \
    --expected-requests 512 --benchmark unconditional --model "$method" --request-prefix "${method}_"
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
    "$ROOT/experiments/prodipa/external_baseline_evaluation/convert_generations.py" \
    --input "$method_root/generations.jsonl" --output "$method_root/generation.tsv" \
    --paradigm unconditional --model "$method" --expected-requests 512
  bash "$HERE/run_method_evaluation.sh" --model-id "$method" \
    --generation-tsv "$method_root/generation.tsv" --gpus "$ALLOWED_GPUS" --publish
}

trap 'write_status STOPPED interrupted; exit 143' INT TERM
trap 'rc=$?; write_status FAILED queue "unexpected queue exit rc=$rc line=$LINENO"; exit "$rc"' ERR
while true; do
  load_runtime
  all_done=true; made_progress=false; blocked=()
  for method in "${methods[@]}"; do
    [[ -s "$OUT/.done/$method" ]] && continue
    all_done=false
    if preflight "$method" >>"$OUT/logs/${method}_preflight.log" 2>&1; then
      write_status WAITING_GPU "$method"
      wait_for_gpu
      write_status RUNNING "$method" "physical_gpu=$SELECTED_GPU uuid=$SELECTED_UUID"
      echo "[$(date --iso-8601=seconds)] starting $method on GPU $SELECTED_GPU"
      set +e
      CUDA_VISIBLE_DEVICES="$SELECTED_UUID" run_generation "$method" \
        > >(tee -a "$OUT/logs/${method}_generation.log") 2>&1
      rc=$?
      set -e
      flock -u 9; exec 9>&-
      (( rc == 0 )) || { write_status FAILED "$method" "generation rc=$rc"; exit "$rc"; }
      audit_convert_evaluate "$method" >>"$OUT/logs/${method}_evaluation.log" 2>&1
      date --iso-8601=seconds > "$OUT/.done/$method"
      made_progress=true
    else
      rc=$?
      (( rc == 2 )) || { write_status FAILED "$method" "preflight rc=$rc"; exit "$rc"; }
      blocked+=("$method")
    fi
  done
  $all_done && break
  if ! $made_progress; then
    write_status WAITING_RESOURCES preflight "${blocked[*]}"
    echo "[$(date --iso-8601=seconds)] waiting for resources: ${blocked[*]}"
    sleep "$POLL_SECONDS"
  fi
done
write_status COMPLETE complete
echo "all requested Unconditional baselines are published"
