#!/usr/bin/env bash
set -euo pipefail

PARADIGM="${1:?usage: run_separate_paradigm_scoring_queue.sh unconditional|conditional}"
[[ "$PARADIGM" == unconditional || "$PARADIGM" == conditional ]] || { echo "invalid paradigm: $PARADIGM" >&2; exit 2; }

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
HERE="$ROOT/experiments/prodipa/conditional_evaluation"
SOURCE="${TEACHER_EVALUATION_ROOT:-$ROOT/outputs/evaluation/teacher_generation_evaluation}"
GENERATION_ROOT="${GENERATION_ROOT:-$SOURCE/generation}"
EVALUATION_MODELS="${EVALUATION_MODELS:-}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-$HERE/prepare_sequences.py}"
if [[ "$PARADIGM" == unconditional ]]; then
  OUT="${PARADIGM_EVALUATION_OUTPUT_ROOT:-$ROOT/outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers}"
else
  OUT="${PARADIGM_EVALUATION_OUTPUT_ROOT:-$ROOT/outputs/evaluation/conditional_prollama_evaluation}"
fi
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TEMBERTURE_PY="${TEMBERTURE_PYTHON:-__PRODIPA_ENVS__/temberture/bin/python}"
PROTGPT2="${PROTGPT2_MODEL:-__PRODIPA_MODELS__/ProtGPT2}"
ESMFOLD="${ESMFOLD_MODEL:-__PRODIPA_MODELS__/esmfold_v1_hf}"
MMSEQS="${MMSEQS_BIN:-$ROOT/tools/mmseqs/bin/mmseqs}"
PARALLEL_GPUS="${PARALLEL_GPUS:-0,1,2,3}"
MIN_TOTAL_MIB="${MIN_TOTAL_MIB:-70000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
LIGHT_MIN_FREE_MIB="${LIGHT_MIN_FREE_MIB:-6000}"
MAX_UTIL_PCT="${MAX_UTIL_PCT:-15}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-1}"
SHARED_MAX_UTIL_PCT="${GPU_SHARED_MAX_UTIL_PCT:-80}"
STABLE_POLLS="${STABLE_POLLS:-3}"
POLL_SECONDS="${GPU_QUEUE_POLL_SECONDS:-${POLL_SECONDS:-60}}"
SHARDS="${ESMFOLD_SHARDS:-16}"
ESMFOLD_CHUNK_SIZE="${ESMFOLD_CHUNK_SIZE:-32}"
ESMFOLD_NUM_RECYCLES="${ESMFOLD_NUM_RECYCLES:-4}"
REQUESTS_PER_MODEL="${REQUESTS_PER_MODEL:-512}"
FOLD_THRESHOLD="${FOLD_THRESHOLD:-85}"
SOL_THRESHOLD="${SOL_THRESHOLD:-0.7}"
THERMO_THRESHOLD="${THERMO_THRESHOLD:-0.6}"

IFS=',' read -r -a GPU_IDS <<<"$PARALLEL_GPUS"
mkdir -p "$OUT"/{prepared,scores,results,logs,work,worker_status,.done}
exec 7>"$OUT/.scoring_queue.lock"
flock -n 7 || { echo "another $PARADIGM scoring queue is active" >&2; exit 1; }
exec > >(tee -a "$OUT/scoring_queue.log") 2>&1

CURRENT_STAGE=prepare
declare -a ACTIVE_CHILDREN=()
write_status() {
  printf '{"state":"%s","paradigm":"%s","stage":"%s","gpus":"%s","updated_at":"%s"}\n' \
    "$1" "$PARADIGM" "$2" "$PARALLEL_GPUS" "$(date --iso-8601=seconds)" > "$OUT/SCORING_STATUS.json"
}
stop_all() {
  local pid
  for pid in "${ACTIVE_CHILDREN[@]:-}"; do [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${ACTIVE_CHILDREN[@]:-}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
}
on_stop() { write_status STOPPED "$CURRENT_STAGE"; stop_all; exit 143; }
on_error() { local rc=$?; write_status FAILED "$CURRENT_STAGE"; stop_all; exit "$rc"; }
trap on_stop INT TERM
trap on_error ERR

run_logged() {
  local task="$1" rc; shift
  echo "[$(date --iso-8601=seconds)] starting task=$task"
  "$@" > >(tee -a "$OUT/logs/$task.log") 2>&1 || {
    rc=$?
    echo "[$(date --iso-8601=seconds)] failed task=$task rc=$rc"
    return "$rc"
  }
  date --iso-8601=seconds > "$OUT/.done/$task"
  echo "[$(date --iso-8601=seconds)] finished task=$task"
}

if [[ ! -s "$OUT/.done/prepare" ]]; then
  write_status RUNNING prepare
  if [[ -n "$EVALUATION_MODELS" ]]; then
    run_logged prepare "$PY" "$PREPARE_SCRIPT" \
      --generation-root "$GENERATION_ROOT" --out-root "$OUT/prepared" \
      --requests-per-model "$REQUESTS_PER_MODEL" --paradigm "$PARADIGM" --models "$EVALUATION_MODELS"
  else
    run_logged prepare "$PY" "$PREPARE_SCRIPT" \
      --generation-root "$GENERATION_ROOT" --out-root "$OUT/prepared" \
      --requests-per-model "$REQUESTS_PER_MODEL" --paradigm "$PARADIGM"
  fi
fi
if [[ ! -s "$OUT/.done/novelty_references" ]]; then
  run_logged novelty_references "$PY" "$HERE/build_novelty_references.py" \
    --project-root "$ROOT" --out-root "$OUT/prepared/references"
fi

gpu_row() {
  local wanted="$1"
  nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' -v wanted="$wanted" \
    '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1==wanted)print $1","$2","$3","$4","$5}'
}

claim_task() {
  local queue="$1" lock="$2" task="" temporary
  exec 6>"$lock"; flock 6
  if [[ -s "$queue" ]]; then
    task="$(head -n1 "$queue")"
    temporary="${queue}.tmp.$$"
    tail -n +2 "$queue" > "$temporary"
    mv "$temporary" "$queue"
  fi
  flock -u 6; exec 6>&-
  printf '%s' "$task"
}

release_gpu_reservation() {
  { flock -u 9; } 2>/dev/null || true
  { exec 9>&-; } 2>/dev/null || true
  { flock -u 8; } 2>/dev/null || true
  { exec 8>&-; } 2>/dev/null || true
}

wait_for_gpu() {
  local gpu="$1" required_free="$2" task="$3" queue="${4:-}" stable=0 row index uuid total free util
  while true; do
    # Do not leave workers polling forever after the remaining tasks have
    # already been claimed by other GPUs.
    [[ -z "$queue" || -s "$queue" ]] || return 3
    row="$(gpu_row "$gpu")"
    if [[ -z "$row" ]]; then stable=0; sleep "$POLL_SECONDS"; continue; fi
    IFS=',' read -r index uuid total free util <<<"$row"
    if (( total >= MIN_TOTAL_MIB && free >= required_free && util <= MAX_UTIL_PCT )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    printf '{"state":"WAITING","gpu":%s,"task":"%s","free_mib":%s,"required_free_mib":%s,"util_pct":%s,"stable":%s,"required":%s,"updated_at":"%s"}\n' \
      "$gpu" "$task" "$free" "$required_free" "$util" "$stable" "$STABLE_POLLS" "$(date --iso-8601=seconds)" \
      > "$OUT/worker_status/gpu_${gpu}.json"
    if (( stable < STABLE_POLLS )); then sleep "$POLL_SECONDS"; continue; fi
    # A per-GPU scoring slot allows one scorer to share a partially occupied
    # card, while preventing several evaluation pipelines from piling onto it.
    exec 8>"/tmp/prodipa_gpu_${uuid}.shared_scoring.lock"
    if ! flock -n 8; then exec 8>&-; stable=0; sleep "$POLL_SECONDS"; continue; fi
    row="$(gpu_row "$gpu")"; IFS=',' read -r index uuid total free util <<<"$row"
    if (( free < required_free || util > MAX_UTIL_PCT )); then
      release_gpu_reservation; stable=0; sleep "$POLL_SECONDS"; continue
    fi
    exec 9>"/tmp/prodipa_gpu_${uuid}.lock"
    if flock -n 9; then
      SELECTED_LOCK_MODE=exclusive
    elif [[ "$ALLOW_SHARED_GPU" == 1 ]] && (( util <= SHARED_MAX_UTIL_PCT )); then
      exec 9>&-
      SELECTED_LOCK_MODE=shared
    else
      release_gpu_reservation; stable=0; sleep "$POLL_SECONDS"; continue
    fi
    SELECTED_UUID="$uuid"
    return 0
  done
}

task_min_free() {
  case "$1" in
    protgpt2_ppl|temberture) printf '%s' "$LIGHT_MIN_FREE_MIB" ;;
    *) printf '%s' "$MIN_FREE_MIB" ;;
  esac
}

run_gpu_task() {
  local task="$1"
  case "$task" in
    protgpt2_ppl)
      run_logged "$task" "$PY" "$HERE/score_protgpt2_ppl.py" \
        --input "$OUT/prepared/unique_valid_sequences.tsv" --model "$PROTGPT2" \
        --out "$OUT/scores/protgpt2_ppl.tsv" ;;
    temberture)
      run_logged "$task" "$TEMBERTURE_PY" "$ROOT/scripts/run_temBERTure_cls.py" \
        --input "$OUT/prepared/unique_valid_sequences.tsv" --repo "$ROOT/tools/TemBERTure" \
        --out "$OUT/scores/temberture.tsv" --batch-size 8 --device cuda ;;
    esmfold_*)
      local index="${task#esmfold_}"
      local precision_args=()
      case "${ESMFOLD_TF32:-inherit}" in
        true|1) precision_args=(--tf32) ;;
        false|0) precision_args=(--no-tf32) ;;
        inherit) ;;
        *) echo "ESMFOLD_TF32 must be inherit, true or false" >&2; return 2 ;;
      esac
      run_logged "$task" "$PY" "$ROOT/scripts/run_esmfold_properties.py" \
        --input "$OUT/prepared/unique_valid_sequences.tsv" \
        --out "$OUT/scores/esmfold_shard_$(printf '%02d' "$index").tsv" \
        --model "$ESMFOLD" --chunk-size "$ESMFOLD_CHUNK_SIZE" \
        --num-recycles "$ESMFOLD_NUM_RECYCLES" \
        --num-shards "$SHARDS" --shard-index "$index" \
        --batch-size "${ESMFOLD_BATCH_SIZE:-1}" --max-tokens-per-batch "${ESMFOLD_MAX_TOKENS:-3200}" \
        "${precision_args[@]}" ;;
    *) echo "unknown GPU task: $task" >&2; return 2 ;;
  esac
}

gpu_worker() {
  local gpu="$1" queue="$2" claim_lock="$3" task required_free rc
  while true; do
    [[ -s "$queue" ]] || return 0
    # Reserve an actually available GPU before removing work from the shared
    # queue.  Previously a busy GPU could claim an ESMFold shard and wait on it
    # indefinitely while an idle GPU exited because the queue looked empty.
    set +e
    wait_for_gpu "$gpu" "$MIN_FREE_MIB" pending_gpu_task "$queue"
    rc=$?
    set -e
    (( rc == 3 )) && return 0
    (( rc == 0 )) || return "$rc"
    task="$(claim_task "$queue" "$claim_lock")"
    if [[ -z "$task" ]]; then
      release_gpu_reservation
      return 0
    fi
    required_free="$(task_min_free "$task")"
    printf '{"state":"RUNNING","gpu":%s,"task":"%s","uuid":"%s","lock_mode":"%s","updated_at":"%s"}\n' \
      "$gpu" "$task" "$SELECTED_UUID" "$SELECTED_LOCK_MODE" "$(date --iso-8601=seconds)" \
      > "$OUT/worker_status/gpu_${gpu}.json"
    set +e; CUDA_VISIBLE_DEVICES="$SELECTED_UUID" run_gpu_task "$task"; rc=$?; set -e
    release_gpu_reservation
    (( rc == 0 )) || return "$rc"
  done
}

CURRENT_STAGE=cpu_scoring
write_status RUNNING "$CURRENT_STAGE"
ACTIVE_CHILDREN=()
if [[ ! -s "$OUT/.done/protein_sol" ]]; then
  (run_logged protein_sol "$PY" "$ROOT/scripts/run_protein_sol.py" \
    --input "$OUT/prepared/unique_valid_sequences.tsv" --work "$OUT/work/protein_sol" \
    --out "$OUT/scores/protein_sol.tsv") & ACTIVE_CHILDREN+=("$!")
fi
if [[ ! -s "$OUT/.done/novelty_t" ]]; then
  reference="${NOVELTY_T_REFERENCE:-$OUT/prepared/references/${PARADIGM}_train_union.fasta}"
  (run_logged novelty_t "$PY" "$HERE/score_mmseqs_novelty.py" \
    --input "$OUT/prepared/unique_valid_sequences.tsv" --reference "$reference" \
    --reference-label "${PARADIGM}_teacher_train_union_300x3" --mmseqs "$MMSEQS" \
    --out "$OUT/scores/novelty_t.tsv") & ACTIVE_CHILDREN+=("$!")
fi
CPU_CHILDREN=("${ACTIVE_CHILDREN[@]}")

CURRENT_STAGE=gpu_scoring
write_status RUNNING "$CURRENT_STAGE"
QUEUE="$OUT/.gpu_tasks_pending"; CLAIM_LOCK="$OUT/.gpu_tasks_claim.lock"; : > "$QUEUE"
# PPL owns a content-addressed manifest and must validate it even when a legacy
# timestamp-only .done marker exists.  A matching complete table exits before
# loading the model; a changed input, scorer, model, tokenizer, or precision is
# archived and recomputed.  Other scorers retain their existing cache behavior.
echo protgpt2_ppl >> "$QUEUE"
[[ -s "$OUT/.done/temberture" ]] || echo temberture >> "$QUEUE"
for ((i=0;i<SHARDS;i++)); do task="esmfold_$i"; [[ -s "$OUT/.done/$task" ]] || echo "$task" >> "$QUEUE"; done
GPU_CHILDREN=()
for gpu in "${GPU_IDS[@]}"; do gpu_worker "$gpu" "$QUEUE" "$CLAIM_LOCK" & GPU_CHILDREN+=("$!"); done
ACTIVE_CHILDREN=("${CPU_CHILDREN[@]}" "${GPU_CHILDREN[@]}")
failed=0
for pid in "${GPU_CHILDREN[@]}"; do if ! wait "$pid"; then failed=1; fi; done
for pid in "${CPU_CHILDREN[@]}"; do if ! wait "$pid"; then failed=1; fi; done
ACTIVE_CHILDREN=()
(( failed == 0 )) || exit 1

CURRENT_STAGE=merge
write_status RUNNING "$CURRENT_STAGE"
run_logged merge_structure "$PY" "$HERE/merge_structure_shards.py" \
  --input "$OUT/prepared/unique_valid_sequences.tsv" --shards-root "$OUT/scores" \
  --num-shards "$SHARDS" --out "$OUT/scores/esmfold.tsv"

CURRENT_STAGE=summarize
write_status RUNNING "$CURRENT_STAGE"
summary_args=(
  --requests "$OUT/prepared/all_requests.tsv" --structure "$OUT/scores/esmfold.tsv"
  --sol "$OUT/scores/protein_sol.tsv" --thermo "$OUT/scores/temberture.tsv"
  --ppl "$OUT/scores/protgpt2_ppl.tsv" --out-root "$OUT/results"
  --fold-threshold "$FOLD_THRESHOLD" --sol-threshold "$SOL_THRESHOLD"
  --thermo-threshold "$THERMO_THRESHOLD"
)
if [[ "$PARADIGM" == unconditional ]]; then
  summary_args+=(--novelty-t-unconditional "$OUT/scores/novelty_t.tsv")
else
  summary_args+=(--novelty-t-conditional "$OUT/scores/novelty_t.tsv")
  [[ -s "$OUT/scores/protrek.tsv" ]] && summary_args+=(--protrek "$OUT/scores/protrek.tsv")
fi
run_logged summarize "$PY" "$HERE/summarize_results.py" "${summary_args[@]}"
write_status COMPLETE complete
echo "$PARADIGM formal scoring complete"
