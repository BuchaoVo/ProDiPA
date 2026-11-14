#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TORCHRUN="${TORCHRUN_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/torchrun}"
GPUS="${BALANCED_GPUS:-0,1,2,3}"
TRAIN_GPUS="${BALANCED_TRAIN_GPUS:-0,1}"
BASE="__PRODIPA_MODELS__/ProtGPT2"
TEACHERS="$ROOT/src/prodipa/unconditional/teachers.yaml"
RUN="$ROOT/outputs/experiments/unconditional_balanced_full_v1"
STUDENT="$ROOT/outputs/unconditional/protgpt2_multiteacher_student"
OFFLINE="$RUN/offline_multiteacher"
ADAPTIVE="$RUN/adaptive_cache"
STATUS="$RUN/STATUS.json"
GENERATOR="$ROOT/experiments/prodipa/conditional_evaluation/generate_model.py"
MERGER="$ROOT/experiments/prodipa/conditional_multiteacher_student/merge_generation_shards.py"

mkdir -p "$RUN/logs"
exec 8>"$RUN/.pipeline.lock"
flock -n 8 || { echo "another balanced Unconditional full-stack pipeline is active" >&2; exit 2; }
exec > >(tee -a "$RUN/pipeline.log") 2>&1

stage=preflight
write_status() {
  "$PY" - "$STATUS" "$1" "$2" "${3:-}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
value={"state":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],
       "updated_at":datetime.now(timezone.utc).isoformat()}
with tempfile.NamedTemporaryFile("w",dir=p.parent,delete=False) as f:
 json.dump(value,f,indent=2);f.write("\n");tmp=f.name
os.replace(tmp,p)
PY
}
on_error() { local rc=$?; write_status FAILED "$stage" "exit code $rc"; exit "$rc"; }
trap on_error ERR

stage=canonical_promotion
write_status RUNNING "$stage" "replace old teacher data, Prefix adapters and stale Unconditional artifacts"
if [[ ! -s "$RUN/CANONICAL_PROMOTION_COMPLETE.json" ]]; then
  "$PY" "$ROOT/experiments/prodipa/unconditional_teacher_training/publish_balanced_release.py" \
    > "$RUN/logs/canonical_promotion.log"
  "$PY" - "$ROOT" "$RUN/CANONICAL_PROMOTION_COMPLETE.json" <<'PY'
import json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
root, marker=map(Path,sys.argv[1:])
targets=[
 root/"outputs/unconditional/protgpt2_multiteacher_student",
 root/"outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers",
 root/"outputs/evaluation/unconditional/methods/protgpt2_multiteacher_student",
 root/"outputs/evaluation/unconditional/methods/reference_theta0",
 root/"outputs/evaluation/unconditional/methods/online_opd_paired",
 root/"outputs/evaluation/unconditional/methods/prodipa_offline",
 root/"outputs/experiments/offline_multiteacher_opd_full/unconditional",
 root/"outputs/experiments/adaptive_cache/unconditional",
]
for target in targets:
 if target.exists(): shutil.rmtree(target)
for target in (
 root/"outputs/evaluation/references/unconditional_train_union.fasta",
 root/"outputs/evaluation/references/novelty_t_reference_manifest.json",
):
 if target.exists(): target.unlink()
marker.write_text(json.dumps({"status":"COMPLETE","completed_at":datetime.now(timezone.utc).isoformat(),
 "removed_old_targets":[str(x) for x in targets]},indent=2)+"\n")
PY
fi

stage=online_student_training
write_status RUNNING "$stage" "256-update Legacy/512-aa Online multi-teacher OPD"
OUTPUT_DIR="$STUDENT" TRAINING_GPUS="$TRAIN_GPUS" SEQUENCE_FORMAT=legacy MAX_LENGTH_AA=512 \
  bash "$ROOT/experiments/prodipa/unconditional_student_training/run_training.sh" \
  2>&1 | tee "$RUN/logs/online_student_training.log"
test -s "$STUDENT/DONE.json"
test -s "$STUDENT/checkpoint-200/optimizer.pt"

stage=teacher_generation_evaluation
write_status RUNNING "$stage" "regenerate Base and three replacement Prefix teachers"
EVALUATE_BASE=0 PARALLEL_GPUS="$GPUS" EVALUATION_OUTPUT_ROOT="$ROOT/outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers" \
  bash "$ROOT/experiments/prodipa/unconditional_evaluation/run_teacher_evaluation.sh" \
  2>&1 | tee "$RUN/logs/teacher_evaluation.log"

stage=student_generation_evaluation
write_status RUNNING "$stage" "regenerate and publish the full Online student"
OUTPUT_DIR="$STUDENT" EVAL_DIR="$ROOT/outputs/evaluation/unconditional/methods/protgpt2_multiteacher_student" \
TRAINING_GPUS="$TRAIN_GPUS" EVALUATION_GPUS="$GPUS" \
  bash "$ROOT/experiments/prodipa/unconditional_student_training/run_full_pipeline.sh" \
  2>&1 | tee "$RUN/logs/student_evaluation.log"

stage=offline_cache
write_status RUNNING "$stage" "new theta0 cache with replacement Prefix teachers"
mkdir -p "$OFFLINE/cache" "$OFFLINE/logs"
if [[ ! -s "$OFFLINE/cache/CACHE_MANIFEST.json" ]]; then
  IFS=',' read -r GPU0 GPU1 _ <<< "$GPUS"
  CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH="$ROOT/src" "$PY" -u \
    "$ROOT/experiments/prodipa/offline_multiteacher_opd/build_unconditional_cache.py" \
    --base-model "$BASE" --teacher-config "$TEACHERS" --theta-zero "$STUDENT/checkpoint-200" \
    --cache-dir "$OFFLINE/cache" --sample-start 0 --sample-stop 256 --sample-count 512 \
    --batch-size 8 --seed 42 --rank 0 --device cuda:0 --max-length-aa 512 \
    > "$OFFLINE/logs/cache_0.log" 2>&1 & p0=$!
  CUDA_VISIBLE_DEVICES="$GPU1" PYTHONPATH="$ROOT/src" "$PY" -u \
    "$ROOT/experiments/prodipa/offline_multiteacher_opd/build_unconditional_cache.py" \
    --base-model "$BASE" --teacher-config "$TEACHERS" --theta-zero "$STUDENT/checkpoint-200" \
    --cache-dir "$OFFLINE/cache" --sample-start 256 --sample-stop 512 --sample-count 512 \
    --batch-size 8 --seed 42 --rank 1 --device cuda:0 --max-length-aa 512 \
    > "$OFFLINE/logs/cache_1.log" 2>&1 & p1=$!
  failed=0; wait "$p0" || failed=1; wait "$p1" || failed=1; ((failed==0))
  PYTHONPATH="$ROOT/src" "$PY" \
    "$ROOT/experiments/prodipa/offline_multiteacher_opd/finalize_unconditional_cache.py" \
    --cache-dir "$OFFLINE/cache" --teacher-config "$TEACHERS" \
    --theta-zero "$STUDENT/checkpoint-200" --base-model "$BASE" --sample-count 512 --shards 2
fi

run_continuation() {
  local mode="$1" output="$2" port="$3"; shift 3
  [[ -s "$output/TRAINING_PROTOCOL.json" ]] && \
    [[ "$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status"))' "$output/TRAINING_PROTOCOL.json")" == COMPLETE ]] && return 0
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONPATH="$ROOT/src:$ROOT/third_party/ProteinOPD/conditional/proteinopd" \
    "$TORCHRUN" --nproc_per_node 2 --master_port "$port" \
    "$ROOT/experiments/prodipa/offline_multiteacher_opd/train_unconditional.py" \
    --mode "$mode" --base-model "$BASE" --teacher-config "$TEACHERS" \
    --theta-zero "$STUDENT/checkpoint-200" --output-dir "$output" \
    --initial-global-step 200 --target-global-step 256 --seed 42 \
    --sequence-format legacy --max-length-aa 512 "$@"
}

stage=paired_online_continuation
write_status RUNNING "$stage" "paired 200-to-256 Online continuation"
run_continuation online "$OFFLINE/training/online" 29831 \
  --online-sample-count 8192 --online-epochs 1 \
  2>&1 | tee "$OFFLINE/logs/online_training.log"

stage=fixed_offline_continuation
write_status RUNNING "$stage" "paired 200-to-256 Fixed Offline continuation"
run_continuation offline "$OFFLINE/training/offline" 29832 \
  --cache-dir "$OFFLINE/cache" --offline-sample-count 512 --offline-epochs 16 \
  2>&1 | tee "$OFFLINE/logs/offline_training.log"

generate_method() {
  local model="$1" adapter="$2" destination="$3"
  local shards="$destination/shards" final="$destination/generation.tsv"
  [[ -s "$final" && $(( $(wc -l < "$final") - 1 )) -eq 512 ]] && return 0
  mkdir -p "$shards" "$destination/logs"
  IFS=',' read -r -a cards <<< "$GPUS"
  local pids=() shard start stop gpu
  for shard in 0 1 2 3; do
    start=$((shard*128)); stop=$((start+128)); gpu="${cards[$shard]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$GENERATOR" \
      --paradigm unconditional --model-name "$model" --base-model "$BASE" --adapter "$adapter" \
      --out "$shards/shard_${shard}.tsv" --num-sequences 512 \
      --sample-id-start "$start" --sample-id-stop "$stop" --base-seed 42 \
      --temperature 0.7 --top-p 0.9 --top-k 200 --repetition-penalty 1.2 \
      --max-new-tokens 512 --sequence-format legacy --generation-max-aa 512 --device cuda \
      > "$destination/logs/generation_${shard}.log" 2>&1 & pids+=("$!")
  done
  local failed=0 pid; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
  "$PY" "$MERGER" --input-dir "$shards" --output "$final" --num-sequences 512 --num-shards 4
}

stage=innovation_one_evaluation
write_status RUNNING "$stage" "Reference, paired Online and Fixed Offline independent evaluation"
declare -A ADAPTERS=(
  [reference_theta0]="$STUDENT/checkpoint-200"
  [online_opd_paired]="$OFFLINE/training/online"
  [prodipa_offline]="$OFFLINE/training/offline"
)
for model in reference_theta0 online_opd_paired prodipa_offline; do
  method_root="$ROOT/outputs/evaluation/unconditional/methods/$model"
  generate_method "$model" "${ADAPTERS[$model]}" "$OFFLINE/evaluation/$model"
  PARALLEL_GPUS="$GPUS" bash "$ROOT/experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh" \
    --model-id "$model" --generation-tsv "$OFFLINE/evaluation/$model/generation.tsv" \
    --output-root "$method_root" --gpus "$GPUS" --publish \
    2>&1 | tee "$OFFLINE/logs/${model}_evaluation.log"
done

# The adaptive-cache accounting interface expects the paired continuation under
# FULL/unconditional/{cache,training}.  Keep a single physical copy while making
# that versioned lineage explicit.
mkdir -p "$OFFLINE/unconditional"
ln -sfn ../cache "$OFFLINE/unconditional/cache"
ln -sfn ../training "$OFFLINE/unconditional/training"

stage=adaptive_cache
write_status RUNNING "$stage" "Fixed, Periodic and quality-aware Adaptive Cache full comparison"
ADAPTIVE_GPUS="$GPUS" ADAPTIVE_PARADIGMS=unconditional \
ADAPTIVE_OUTPUT_ROOT="$ADAPTIVE" ADAPTIVE_FULL_ROOT="$OFFLINE" \
ADAPTIVE_UNCONDITIONAL_THETA="$STUDENT/checkpoint-200" \
ADAPTIVE_UNCONDITIONAL_CACHE="$OFFLINE/cache" \
  "$PY" -u "$ROOT/experiments/prodipa/adaptive_cache/run.py" \
  2>&1 | tee "$RUN/logs/adaptive_cache.log"

stage=complete
write_status COMPLETE "$stage" "teachers, Online OPD, Offline OPD, Adaptive Cache and canonical tables complete"
date --iso-8601=seconds > "$RUN/COMPLETE"
