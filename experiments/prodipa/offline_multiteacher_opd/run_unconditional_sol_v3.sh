#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TORCHRUN="${TORCHRUN_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/torchrun}"
HERE="$ROOT/experiments/prodipa/offline_multiteacher_opd"
RUNTIME_SRC="$ROOT/third_party/ProteinOPD/conditional/proteinopd"
PYTHONPATH_VALUE="$ROOT/src:$RUNTIME_SRC${PYTHONPATH:+:$PYTHONPATH}"
OUT="${EXPERIMENT_OUT:-$ROOT/outputs/experiments/offline_multiteacher_opd_unconditional_sol_v3}"
BASE="${PROTGPT2_MODEL:-__PRODIPA_MODELS__/ProtGPT2}"
TEACHERS="$ROOT/src/prodipa/unconditional/teachers.yaml"
THETA="$ROOT/outputs/unconditional/protgpt2_multiteacher_student/checkpoint-200"
HV_MANIFEST="$ROOT/outputs/evaluation/unconditional/results/unconditional_hv_manifest.json"
STATUS="$OUT/STATUS.json"
TRAIN_MIN_FREE_MIB="${TRAIN_MIN_FREE_MIB:-40000}"
TRAIN_MAX_UTIL="${TRAIN_MAX_UTIL:-100}"
POLL_SECONDS="${POLL_SECONDS:-15}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"

mkdir -p "$OUT"/{logs,cache,training/online,training/offline,evaluation/generation,evaluation/methods,evaluation/novelty,results}
exec 9>"$OUT/.pipeline.lock"
flock -n 9 || { echo "another Unconditional Sol-V3 Innovation-I pipeline is active" >&2; exit 2; }
exec > >(tee -a "$OUT/pipeline.log") 2>&1

write_status() {
  "$PY" - "$STATUS" "$1" "$2" "${3:-}" <<'PY'
import json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
payload={"state":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],
         "updated_at":datetime.now(timezone.utc).isoformat()}
with tempfile.NamedTemporaryFile("w",dir=p.parent,delete=False) as f:
 json.dump(payload,f,indent=2); f.write("\n"); temporary=f.name
os.replace(temporary,p)
PY
}

stage=preflight
declare -a TRAIN_LOCK_FDS=()
release_training_gpus() {
  local fd
  for fd in "${TRAIN_LOCK_FDS[@]}"; do flock -u "$fd" 2>/dev/null || true; eval "exec ${fd}>&-"; done
  TRAIN_LOCK_FDS=()
}
on_error() { local rc=$?; release_training_gpus; write_status FAILED "$stage" "exit code $rc"; exit "$rc"; }
trap on_error ERR INT TERM

write_status RUNNING "$stage" "verifying common checkpoint and new teacher hashes"
for path in "$BASE/config.json" "$TEACHERS" "$THETA/adapter_model.safetensors" \
  "$THETA/optimizer.pt" "$THETA/scheduler.pt" "$THETA/trainer_state.json" \
  "$THETA/SEQUENCE_PROTOCOL.json" "$HV_MANIFEST"; do
  [[ -s "$path" ]] || { echo "missing required artifact: $path" >&2; exit 1; }
done
"$PY" - "$ROOT" "$OUT/PROTOCOL.json" "$THETA" "$TEACHERS" "$HV_MANIFEST" "$OUT/FROZEN_HV_MANIFEST.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
root,out,theta,config,hv_manifest,frozen_hv=map(Path,sys.argv[1:])
state=json.loads((theta/'trainer_state.json').read_text())
assert (state['global_step'],state['max_steps']) == (200,256)
sequence=json.loads((theta/'SEQUENCE_PROTOCOL.json').read_text())
assert sequence['sequence_format'] == 'legacy'
teachers=[]
for name in ('fold','sol','thermo'):
 p=root/f'outputs/unconditional/protgpt2/{name}/best_adapter/adapter_model.safetensors'
 teachers.append({'name':name,'path':str(p.parent.resolve()),
                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
payload={'status':'PASS','checked_at':datetime.now(timezone.utc).isoformat(),
 'paradigm':'unconditional','theta_zero':str(theta.resolve()),
 'theta_zero_sha256':hashlib.sha256((theta/'adapter_model.safetensors').read_bytes()).hexdigest(),
 'initial_global_step':200,'target_global_step':256,'student_updates':56,
 'cache_trajectories':512,'sequence_format':'legacy','max_length_aa':512,
 'teacher_config':str(config.resolve()),
 'teacher_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
 'frozen_hv_manifest':str(frozen_hv.resolve()),
 'frozen_hv_manifest_sha256':hashlib.sha256(hv_manifest.read_bytes()).hexdigest(),
 'teachers':teachers,'loss':'full-vocabulary JSD','aggregation':'equal-weight PoE'}
out.write_text(json.dumps(payload,indent=2)+'\n')
frozen_hv.write_bytes(hv_manifest.read_bytes())
PY

gpu_rows() {
  nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1>=0 && $1<=3)print $1","$2","$3","$4}' | \
    sort -t',' -k3,3nr
}
claim_training_pair() {
  local index uuid free util fd
  while true; do
    TRAIN_GPU_IDS=(); TRAIN_GPU_UUIDS=(); TRAIN_LOCK_FDS=()
    while IFS=',' read -r index uuid free util; do
      (( free >= TRAIN_MIN_FREE_MIB && util <= TRAIN_MAX_UTIL )) || continue
      exec {fd}>"/tmp/prodipa_gpu_${uuid}.lock"
      if flock -n "$fd"; then
        TRAIN_GPU_IDS+=("$index"); TRAIN_GPU_UUIDS+=("$uuid"); TRAIN_LOCK_FDS+=("$fd")
        (( ${#TRAIN_GPU_IDS[@]} == 2 )) && return 0
      else
        exec {fd}>&-
      fi
    done < <(gpu_rows)
    release_training_gpus
    write_status WAITING wait_for_training_gpus "need two GPUs with >=${TRAIN_MIN_FREE_MIB} MiB free"
    sleep "$POLL_SECONDS"
  done
}

stage=wait_for_training_gpus
claim_training_pair
TRAIN_GPUS="${TRAIN_GPU_IDS[0]},${TRAIN_GPU_IDS[1]}"
write_status RUNNING cache "building 512 fixed trajectories and full-vocabulary PoE targets on GPUs $TRAIN_GPUS"

stage=cache
if [[ ! -s "$OUT/cache/CACHE_MANIFEST.json" ]]; then
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS[0]}" PYTHONPATH="$PYTHONPATH_VALUE" "$PY" "$HERE/build_unconditional_cache.py" \
    --base-model "$BASE" --teacher-config "$TEACHERS" --theta-zero "$THETA" \
    --cache-dir "$OUT/cache" --sample-start 0 --sample-stop 256 --sample-count 512 \
    --batch-size 8 --seed 42 --rank 0 --device cuda:0 --max-length-aa 512 \
    > "$OUT/logs/cache_rank_0.log" 2>&1 & p0=$!
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS[1]}" PYTHONPATH="$PYTHONPATH_VALUE" "$PY" "$HERE/build_unconditional_cache.py" \
    --base-model "$BASE" --teacher-config "$TEACHERS" --theta-zero "$THETA" \
    --cache-dir "$OUT/cache" --sample-start 256 --sample-stop 512 --sample-count 512 \
    --batch-size 8 --seed 42 --rank 1 --device cuda:0 --max-length-aa 512 \
    > "$OUT/logs/cache_rank_1.log" 2>&1 & p1=$!
  failed=0; wait "$p0" || failed=1; wait "$p1" || failed=1; (( failed == 0 ))
  PYTHONPATH="$PYTHONPATH_VALUE" "$PY" "$HERE/finalize_unconditional_cache.py" \
    --cache-dir "$OUT/cache" --teacher-config "$TEACHERS" --theta-zero "$THETA" \
    --base-model "$BASE" --sample-count 512 --shards 2
fi

export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export PYTHONPATH="$PYTHONPATH_VALUE"
export WANDB_MODE=disabled

stage=online_training
write_status RUNNING "$stage" "paired Online continuation step 200 to 256 on GPUs $TRAIN_GPUS"
if ! grep -q '"status": "COMPLETE"' "$OUT/training/online/TRAINING_PROTOCOL.json" 2>/dev/null; then
  "$TORCHRUN" --nproc_per_node 2 --master_port 29831 "$HERE/train_unconditional.py" \
    --mode online --base-model "$BASE" --teacher-config "$TEACHERS" --theta-zero "$THETA" \
    --output-dir "$OUT/training/online" --online-sample-count 8192 --online-epochs 1 \
    --initial-global-step 200 --target-global-step 256 --seed 42 \
    --sequence-format legacy --max-length-aa 512 2>&1 | tee "$OUT/logs/online_training.log"
fi

stage=offline_training
write_status RUNNING "$stage" "paired Offline continuation step 200 to 256 with fixed cache"
if ! grep -q '"status": "COMPLETE"' "$OUT/training/offline/TRAINING_PROTOCOL.json" 2>/dev/null; then
  "$TORCHRUN" --nproc_per_node 2 --master_port 29832 "$HERE/train_unconditional.py" \
    --mode offline --base-model "$BASE" --teacher-config "$TEACHERS" --theta-zero "$THETA" \
    --cache-dir "$OUT/cache" --output-dir "$OUT/training/offline" \
    --offline-sample-count 512 --offline-epochs 16 \
    --initial-global-step 200 --target-global-step 256 --seed 42 \
    --sequence-format legacy --max-length-aa 512 2>&1 | tee "$OUT/logs/offline_training.log"
fi
release_training_gpus
unset CUDA_VISIBLE_DEVICES

generate_model() {
  local model="$1" adapter="$2" shard start stop pid failed=0
  local dir="$OUT/evaluation/generation/$model" final="$OUT/evaluation/generation/$model/unconditional/$model.tsv"
  mkdir -p "$dir/shards" "$dir/unconditional" "$OUT/logs/generation/$model"
  if [[ -s "$final" && $(( $(wc -l < "$final") - 1 )) -eq 512 ]]; then return 0; fi
  "$PY" "$ROOT/experiments/prodipa/conditional_multiteacher_student/prepare_generation_shards.py" \
    --existing "$final" --output-dir "$dir/shards" --num-sequences 512 --num-shards 4
  IFS=',' read -r -a eval_ids <<< "$EVAL_GPUS"
  pids=()
  for shard in 0 1 2 3; do
    start=$((shard*128)); stop=$((start+128))
    CUDA_VISIBLE_DEVICES="${eval_ids[$shard]}" "$PY" \
      "$ROOT/experiments/prodipa/conditional_evaluation/generate_model.py" \
      --paradigm unconditional --model-name "$model" --base-model "$BASE" --adapter "$adapter" \
      --out "$dir/shards/shard_${shard}.tsv" --num-sequences 512 \
      --sample-id-start "$start" --sample-id-stop "$stop" --base-seed 42 \
      --temperature 0.7 --top-p 0.9 --top-k 200 --repetition-penalty 1.2 \
      --max-new-tokens 512 --sequence-format legacy --generation-max-aa 512 --device cuda \
      > "$OUT/logs/generation/$model/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 ))
  "$PY" "$ROOT/experiments/prodipa/conditional_multiteacher_student/merge_generation_shards.py" \
    --input-dir "$dir/shards" --output "$final" --num-sequences 512 --num-shards 4
}

stage=generation
write_status RUNNING "$stage" "512 independent requests for Reference, Online and Offline"
generate_model reference_theta0 "$THETA"
generate_model online_opd_paired "$OUT/training/online"
generate_model prodipa_offline "$OUT/training/offline"

stage=scoring
write_status RUNNING "$stage" "formal PPL, Novelty-T, structure, Sol and Thermo scoring"
for model in reference_theta0 online_opd_paired prodipa_offline; do
  generation="$OUT/evaluation/generation/$model/unconditional/$model.tsv"
  PARALLEL_GPUS="$EVAL_GPUS" bash "$ROOT/experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh" \
    --model-id "$model" --generation-tsv "$generation" \
    --output-root "$OUT/evaluation/methods/$model" --gpus "$EVAL_GPUS" \
    2>&1 | tee "$OUT/logs/${model}_evaluation.log"
done

stage=novelty_u
write_status RUNNING "$stage" "joint UniProt Novelty-U scoring"
"$PY" "$HERE/combine_sequences.py" \
  --input "$OUT/evaluation/methods/reference_theta0/prepared/unique_valid_sequences.tsv" \
          "$OUT/evaluation/methods/online_opd_paired/prepared/unique_valid_sequences.tsv" \
          "$OUT/evaluation/methods/prodipa_offline/prepared/unique_valid_sequences.tsv" \
  --output "$OUT/evaluation/novelty/unconditional_combined_sequences.tsv"
"$PY" "$ROOT/experiments/prodipa/conditional_evaluation/score_mmseqs_novelty.py" \
  --input "$OUT/evaluation/novelty/unconditional_combined_sequences.tsv" \
  --reference-db __PRODIPA_DATABASES__/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03 \
  --out "$OUT/evaluation/novelty/unconditional_novelty_u.tsv" \
  --mmseqs "$ROOT/tools/mmseqs/bin/mmseqs" --threads 32 --split-memory-limit 300G \
  --tmp-dir "$OUT/evaluation/novelty/mmseqs_tmp" --reference-label UniProtKB_2026_03 \
  2>&1 | tee "$OUT/logs/novelty_u.log"

stage=publication
write_status RUNNING "$stage" "frozen HV, GPU-hours, retention and main-table publication"
"$PY" "$HERE/finalize_unconditional_sol_v3.py" --project-root "$ROOT" --experiment-root "$OUT"
touch "$OUT/COMPLETE"
write_status COMPLETE complete "new-teacher Unconditional Innovation-I results published"
trap - ERR INT TERM
