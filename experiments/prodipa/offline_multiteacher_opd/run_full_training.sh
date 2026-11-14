#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TORCHRUN="${TORCHRUN_BIN:-__PRODIPA_ENVS__/ProDiPA/bin/torchrun}"
HERE="$ROOT/experiments/prodipa/offline_multiteacher_opd"
RUNTIME_SRC="$ROOT/third_party/ProteinOPD/conditional/proteinopd"
PYTHONPATH_VALUE="$ROOT/src:$RUNTIME_SRC${PYTHONPATH:+:$PYTHONPATH}"
OUT="$ROOT/outputs/experiments/offline_multiteacher_opd_full"
COND_CONFIG="$ROOT/experiments/prodipa/conditional_multiteacher_student/opd_config.yaml"
COND_THETA="$ROOT/outputs/conditional/prollama_multiteacher_student/checkpoint-192"
COND_CACHE="$ROOT/outputs/experiments/offline_multiteacher_opd/cache"
UNCOND_BASE="__PRODIPA_MODELS__/ProtGPT2"
UNCOND_TEACHERS="$ROOT/src/prodipa/unconditional/teachers.yaml"
UNCOND_THETA="$ROOT/outputs/unconditional/protgpt2_multiteacher_student/checkpoint-200"
GPUS="${TRAIN_GPUS:-0,1}"
STATUS="$OUT/TRAINING_STATUS.json"

mkdir -p "$OUT"/{logs,conditional/training,unconditional/training,unconditional/cache}
printf '%s\n' "$$" > "$OUT/training_pipeline.pid"
exec 9>"$OUT/.training_pipeline.lock"
flock -n 9 || { echo "another full paired training pipeline is active" >&2; exit 2; }
exec > >(tee -a "$OUT/full_training.log") 2>&1

stage=preflight
write_status() {
  "$PY" - "$STATUS" "$1" "$2" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
path,state,stage=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
path.write_text(json.dumps({"state":state,"stage":stage,"updated_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
PY
}
on_error() { local rc=$?; write_status FAILED "$stage"; exit "$rc"; }
trap on_error ERR

stage=preflight
write_status RUNNING "$stage"
for path in \
  "$COND_CONFIG" "$COND_THETA/adapter_model.bin" "$COND_THETA/optimizer.pt" \
  "$COND_THETA/scheduler.pt" "$COND_THETA/trainer_state.json" \
  "$COND_CACHE/CACHE_MANIFEST.json" "$UNCOND_BASE/config.json" \
  "$UNCOND_TEACHERS" "$UNCOND_THETA/adapter_model.safetensors" \
  "$UNCOND_THETA/optimizer.pt" "$UNCOND_THETA/scheduler.pt" \
  "$UNCOND_THETA/trainer_state.json"; do
  [[ -s "$path" ]] || { echo "missing required artifact: $path" >&2; exit 1; }
done
"$PY" - "$OUT/PREFLIGHT.json" "$COND_THETA/trainer_state.json" \
  "$UNCOND_THETA/trainer_state.json" "$COND_CACHE/CACHE_MANIFEST.json" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
out,cond,uncond,cache=map(Path,sys.argv[1:])
c=json.loads(cond.read_text()); u=json.loads(uncond.read_text()); m=json.loads(cache.read_text())
assert (c["global_step"],c["max_steps"]) == (192,256)
assert (u["global_step"],u["max_steps"]) == (200,256)
assert m["status"] == "COMPLETE" and m["sample_count"] == 512
out.write_text(json.dumps({
 "status":"PASS","checked_at":datetime.now(timezone.utc).isoformat(),
 "conditional":{"initial_global_step":192,"formal_endpoint":256,"continuation_updates":64,"cache_trajectories":512},
 "unconditional":{"initial_global_step":200,"formal_endpoint":256,"continuation_updates":56,"cache_trajectories":512},
 "cache_consistency":m["consistency_audit"],
},indent=2)+"\n")
PY

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$PYTHONPATH_VALUE"
export WANDB_MODE=disabled

stage=conditional_online
write_status RUNNING "$stage"
COND_ONLINE="$OUT/conditional/training/online"
if [[ "$("$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status","") if p.is_file() else "")' "$COND_ONLINE/TRAINING_PROTOCOL.json")" != COMPLETE ]]; then
  "$TORCHRUN" --nproc_per_node 2 --master_port 29731 "$HERE/train.py" \
    --mode online --config "$COND_CONFIG" --theta-zero "$COND_THETA" \
    --output-dir "$COND_ONLINE" --seed 42 \
    --online-sample-count 8192 --online-epochs 1 \
    --resume-optimizer-state --initial-global-step 192 --target-global-step 256 \
    2>&1 | tee "$OUT/logs/conditional_online_training.log"
fi

stage=conditional_offline
write_status RUNNING "$stage"
COND_OFFLINE="$OUT/conditional/training/offline"
if [[ "$("$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status","") if p.is_file() else "")' "$COND_OFFLINE/TRAINING_PROTOCOL.json")" != COMPLETE ]]; then
  "$TORCHRUN" --nproc_per_node 2 --master_port 29732 "$HERE/train.py" \
    --mode offline --config "$COND_CONFIG" --theta-zero "$COND_THETA" \
    --cache-dir "$COND_CACHE" --output-dir "$COND_OFFLINE" --seed 42 \
    --offline-sample-count 512 --offline-epochs 16 \
    --resume-optimizer-state --initial-global-step 192 --target-global-step 256 \
    2>&1 | tee "$OUT/logs/conditional_offline_training.log"
fi

stage=unconditional_cache
write_status RUNNING "$stage"
UNCOND_CACHE="$OUT/unconditional/cache"
if [[ ! -s "$UNCOND_CACHE/CACHE_MANIFEST.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PY" "$HERE/build_unconditional_cache.py" \
    --base-model "$UNCOND_BASE" --teacher-config "$UNCOND_TEACHERS" \
    --theta-zero "$UNCOND_THETA" --cache-dir "$UNCOND_CACHE" \
    --sample-start 0 --sample-stop 256 --sample-count 512 --batch-size 8 \
    --seed 42 --rank 0 --device cuda:0 > "$OUT/logs/unconditional_cache_0.log" 2>&1 &
  p0=$!
  CUDA_VISIBLE_DEVICES="${GPUS##*,}" "$PY" "$HERE/build_unconditional_cache.py" \
    --base-model "$UNCOND_BASE" --teacher-config "$UNCOND_TEACHERS" \
    --theta-zero "$UNCOND_THETA" --cache-dir "$UNCOND_CACHE" \
    --sample-start 256 --sample-stop 512 --sample-count 512 --batch-size 8 \
    --seed 42 --rank 1 --device cuda:0 > "$OUT/logs/unconditional_cache_1.log" 2>&1 &
  p1=$!
  failed=0
  wait "$p0" || failed=1
  wait "$p1" || failed=1
  (( failed == 0 ))
  "$PY" "$HERE/finalize_unconditional_cache.py" \
    --cache-dir "$UNCOND_CACHE" --teacher-config "$UNCOND_TEACHERS" \
    --theta-zero "$UNCOND_THETA" --base-model "$UNCOND_BASE" \
    --sample-count 512 --shards 2
fi

stage=unconditional_online
write_status RUNNING "$stage"
UNCOND_ONLINE="$OUT/unconditional/training/online"
if [[ "$("$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status","") if p.is_file() else "")' "$UNCOND_ONLINE/TRAINING_PROTOCOL.json")" != COMPLETE ]]; then
  "$TORCHRUN" --nproc_per_node 2 --master_port 29733 "$HERE/train_unconditional.py" \
    --mode online --base-model "$UNCOND_BASE" --teacher-config "$UNCOND_TEACHERS" \
    --theta-zero "$UNCOND_THETA" --output-dir "$UNCOND_ONLINE" \
    --online-sample-count 8192 --online-epochs 1 \
    --initial-global-step 200 --target-global-step 256 --seed 42 \
    2>&1 | tee "$OUT/logs/unconditional_online_training.log"
fi

stage=unconditional_offline
write_status RUNNING "$stage"
UNCOND_OFFLINE="$OUT/unconditional/training/offline"
if [[ "$("$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("status","") if p.is_file() else "")' "$UNCOND_OFFLINE/TRAINING_PROTOCOL.json")" != COMPLETE ]]; then
  "$TORCHRUN" --nproc_per_node 2 --master_port 29734 "$HERE/train_unconditional.py" \
    --mode offline --base-model "$UNCOND_BASE" --teacher-config "$UNCOND_TEACHERS" \
    --theta-zero "$UNCOND_THETA" --cache-dir "$UNCOND_CACHE" \
    --output-dir "$UNCOND_OFFLINE" --offline-sample-count 512 --offline-epochs 16 \
    --initial-global-step 200 --target-global-step 256 --seed 42 \
    2>&1 | tee "$OUT/logs/unconditional_offline_training.log"
fi

stage=training_complete
write_status COMPLETE "$stage"
touch "$OUT/TRAINING_COMPLETE"
