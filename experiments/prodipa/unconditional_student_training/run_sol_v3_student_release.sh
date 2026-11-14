#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
TEACHER_EXP="$ROOT/outputs/experiments/unconditional_teacher_sol_v3_fasta60_v1"
RUN="$ROOT/outputs/experiments/unconditional_sol_v3_student_release"
STUDENT="$ROOT/outputs/unconditional/protgpt2_multiteacher_student"
STUDENT_EVAL="$ROOT/outputs/evaluation/unconditional/methods/protgpt2_multiteacher_student"
STATUS="$RUN/STATUS.json"
GPU_MEMORY_LOG="$RUN/logs/student_training_gpu_memory.tsv"
POLL_SECONDS="${POLL_SECONDS:-30}"
MIN_TRAIN_FREE_MIB="${MIN_TRAIN_FREE_MIB:-60000}"
MAX_TRAIN_UTIL="${MAX_TRAIN_UTIL:-20}"
MIN_EVAL_FREE_MIB="${MIN_EVAL_FREE_MIB:-50000}"
MAX_EVAL_UTIL="${MAX_EVAL_UTIL:-25}"

mkdir -p "$RUN/logs"
exec 8>"$RUN/.pipeline.lock"
flock -n 8 || { echo "another Sol V3 student release is active" >&2; exit 2; }
exec > >(tee -a "$RUN/pipeline.log") 2>&1

write_status() {
  "$PY" - "$STATUS" "$1" "$2" "${3:-}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
v={"state":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],
   "updated_at":datetime.now(timezone.utc).isoformat()}
with tempfile.NamedTemporaryFile("w",dir=p.parent,delete=False) as f:
 json.dump(v,f,indent=2);f.write("\n");tmp=f.name
os.replace(tmp,p)
PY
}

GPU_MONITOR_PID=""
stop_gpu_monitor() {
  if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  GPU_MONITOR_PID=""
}

start_gpu_monitor() {
  printf 'timestamp\tindex\tmemory_used_mib\tmemory_free_mib\tutilization_pct\n' > "$GPU_MEMORY_LOG"
  (
    while true; do
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits | awk -F',' -v ts="$(date --iso-8601=seconds)" \
        -v selected="$TRAIN_GPUS" '
          BEGIN { split(selected, wanted, ","); for (i in wanted) keep[wanted[i]]=1 }
          { for(i=1;i<=NF;i++) gsub(/^ +| +$/, "", $i); if ($1 in keep) print ts "\t" $1 "\t" $2 "\t" $3 "\t" $4 }
        ' >> "$GPU_MEMORY_LOG"
      sleep 2
    done
  ) &
  GPU_MONITOR_PID=$!
}

stage=wait_for_teacher_evaluation
on_error() { local rc=$?; stop_gpu_monitor; write_status FAILED "$stage" "exit code $rc"; exit "$rc"; }
trap on_error ERR
write_status WAITING "$stage" "waiting for frozen Novelty-U/HV and teacher evaluation audit"
while [[ ! -s "$TEACHER_EXP/evaluation/EVALUATION_COMPLETE.json" ]]; do
  if [[ -s "$TEACHER_EXP/evaluation/EVALUATION_STATUS.json" ]] && \
     grep -q '"state": "FAILED"' "$TEACHER_EXP/evaluation/EVALUATION_STATUS.json"; then
    echo "teacher evaluation failed" >&2; exit 1
  fi
  sleep "$POLL_SECONDS"
done

stage=canonical_teacher_promotion
write_status RUNNING "$stage" "publishing three new Prefix teachers and replacing teacher scores"
"$PY" "$ROOT/experiments/prodipa/unconditional_teacher_training/publish_sol_v3_teachers.py" \
  > "$RUN/logs/teacher_promotion.log"

gpu_rows() {
  nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{for(i=1;i<=NF;i++)gsub(/^ +| +$/, "", $i); if($1>=0 && $1<=3)print $1","$2","$3","$4}'
}

choose_training_pair() {
  local line index uuid free util fd candidates=() fds=()
  while true; do
    candidates=(); fds=()
    while IFS=',' read -r index uuid free util; do
      if (( free >= MIN_TRAIN_FREE_MIB && util <= MAX_TRAIN_UTIL )); then
        exec {fd}>"/tmp/prodipa_gpu_${uuid}.lock"
        if flock -n "$fd"; then
          candidates+=("$index"); fds+=("$fd")
          if (( ${#candidates[@]} == 2 )); then
            TRAIN_GPUS="${candidates[0]},${candidates[1]}"
            TRAIN_LOCK_FDS=("${fds[@]}")
            return 0
          fi
        else
          exec {fd}>&-
        fi
      fi
    done < <(gpu_rows | sort -t',' -k3,3nr)
    if (( ${#fds[@]} )); then
      for fd in "${fds[@]}"; do flock -u "$fd" || true; eval "exec ${fd}>&-"; done
    fi
    write_status WAITING wait_for_training_gpus "need two GPUs among 0-3 with >=${MIN_TRAIN_FREE_MIB} MiB free and <=${MAX_TRAIN_UTIL}% util"
    sleep "$POLL_SECONDS"
  done
}

stage=wait_for_training_gpus
choose_training_pair
stage=student_training
write_status RUNNING "$stage" "fresh 256-update OPD with new teachers on GPUs $TRAIN_GPUS"
start_gpu_monitor
OUTPUT_DIR="$STUDENT" TRAINING_GPUS="$TRAIN_GPUS" SEQUENCE_FORMAT=legacy MAX_LENGTH_AA=512 \
  bash "$ROOT/experiments/prodipa/unconditional_student_training/run_training.sh" \
  2>&1 | tee "$RUN/logs/student_training.log"
stop_gpu_monitor
for fd in "${TRAIN_LOCK_FDS[@]}"; do flock -u "$fd" || true; eval "exec ${fd}>&-"; done
test -s "$STUDENT/DONE.json"

stage=wait_for_evaluation_gpus
while true; do
  ready=0
  while IFS=',' read -r index uuid free util; do
    (( free >= MIN_EVAL_FREE_MIB && util <= MAX_EVAL_UTIL )) && ready=$((ready+1))
  done < <(gpu_rows)
  (( ready == 4 )) && break
  write_status WAITING "$stage" "need GPU 0-3 with >=${MIN_EVAL_FREE_MIB} MiB free and <=${MAX_EVAL_UTIL}% util"
  sleep "$POLL_SECONDS"
done

stage=student_generation_scoring_publication
write_status RUNNING "$stage" "four-GPU generation, full scoring, Novelty-U/HV, and canonical main table"
OUTPUT_DIR="$STUDENT" EVAL_DIR="$STUDENT_EVAL" TRAINING_GPUS="$TRAIN_GPUS" \
EVALUATION_GPUS=0,1,2,3 SEQUENCE_FORMAT=legacy MAX_LENGTH_AA=512 \
  bash "$ROOT/experiments/prodipa/unconditional_student_training/run_full_pipeline.sh" \
  2>&1 | tee "$RUN/logs/student_full_pipeline.log"

stage=paper_table_sync
write_status RUNNING "$stage" "syncing comparable rows and invalidating stale paired old-teacher rows"
"$PY" "$ROOT/experiments/prodipa/unconditional_student_training/sync_paper_main.py"

stage=finalize_release
"$PY" - "$ROOT/outputs/unconditional/protgpt2/CANONICAL_RELEASE.json" \
  "$STUDENT/adapter_model.safetensors" \
  "$ROOT/outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv" <<'PY'
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
release_path, adapter, table = map(Path,sys.argv[1:])
release=json.loads(release_path.read_text())
with adapter.open('rb') as f:
 h=hashlib.sha256(); [h.update(x) for x in iter(lambda:f.read(1024*1024),b'')]
with table.open(newline='',encoding='utf-8') as f:
 rows=list(csv.DictReader(f,delimiter='\t'))
student=next(row for row in rows if row['Model']=='protgpt2_multiteacher_student')
release.update({"status":"PUBLISHED_SOL_V3_TEACHERS_AND_STUDENT_COMPLETE",
 "student_adapter_sha256":h.hexdigest(),"student_metrics":student,
 "student_published_at":datetime.now(timezone.utc).isoformat()})
release_path.write_text(json.dumps(release,indent=2)+"\n")
PY

write_status COMPLETE complete "new teachers, OPD student, evaluations and main tables published"
trap - ERR
echo "Sol V3 Unconditional teacher/student release complete"
