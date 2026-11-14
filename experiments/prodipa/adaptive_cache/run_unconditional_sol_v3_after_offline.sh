#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
HERE="$ROOT/experiments/prodipa/adaptive_cache"
UPSTREAM="${OFFLINE_SOL_V3_ROOT:-$ROOT/outputs/experiments/offline_multiteacher_opd_unconditional_sol_v3}"
OUT="${ADAPTIVE_OUTPUT_ROOT:-$ROOT/outputs/experiments/adaptive_cache_unconditional_sol_v3}"
THETA="$ROOT/outputs/unconditional/protgpt2_multiteacher_student/checkpoint-200"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$OUT/logs"
exec 9>"$OUT/.handoff.lock"
flock -n 9 || { echo "another Sol-V3 Unconditional Adaptive Cache handoff is active" >&2; exit 2; }
exec > >(tee -a "$OUT/pipeline.log") 2>&1

write_status() {
  "$PY" - "$OUT/HANDOFF_STATUS.json" "$1" "$2" "${3:-}" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
path=Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True)
payload={"state":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],
         "updated_at":datetime.now(timezone.utc).isoformat()}
with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False) as handle:
    json.dump(payload,handle,indent=2); handle.write("\n"); temporary=handle.name
os.replace(temporary,path)
PY
}

stage=wait_for_offline_comparison
on_error() { local rc=$?; write_status FAILED "$stage" "exit code $rc"; exit "$rc"; }
trap on_error ERR INT TERM

write_status WAITING "$stage" "waiting for paired Unconditional Online/Offline scoring and publication"
while [[ ! -f "$UPSTREAM/COMPLETE" ]]; do
  if [[ -f "$UPSTREAM/STATUS.json" ]] && grep -q '"state": "FAILED"' "$UPSTREAM/STATUS.json"; then
    echo "upstream paired experiment failed" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

stage=preflight
write_status RUNNING "$stage" "validating new teacher lineage, cache and paired Online endpoint"
"$PY" - "$ROOT" "$UPSTREAM" "$THETA" "$OUT/UPSTREAM_MANIFEST.json" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

root, upstream, theta, destination = map(Path, sys.argv[1:])
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

status=json.loads((upstream/"STATUS.json").read_text())
if status.get("state") != "COMPLETE":
    raise ValueError(f"upstream is not complete: {status}")
cache=json.loads((upstream/"cache/CACHE_MANIFEST.json").read_text())
if cache.get("status") != "COMPLETE" or cache.get("sample_count") != 512:
    raise ValueError("upstream fixed cache is incomplete")
state=json.loads((theta/"trainer_state.json").read_text())
if (state.get("global_step"),state.get("max_steps")) != (200,256):
    raise ValueError("theta0 is not the paired step-200 checkpoint")
if sha(theta/"adapter_model.safetensors") != cache["theta_zero"]["adapter_weights_sha256"]:
    raise ValueError("theta0/cache weight mismatch")
teachers=[]
for item in cache["teachers"]:
    path=Path(item["adapter_path"])/"adapter_model.safetensors"
    actual=sha(path)
    if actual != item["adapter_weights_sha256"]:
        raise ValueError(f"teacher hash mismatch: {item['name']}")
    teachers.append({"name":item["name"],"path":str(path.parent),"sha256":actual})
online=upstream/"training/online"
for path in [online/"adapter_model.safetensors",online/"timings/rank_0.json",online/"timings/rank_1.json"]:
    if not path.is_file(): raise FileNotFoundError(path)
payload={"status":"PASS","checked_at":datetime.now(timezone.utc).isoformat(),
         "upstream":str(upstream.resolve()),"theta_zero":str(theta.resolve()),
         "cache":str((upstream/"cache").resolve()),"online_endpoint":str(online.resolve()),
         "teachers":teachers,"adaptive_output":str(destination.parent.resolve())}
destination.write_text(json.dumps(payload,indent=2)+"\n")
PY

stage=adaptive_cache
write_status RUNNING "$stage" "starting Fixed/Periodic/Adaptive full Unconditional comparison on GPUs 0-3"
export ADAPTIVE_PARADIGMS=unconditional
export ADAPTIVE_GPUS="${ADAPTIVE_GPUS:-0,1,2,3}"
export ADAPTIVE_OUTPUT_ROOT="$OUT"
export ADAPTIVE_FULL_ROOT="$UPSTREAM"
export ADAPTIVE_UNCONDITIONAL_THETA="$THETA"
export ADAPTIVE_UNCONDITIONAL_CACHE="$UPSTREAM/cache"
export ADAPTIVE_UNCONDITIONAL_ONLINE="$UPSTREAM/training/online"

"$PY" -u "$HERE/run.py"
write_status COMPLETE complete "Unconditional Adaptive Cache training, evaluation and publication complete"
trap - ERR INT TERM
