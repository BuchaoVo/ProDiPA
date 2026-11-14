"""Frozen protocol, GPU allocation and measured jobs for the cache comparison."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "experiments/prodipa/offline_multiteacher_opd"
FULL = Path(os.environ.get(
    "ADAPTIVE_FULL_ROOT", ROOT / "outputs/experiments/offline_multiteacher_opd_full"
))
OUT = Path(os.environ.get(
    "ADAPTIVE_OUTPUT_ROOT", ROOT / "outputs/experiments/adaptive_cache"
))
PY = "__PRODIPA_ENVS__/ProDiPA/bin/python"
TORCHRUN = "__PRODIPA_ENVS__/ProDiPA/bin/torchrun"
TEMBERTURE_PY = "__PRODIPA_ENVS__/temberture/bin/python"
METHODS = ("fixed", "periodic", "adaptive")
_ALL_STARTS = {"conditional": 192, "unconditional": 200}
_requested = tuple(
    item.strip() for item in os.environ.get(
        "ADAPTIVE_PARADIGMS", "conditional,unconditional"
    ).split(",") if item.strip()
)
if not _requested or any(item not in _ALL_STARTS for item in _requested):
    raise ValueError(f"invalid ADAPTIVE_PARADIGMS={_requested!r}")
STARTS = {item: _ALL_STARTS[item] for item in _requested}
BASES = {"conditional": "__PRODIPA_MODELS__/ProLLaMA", "unconditional": "__PRODIPA_MODELS__/ProtGPT2"}
THETAS = {
    "conditional": ROOT / "outputs/conditional/prollama_multiteacher_student/checkpoint-192",
    "unconditional": Path(os.environ.get(
        "ADAPTIVE_UNCONDITIONAL_THETA",
        ROOT / "outputs/unconditional/protgpt2_multiteacher_student/checkpoint-200",
    )),
}
CACHES = {
    "conditional": ROOT / "outputs/experiments/offline_multiteacher_opd/cache",
    "unconditional": Path(os.environ.get(
        "ADAPTIVE_UNCONDITIONAL_CACHE", FULL / "unconditional/cache"
    )),
}
ONLINE_MODELS = {
    "conditional": FULL / "conditional/training/online",
    "unconditional": Path(os.environ.get(
        "ADAPTIVE_UNCONDITIONAL_ONLINE",
        FULL / "unconditional/training/online",
    )),
}
CONFIG = ROOT / "experiments/prodipa/conditional_multiteacher_student/opd_config.yaml"
TEACHERS = ROOT / "src/prodipa/unconditional/teachers.yaml"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [{k.strip(): v.strip() if isinstance(v, str) else v
                 for k, v in r.items() if k is not None}
                for r in csv.DictReader(f, delimiter="\t")]


def tsv(path, values, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(values[0])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seqid(sequence):
    return "sha256:" + hashlib.sha256(sequence.encode()).hexdigest()


_status_lock = threading.Lock()
_training_states = {}


def status(stage, state="RUNNING", **kw):
    with _status_lock:
        payload = {"state": state, "stage": stage, "updated_at_unix": time.time(), **kw}
        paradigm = stage.split("/", 1)[0]
        if threading.current_thread().name.startswith("paradigm") and paradigm in STARTS:
            _training_states[paradigm] = payload
            dump(OUT / paradigm / "STATUS.json", payload)
            aggregate = "FAILED" if any(s["state"] == "FAILED" for s in _training_states.values()) else "RUNNING"
            payload = {"state": aggregate, "stage": "parallel_training_and_evaluation",
                       "updated_at_unix": time.time(), "paradigms": dict(_training_states)}
        dump(OUT / "STATUS.json", payload)
    print(f"[{time.strftime('%F %T')}] {state} {stage}", flush=True)


def env():
    e = os.environ.copy()
    e.update(PYTHONUNBUFFERED="1", WANDB_MODE="disabled", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    e["PYTHONPATH"] = ":".join(map(str, [ROOT / "src", ROOT / "third_party/ProteinOPD/conditional/proteinopd", OLD, HERE]))
    return e


def acquire(count, free_mib, paradigm=None):
    allowed = set(os.environ.get("ADAPTIVE_GPUS", "0,1,2,3").split(","))
    while True:
        if paradigm and (OUT / paradigm / "PAUSED.json").exists():
            raise RuntimeError(f"{paradigm} is paused by user; no further GPU allocation")
        found = []
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        candidates = [tuple(x.strip() for x in line.split(",")) for line in output.splitlines()]
        for index, uuid, free, util in sorted(candidates, key=lambda x: -int(x[2])):
            if index not in allowed or int(free) < free_mib or int(util) > 95:
                continue
            handle = open(f"/tmp/prodipa_gpu_{uuid}.lock", "a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            found.append((uuid, handle))
            if len(found) == count:
                return found
        for _, handle in found:
            handle.close()
        time.sleep(30)


def job(name, command, paradigm, method, phase, node, gpus=0, free_mib=8000):
    """Account allocated device time, including launch, loading and output I/O."""
    if (OUT / paradigm / "PAUSED.json").exists():
        raise RuntimeError(f"{paradigm} is paused by user; refusing job {name}")
    record = OUT / "jobs" / f"{name}.json"
    if record.exists() and read(record).get("status") == "COMPLETE":
        return
    log = OUT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    allocation = acquire(gpus, free_mib, paradigm) if gpus else []
    e = env()
    if allocation:
        e["CUDA_VISIBLE_DEVICES"] = ",".join(u for u, _ in allocation)
    started = time.time()
    previous = read(record).get("attempts", []) if record.exists() else []
    payload = dict(name=name, paradigm=paradigm, method=method, phase=phase, node=node,
                   gpu_ids=[u for u, _ in allocation], started_at_unix=started,
                   command=list(map(str, command)), status="RUNNING", attempts=previous)
    dump(record, payload)
    print(f"start {name} GPUs={payload['gpu_ids']} log={log}", flush=True)
    try:
        with log.open("a") as f:
            rc = subprocess.call(list(map(str, command)), cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT)
        if rc:
            raise RuntimeError(f"{name} exited {rc}; see {log}")
        payload["status"] = "COMPLETE"
    except BaseException:
        payload["status"] = "FAILED"
        raise
    finally:
        payload["finished_at_unix"] = time.time()
        payload["gpu_seconds"] = len(allocation) * (payload["finished_at_unix"] - started)
        previous.append({k: payload[k] for k in ("status", "gpu_ids", "started_at_unix", "finished_at_unix", "gpu_seconds")})
        dump(record, payload)
        for _, handle in allocation:
            handle.close()


def cost(paradigm, method, node, phase="training"):
    total = 0.0
    for path in (OUT / "jobs").glob("*.json"):
        r = read(path)
        if r["paradigm"] == paradigm and r["method"] == method and r["phase"] == phase and r["node"] <= node:
            total += sum(a["gpu_seconds"] for a in r["attempts"])
    if phase == "training":
        total += read(OUT / "PROTOCOL.json")["initial_cache_gpu_hours"][paradigm] * 3600
    return total / 3600
