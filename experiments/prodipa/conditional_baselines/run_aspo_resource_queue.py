#!/usr/bin/env python3
"""Prepare frozen TPR steering vectors on the available evaluation GPUs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def pipeline_state(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "MISSING"))
    except (OSError, json.JSONDecodeError):
        return "MISSING"


def eligible_gpus(allowed: set[str], minimum_free: int) -> list[tuple[str, str, int]]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    candidates = []
    for line in output.splitlines():
        index, uuid, total, free, utilization = (value.strip() for value in line.split(","))
        if index in allowed and int(total) >= 70000 and int(free) >= minimum_free:
            candidates.append((int(free), -int(utilization), index, uuid))
    candidates.sort(reverse=True)
    return [(index, uuid, free) for free, _negative_utilization, index, uuid in candidates]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--allowed-gpus", default="2,3")
    parser.add_argument("--minimum-free-mib", type=int, default=30000)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument(
        "--wait-for-upstream", action="store_true",
        help="Wait for the unrelated Unconditional student pipeline before allocating GPUs.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/evaluation/conditional/requested_baselines"
    vectors = output / "aspo_vectors"
    status = output / "ASPO_RESOURCE_STATUS.json"
    log = output / "aspo_resource_queue.log"
    lock = (output / ".aspo_resource_queue.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another steering-vector resource queue is active")
    upstream = root / "outputs/unconditional/protgpt2_multiteacher_student/PIPELINE_STATUS.json"
    properties = ("fold", "sol", "thermo")
    while args.wait_for_upstream and pipeline_state(upstream) == "RUNNING":
        atomic_json(status, {
            "state": "WAITING_UPSTREAM_PIPELINE",
            "upstream": str(upstream),
            "updated_at": datetime.now().astimezone().isoformat(),
        })
        time.sleep(args.poll_seconds)
    missing = [name for name in properties if not (vectors / f"{name}.pt").is_file()]
    if not missing:
        atomic_json(status, {"state": "COMPLETE", "updated_at": datetime.now().astimezone().isoformat()})
        return
    allowed = {value.strip() for value in args.allowed_gpus.split(",") if value.strip()}
    while True:
        selected = eligible_gpus(allowed, args.minimum_free_mib)
        if not selected:
            atomic_json(status, {
                "state": "WAITING_GPU", "allowed_gpus": sorted(allowed),
                "minimum_free_mib": args.minimum_free_mib,
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            time.sleep(args.poll_seconds)
            continue
        locked: list[tuple[str, str, int, object]] = []
        for index, uuid, free in selected:
            gpu_lock = Path(f"/tmp/prodipa_gpu_{uuid}.lock").open("w", encoding="utf-8")
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gpu_lock.close()
                continue
            locked.append((index, uuid, free, gpu_lock))
            if len(locked) == min(len(missing), len(allowed)):
                break
        if locked:
            break
        time.sleep(args.poll_seconds)

    # Spread properties deterministically.  For three properties and two GPUs,
    # GPU 2 receives Fold/Thermo and GPU 3 receives Sol.
    assignments: list[list[str]] = [[] for _ in locked]
    for property_index, name in enumerate(missing):
        assignments[property_index % len(locked)].append(name)
    jobs = []
    try:
        for (index, uuid, free, gpu_lock), names in zip(locked, assignments):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = uuid
            command = [
                "__PRODIPA_ENVS__/ProDiPA/bin/python",
                str(root / "src/prodipa/baselines/aspo_prollama_hf.py"), "extract",
                "--config", str(root / "configs/baselines/conditional_aspo.json"),
                "--project-root", str(root), "--device", "cuda:0",
                "--properties", ",".join(names),
            ]
            job_log = output / f"aspo_extract_gpu_{index}.log"
            handle = job_log.open("a", encoding="utf-8")
            print("+", " ".join(command), file=handle, flush=True)
            process = subprocess.Popen(
                command, env=environment, stdout=handle, stderr=subprocess.STDOUT,
            )
            jobs.append({
                "process": process, "handle": handle, "lock": gpu_lock,
                "physical_gpu": index, "gpu_uuid": uuid,
                "free_mib_at_launch": free, "properties": names,
                "log": str(job_log),
            })
        atomic_json(status, {
            "state": "RUNNING",
            "jobs": [{key: value for key, value in job.items() if key not in {"process", "handle", "lock"}}
                     | {"pid": job["process"].pid} for job in jobs],
            "updated_at": datetime.now().astimezone().isoformat(),
        })
        failures = []
        for job in jobs:
            returncode = job["process"].wait()
            if returncode != 0:
                failures.append({"physical_gpu": job["physical_gpu"], "returncode": returncode})
        if failures:
            raise RuntimeError(f"steering-vector extraction failed: {failures}")
    except Exception as error:
        atomic_json(status, {
            "state": "FAILED", "error": repr(error),
            "updated_at": datetime.now().astimezone().isoformat(),
        })
        raise
    finally:
        for job in jobs:
            job["handle"].close()
            job["lock"].close()
    missing = [name for name in properties if not (vectors / f"{name}.pt").is_file()]
    final_state = "COMPLETE" if not missing else "FAILED"
    atomic_json(status, {
        "state": final_state, "missing": missing,
        "updated_at": datetime.now().astimezone().isoformat(),
    })
    if missing:
        raise SystemExit(f"missing vector artifacts after extraction: {missing}")


if __name__ == "__main__":
    main()
