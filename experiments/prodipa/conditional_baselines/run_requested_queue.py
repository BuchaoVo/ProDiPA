#!/usr/bin/env python3
"""Wait for audited resources, then generate and score requested Conditional methods."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from prodipa.baselines.audit_resources import audit_config
from prodipa.baselines.common import load_config


CONFIGS = {
    "aspo_tpr_hf_compat": "conditional_aspo.json",
    "ctrlprot_tpr": "conditional_ctrlprot.json",
    "prefixprot_tpr": "conditional_prefixprot.json",
    "protgpt3_msa_tpr": "conditional_protgpt3_msa.json",
    "lineageflow_tpr": "conditional_lineageflow.json",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def state(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("state", "MISSING"))
    except (OSError, json.JSONDecodeError):
        return "MISSING"


def resource_path(report: dict, name: str) -> Path:
    for item in report["resources"]:
        if item["name"] == name and item["status"] == "ready":
            return Path(item["resolved_path"])
    raise FileNotFoundError(name)


def select_gpu(min_free_mib: int) -> tuple[str, str, int] | None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    candidates = []
    allowed = {item.strip() for item in os.environ.get("CONDITIONAL_BASELINE_GPUS", "0,1,2,3").split(",")}
    for line in completed.stdout.splitlines():
        index, uuid, total, free = (item.strip() for item in line.split(","))
        if index in allowed and int(total) >= 70000 and int(free) >= min_free_mib:
            candidates.append((int(free), index, uuid))
    if not candidates:
        return None
    free, index, uuid = max(candidates)
    return index, uuid, free


def run(command: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        print("+", " ".join(command), file=handle, flush=True)
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--min-free-mib", type=int, default=28000)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/evaluation/conditional/requested_baselines"
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / ".queue.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another requested Conditional baseline queue is active")
    status_path = output / "STATUS.json"
    stopped = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    python = os.environ.get("PRODIPA_PYTHON", "__PRODIPA_ENVS__/ProDiPA/bin/python")
    config_root = root / "configs/baselines"
    current_status = root / "outputs/evaluation/conditional/SCORING_STATUS.json"
    scoring_gpus = os.environ.get("CONDITIONAL_BASELINE_GPUS", "0,1,2,3")

    while not stopped:
        if state(current_status) != "COMPLETE":
            atomic_json(status_path, {
                "state": "WAITING_CURRENT_EVALUATION", "current_state": state(current_status),
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            if args.once:
                return
            time.sleep(args.poll_seconds)
            continue

        reports = {
            method: audit_config(config_root / config_name, root)
            for method, config_name in CONFIGS.items()
        }
        pending = [method for method in CONFIGS if not (output / ".done" / method).is_file()]
        if not pending:
            atomic_json(status_path, {
                "state": "COMPLETE", "methods": list(CONFIGS),
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            return
        made_progress = False
        for method in pending:
            report = reports[method]
            if report["status"] != "ready":
                continue
            selected = select_gpu(args.min_free_mib)
            if selected is None:
                break
            index, uuid, free = selected
            config_path = config_root / CONFIGS[method]
            config = load_config(config_path)
            generation = root / config["protocol"]["output"]
            generation.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = uuid
            log = output / "logs" / f"{method}.log"
            atomic_json(status_path, {
                "state": "RUNNING", "method": method, "stage": "generation",
                "physical_gpu": index, "gpu_uuid": uuid, "free_mib_at_launch": free,
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            if method == "protgpt3_msa_tpr":
                command = [
                    python, str(root / "src/prodipa/baselines/run_protgpt3_msa_tpr.py"), "run",
                    "--config", str(config_path), "--project-root", str(root),
                    "--report", str(output / f"{method}_preflight.json"), "--device", "cuda:0",
                ]
            else:
                entrypoint = resource_path(report, "generation_entrypoint")
                command = [
                    python, str(entrypoint), "--config", str(config_path),
                    "--project-root", str(root), "--output", str(generation), "--device", "cuda:0",
                ]
            run(command, log, env)

            generation_tsv = output / "generation" / f"{method}.tsv"
            run([
                python, str(root / "experiments/prodipa/external_baseline_evaluation/convert_generations.py"),
                "--input", str(generation), "--output", str(generation_tsv),
                "--paradigm", "conditional", "--model", method, "--expected-requests", "512",
            ], log)
            atomic_json(status_path, {
                "state": "RUNNING", "method": method, "stage": "scoring",
                "updated_at": datetime.now().astimezone().isoformat(),
            })
            run([
                "bash", str(root / "experiments/prodipa/conditional_evaluation/run_method_evaluation.sh"),
                "--model-id", method, "--generation-tsv", str(generation_tsv),
                "--output-root", str(root / f"outputs/evaluation/conditional/methods/{method}"),
                "--gpus", scoring_gpus,
            ], log, env)
            method_root = root / f"outputs/evaluation/conditional/methods/{method}"
            protrek = method_root / "scores/protrek.tsv"
            run([
                python,
                str(root / "experiments/prodipa/conditional_evaluation/score_protrek_similarity.py"),
                "--input", str(method_root / "prepared/unique_valid_sequences.tsv"),
                "--output", str(protrek),
                "--repo", str(root / "third_party/ProTrek"),
                "--model", "__PRODIPA_MODELS__/ProTrek_35M",
                "--batch-size", "32", "--device", "cuda:0",
            ], log, env)
            run([
                "bash", str(root / "experiments/prodipa/conditional_evaluation/run_method_evaluation.sh"),
                "--model-id", method, "--generation-tsv", str(generation_tsv),
                "--protrek-tsv", str(protrek),
                "--output-root", str(method_root), "--gpus", scoring_gpus, "--publish",
            ], log, env)
            marker = output / ".done" / method
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(datetime.now().astimezone().isoformat() + "\n", encoding="utf-8")
            made_progress = True

        blockers = {method: reports[method]["blockers"] for method in pending if reports[method]["status"] != "ready"}
        atomic_json(status_path, {
            "state": "WAITING_RESOURCES" if blockers else "WAITING_GPU",
            "pending": pending, "resource_blockers": blockers,
            "updated_at": datetime.now().astimezone().isoformat(),
        })
        if args.once:
            return
        if not made_progress:
            time.sleep(args.poll_seconds)

    atomic_json(status_path, {
        "state": "STOPPED", "updated_at": datetime.now().astimezone().isoformat(),
    })


if __name__ == "__main__":
    main()
