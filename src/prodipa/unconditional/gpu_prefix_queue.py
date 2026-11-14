#!/usr/bin/env python3
"""Resource-safe queue for the three formal ProtGPT2 Prefix teachers.

The supervisor performs CPU-only data/tokenizer preflight first. It creates no
CUDA context while waiting, locks a GPU by UUID, runs one BF16 engineering check,
cleans its temporary checkpoint directory, then trains Fold -> Sol -> Thermo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import yaml

HERE = Path(__file__).resolve().parent
from prefix_data import file_for, run_preflight, sha256

JOBS = (("fold", "plddt"), ("sol", "sol"), ("thermo", "thermo"))
ACTIVE_CHILD: subprocess.Popen | None = None


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def notice(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def terminate_queue(signum, _frame) -> None:
    global ACTIVE_CHILD
    notice(f"Received signal {signum}; stopping active trainer")
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        try:
            os.killpg(ACTIVE_CHILD.pid, signal.SIGTERM)
            ACTIVE_CHILD.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE_CHILD.pid, signal.SIGKILL)
            ACTIVE_CHILD.wait()
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


def list_gpus() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 6 or any(value in {"N/A", "[Not Supported]"} for value in values[3:]):
            continue
        index, uuid, name, total, free, utilization = values
        rows.append({
            "index": int(index), "uuid": uuid, "name": name,
            "total_mib": int(total), "free_mib": int(free), "util_pct": int(utilization),
        })
    return rows


def eligible(gpu: dict, args) -> bool:
    return (
        gpu["index"] in args.allowed_gpu_ids
        and gpu["total_mib"] >= args.min_total_mib
        and gpu["free_mib"] >= args.min_free_mib
        and gpu["util_pct"] <= args.max_util_pct
    )


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("a+")
    try:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        descriptor.seek(0)
        descriptor.truncate()
        descriptor.write(f"{os.getpid()} {stamp()}\n")
        descriptor.flush()
        return descriptor
    except BlockingIOError:
        descriptor.close()
        return None


def wait_for_gpu(args, purpose: str) -> tuple[dict, object]:
    notice(
        f"Waiting for GPU ({purpose}): free>={args.min_free_mib} MiB, "
        f"util<={args.max_util_pct}%, stable={args.stable_polls}"
    )
    stable = 0
    previous_uuid = None
    while True:
        try:
            candidates = sorted(
                (gpu for gpu in list_gpus() if eligible(gpu, args)),
                key=lambda gpu: gpu["free_mib"], reverse=True,
            )
            choice = candidates[0] if candidates else None
            if choice is None:
                stable = 0
                previous_uuid = None
                notice("No GPU currently meets formal resource thresholds")
            else:
                stable = stable + 1 if choice["uuid"] == previous_uuid else 1
                previous_uuid = choice["uuid"]
                notice(
                    f"GPU {choice['index']} free={choice['free_mib']} MiB "
                    f"util={choice['util_pct']}% stable={stable}/{args.stable_polls}"
                )
                if stable >= args.stable_polls:
                    lock_path = Path("/tmp") / (
                        "prodipa_gpu_" + re.sub(r"[^A-Za-z0-9_-]", "_", choice["uuid"]) + ".lock"
                    )
                    lock = acquire_lock(lock_path)
                    if lock is not None:
                        refreshed = next((gpu for gpu in list_gpus() if gpu["uuid"] == choice["uuid"]), None)
                        if refreshed is not None and eligible(refreshed, args):
                            return refreshed, lock
                        lock.close()
                        stable = 0
        except Exception as error:
            notice(f"GPU query transient failure: {type(error).__name__}: {error}")
            stable = 0
            previous_uuid = None
        time.sleep(args.poll_seconds)


def directory_fingerprint(model_dir: Path) -> str:
    important = sorted(
        path for path in model_dir.iterdir()
        if path.is_file() and (
            path.name in {"config.json", "tokenizer.json", "tokenizer_config.json"}
            or path.suffix in {".bin", ".safetensors"}
        )
    )
    if not important:
        raise ValueError(f"No model/config files found in {model_dir}")
    digest = hashlib.sha256()
    for path in important:
        digest.update(path.name.encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def build_config(
    property_name, attribute, data_dir, model, output_dir, base_fingerprint,
    source_revision, sequence_protocol="plain_direct_eos_v1",
):
    return {
        "property_name": property_name,
        "model_name_or_path": str(model),
        "tokenizer_name_or_path": str(model),
        "train_path": str(file_for(data_dir, "train", attribute)),
        "validation_path": str(file_for(data_dir, "test", attribute)),
        "output_dir": str(output_dir),
        "text_column": "sequence", "id_column": "sequence_id",
        "num_virtual_tokens": 100, "max_length": 512,
        "learning_rate": 0.005, "lr_end": 1e-7, "lr_power": 3.0,
        "max_epochs": 50, "micro_batch_size": 8, "gradient_accumulation_steps": 1,
        "seed": 42, "early_stopping_patience": 5, "early_stopping_min_delta": 1e-4,
        "dtype": "bfloat16", "num_workers": 0, "log_every_optimizer_steps": 10,
        "resume": True, "base_model_sha256": base_fingerprint,
        "source_code_revision": source_revision, "generation_samples_per_epoch": 0,
        "sequence_protocol": sequence_protocol,
    }


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def run_child(command: list[str], env: dict, log_path: Path) -> int:
    global ACTIVE_CHILD
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, start_new_session=True,
        ) as process:
            ACTIVE_CHILD = process
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
            ACTIVE_CHILD = None
            return code


def trainer_environment(gpu: dict) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WANDB_MODE", "disabled")
    return env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="__PRODIPA_ROOT__/data/unconditional")
    parser.add_argument("--model", default=os.getenv("PROTGPT2_MODEL", ""))
    parser.add_argument(
        "--output-root",
        default="__PRODIPA_ROOT__/outputs/unconditional/protgpt2",
    )
    parser.add_argument("--allowed-gpus", default="", help="Physical nvidia-smi indices; default all")
    parser.add_argument("--min-total-mib", type=int, default=70000)
    parser.add_argument("--min-free-mib", type=int, default=51200)
    parser.add_argument("--max-util-pct", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--smoke-optimizer-steps", type=int, default=2)
    parser.add_argument(
        "--sequence-protocol",
        choices=("plain_direct_eos_v1", "protgpt2_fasta60_v1"),
        default="plain_direct_eos_v1",
        help=(
            "Teacher token/EOS protocol. Historical runs retain plain_direct_eos_v1; "
            "new pretraining-compatible runs must explicitly select protgpt2_fasta60_v1."
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGTERM, terminate_queue)
    signal.signal(signal.SIGINT, terminate_queue)
    args = parse_args()
    if args.poll_seconds < 1 or args.stable_polls < 1 or args.smoke_optimizer_steps < 1:
        raise ValueError("poll-seconds, stable-polls and smoke-optimizer-steps must be positive")
    all_indices = {gpu["index"] for gpu in list_gpus()}
    args.allowed_gpu_ids = (
        {int(value.strip()) for value in args.allowed_gpus.split(",") if value.strip()}
        if args.allowed_gpus.strip() else all_indices
    )
    if args.once:
        for gpu in list_gpus():
            notice(
                f"GPU {gpu['index']} {gpu['name']}: free={gpu['free_mib']} MiB "
                f"util={gpu['util_pct']}% eligible={eligible(gpu, args)}"
            )
        return
    if not args.model:
        raise ValueError("Set PROTGPT2_MODEL to the original pretrained ProtGPT2 directory")
    model = Path(args.model).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    trainer = HERE / "train_teacher.py"
    if not trainer.is_file():
        raise FileNotFoundError(trainer)

    output_root.mkdir(parents=True, exist_ok=True)
    queue_lock = acquire_lock(output_root / ".queue.lock")
    if queue_lock is None:
        raise RuntimeError(f"Another queue already owns {output_root}")
    try:
        audit = run_preflight(data_dir)
        atomic_json(output_root / "data_preflight.json", audit)
        base_fingerprint = directory_fingerprint(model)
        source_revision = "sha256:" + sha256(trainer)
        configs = {}
        for property_name, attribute in JOBS:
            config = build_config(
                property_name, attribute, data_dir, model, output_root / property_name,
                base_fingerprint, source_revision, args.sequence_protocol,
            )
            config_path = output_root / "configs" / f"{property_name}.yaml"
            write_yaml(config_path, config)
            configs[property_name] = config_path
            result = subprocess.run(
                [sys.executable, str(trainer), "--config", str(config_path), "--preflight-only"],
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"CPU/tokenizer preflight failed for {property_name}")
        notice("CPU/data/tokenizer preflight passed for Fold, Sol and Thermo")
        if args.dry_run:
            for property_name, _ in JOBS:
                notice(f"DRY_RUN {property_name}: {sys.executable} {trainer} --config {configs[property_name]}")
            return

        engineering_report = output_root / "engineering_check.json"
        if not engineering_report.is_file():
            gpu, lock = wait_for_gpu(args, "BF16 engineering check")
            smoke_root = output_root / ".engineering_check_tmp"
            if smoke_root.exists():
                shutil.rmtree(smoke_root)
            smoke_config = yaml.safe_load(configs["fold"].read_text(encoding="utf-8"))
            smoke_config["output_dir"] = str(smoke_root / "fold")
            smoke_config["resume"] = False
            smoke_config_path = output_root / "configs" / "engineering_check_fold.yaml"
            write_yaml(smoke_config_path, smoke_config)
            try:
                notice(f"START BF16 engineering check on GPU {gpu['index']} ({gpu['uuid']})")
                code = run_child(
                    [sys.executable, "-u", str(trainer), "--config", str(smoke_config_path),
                     "--smoke-optimizer-steps", str(args.smoke_optimizer_steps)],
                    trainer_environment(gpu), output_root / "engineering_check.log",
                )
                smoke_status = smoke_root / "fold" / "SMOKE_OK.json"
                if code or not smoke_status.is_file():
                    raise RuntimeError(f"BF16 engineering check failed (exit={code})")
                report = json.loads(smoke_status.read_text(encoding="utf-8"))
                report.update({"gpu_index": gpu["index"], "gpu_uuid": gpu["uuid"], "completed": stamp()})
                atomic_json(engineering_report, report)
                notice("BF16 engineering check passed; removing temporary smoke checkpoints")
            finally:
                lock.close()
                if smoke_root.exists():
                    shutil.rmtree(smoke_root)

        for property_name, _ in JOBS:
            job_dir = output_root / property_name
            done = job_dir / "DONE.json"
            if done.is_file() and (job_dir / "best_adapter" / "adapter_config.json").is_file():
                notice(f"SKIP {property_name}: completed adapter already present")
                continue
            failed = job_dir / "FAILED.json"
            resumable = job_dir / "latest_resume" / "training_state.pt"
            if failed.is_file() and not resumable.is_file():
                raise RuntimeError(
                    f"Non-resumable failed run at {job_dir}; inspect it and use a new versioned output root"
                )
            gpu, lock = wait_for_gpu(args, property_name)
            state_path = job_dir / "queue_state.json"
            state = {
                "property": property_name, "state": "running", "updated": stamp(),
                "gpu_index": gpu["index"], "gpu_uuid": gpu["uuid"],
                "config": str(configs[property_name]), "resume": resumable.is_file(),
            }
            atomic_json(state_path, state)
            if failed.exists():
                failed.unlink()
            try:
                notice(f"START {property_name} on GPU {gpu['index']} ({gpu['uuid']})")
                code = run_child(
                    [sys.executable, "-u", str(trainer), "--config", str(configs[property_name])],
                    trainer_environment(gpu), job_dir / "queue_stdout.log",
                )
                if code or not done.is_file() or not (job_dir / "best_adapter" / "adapter_config.json").is_file():
                    state.update({"state": "failed", "return_code": code, "updated": stamp()})
                    atomic_json(state_path, state)
                    raise RuntimeError(f"{property_name} failed exit={code}; see {job_dir / 'queue_stdout.log'}")
                state.update({"state": "complete", "return_code": 0, "updated": stamp()})
                atomic_json(state_path, state)
                notice(f"COMPLETED {property_name}")
            finally:
                lock.close()
        atomic_json(output_root / "ALL_DONE.json", {"state": "complete", "completed": stamp()})
        notice("All three formal Prefix teachers completed")
    finally:
        queue_lock.close()


if __name__ == "__main__":
    main()
