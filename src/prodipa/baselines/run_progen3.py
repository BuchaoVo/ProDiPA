#!/usr/bin/env python3
"""Run official ProGen3-762M unconditional autoregressive generation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import (
        atomic_write_json,
        atomic_write_jsonl,
        build_generation_record,
        expand_path,
        load_config,
        read_jsonl,
        seed_everything,
    )
except ImportError:
    from audit_resources import audit_config
    from common import (
        atomic_write_json,
        atomic_write_jsonl,
        build_generation_record,
        expand_path,
        load_config,
        read_jsonl,
        seed_everything,
    )


def resolved(audit: dict[str, Any], name: str) -> Path:
    for item in audit["resources"]:
        if item["name"] == name and item["status"] == "ready":
            return Path(item["resolved_path"])
    raise RuntimeError(f"resource is not ready: {name}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path, project_root: Path) -> None:
    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"ProGen3 resources are incomplete: {audit['blockers']}")
    repo = resolved(audit, "official_repo")
    checkpoint = resolved(audit, "checkpoint")
    weights = resolved(audit, "checkpoint_weights")
    expected_sha256 = str(config["resources"]["checkpoint_weights"]["expected_sha256"])
    observed_sha256 = file_sha256(weights)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"ProGen3 checkpoint SHA256 is {observed_sha256}, expected {expected_sha256}"
        )

    sys.path.insert(0, str(repo / "src"))
    from progen3.generator import ProGen3Generator
    from progen3.tools.utils import get_progen3_model

    protocol = config["protocol"]
    total = int(protocol["requests"])
    seed = int(protocol["seed"])
    chunk_size = int(protocol["resume_chunk_size"])
    output = expand_path(protocol["output"], base=project_root)
    records = list(read_jsonl(output) if output.exists() else [])
    if len(records) > total:
        raise RuntimeError(f"existing output has {len(records)} rows, expected at most {total}")
    if len(records) == total:
        print(f"already complete: {output}")
        return

    model = get_progen3_model(str(checkpoint), use_fsdp=False)
    generator = ProGen3Generator(
        model=model,
        max_batch_tokens=int(protocol["max_batch_tokens"]),
        temperature=float(protocol["temperature"]),
        top_p=float(protocol["top_p"]),
    )
    while len(records) < total:
        start = len(records)
        count = min(chunk_size, total - start)
        batch_seed = seed + start
        seed_everything(batch_seed)
        iterator = generator.generate(
            prompt="1",
            num_sequences=count,
            min_new_tokens=int(protocol["min_new_tokens"]),
            max_new_tokens=int(protocol["max_new_tokens"]),
        )
        results = list(iterator)
        if len(results) != count:
            raise RuntimeError(f"ProGen3 returned {len(results)} requests, expected {count}")
        for offset, result in enumerate(results, start=1):
            index = start + offset
            sequence = result.sequence or ""
            eos_reached = result.sequence is not None
            records.append(
                build_generation_record(
                    request_id=f"progen3_762m_{index:06d}",
                    benchmark="unconditional",
                    model="progen3_762m",
                    seed=seed + index - 1,
                    condition="unconditional_forward",
                    raw_output=result.generation,
                    sequence=sequence,
                    terminated=eos_reached,
                    termination_reason="eos" if eos_reached else "invalid_or_no_eos",
                    generation_method="autoregressive_causal_language_model",
                    generation_config_id=str(config["generation_config_id"]),
                    method_metadata={
                        "temperature": protocol["temperature"],
                        "top_p": protocol["top_p"],
                        "min_new_tokens": protocol["min_new_tokens"],
                        "max_new_tokens": protocol["max_new_tokens"],
                        "direction": "N_to_C",
                        "batch_seed": batch_seed,
                    },
                )
            )
        atomic_write_jsonl(output, records)
        print(f"ProGen3 requests complete: {len(records)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/progen3_762m.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0", help="CUDA visibility is set by the launcher")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"ProGen3 preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
