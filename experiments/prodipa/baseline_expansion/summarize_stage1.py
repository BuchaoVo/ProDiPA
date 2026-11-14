#!/usr/bin/env python3
"""Summarize the six Stage-1 baseline pipelines without changing their state."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


METHODS = {
    "conditional": (
        "protdat_tpr_text",
        "pinal_1.2b_tpr_text",
        "paag_tpr_annotation",
    ),
    "unconditional": (
        "progen3_762m",
        "evodiff_oa_dm_640m",
        "dplm2_650m",
    ),
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"state": "INVALID_STATUS", "path": str(path)}


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def pid_state(path: Path) -> tuple[int | None, bool]:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid, True
    except (OSError, ValueError):
        return None, False


def generation_path(root: Path, paradigm: str, method: str) -> Path:
    if paradigm == "conditional":
        return root / "outputs/evaluation/conditional/baseline_generation" / f"{method}.jsonl"
    return root / "outputs/unconditional/external_baselines" / method / "generations.jsonl"


def summarize(root: Path) -> dict:
    result = {
        "schema_version": 1,
        "updated_at": datetime.now().astimezone().isoformat(),
        "expected_requests_per_method": 512,
        "methods": {},
    }
    counts: dict[str, int] = {}
    for paradigm, methods in METHODS.items():
        for method in methods:
            pipeline = root / "outputs/evaluation" / paradigm / "requested_baselines" / method
            status = load_json(pipeline / "STATUS.json")
            pid, alive = pid_state(pipeline / "pipeline.pid")
            generated = generation_path(root, paradigm, method)
            generated_rows = line_count(generated)
            method_result = {
                "paradigm": paradigm,
                "state": status.get("state", "NOT_STARTED"),
                "stage": status.get("stage", "unknown"),
                "pipeline_pid": pid,
                "pipeline_alive": alive,
                "generated_requests": generated_rows,
                "generation_complete": generated_rows == 512,
                "published": (pipeline / "COMPLETE.json").is_file(),
                "pipeline_log": str(pipeline / "pipeline.log"),
                "generation_log": str(pipeline / "logs/generation.log"),
                "resource_audit": str(pipeline / "preflight/resources.json"),
            }
            if status.get("updated_at"):
                method_result["status_updated_at"] = status["updated_at"]
            result["methods"][method] = method_result
            counts[method_result["state"]] = counts.get(method_result["state"], 0) + 1
    result["summary"] = {
        "method_count": sum(len(value) for value in METHODS.values()),
        "state_counts": counts,
        "all_published": all(item["published"] for item in result["methods"].values()),
    }
    return result


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/baseline_expansion/STAGE1_STATUS.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    payload = summarize(root)
    atomic_write(output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
