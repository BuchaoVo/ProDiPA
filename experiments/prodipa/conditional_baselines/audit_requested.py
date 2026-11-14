#!/usr/bin/env python3
"""Audit resources for all requested Conditional comparison methods."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from prodipa.baselines.audit_resources import audit_config


CONFIGS = (
    "conditional_aspo.json",
    "conditional_ctrlprot.json",
    "conditional_prefixprot.json",
    "conditional_protgpt3_msa.json",
    "conditional_lineageflow.json",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/evaluation/conditional/requested_baselines/RESOURCE_AUDIT.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_root = root / "configs/baselines"
    reports = [audit_config(config_root / name, root) for name in CONFIGS]
    result = {
        "state": "READY" if all(report["status"] == "ready" for report in reports) else "WAITING_RESOURCES",
        "methods": reports,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    output = args.output if args.output.is_absolute() else root / args.output
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
