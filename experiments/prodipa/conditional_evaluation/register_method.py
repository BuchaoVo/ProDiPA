#!/usr/bin/env python3
"""Atomically register one completed method in the canonical Conditional cohort."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path


MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    evaluation_root = args.evaluation_root.resolve()
    if not MODEL_ID.fullmatch(args.model_id):
        raise ValueError(f"invalid readable model id: {args.model_id!r}")
    required = (
        evaluation_root / "results/sequence_scores.tsv",
        evaluation_root / "results/supplementary_metrics.tsv",
        evaluation_root / "results/scoring_completeness.tsv",
        evaluation_root / "prepared/unique_valid_sequences.tsv",
        evaluation_root / "SCORING_STATUS.json",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"method evaluation is incomplete: {missing}")

    lock_path = config_path.parent / f".{config_path.name}.register.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("evaluation_name") != "conditional":
        raise ValueError("unsupported Conditional evaluation config")
    with (evaluation_root / "results/sequence_scores.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != int(config["requests_per_model"]):
        raise ValueError(f"{args.model_id} has {len(rows)} requests")
    if {row.get("model") for row in rows} != {args.model_id}:
        raise ValueError("sequence-score model column does not match --model-id")
    if {row.get("paradigm") for row in rows} != {"conditional"}:
        raise ValueError("sequence-score paradigm must be conditional")
    status = json.loads((evaluation_root / "SCORING_STATUS.json").read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE":
        raise ValueError("method scoring status is not COMPLETE")

    required_extensions = [
        field for field, enabled in config.get("required_extensions", {}).items() if enabled
    ]
    valid = [row for row in rows if row.get("valid_complete", "").lower() == "true"]
    for field in required_extensions:
        missing_count = sum(row.get(field, "") == "" for row in valid)
        if missing_count:
            raise ValueError(
                f"cannot publish {args.model_id}: {missing_count}/{len(valid)} valid rows "
                f"lack required {field} evidence"
            )

    try:
        relative_root = str(evaluation_root.relative_to(root))
    except ValueError:
        relative_root = str(evaluation_root)
    groups = config.setdefault("score_groups", [])
    registrations = [
        (group, model) for group in groups for model in group.get("models", [])
        if model == args.model_id
    ]
    if registrations:
        group = registrations[0][0]
        if len(registrations) != 1 or group.get("evaluation_root") != relative_root:
            raise ValueError(f"model {args.model_id!r} is registered elsewhere")
        print(f"already registered: {args.model_id} -> {relative_root}")
        return
    groups.append({
        "name": f"method_{args.model_id}",
        "models": [args.model_id],
        "evaluation_root": relative_root,
    })
    atomic_json(config_path, config)
    print(f"registered: {args.model_id} -> {relative_root}")


if __name__ == "__main__":
    main()
