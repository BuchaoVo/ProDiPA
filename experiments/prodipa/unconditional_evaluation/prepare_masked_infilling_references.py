#!/usr/bin/env python3
"""Freeze the 512 natural reference sequences used by ESM2/ESM3/ASPO infilling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path


AA = set("ACDEFGHIKLMNPQRSTVWY")
FIELDS = ["sample_id", "sequence_id", "sequence", "length_aa", "homology_group", "source", "seed"]


def atomic_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        delete=False, prefix=f".{path.name}."
    ) as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))
    eligible: list[dict[str, str]] = []
    seen_sequences: set[str] = set()
    seen_groups: set[str] = set()
    for row in source_rows:
        sequence = "".join(row.get("sequence", "").split()).upper()
        group = row.get("homology_group", "")
        if not (50 <= len(sequence) <= 512 and set(sequence) <= AA):
            continue
        if row.get("split") != "train" or row.get("valid_complete", "").lower() != "true":
            continue
        if not group or group in seen_groups or sequence in seen_sequences:
            continue
        seen_groups.add(group)
        seen_sequences.add(sequence)
        eligible.append({**row, "sequence": sequence})
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[: args.requests]
    if len(selected) != args.requests:
        raise RuntimeError(f"only {len(selected)} eligible independent reference sequences")
    output_rows = [{
        "sample_id": index,
        "sequence_id": row["sequence_id"],
        "sequence": row["sequence"],
        "length_aa": len(row["sequence"]),
        "homology_group": row["homology_group"],
        "source": row.get("candidate_source", "frozen_train_pool"),
        "seed": args.seed,
    } for index, row in enumerate(selected)]
    atomic_tsv(args.output, output_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "COMPLETE",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "seed": args.seed,
        "requests": args.requests,
        "eligible_independent_sequences": len(eligible),
        "selection": "seeded shuffle after stable source order; one sequence per homology group",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
