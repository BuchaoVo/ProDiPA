#!/usr/bin/env python3
"""Validate arbitrary external baseline request TSVs and deduplicate scorer input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--paradigm", choices=("unconditional", "conditional"), required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model identifiers")
    parser.add_argument("--requests-per-model", type=int, default=512)
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        raise ValueError("--models is empty")
    all_rows: list[dict[str, str]] = []
    counts = {}
    for model in models:
        path = args.generation_root / args.paradigm / f"{model}.tsv"
        rows = read_tsv(path)
        if len(rows) != args.requests_per_model:
            raise RuntimeError(f"{path}: expected {args.requests_per_model}, got {len(rows)}")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise RuntimeError(f"duplicate sample_id in {path}")
        counts[model] = {
            "requests": len(rows),
            "valid_complete": sum(row["valid_complete"].lower() == "true" for row in rows),
            "eos_reached": sum(row["eos_reached"].lower() == "true" for row in rows),
        }
        for row in rows:
            row = dict(row)
            row["request_id"] = f"{args.paradigm}:{model}:{int(row['sample_id']):04d}"
            row["sequence_id"] = "sha256:" + hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest()
            all_rows.append(row)

    fields = ["request_id", "sequence_id"] + [
        field for field in all_rows[0] if field not in {"request_id", "sequence_id"}
    ]
    write_tsv(args.out_root / "all_requests.tsv", all_rows, fields)
    unique = {}
    for row in all_rows:
        if row["valid_complete"].lower() != "true":
            continue
        unique.setdefault(row["sequence_id"], {
            "sequence_id": row["sequence_id"], "sequence": row["sequence"],
            "length": row["length_aa"], "valid_complete": "true",
        })
    unique_rows = sorted(unique.values(), key=lambda row: row["sequence_id"])
    write_tsv(
        args.out_root / "unique_valid_sequences.tsv", unique_rows,
        ["sequence_id", "sequence", "length", "valid_complete"],
    )
    manifest = {
        "paradigm": args.paradigm,
        "models": counts,
        "expected_requests": len(models) * args.requests_per_model,
        "observed_requests": len(all_rows),
        "unique_valid_sequences": len(unique_rows),
    }
    (args.out_root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
