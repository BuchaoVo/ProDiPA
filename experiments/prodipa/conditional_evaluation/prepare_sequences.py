#!/usr/bin/env python3
"""Validate the 8 x 512 generation budget and build deduplicated scorer input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


MODELS = {
    "unconditional": ("base", "fold_teacher", "sol_teacher", "thermo_teacher"),
    "conditional": ("base", "fold_teacher", "sol_teacher", "thermo_teacher"),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--requests-per-model", type=int, default=512)
    parser.add_argument(
        "--paradigm",
        choices=("unconditional", "conditional"),
        help="Prepare one generation paradigm only; default prepares both.",
    )
    parser.add_argument(
        "--models",
        help="Optional comma-separated model identifiers for the selected paradigm.",
    )
    args = parser.parse_args()

    requested_models = None
    if args.models:
        if args.paradigm is None:
            raise ValueError("--models requires --paradigm")
        requested_models = tuple(item.strip() for item in args.models.split(",") if item.strip())
        if not requested_models:
            raise ValueError("--models is empty")

    all_rows: list[dict[str, str]] = []
    counts = {}
    selected_models = {
        paradigm: requested_models if requested_models is not None else models
        for paradigm, models in MODELS.items()
        if args.paradigm is None or paradigm == args.paradigm
    }
    for paradigm, models in selected_models.items():
        for model in models:
            path = args.generation_root / paradigm / f"{model}.tsv"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = read_tsv(path)
            if len(rows) != args.requests_per_model:
                raise RuntimeError(f"{path}: expected {args.requests_per_model}, got {len(rows)}")
            if len({row["sample_id"] for row in rows}) != len(rows):
                raise RuntimeError(f"duplicate sample_id in {path}")
            counts[f"{paradigm}/{model}"] = {
                "requests": len(rows),
                "valid_complete": sum(row["valid_complete"].lower() == "true" for row in rows),
                "eos_reached": sum(row["eos_reached"].lower() == "true" for row in rows),
            }
            for row in rows:
                row = dict(row)
                row["request_id"] = f"{paradigm}:{model}:{int(row['sample_id']):04d}"
                sequence = row["sequence"]
                row["sequence_id"] = "sha256:" + hashlib.sha256(sequence.encode("ascii")).hexdigest()
                all_rows.append(row)

    request_fields = ["request_id", "sequence_id"] + [
        field for field in all_rows[0] if field not in {"request_id", "sequence_id"}
    ]
    write_tsv(args.out_root / "all_requests.tsv", all_rows, request_fields)

    unique = {}
    for row in all_rows:
        if row["valid_complete"].lower() != "true":
            continue
        unique.setdefault(row["sequence_id"], {
            "sequence_id": row["sequence_id"],
            "sequence": row["sequence"],
            "length": row["length_aa"],
            "valid_complete": "true",
        })
    unique_rows = sorted(unique.values(), key=lambda row: row["sequence_id"])
    write_tsv(
        args.out_root / "unique_valid_sequences.tsv",
        unique_rows,
        ["sequence_id", "sequence", "length", "valid_complete"],
    )
    manifest = {
        "paradigm": args.paradigm or "both",
        "expected_requests": sum(len(models) for models in selected_models.values()) * args.requests_per_model,
        "observed_requests": len(all_rows),
        "unique_valid_sequences": len(unique_rows),
        "models": counts,
    }
    (args.out_root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
