#!/usr/bin/env python3
"""Fail-closed validation for one standardized Unconditional generation TSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REQUIRED_FIELDS = (
    "paradigm", "model", "sample_id", "seed", "sequence", "length_aa",
    "valid_sequence", "eos_reached", "truncated", "valid_complete",
    "wrapper_closed", "invalid_reason", "raw_completion", "temperature",
    "top_p", "top_k", "repetition_penalty", "max_new_tokens",
)
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def boolean(value: str, field: str, row_number: int) -> bool:
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"row {row_number}: {field} must be true/false, got {value!r}")
    return lowered == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in REQUIRED_FIELDS if field not in fields]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        rows = list(reader)

    if len(rows) != args.requests:
        raise ValueError(f"expected {args.requests} requests, observed {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id values are not unique")

    valid = 0
    eos = 0
    lengths: list[int] = []
    for row_number, row in enumerate(rows, start=2):
        if row["paradigm"] != "unconditional":
            raise ValueError(f"row {row_number}: paradigm must be unconditional")
        if row["model"] != args.model_id:
            raise ValueError(
                f"row {row_number}: model={row['model']!r}, expected {args.model_id!r}"
            )
        sequence = row["sequence"].strip().upper()
        length = int(row["length_aa"])
        if length != len(sequence):
            raise ValueError(
                f"row {row_number}: length_aa={length}, actual sequence length={len(sequence)}"
            )
        is_valid = boolean(row["valid_complete"], "valid_complete", row_number)
        valid_sequence = boolean(row["valid_sequence"], "valid_sequence", row_number)
        reached_eos = boolean(row["eos_reached"], "eos_reached", row_number)
        boolean(row["truncated"], "truncated", row_number)
        boolean(row["wrapper_closed"], "wrapper_closed", row_number)
        if is_valid:
            if not valid_sequence or not AA.fullmatch(sequence):
                raise ValueError(f"row {row_number}: valid_complete row has invalid AA sequence")
            if not args.min_length <= length <= args.max_length:
                raise ValueError(
                    f"row {row_number}: valid_complete length {length} outside "
                    f"[{args.min_length}, {args.max_length}]"
                )
            valid += 1
            lengths.append(length)
        eos += int(reached_eos)

    report = {
        "input": str(args.input.resolve()),
        "model": args.model_id,
        "requests": len(rows),
        "valid_complete": valid,
        "valid_rate": valid / len(rows),
        "eos_reached": eos,
        "eos_rate": eos / len(rows),
        "valid_length_min": min(lengths) if lengths else None,
        "valid_length_max": max(lengths) if lengths else None,
        "status": "PASS",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
