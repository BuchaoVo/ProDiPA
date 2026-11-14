#!/usr/bin/env python3
"""Convert audited external-baseline JSONL into the frozen scorer request TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from prodipa.baselines.common import read_jsonl


FIELDS = [
    "paradigm", "model", "sample_id", "seed", "sequence", "length_aa",
    "valid_sequence", "eos_reached", "truncated", "valid_complete",
    "wrapper_closed", "invalid_reason", "raw_completion", "temperature",
    "top_p", "top_k", "repetition_penalty", "max_new_tokens", "condition",
    "generation_method", "generation_config_id", "source_request_id",
]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def completed_generation(row: dict) -> bool:
    """Return whether the method reached its declared stopping condition.

    Autoregressive methods normally stop on EOS.  Fixed-length protocols such
    as ProtDAT instead finish when the requested target length is reached; that
    is a successful completion, not a max-token truncation.
    """
    if bool(row["terminated"]):
        return True
    return row.get("termination_reason") == "target_length_reached"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paradigm", choices=("unconditional", "conditional"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-requests", type=int, default=512)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if len(rows) != args.expected_requests:
        raise RuntimeError(f"expected {args.expected_requests} rows, found {len(rows)}")
    converted = []
    for sample_id, row in enumerate(rows):
        metadata = row.get("method_metadata") or {}
        valid = bool(row["valid"])
        completed = completed_generation(row)
        valid_complete = valid and completed
        converted.append({
            "paradigm": args.paradigm,
            "model": args.model,
            "sample_id": sample_id,
            "seed": row["seed"],
            "sequence": row["sequence"],
            "length_aa": row["length_aa"],
            "valid_sequence": bool_text(valid),
            # The frozen scorer schema uses eos_reached as its generic
            # successful-stop flag, including non-EOS fixed-length methods.
            "eos_reached": bool_text(completed),
            "truncated": bool_text(not completed),
            "valid_complete": bool_text(valid_complete),
            "wrapper_closed": bool_text(bool(row["terminated"])),
            "invalid_reason": (
                "" if valid_complete else
                "invalid_length_or_alphabet" if not valid else
                "generation_incomplete"
            ),
            "raw_completion": row["raw_output"],
            "temperature": metadata.get("temperature", ""),
            "top_p": metadata.get("top_p", ""),
            "top_k": metadata.get("top_k", ""),
            "repetition_penalty": metadata.get("repetition_penalty", ""),
            "max_new_tokens": metadata.get("max_new_tokens", ""),
            "condition": row["condition"],
            "generation_method": row["generation_method"],
            "generation_config_id": row["generation_config_id"],
            "source_request_id": row["request_id"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(converted)


if __name__ == "__main__":
    main()
