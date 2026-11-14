from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .common import atomic_write_jsonl, build_generation_record, read_jsonl
except ImportError:  # direct script execution
    from common import atomic_write_jsonl, build_generation_record, read_jsonl


COMMON_INPUT_FIELDS = {
    "request_id",
    "raw_output",
    "sequence",
    "terminated",
    "termination_reason",
    "seed",
    "condition",
}


def normalize_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(args.input), start=1):
        sequence = row.get("sequence", "")
        raw_output = row.get("raw_output", sequence)
        metadata = {key: value for key, value in row.items() if key not in COMMON_INPUT_FIELDS}
        normalized.append(
            build_generation_record(
                request_id=row.get("request_id", f"{args.request_prefix}_{index:06d}"),
                benchmark=args.benchmark,
                model=args.model,
                seed=int(row.get("seed", args.seed)),
                condition=row.get("condition", args.condition),
                raw_output=raw_output,
                sequence=sequence,
                terminated=bool(row.get("terminated", False)),
                termination_reason=row.get("termination_reason", "not_reported"),
                generation_method=args.generation_method,
                generation_config_id=args.generation_config_id,
                method_metadata=metadata,
                min_length=args.min_length,
                max_length=args.max_length,
            )
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize external baseline output to ProDiPA JSONL")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation-method", required=True)
    parser.add_argument("--generation-config-id", required=True)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", default="")
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    rows = normalize_rows(args)
    atomic_write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} standardized requests to {args.output}")


if __name__ == "__main__":
    main()
