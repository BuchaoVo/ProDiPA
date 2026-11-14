from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (
        AA_ALPHABET,
        REQUIRED_GENERATION_FIELDS,
        atomic_write_json,
        read_jsonl,
        sequence_is_valid,
        sha256_file,
        sha256_text,
    )
except ImportError:  # direct script execution
    from common import (
        AA_ALPHABET,
        REQUIRED_GENERATION_FIELDS,
        atomic_write_json,
        read_jsonl,
        sequence_is_valid,
        sha256_file,
        sha256_text,
    )


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    expected_requests: int,
    benchmark: str | None,
    model: str | None,
    request_prefix: str | None,
    min_length: int,
    max_length: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    request_ids: list[str] = []
    sequences: list[str] = []
    valid_count = 0
    terminated_count = 0

    if len(rows) != expected_requests:
        errors.append(f"request_count={len(rows)} expected={expected_requests}")

    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_GENERATION_FIELDS if field not in row]
        if missing:
            errors.append(f"row={index} missing_fields={','.join(missing)}")
            continue
        request_id = str(row["request_id"])
        request_ids.append(request_id)
        sequence = str(row["sequence"])
        sequences.append(sequence)
        if request_prefix and not request_id.startswith(request_prefix):
            errors.append(f"row={index} unexpected_request_id={request_id}")
        if benchmark and row["benchmark"] != benchmark:
            errors.append(f"row={index} benchmark={row['benchmark']} expected={benchmark}")
        if model and row["model"] != model:
            errors.append(f"row={index} model={row['model']} expected={model}")
        if row["length_aa"] != len(sequence):
            errors.append(f"row={index} length_aa_mismatch")
        if row["sequence_sha256"] != sha256_text(sequence):
            errors.append(f"row={index} sequence_sha256_mismatch")
        computed_valid = sequence_is_valid(sequence, min_length=min_length, max_length=max_length)
        if bool(row["valid"]) != computed_valid:
            errors.append(f"row={index} valid_flag_mismatch")
        if computed_valid:
            valid_count += 1
        if bool(row["terminated"]):
            terminated_count += 1
        illegal = sorted(set(sequence) - AA_ALPHABET)
        if illegal:
            warnings.append(f"row={index} illegal_symbols={''.join(illegal)}")

    duplicate_request_ids = sorted(key for key, count in Counter(request_ids).items() if count > 1)
    if duplicate_request_ids:
        errors.append(f"duplicate_request_ids={len(duplicate_request_ids)}")
    duplicate_sequence_instances = sum(count - 1 for count in Counter(sequences).values() if count > 1)

    return {
        "status": "passed" if not errors else "failed",
        "request_count": len(rows),
        "expected_requests": expected_requests,
        "valid_count": valid_count,
        "valid_rate": valid_count / len(rows) if rows else 0.0,
        "terminated_count": terminated_count,
        "termination_rate": terminated_count / len(rows) if rows else 0.0,
        "unique_sequence_count": len(set(sequences)),
        "duplicate_sequence_instances": duplicate_sequence_instances,
        "duplicate_request_ids": duplicate_request_ids,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed audit of standardized baseline generations")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-requests", type=int, default=512)
    parser.add_argument("--benchmark")
    parser.add_argument("--model")
    parser.add_argument("--request-prefix")
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    report = audit_rows(
        rows,
        expected_requests=args.expected_requests,
        benchmark=args.benchmark,
        model=args.model,
        request_prefix=args.request_prefix,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    report["input"] = str(args.input.resolve())
    report["input_sha256"] = sha256_file(args.input)
    atomic_write_json(args.output, report)
    print(f"{report['status']}: {args.input} ({report['request_count']} requests)")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
