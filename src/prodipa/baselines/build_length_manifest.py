from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

try:
    from .common import atomic_write_json, atomic_write_jsonl, normalize_sequence, sha256_file, sha256_text
except ImportError:  # direct script execution
    from common import atomic_write_json, atomic_write_jsonl, normalize_sequence, sha256_file, sha256_text


SEQUENCE_KEYS = ("sequence", "seq", "output")
ID_KEYS = ("sequence_id", "id", "protein_id", "request_id")


def _extract_sequence(row: dict[str, Any]) -> str:
    value: Any = None
    for key in SEQUENCE_KEYS:
        if key in row:
            value = row[key]
            break
    if isinstance(value, str) and "Seq=<" in value:
        value = value.split("Seq=<", 1)[1].split(">", 1)[0]
    return normalize_sequence(value)


def _extract_id(row: dict[str, Any], sequence: str) -> str:
    for key in ID_KEYS:
        if row.get(key) not in (None, ""):
            return str(row[key])
    return sha256_text(sequence)


def read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("JSON source must be an array")
        return value
    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a deterministic target-length benchmark manifest")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--request-prefix", default="dplm_650m")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"Train25K source is missing: {args.source}")
    source_rows = read_table(args.source)
    eligible: list[tuple[str, str]] = []
    excluded = 0
    for row in source_rows:
        sequence = _extract_sequence(row)
        if args.min_length <= len(sequence) <= args.max_length:
            eligible.append((_extract_id(row, sequence), sequence))
        else:
            excluded += 1
    if len(eligible) < args.requests:
        raise ValueError(f"only {len(eligible)} eligible sequences for {args.requests} requests")

    selected = random.Random(args.seed).sample(eligible, args.requests)
    manifest = [
        {
            "request_id": f"{args.request_prefix}_{index:06d}",
            "target_length": len(sequence),
            "length_source_id": source_id,
            "length_source_sequence_sha256": sha256_text(sequence),
            "seed": args.seed,
        }
        for index, (source_id, sequence) in enumerate(selected, start=1)
    ]
    atomic_write_jsonl(args.output, manifest)
    atomic_write_json(
        args.report,
        {
            "status": "complete",
            "source": str(args.source.resolve()),
            "source_sha256": sha256_file(args.source),
            "source_rows": len(source_rows),
            "eligible_rows": len(eligible),
            "excluded_outside_length_range": excluded,
            "requests": args.requests,
            "seed": args.seed,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "request_prefix": args.request_prefix,
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
        },
    )
    print(f"wrote {len(manifest)} frozen target lengths to {args.output}")


if __name__ == "__main__":
    main()
