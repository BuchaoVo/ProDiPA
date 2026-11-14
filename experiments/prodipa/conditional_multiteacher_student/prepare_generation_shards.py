#!/usr/bin/env python3
"""Seed independent generation shards from an existing partial TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    if not fields:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_sequences % args.num_shards:
        raise ValueError("num-sequences must be divisible by num-shards")

    fields, existing = read_rows(args.existing)
    width = args.num_sequences // args.num_shards
    seen: set[int] = set()
    for row in existing:
        sample_id = int(row["sample_id"])
        if sample_id in seen or not 0 <= sample_id < args.num_sequences:
            raise ValueError(f"Invalid or duplicate sample_id: {sample_id}")
        seen.add(sample_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(args.num_shards):
        start, stop = shard * width, (shard + 1) * width
        target = args.output_dir / f"shard_{shard}.tsv"
        shard_fields, shard_rows = read_rows(target)
        if shard_rows:
            if fields and shard_fields != fields:
                raise ValueError(f"Schema mismatch in {target}")
            for row in shard_rows:
                sample_id = int(row["sample_id"])
                if not start <= sample_id < stop:
                    raise ValueError(f"Out-of-range sample_id {sample_id} in {target}")
            continue
        rows = [row for row in existing if start <= int(row["sample_id"]) < stop]
        write_rows(target, fields, rows)


if __name__ == "__main__":
    main()
