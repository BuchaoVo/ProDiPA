#!/usr/bin/env python3
"""Merge ESMFold shards and require exactly one row per scorer input sequence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = {row["sequence_id"] for row in read(args.input)}
    merged = {}
    fields = None
    for index in range(args.num_shards):
        path = args.shards_root / f"esmfold_shard_{index:02d}.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read(path)
        if rows and fields is None:
            fields = list(rows[0])
        for row in rows:
            if row["sequence_id"] in merged:
                raise RuntimeError(f"duplicate structure row: {row['sequence_id']}")
            merged[row["sequence_id"]] = row
    if set(merged) != expected:
        raise RuntimeError(
            f"structure IDs differ: expected={len(expected)} observed={len(merged)} "
            f"missing={len(expected-set(merged))} extra={len(set(merged)-expected)}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged[key] for key in sorted(merged))
    failed = sum(row["status"] == "failed" for row in merged.values())
    print(f"merged={len(merged)} failed={failed} out={args.out}")


if __name__ == "__main__":
    main()
