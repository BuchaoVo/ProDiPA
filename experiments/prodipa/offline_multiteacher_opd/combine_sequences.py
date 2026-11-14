#!/usr/bin/env python3
"""Combine per-method valid sequence tables without changing sequence identifiers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields = None
    rows = []
    seen: dict[str, str] = {}
    for path in args.input:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            current = list(reader.fieldnames or [])
            if fields is None:
                fields = current
            elif current != fields:
                raise ValueError(f"schema mismatch: {path}")
            for row in reader:
                sequence_id = row["sequence_id"]
                if sequence_id in seen:
                    if seen[sequence_id] != row["sequence"]:
                        raise ValueError(f"hash collision for sequence_id: {sequence_id}")
                    continue
                seen[sequence_id] = row["sequence"]
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields or [], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"combined {len(rows)} unique valid sequences")


if __name__ == "__main__":
    main()
