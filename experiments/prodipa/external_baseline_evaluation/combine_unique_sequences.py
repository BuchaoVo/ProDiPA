#!/usr/bin/env python3
"""Combine prepared unique-sequence TSVs without duplicating sequence hashes."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_id: dict[str, dict[str, str]] = {}
    fields: list[str] | None = None
    for path in args.input:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            current_fields = list(reader.fieldnames or [])
            if fields is None:
                fields = current_fields
            elif current_fields != fields:
                raise ValueError(f"schema mismatch: {path}")
            for row in reader:
                sequence_id = row["sequence_id"]
                previous = by_id.get(sequence_id)
                if previous is not None and previous["sequence"] != row["sequence"]:
                    raise ValueError(f"sequence hash collision: {sequence_id}")
                by_id[sequence_id] = row
    if not fields or not by_id:
        raise ValueError("no sequences were loaded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=args.output.parent,
        delete=False, prefix=f".{args.output.name}.",
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(by_id[key] for key in sorted(by_id))
        temporary = handle.name
    os.replace(temporary, args.output)
    print(f"combined unique sequences: {len(by_id)} -> {args.output}")


if __name__ == "__main__":
    main()
