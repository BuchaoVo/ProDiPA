#!/usr/bin/env python3
"""Atomically append verified external-baseline rows to one existing paradigm main table."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--expected-models", required=True)
    args = parser.parse_args()

    fields, existing = read(args.existing)
    new_fields, additional = read(args.additional)
    if fields != new_fields:
        raise RuntimeError(f"table schema mismatch: {fields} != {new_fields}")
    expected = [item.strip() for item in args.expected_models.split(",") if item.strip()]
    found = [row["Model"] for row in additional]
    if sorted(found) != sorted(expected):
        raise RuntimeError(f"additional table models {found} != expected {expected}")
    by_model = {row["Model"]: row for row in existing}
    for row in additional:
        by_model[row["Model"]] = row
    order = [row["Model"] for row in existing]
    order.extend(model for model in expected if model not in order)
    rows = [by_model[model] for model in order]

    args.existing.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=args.existing.parent,
        prefix=f".{args.existing.name}.", delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, args.existing)


if __name__ == "__main__":
    main()
