#!/usr/bin/env python3
"""Verify that Conditional extension scores exactly cover valid unique sequences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def check(reference: set[str], path: Path, field: str) -> None:
    rows = read(path)
    identifiers = [row.get("sequence_id", "") for row in rows]
    if not identifiers or any(not item for item in identifiers):
        raise ValueError(f"{path}: missing sequence_id")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: duplicate sequence_id")
    if set(identifiers) != reference:
        raise ValueError(
            f"{path}: ID mismatch missing={len(reference-set(identifiers))} "
            f"extra={len(set(identifiers)-reference)}"
        )
    values = [row.get(field, "") for row in rows]
    if any(value == "" for value in values):
        raise ValueError(f"{path}: missing {field} values")
    for value in values:
        score = float(value)
        if not -100 <= score <= 100:
            raise ValueError(f"{path}: {field} outside [-100,100]: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--protrek", type=Path)
    args = parser.parse_args()
    reference = {row["sequence_id"] for row in read(args.sequences)}
    if not reference:
        raise ValueError("reference sequence table is empty")
    if args.protrek:
        check(reference, args.protrek, "protrek_percent")
    print(f"extension score coverage PASS: {len(reference)} unique valid sequences")


if __name__ == "__main__":
    main()
