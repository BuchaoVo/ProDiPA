#!/usr/bin/env python3
"""Merge legacy and direct InterPro TPR candidates without losing evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "sequence_id", "sequence", "length_aa", "legacy_split", "legacy_attributes",
    "accession", "source_database", "organism_tax_id", "cath_superfamily",
    "tpr_coverage_union", "cath_models", "best_cath_evalue", "family_evidence",
    "interpro_version", "fold_score", "sol_score", "thermo_score", "valid_complete",
]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--interpro", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    merged: dict[str, dict[str, str]] = {}
    for source in [args.legacy, *args.interpro]:
        for raw in read(source):
            sequence = raw["sequence"].strip().upper()
            row = merged.setdefault(sequence, {field: "" for field in FIELDS})
            direct = raw.get("family_evidence", "").startswith("InterPro cathgene3d")
            if not row["sequence_id"] or direct:
                for field in FIELDS:
                    if raw.get(field):
                        row[field] = raw[field]
            else:
                for field in FIELDS:
                    if not row[field] and raw.get(field):
                        row[field] = raw[field]
            row["sequence"] = sequence
            row["length_aa"] = str(len(sequence))
            row["valid_complete"] = "true"
            if raw.get("legacy_split"):
                if row["legacy_split"] and row["legacy_split"] != raw["legacy_split"]:
                    raise ValueError(f"Conflicting legacy split for {row['sequence_id']}")
                row["legacy_split"] = raw["legacy_split"]
            if raw.get("legacy_attributes"):
                values = set(filter(None, row["legacy_attributes"].split(";")))
                values.update(filter(None, raw["legacy_attributes"].split(";")))
                row["legacy_attributes"] = ";".join(sorted(values))
    all_rows = sorted(merged.values(), key=lambda row: row["sequence_id"])
    rows = [
        row for row in all_rows
        if args.min_length <= len(row["sequence"]) <= args.max_length
    ]
    if not rows:
        raise RuntimeError("No candidates to merge")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(
        f"written={len(rows)} excluded_by_length={len(all_rows) - len(rows)} "
        f"length_range={args.min_length}-{args.max_length} path={args.out}"
    )


if __name__ == "__main__":
    main()
