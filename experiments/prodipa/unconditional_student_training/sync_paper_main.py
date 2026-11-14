#!/usr/bin/env python3
"""Synchronize comparable canonical Unconditional rows into the paper table."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[3]
CANONICAL = ROOT / "outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv"
PAPER = ROOT / "outputs/evaluation/paper/unconditional_main.tsv"
METRICS = ("PPL", "Novelty-U", "Novelty-T", "pLDDT", "pAE", "Sol", "Thermo", "HV")
STALE_PAIRED = {
    "Reference θ0", "Online OPD (paired)", "ProDiPA (Offline)",
    "Periodic Cache OPD", "ProDiPA (Adaptive Cache)",
}


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    fields, paper_rows = read(PAPER)
    _, canonical_rows = read(CANONICAL)
    latest = {row["Model"]: row for row in canonical_rows}
    for row in paper_rows:
        current = latest.get(row["Model"])
        if current:
            for metric in METRICS:
                if metric in row and metric in current:
                    row[metric] = current[metric]
        if row["Model"] in STALE_PAIRED:
            for metric in METRICS:
                if metric in row:
                    row[metric] = "--"
            if "HV Ret. (%)" in row:
                row["HV Ret. (%)"] = "N/A"
            for field in ("Train GPU-h", "HV-AUC", "HQ Disc."):
                if field in row:
                    row[field] = "--"
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=PAPER.parent,
        delete=False, prefix=f".{PAPER.name}.",
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(paper_rows); temporary=handle.name
    os.replace(temporary, PAPER)


if __name__ == "__main__":
    main()
