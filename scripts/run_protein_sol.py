#!/usr/bin/env python3
"""Run the local Protein-Sol sequence predictor for a ProDiPA TSV."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_predictions(path: Path) -> dict[str, dict[str, str]]:
    predictions = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for fields in csv.reader(handle):
            if not fields or fields[0] != "SEQUENCE PREDICTIONS":
                continue
            sequence_id = fields[1].strip().removeprefix(">")
            predictions[sequence_id] = {
                "protein_sol_percent": fields[2].strip(),
                "solubility_scaled_01": fields[3].strip(),
                "protein_sol_population": fields[4].strip(),
                "protein_sol_pi": fields[5].strip(),
            }
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/properties/pilot_1000.tsv"))
    parser.add_argument(
        "--software", type=Path,
        default=Path("tools/protein_sol/protein-sol-sequence-prediction-software"),
    )
    parser.add_argument("--work", type=Path, default=Path("work/property_annotation/protein_sol"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/properties/pilot_1000_protein_sol.tsv"),
    )
    parser.add_argument("--run-summary", type=Path)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[:args.limit]
    predictions = {}
    if args.reuse:
        predictions.update({row["sequence_id"]: row for row in read_rows(args.reuse) if row["status"] == "ok"})
    eligible = [row for row in rows if row.get("valid_complete", "true").lower() == "true"]
    pending = [row for row in eligible if row["sequence_id"] not in predictions]
    started = time.perf_counter()
    if pending:
        if args.work.exists():
            shutil.rmtree(args.work)
        shutil.copytree(args.software, args.work)
        fasta = args.work / "input.fasta"
        with fasta.open("w", encoding="ascii") as handle:
            for row in pending:
                handle.write(f">{row['sequence_id']}\n{row['sequence']}\n")
        subprocess.run(
            ["bash", "multiple_prediction_wrapper_export.sh", fasta.name],
            cwd=args.work,
            check=True,
        )
        predictions.update(parse_predictions(args.work / "seq_prediction.txt"))
    elapsed = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence_id", "solubility_scaled_01", "protein_sol_percent",
        "protein_sol_population", "protein_sol_pi", "status", "error_reason",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            prediction = predictions.get(row["sequence_id"])
            eligible_row = row.get("valid_complete", "true").lower() == "true"
            writer.writerow({
                "sequence_id": row["sequence_id"],
                **(prediction or {}),
                "status": "ok" if prediction else ("failed" if eligible_row else "skipped"),
                "error_reason": "" if prediction else (
                    "missing_output" if eligible_row else "invalid_or_incomplete_generation"
                ),
            })

    valid = sum(row["sequence_id"] in predictions for row in eligible)
    print(f"sequences={len(rows)} reused={len(rows) - len(pending)} valid={valid} elapsed_s={elapsed:.3f}")
    if args.run_summary:
        write_fields = ["predictor", "sequences", "valid", "wall_time_s"]
        args.run_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.run_summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, write_fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({
                "predictor": "Protein-Sol",
                "sequences": len(rows),
                "valid": valid,
                "wall_time_s": f"{elapsed:.6f}",
            })


if __name__ == "__main__":
    main()
