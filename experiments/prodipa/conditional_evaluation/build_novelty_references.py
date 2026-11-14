#!/usr/bin/env python3
"""Build frozen common Novelty-T references for fair within-paradigm comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def csv_sequences(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row["sequence"].strip().upper() for row in rows]


def json_sequences(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    sequences = []
    for row in rows:
        output = row["output"]
        if "Seq=<" not in output:
            raise ValueError(f"Unrecognized output in {path}: {output[:80]}")
        sequences.append(output.split("Seq=<", 1)[1].split(">", 1)[0].strip().upper())
    return sequences


def write_fasta(path: Path, sequences: list[str]) -> dict:
    unique = sorted(set(sequences))
    with path.open("w", encoding="ascii") as handle:
        for sequence in unique:
            identifier = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            handle.write(f">sha256:{identifier}\n{sequence}\n")
    return {
        "path": str(path.resolve()),
        "input_records": len(sequences),
        "unique_sequences": len(unique),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    uncond = []
    for relative in (
        "data/unconditional/fold/train_plddt_300.csv",
        "data/unconditional/sol/train_sol_300.csv",
        "data/unconditional/thermo/train_thermo_300.csv",
    ):
        uncond.extend(csv_sequences(args.project_root / relative))
    cond = []
    for relative in (
        "data/conditional/fold/train_plddt_300.json",
        "data/conditional/sol/train_sol_300.json",
        "data/conditional/thermo/train_thermo_300.json",
    ):
        cond.extend(json_sequences(args.project_root / relative))

    manifest = {
        "definition": (
            "Novelty-T uses the union of all three 300-sequence teacher training sets "
            "within each paradigm, identically for Base and all teachers."
        ),
        "unconditional": write_fasta(args.out_root / "unconditional_train_union.fasta", uncond),
        "conditional": write_fasta(args.out_root / "conditional_train_union.fasta", cond),
    }
    (args.out_root / "novelty_t_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
