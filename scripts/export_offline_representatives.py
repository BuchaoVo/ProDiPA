#!/usr/bin/env python3
"""Freeze and export the two strongest ProDiPA Offline showcase proteins.

This is a post-hoc case-study export. It does not alter evaluation tables,
training outputs, or the frozen scores used to select the representatives.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

from collect_opd_showcase import dump, export, write_tsv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/showcase/prodipa_offline_representatives_20261113"

SPECS = (
    {
        "candidate_id": "conditional_offline_best_0395",
        "paradigm": "conditional",
        "sample_id": "395",
        "sequence_id": "sha256:5b38a7d23b0cb0e1bfe6b3b2ea5f69e4d9292f4192c6ce9da47dd6ad489140e3",
        "source": ROOT / "outputs/experiments/adaptive_cache/evaluation/conditional/fixed/node_4/scores/sequence_scores.tsv",
        "standalone_hv_box": 0.2006761755,
        "archive_marginal_hv": 0.00164401418,
    },
    {
        "candidate_id": "unconditional_offline_best_0359",
        "paradigm": "unconditional",
        "sample_id": "359",
        "sequence_id": "sha256:f1176c68ce8887112685a71cba07522178cec5763a0b111c1246799f42032554",
        "source": ROOT / "outputs/experiments/adaptive_cache_unconditional_sol_v3/evaluation/unconditional/fixed/node_4/scores/sequence_scores.tsv",
        "standalone_hv_box": 0.4740201704,
        "archive_marginal_hv": 0.00625254359,
    },
)


def read_exact_row(path: Path, sample_id: str, sequence_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["sample_id"] == sample_id and row["sequence_id"] == sequence_id
        ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source row, found {len(matches)}: {path}")
    row = matches[0]
    digest = "sha256:" + hashlib.sha256(row["sequence"].encode()).hexdigest()
    if digest != sequence_id:
        raise RuntimeError(f"sequence hash mismatch: {path} sample={sample_id}")
    if row.get("valid_complete", "").lower() != "true":
        raise RuntimeError(f"representative is not a valid complete generation: {path}")
    return row


def prepare(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"use a new empty output directory: {out}")
    for folder in ("fasta", "pdb", "confidence", "figures", "pymol"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    candidates = []
    for spec in SPECS:
        source = spec["source"]
        row = read_exact_row(source, spec["sample_id"], spec["sequence_id"])
        candidate = {
            "candidate_id": spec["candidate_id"],
            "paradigm": spec["paradigm"],
            "method": "Fixed Offline OPD",
            "sample_id": row["sample_id"],
            "seed": row["seed"],
            "sequence_id": row["sequence_id"],
            "length_aa": int(row["length_aa"]),
            "ppl": float(row["ppl"]),
            "protrek_percent": row.get("protrek_percent", ""),
            "novelty_u_percent": row.get("novelty_u_percent", ""),
            "novelty_t_percent": row.get("novelty_t_percent", ""),
            "plddt": float(row["plddt"]),
            "pae": float(row["pae"]),
            "sol": float(row["sol"]),
            "thermo": float(row["thermo"]),
            "standalone_hv_box": spec["standalone_hv_box"],
            "archive_marginal_hv": spec["archive_marginal_hv"],
            "source_scores": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "sequence": row["sequence"],
        }
        candidates.append(candidate)
        fasta = f">{candidate['candidate_id']} {candidate['sequence_id']}\n{candidate['sequence']}\n"
        (out / "fasta" / f"{candidate['candidate_id']}.fasta").write_text(fasta)

    (out / "selected.fasta").write_text("".join(
        f">{row['candidate_id']} {row['sequence_id']}\n{row['sequence']}\n" for row in candidates
    ))
    write_tsv(out / "selected_candidates.tsv", candidates)
    dump(out / "selection.json", {
        "created_at": time.strftime("%F %T %z"),
        "selection": (
            "Post-hoc Fixed Offline OPD case-study representatives: strongest balanced "
            "multi-objective sequence under each paradigm's frozen score/HV protocol."
        ),
        "caveat": (
            "The standalone box and archive-marginal HV values are ranking aids under the "
            "existing frozen normalization, not per-protein biological measurements."
        ),
        "historical_scores_immutable": True,
        "candidates": candidates,
    })
    dump(out / "STATUS.json", {"stage": "SEQUENCES_READY_PDB_PENDING", "selected": len(candidates)})


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.out)
    export(args.out)


if __name__ == "__main__":
    main()
