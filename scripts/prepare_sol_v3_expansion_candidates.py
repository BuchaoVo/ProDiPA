#!/usr/bin/env python3
"""Prepare unscored 250–512-aa Sol V3 expansion candidates from local provenance."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

from build_unconditional_teacher_datasets_balanced import AA, complexity_metrics, read_release, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", type=Path, default=Path("data/provenance/split_membership.tsv.gz"))
    parser.add_argument("--release-archive", type=Path, default=Path("releases/zenodo/ProDiPA_frozen_dataset_2026-11-02.tar.gz"))
    parser.add_argument("--strict-manifest", type=Path, default=Path("data/rebuild/sol_designable_v3_strict70_15/strict_candidate_manifest.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/rebuild/sol_designable_v3_strict70_15/P1_prefiltered_candidates.tsv"))
    args = parser.parse_args()

    scored = read_release(args.release_archive, "train_25000_complete.tsv.gz")
    scored += read_release(args.release_archive, "validation_1000_complete.tsv.gz")
    scored_ids = {row["sequence_id"] for row in scored}
    with args.strict_manifest.open(encoding="utf-8", newline="") as handle:
        strict_groups = {row["homology_group"] for row in csv.DictReader(handle, delimiter="\t")}

    stages = {"membership": 0, "allowed_source_split": 0, "unscored": 0, "length_250_512": 0,
              "canonical": 0, "complexity_pass": 0, "new_homology_group": 0}
    candidates = []
    with gzip.open(args.membership, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            stages["membership"] += 1
            if row["split"] not in {"train", "validation"}:
                continue
            stages["allowed_source_split"] += 1
            if row["sequence_id"] in scored_ids:
                continue
            stages["unscored"] += 1
            sequence = row["sequence"].strip().upper()
            if not 250 <= len(sequence) <= 512:
                continue
            stages["length_250_512"] += 1
            if not set(sequence) <= AA:
                continue
            stages["canonical"] += 1
            metrics = complexity_metrics(sequence)
            if not metrics["passes_complexity_gate"]:
                continue
            stages["complexity_pass"] += 1
            if row["homology_group"] in strict_groups:
                continue
            stages["new_homology_group"] += 1
            candidates.append(row | {"sequence": sequence, "_complexity": metrics})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence_id", "sequence", "length_aa", "length_bin", "representative_accession",
              "uniref_cluster", "homology_group", "source_split", "source_release",
              "global_entropy_bits", "max_single_residue_fraction", "max_homopolymer_run",
              "low_complexity_window_fraction", "unique_residues", "valid_complete"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in sorted(candidates, key=lambda item: (len(item["sequence"]), item["homology_group"], item["sequence_id"])):
            metrics = row["_complexity"]
            writer.writerow({
                "sequence_id": row["sequence_id"], "sequence": row["sequence"],
                "length_aa": len(row["sequence"]),
                "length_bin": "250-349" if len(row["sequence"]) <= 349 else "350-512",
                "representative_accession": row["representative_accession"],
                "uniref_cluster": row["uniref_cluster"], "homology_group": row["homology_group"],
                "source_split": row["split"], "source_release": row["source_release"],
                "global_entropy_bits": f"{metrics['global_entropy_bits']:.9g}",
                "max_single_residue_fraction": f"{metrics['max_single_residue_fraction']:.9g}",
                "max_homopolymer_run": metrics["max_homopolymer_run"],
                "low_complexity_window_fraction": f"{metrics['low_complexity_window_fraction']:.9g}",
                "unique_residues": metrics["unique_residues"], "valid_complete": "true",
            })
    counts = {
        "250-349": sum(250 <= len(row["sequence"]) <= 349 for row in candidates),
        "350-512": sum(350 <= len(row["sequence"]) <= 512 for row in candidates),
    }
    status = {
        "status": "READY_FOR_PROTEIN_SOL",
        "source_membership": str(args.membership.resolve()),
        "source_membership_sha256": sha256_file(args.membership),
        "excluded_scored_ids": len(scored_ids),
        "excluded_existing_strict_homology_groups": len(strict_groups),
        "stages": stages,
        "candidates": len(candidates),
        "length_bins": counts,
        "output": str(args.output.resolve()),
        "training_started": False,
    }
    (args.output.parent / "P1_PREFILTER_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
