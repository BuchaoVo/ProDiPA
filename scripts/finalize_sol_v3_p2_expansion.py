#!/usr/bin/env python3
"""Finalize the reviewed long-only Sol V3 P2 capacity expansion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


BIN_NAMES = ("50-149", "150-249", "250-349", "350-512")
POOLED_TARGET = {"50-149": 60, "150-249": 176, "250-349": 87, "350-512": 77}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(rows: list[dict[str, str]]) -> dict[str, int]:
    values = Counter(row["length_bin"] for row in rows)
    return {name: values[name] for name in BIN_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--existing", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/strict_candidate_manifest_expanded.tsv"),
    )
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_structure_queue.tsv"),
    )
    parser.add_argument(
        "--structure", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_esmfold.tsv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15"),
    )
    args = parser.parse_args()

    existing = read_tsv(args.existing)
    candidates = read_tsv(args.candidates)
    structure_rows = read_tsv(args.structure)
    structure = {row["sequence_id"]: row for row in structure_rows}
    if len(structure_rows) != len(candidates) or len(structure) != len(candidates):
        raise RuntimeError(
            f"incomplete/nonunique ESMFold output: rows={len(structure_rows)} "
            f"unique={len(structure)} expected={len(candidates)}"
        )
    failed = [row for row in structure_rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"ESMFold has {len(failed)} failed rows")

    intersection = Counter()
    strict = []
    for row in candidates:
        score = structure[row["sequence_id"]]
        plddt = float(score["fold_plddt_100"])
        pae = float(score["fold_mean_pae"])
        pass_plddt = plddt >= 70.0
        pass_pae = pae <= 15.0
        if pass_plddt and pass_pae:
            intersection["pass_both"] += 1
            strict.append(row | {
                "fold_plddt_100": score["fold_plddt_100"],
                "fold_mean_pae": score["fold_mean_pae"],
                "fold_ptm_01": score["fold_ptm_01"],
            })
        elif pass_plddt:
            intersection["fail_pae_only"] += 1
        elif pass_pae:
            intersection["fail_plddt_only"] += 1
        else:
            intersection["fail_both"] += 1

    fields = [
        "source_stage", "source_partition", "sequence_id", "sequence_sha256",
        "representative_accession", "cluster_name", "homology_group", "length_aa",
        "length_bin", "sol_score", "fold_plddt_100", "fold_mean_pae", "fold_ptm_01",
        "global_entropy_bits", "max_single_residue_fraction", "max_homopolymer_run",
        "low_complexity_window_fraction", "unique_residues", "sequence",
    ]
    new_rows = []
    for row in strict:
        new_rows.append({
            "source_stage": "p2_reviewed_swissprot_long_expansion",
            "source_partition": "unassigned_external",
            "sequence_id": row["sequence_id"],
            "sequence_sha256": row["sequence_sha256"],
            "representative_accession": row["accession"],
            "cluster_name": row["cluster_representative"],
            "homology_group": row["homology_group"],
            "length_aa": row["length_aa"],
            "length_bin": row["length_bin"],
            "sol_score": row["solubility_scaled_01"],
            "fold_plddt_100": row["fold_plddt_100"],
            "fold_mean_pae": row["fold_mean_pae"],
            "fold_ptm_01": row["fold_ptm_01"],
            "global_entropy_bits": row["global_entropy_bits"],
            "max_single_residue_fraction": row["max_single_residue_fraction"],
            "max_homopolymer_run": row["max_homopolymer_run"],
            "low_complexity_window_fraction": row["low_complexity_window_fraction"],
            "unique_residues": row["unique_residues"],
            "sequence": row["sequence"],
        })
    old_rows = [{field: row.get(field, "") for field in fields} for row in existing]
    combined = old_rows + new_rows

    new_manifest = args.output_dir / "P2_strict_expansion_manifest.tsv"
    combined_manifest = args.output_dir / "strict_candidate_manifest_expanded_p2.tsv"
    for path, rows in ((new_manifest, new_rows), (combined_manifest, combined)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    capacity = counts(combined)
    deficits = {name: max(0, POOLED_TARGET[name] - capacity[name]) for name in BIN_NAMES}
    long_capacity = capacity["350-512"]
    if long_capacity >= 120:
        status = "RECOMMENDED_LONG_CAPACITY_MET"
    elif long_capacity >= 77:
        status = "MINIMUM_LONG_CAPACITY_MET"
    else:
        status = "EXPANSION_STILL_INSUFFICIENT"
    report = {
        "protocol": "Sol-Designable-V3-Strict70-15-P2-reviewed-long-only",
        "status": status,
        "thresholds": {"sol_min": 0.70, "plddt_min": 70.0, "pae_max": 15.0},
        "scored": len(candidates),
        "structure_gate_intersection": dict(sorted(intersection.items())),
        "new_strict_independent_long": len(new_rows),
        "combined_capacity_by_length": capacity,
        "pooled_target": POOLED_TARGET,
        "pooled_deficit": deficits,
        "minimum_long_pool_met": long_capacity >= 77,
        "recommended_long_pool_lower_bound_met": long_capacity >= 120,
        "recommended_long_pool_target": [120, 150],
        "source_quality": {
            "reviewed_swissprot_only": True,
            "fragment_annotation_excluded": True,
            "canonical_only": True,
            "complexity_gate_relaxed": False,
            "homology_grouping": "MMseqs2 30% identity, 80% bidirectional coverage, sensitivity 7.5",
            "one_scored_representative_per_new_group": True,
        },
        "note": (
            "Capacity evidence only. P3 must assign complete homology groups to train/validation "
            "and re-audit cross-split and cross-attribute leakage before any Teacher training."
        ),
        "inputs": {
            "existing_manifest_sha256": sha256_file(args.existing),
            "candidate_queue_sha256": sha256_file(args.candidates),
            "esmfold_sha256": sha256_file(args.structure),
        },
        "outputs": {
            "new_strict_manifest": str(new_manifest.resolve()),
            "new_strict_manifest_sha256": sha256_file(new_manifest),
            "combined_manifest": str(combined_manifest.resolve()),
            "combined_manifest_sha256": sha256_file(combined_manifest),
        },
        "training_started": False,
    }
    result = args.output_dir / "P2_EXPANSION_RESULT.json"
    result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
