#!/usr/bin/env python3
"""Finalize scored Sol V3 expansion capacity without creating training data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


BINS = ((50, 149), (150, 249), (250, 349), (350, 512))
BIN_NAMES = ("50-149", "150-249", "250-349", "350-512")
POOLED_TARGET = (60, 176, 87, 77)
SOL_MIN = 0.70
PLDDT_MIN = 70.0
PAE_MAX = 15.0


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bin_name(length: int) -> str:
    for name, (low, high) in zip(BIN_NAMES, BINS):
        if low <= length <= high:
            return name
    raise ValueError(f"length outside protocol: {length}")


def counts(rows: list[dict[str, str]]) -> dict[str, int]:
    result = Counter(row["length_bin"] for row in rows)
    return {name: result[name] for name in BIN_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-manifest", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/strict_candidate_manifest.tsv"),
    )
    parser.add_argument(
        "--new-candidates", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P1_high_sol_candidates.tsv"),
    )
    parser.add_argument(
        "--structure", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P1_esmfold.tsv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15"),
    )
    args = parser.parse_args()

    old = read_tsv(args.old_manifest)
    candidates = read_tsv(args.new_candidates)
    structure_rows = read_tsv(args.structure)
    structure = {row["sequence_id"]: row for row in structure_rows}
    if len(structure_rows) != len(candidates) or len(structure) != len(candidates):
        raise RuntimeError(
            f"incomplete/nonunique structure scores: rows={len(structure_rows)} "
            f"unique={len(structure)} expected={len(candidates)}"
        )
    failed = [row for row in candidates if structure[row["sequence_id"]]["status"] != "ok"]
    if failed:
        raise RuntimeError(f"structure scoring has {len(failed)} failed rows")

    intersections = Counter()
    strict_raw: list[dict[str, str]] = []
    for row in candidates:
        score = structure[row["sequence_id"]]
        plddt = float(score["fold_plddt_100"])
        pae = float(score["fold_mean_pae"])
        pass_plddt = plddt >= PLDDT_MIN
        pass_pae = pae <= PAE_MAX
        if pass_plddt and pass_pae:
            intersections["pass_both"] += 1
        elif pass_plddt:
            intersections["fail_pae_only"] += 1
        elif pass_pae:
            intersections["fail_plddt_only"] += 1
        else:
            intersections["fail_both"] += 1
        if pass_plddt and pass_pae:
            strict_raw.append(row | {
                "fold_plddt_100": score["fold_plddt_100"],
                "fold_mean_pae": score["fold_mean_pae"],
                "fold_ptm_01": score["fold_ptm_01"],
            })

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in strict_raw:
        by_group[row["homology_group"]].append(row)
    strict_new = []
    for group, members in by_group.items():
        strict_new.append(min(members, key=lambda row: (
            -float(row["solubility_scaled_01"]),
            -float(row["fold_plddt_100"]),
            float(row["fold_mean_pae"]),
            row["sequence_id"],
        )))

    old_groups = {row["homology_group"] for row in old}
    overlap = old_groups & set(by_group)
    if overlap:
        raise RuntimeError(f"new expansion overlaps {len(overlap)} old strict homology groups")

    fields = [
        "source_stage", "source_partition", "sequence_id", "sequence_sha256",
        "representative_accession", "cluster_name", "homology_group", "length_aa",
        "length_bin", "sol_score", "fold_plddt_100", "fold_mean_pae", "fold_ptm_01",
        "global_entropy_bits", "max_single_residue_fraction", "max_homopolymer_run",
        "low_complexity_window_fraction", "unique_residues", "sequence",
    ]
    combined = []
    for row in old:
        combined.append({field: row.get(field, "") for field in fields} | {"source_stage": "frozen_26k"})
    for row in strict_new:
        sequence = row["sequence"]
        combined.append({
            "source_stage": "p1_local_unscored_expansion",
            "source_partition": row["source_split"],
            "sequence_id": row["sequence_id"],
            "sequence_sha256": sha256_text(sequence),
            "representative_accession": row["representative_accession"],
            "cluster_name": row["uniref_cluster"],
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
            "sequence": sequence,
        })

    manifest = args.output_dir / "strict_candidate_manifest_expanded.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(combined, key=lambda row: (
            BIN_NAMES.index(row["length_bin"]), int(row["length_aa"]), row["homology_group"]
        )))

    new_manifest = args.output_dir / "P1_strict_expansion_manifest.tsv"
    with new_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in combined if row["source_stage"] == "p1_local_unscored_expansion")

    combined_counts = counts(combined)
    target = dict(zip(BIN_NAMES, POOLED_TARGET))
    deficits = {name: max(0, target[name] - combined_counts[name]) for name in BIN_NAMES}
    by_partition = {
        partition: counts([row for row in combined if row["source_partition"] == partition])
        for partition in ("train", "validation")
    }
    report = {
        "protocol": "Sol-Designable-V3-Strict70-15",
        "status": "POOLED_QUOTA_CAPACITY_MET" if not any(deficits.values()) else "EXPANSION_STILL_INSUFFICIENT",
        "thresholds": {"sol_min": SOL_MIN, "plddt_min": PLDDT_MIN, "pae_max": PAE_MAX},
        "scored": len(candidates),
        "structure_gate_intersection": dict(sorted(intersections.items())),
        "new_strict_before_homology_dedup": len(strict_raw),
        "new_strict_independent": len(strict_new),
        "new_strict_by_length": counts(strict_new),
        "combined_strict_independent": len(combined),
        "combined_capacity_by_length": combined_counts,
        "combined_capacity_by_source_partition": by_partition,
        "pooled_target": target,
        "pooled_deficit": deficits,
        "minimum_long_pool_met": combined_counts["350-512"] >= 77,
        "recommended_long_pool_lower_bound_met": combined_counts["350-512"] >= 120,
        "recommended_long_pool_target": [120, 150],
        "cross_old_new_homology_overlap": len(overlap),
        "note": (
            "Capacity only. Train/validation assignment must be rebuilt at homology-group level; "
            "this script does not authorize training or move records across source partitions."
        ),
        "inputs": {
            "old_manifest_sha256": sha256_file(args.old_manifest),
            "new_candidates_sha256": sha256_file(args.new_candidates),
            "structure_sha256": sha256_file(args.structure),
        },
        "outputs": {
            "expanded_manifest": str(manifest.resolve()),
            "expanded_manifest_sha256": sha256_file(manifest),
            "new_strict_manifest": str(new_manifest.resolve()),
            "new_strict_manifest_sha256": sha256_file(new_manifest),
        },
        "training_started": False,
    }
    result = args.output_dir / "P1_EXPANSION_RESULT.json"
    result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
