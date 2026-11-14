#!/usr/bin/env python3
"""Audit and stage strict designable Sol V3 candidates without publishing.

The builder never relaxes the Sol/structure/complexity gates and never moves a
record across the frozen source train/validation partitions.  If a requested
length quota cannot be filled, it writes the real capacity and a partial
selection, marks the package not training-ready, and exits successfully so the
scientific capacity result remains inspectable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from build_unconditional_teacher_datasets_balanced import (
    AA,
    BINS,
    BIN_NAMES,
    COMPLEXITY_GATE,
    complexity_metrics,
    empirical_quantile,
    length_bin,
    read_release,
    safe_float,
    select_bin,
    sha256_file,
    stable_key,
)


PROTOCOL = "Sol-Designable-V3-Strict70-15"
TRAIN_QUOTA = (45, 132, 65, 58)
VALIDATION_QUOTA = (15, 44, 22, 19)
SOL_MIN = 0.70
PLDDT_MIN = 70.0
PAE_MAX = 15.0


def counts_by_bin(rows: Iterable[dict[str, object]]) -> list[int]:
    rows = list(rows)
    return [
        sum(low <= len(str(row["sequence"])) <= high for row in rows)
        for low, high in BINS
    ]


def strict_candidates(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    stages: dict[str, list[dict[str, str]]] = {"raw": rows}
    stages["sol"] = [row for row in rows if safe_float(row, "sol_score", -math.inf) >= SOL_MIN]
    stages["structure"] = [
        row for row in stages["sol"]
        if safe_float(row, "fold_score", -math.inf) >= PLDDT_MIN
        and safe_float(row, "fold_mean_pae", math.inf) <= PAE_MAX
    ]
    stages["canonical_length"] = [
        row for row in stages["structure"]
        if 50 <= len(row["sequence"].strip()) <= 512
        and set(row["sequence"].strip().upper()) <= AA
    ]
    complexity_rows: list[dict[str, object]] = []
    for row in stages["canonical_length"]:
        sequence = row["sequence"].strip().upper()
        metrics = complexity_metrics(sequence)
        if metrics["passes_complexity_gate"]:
            complexity_rows.append(row | {"sequence": sequence, "_complexity": metrics})

    groups: dict[str, list[dict[str, object]]] = {}
    for row in complexity_rows:
        groups.setdefault(str(row["homology_group"]), []).append(row)
    representatives = []
    for group, members in groups.items():
        representatives.append(min(members, key=lambda row: (
            -safe_float(row, "sol_score"),
            -safe_float(row, "fold_score"),
            safe_float(row, "fold_mean_pae"),
            stable_key(42, PROTOCOL, group, str(row["sequence_id"])),
        )))

    def stage_summary(stage_rows: list[dict[str, object]]) -> dict[str, object]:
        return {"n": len(stage_rows), "length_bins": counts_by_bin(stage_rows)}

    report = {
        "raw": stage_summary(list(stages["raw"])),
        "sol_at_least_0_70": stage_summary(list(stages["sol"])),
        "strict_structure_70_15": stage_summary(list(stages["structure"])),
        "canonical_50_512": stage_summary(list(stages["canonical_length"])),
        "complexity_pass": stage_summary(complexity_rows),
        "independent_homology_representatives": stage_summary(representatives),
    }
    return representatives, report


def select_partial(
    candidates: list[dict[str, object]], quotas: tuple[int, int, int, int], seed: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for index, requested in enumerate(quotas):
        in_bin = [row for row in candidates if length_bin(len(str(row["sequence"]))) == index]
        quota = min(requested, len(in_bin))
        if not quota:
            continue
        lengths = [len(str(row["sequence"])) for row in in_bin]
        targets = [empirical_quantile(lengths, (slot + 0.5) / quota) for slot in range(quota)]
        selected.extend(select_bin(in_bin, "sol", quota, targets, seed + index))
    return selected


def write_candidates(path: Path, rows: list[dict[str, object]], partition: str, selected: set[str]) -> None:
    fields = [
        "source_partition", "selected_partial", "sequence_id", "sequence_sha256",
        "representative_accession", "cluster_name", "homology_group", "length_aa",
        "length_bin", "starts_with_m", "sol_score", "fold_plddt_100",
        "fold_mean_pae", "fold_ptm_01", "global_entropy_bits",
        "max_single_residue_fraction", "max_homopolymer_run",
        "low_complexity_window_fraction", "unique_residues", "sequence",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        if path.stat().st_size == 0:
            writer.writeheader()
        for row in sorted(rows, key=lambda item: (length_bin(len(str(item["sequence"]))), len(str(item["sequence"])), str(item["sequence_id"]))):
            metrics = row["_complexity"]
            sequence = str(row["sequence"])
            writer.writerow({
                "source_partition": partition,
                "selected_partial": str(str(row["sequence_id"]) in selected).lower(),
                "sequence_id": row["sequence_id"],
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "representative_accession": row.get("representative_accession", ""),
                "cluster_name": row.get("cluster_name", ""),
                "homology_group": row["homology_group"],
                "length_aa": len(sequence),
                "length_bin": BIN_NAMES[length_bin(len(sequence))],
                "starts_with_m": str(sequence.startswith("M")).lower(),
                "sol_score": row["sol_score"],
                "fold_plddt_100": row["fold_score"],
                "fold_mean_pae": row["fold_mean_pae"],
                "fold_ptm_01": row.get("fold_ptm_01", ""),
                "global_entropy_bits": f"{metrics['global_entropy_bits']:.9g}",
                "max_single_residue_fraction": f"{metrics['max_single_residue_fraction']:.9g}",
                "max_homopolymer_run": metrics["max_homopolymer_run"],
                "low_complexity_window_fraction": f"{metrics['low_complexity_window_fraction']:.9g}",
                "unique_residues": metrics["unique_residues"],
                "sequence": sequence,
            })


def write_partial_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ["sequence", "sol"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"sequence": row["sequence"], "sol": f"{safe_float(row, 'sol_score'):.9g}"})


def summarize_selected(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"n": 0, "length_bins": [0, 0, 0, 0]}
    lengths = [len(str(row["sequence"])) for row in rows]
    return {
        "n": len(rows),
        "length_bins": counts_by_bin(rows),
        "length_mean": sum(lengths) / len(lengths),
        "length_median": sorted(lengths)[len(lengths) // 2],
        "sol_mean": sum(safe_float(row, "sol_score") for row in rows) / len(rows),
        "plddt_mean": sum(safe_float(row, "fold_score") for row in rows) / len(rows),
        "pae_mean": sum(safe_float(row, "fold_mean_pae") for row in rows) / len(rows),
        "unique_homology_groups": len({str(row["homology_group"]) for row in rows}),
        "complexity_failures": sum(not bool(row["_complexity"]["passes_complexity_gate"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-archive", type=Path, default=Path("releases/zenodo/ProDiPA_frozen_dataset_2026-11-02.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/rebuild/sol_designable_v3_strict70_15"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train_rows = read_release(args.release_archive, "train_25000_complete.tsv.gz")
    validation_rows = read_release(args.release_archive, "validation_1000_complete.tsv.gz")
    train, train_stages = strict_candidates(train_rows)
    validation, validation_stages = strict_candidates(validation_rows)
    selected_train = select_partial(train, TRAIN_QUOTA, args.seed)
    selected_validation = select_partial(validation, VALIDATION_QUOTA, args.seed + 100)

    train_capacity = counts_by_bin(train)
    validation_capacity = counts_by_bin(validation)
    pooled_capacity = [left + right for left, right in zip(train_capacity, validation_capacity)]
    pooled_target = [left + right for left, right in zip(TRAIN_QUOTA, VALIDATION_QUOTA)]
    train_deficit = [max(0, target - actual) for target, actual in zip(TRAIN_QUOTA, train_capacity)]
    validation_deficit = [max(0, target - actual) for target, actual in zip(VALIDATION_QUOTA, validation_capacity)]
    train_ready = not any(train_deficit)
    validation_ready = not any(validation_deficit)
    cross_split = {str(row["homology_group"]) for row in train} & {str(row["homology_group"]) for row in validation}

    candidate_path = args.output / "strict_candidate_manifest.tsv"
    candidate_path.write_text("", encoding="utf-8")
    write_candidates(candidate_path, train, "train", {str(row["sequence_id"]) for row in selected_train})
    write_candidates(candidate_path, validation, "validation", {str(row["sequence_id"]) for row in selected_validation})
    write_partial_csv(args.output / "train_sol_partial.csv", selected_train)
    write_partial_csv(args.output / "test_sol_partial.csv", selected_validation)

    report = {
        "protocol": PROTOCOL,
        "status": "TRAINING_READY" if train_ready and validation_ready and not cross_split else "INSUFFICIENT_CAPACITY_NOT_TRAINING_READY",
        "source_archive": str(args.release_archive.resolve()),
        "source_archive_sha256": sha256_file(args.release_archive),
        "thresholds": {"sol_min": SOL_MIN, "plddt_min": PLDDT_MIN, "pae_max": PAE_MAX},
        "complexity_gate": COMPLEXITY_GATE,
        "length_bins": [list(item) for item in BINS],
        "target_quotas": {"train": list(TRAIN_QUOTA), "validation": list(VALIDATION_QUOTA)},
        "stages": {"train": train_stages, "validation": validation_stages},
        "strict_independent_capacity": {"train": train_capacity, "validation": validation_capacity},
        "strict_pooled_capacity": pooled_capacity,
        "pooled_target": pooled_target,
        "pooled_quota_deficit_even_if_repartitioned": [
            max(0, target - actual) for target, actual in zip(pooled_target, pooled_capacity)
        ],
        "quota_deficit": {"train": train_deficit, "validation": validation_deficit},
        "partial_selection": {
            "train": summarize_selected(selected_train),
            "validation": summarize_selected(selected_validation),
        },
        "cross_source_partition_homology_overlap": len(cross_split),
        "rules": {
            "threshold_relaxation": False,
            "cropping": False,
            "duplication": False,
            "cross_partition_borrowing": False,
            "canonical_v2_modified": False,
        },
        "generated_teacher_acceptance_targets": {
            "sol_mean_min": 0.70,
            "structure_pass_plddt70_pae15_rate_min": 0.48,
            "structure_pass_reference_range": [0.48, 0.63],
            "noncanonical_requests": 0,
        },
    }
    (args.output / "CAPACITY_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "BUILD_STATUS.json").write_text(json.dumps({
        "status": report["status"],
        "protocol": PROTOCOL,
        "train_selected": len(selected_train),
        "validation_selected": len(selected_validation),
        "training_started": False,
        "canonical_v2_modified": False,
    }, indent=2) + "\n", encoding="utf-8")
    audited_files = [
        args.output / "BUILD_STATUS.json",
        args.output / "CAPACITY_AUDIT.json",
        args.output / "strict_candidate_manifest.tsv",
        args.output / "train_sol_partial.csv",
        args.output / "test_sol_partial.csv",
    ]
    (args.output / "FINAL_AUDIT.json").write_text(json.dumps({
        "status": "PASS_CAPACITY_AUDIT_NOT_TRAINING_READY",
        "protocol": PROTOCOL,
        "files": {path.name: sha256_file(path) for path in audited_files},
        "strict_candidate_rows": len(train) + len(validation),
        "partial_selection_rows": len(selected_train) + len(selected_validation),
        "cross_source_partition_homology_overlap": len(cross_split),
        "canonical_v2_modified": False,
        "training_started": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
