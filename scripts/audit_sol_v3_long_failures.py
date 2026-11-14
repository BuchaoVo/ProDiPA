#!/usr/bin/env python3
"""Decompose the strict Sol V3 350–512-aa capacity losses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from build_unconditional_teacher_datasets_balanced import (
    AA,
    complexity_metrics,
    read_release,
    safe_float,
    sha256_file,
)


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for index in order[start:stop]:
            output[index] = rank
        start = stop
    return output


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "q05": quantile(values, 0.05),
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "q75": quantile(values, 0.75),
        "q95": quantile(values, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-archive", type=Path, default=Path("releases/zenodo/ProDiPA_frozen_dataset_2026-11-02.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/rebuild/sol_designable_v3_strict70_15"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_release(args.release_archive, "train_25000_complete.tsv.gz")
    rows += read_release(args.release_archive, "validation_1000_complete.tsv.gz")
    long_rows = [row for row in rows if 350 <= len(row["sequence"].strip()) <= 512]
    prepared = []
    for row in long_rows:
        sequence = row["sequence"].strip().upper()
        canonical = set(sequence) <= AA
        metrics = complexity_metrics(sequence) if canonical else None
        prepared.append(row | {
            "sequence": sequence,
            "_canonical": canonical,
            "_complexity": metrics,
            "_complexity_pass": bool(metrics and metrics["passes_complexity_gate"]),
            "_sol_pass": safe_float(row, "sol_score", -math.inf) >= 0.70,
            "_plddt_pass": safe_float(row, "fold_score", -math.inf) >= 70.0,
            "_pae_pass": safe_float(row, "fold_mean_pae", math.inf) <= 15.0,
        })

    first_failure = Counter()
    sequential = Counter(raw=len(prepared))
    strict_pre_homology = []
    for row in prepared:
        if not row["_canonical"]:
            first_failure["noncanonical"] += 1
            continue
        sequential["canonical"] += 1
        if not row["_complexity_pass"]:
            first_failure["complexity"] += 1
            continue
        sequential["complexity"] += 1
        if not row["_sol_pass"]:
            first_failure["sol_below_0_70"] += 1
            continue
        sequential["sol"] += 1
        if not row["_plddt_pass"]:
            first_failure["plddt_below_70"] += 1
            continue
        sequential["plddt"] += 1
        if not row["_pae_pass"]:
            first_failure["pae_above_15"] += 1
            continue
        sequential["pae"] += 1
        strict_pre_homology.append(row)

    groups: dict[str, list[dict[str, object]]] = {}
    for row in strict_pre_homology:
        groups.setdefault(str(row["homology_group"]), []).append(row)
    strict_representatives = []
    for group, members in groups.items():
        strict_representatives.append(min(members, key=lambda row: (
            -safe_float(row, "sol_score"),
            -safe_float(row, "fold_score"),
            safe_float(row, "fold_mean_pae"),
            str(row["sequence_id"]),
        )))
        first_failure["homology_redundant_after_all_gates"] += len(members) - 1
    first_failure["strict_independent_eligible"] = len(strict_representatives)

    high_sol = [row for row in prepared if row["_canonical"] and row["_complexity_pass"] and row["_sol_pass"]]
    joint = Counter()
    for row in high_sol:
        key = (
            "pass_both" if row["_plddt_pass"] and row["_pae_pass"] else
            "fail_plddt_only" if not row["_plddt_pass"] and row["_pae_pass"] else
            "fail_pae_only" if row["_plddt_pass"] and not row["_pae_pass"] else
            "fail_both_structure_gates"
        )
        joint[key] += 1

    profile_rows = []
    for label, low, high in (("250-299", 250, 299), ("300-349", 300, 349), ("350-399", 350, 399), ("400-449", 400, 449), ("450-512", 450, 512)):
        subset = [
            row for row in rows
            if low <= len(row["sequence"].strip()) <= high
            and safe_float(row, "sol_score", -math.inf) >= 0.70
        ]
        pae = [safe_float(row, "fold_mean_pae") for row in subset]
        plddt = [safe_float(row, "fold_score") for row in subset]
        profile_rows.append({
            "length_band": label,
            "n_high_sol": len(subset),
            "pae_mean": statistics.fmean(pae) if pae else None,
            "pae_median": quantile(pae, 0.5),
            "pae_q75": quantile(pae, 0.75),
            "pae_le_15": sum(value <= 15 for value in pae),
            "pae_le_15_rate": sum(value <= 15 for value in pae) / len(pae) if pae else None,
            "plddt_mean": statistics.fmean(plddt) if plddt else None,
            "plddt_ge_70": sum(value >= 70 for value in plddt),
            "plddt_ge_70_rate": sum(value >= 70 for value in plddt) / len(plddt) if plddt else None,
            "strict_both": sum(p >= 70 and e <= 15 for p, e in zip(plddt, pae)),
            "strict_both_rate": sum(p >= 70 and e <= 15 for p, e in zip(plddt, pae)) / len(pae) if pae else None,
        })

    with (args.output / "P0_LONG_FAILURE_FUNNEL.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["stage", "remaining", "removed_from_previous", "remaining_rate_of_raw"])
        ordered = [
            ("raw_350_512", sequential["raw"]),
            ("canonical", sequential["canonical"]),
            ("complexity_pass", sequential["complexity"]),
            ("sol_ge_0_70", sequential["sol"]),
            ("plddt_ge_70", sequential["plddt"]),
            ("pae_le_15", sequential["pae"]),
            ("independent_homology", len(strict_representatives)),
        ]
        previous = ordered[0][1]
        for stage, count in ordered:
            writer.writerow([stage, count, previous - count, f"{count / ordered[0][1]:.9g}"])
            previous = count

    with (args.output / "P0_PAE_LENGTH_PROFILE.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, profile_rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(profile_rows)

    families = Counter(str(row.get("cluster_name", "")) or "unannotated" for row in strict_representatives)
    with (args.output / "P0_STRICT_FAMILY_COUNTS.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["cluster_name", "strict_independent_sequences"])
        writer.writerows(families.most_common())

    high_sol_lengths = [float(len(str(row["sequence"]))) for row in high_sol]
    high_sol_pae = [safe_float(row, "fold_mean_pae") for row in high_sol]
    report = {
        "status": "COMPLETE",
        "scope": "frozen train_25000 plus validation_1000, 350-512 aa",
        "source_archive": str(args.release_archive.resolve()),
        "source_archive_sha256": sha256_file(args.release_archive),
        "first_failure_exclusive": dict(first_failure),
        "high_sol_structure_gate_intersection": dict(joint),
        "high_sol_pae": summary(high_sol_pae),
        "high_sol_plddt": summary([safe_float(row, "fold_score") for row in high_sol]),
        "high_sol_length_vs_pae_spearman": pearson(ranks(high_sol_lengths), ranks(high_sol_pae)),
        "strict_independent": len(strict_representatives),
        "strict_unique_cluster_names": len(families),
        "strict_largest_cluster_name_count": max(families.values(), default=0),
        "interpretation_guard": "A length-pAE association does not identify domain count or justify relaxing pAE<=15.",
    }
    (args.output / "P0_FAILURE_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
