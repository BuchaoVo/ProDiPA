#!/usr/bin/env python3
"""Build Sol V3 300/100 splits by complete homology group and audit leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


BIN_NAMES = ("50-149", "150-249", "250-349", "350-512")
QUOTAS = {
    "train": dict(zip(BIN_NAMES, (45, 132, 65, 58))),
    "validation": dict(zip(BIN_NAMES, (15, 44, 22, 19))),
}
REFERENCE_FILES = {
    "fold_train": Path("data/unconditional/fold/train_plddt_300.csv"),
    "fold_validation": Path("data/unconditional/fold/test_plddt_100.csv"),
    "thermo_train": Path("data/unconditional/thermo/train_thermo_300.csv"),
    "thermo_validation": Path("data/unconditional/thermo/test_thermo_100.csv"),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_value(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def ks_statistic(left: list[int], right: list[int]) -> float:
    values = sorted(set(left + right))
    return max(
        abs(sum(item <= value for item in left) / len(left) - sum(item <= value for item in right) / len(right))
        for value in values
    )


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    lengths = [int(row["length_aa"]) for row in rows]
    return {
        "n": len(rows),
        "length_bins": {name: sum(row["length_bin"] == name for row in rows) for name in BIN_NAMES},
        "length_mean": sum(lengths) / len(lengths),
        "length_median": sorted(lengths)[len(lengths) // 2],
        "length_min": min(lengths),
        "length_max": max(lengths),
        "sol_mean": sum(float(row["sol_score"]) for row in rows) / len(rows),
        "plddt_mean": sum(float(row["fold_plddt_100"]) for row in rows) / len(rows),
        "pae_mean": sum(float(row["fold_mean_pae"]) for row in rows) / len(rows),
        "starts_with_m": sum(row["sequence"].startswith("M") for row in rows),
        "unique_sequences": len({row["sequence_sha256"] for row in rows}),
        "unique_homology_groups": len({row["p3_homology_group"] for row in rows}),
        "source_stages": dict(sorted(Counter(row["source_stage"] for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/strict_candidate_manifest_expanded_p2.tsv"),
    )
    parser.add_argument("--mmseqs", type=Path, default=Path("tools/mmseqs/bin/mmseqs"))
    parser.add_argument(
        "--work", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P3_homology_work"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15"),
    )
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    raw_candidates = read_tsv(args.candidates)
    fragment_rows = [row for row in raw_candidates if re.search(r"\bfragments?\b", row["cluster_name"], re.I)]
    candidates = [row for row in raw_candidates if row not in fragment_rows]
    sequence_meta: dict[str, dict[str, object]] = {}
    for row in candidates:
        sequence_meta.setdefault(row["sequence"], {"candidates": [], "reference_roles": set()})["candidates"].append(row)
    reference_counts = {}
    reference_rows_by_role = {}
    for role, path in REFERENCE_FILES.items():
        rows = read_csv(path)
        reference_rows_by_role[role] = rows
        reference_counts[role] = len(rows)
        for row in rows:
            sequence_meta.setdefault(row["sequence"], {"candidates": [], "reference_roles": set()})["reference_roles"].add(role)

    args.work.mkdir(parents=True, exist_ok=True)
    fasta = args.work / "combined_candidates_fold_thermo.fasta"
    internal_meta = {}
    with fasta.open("w", encoding="ascii") as handle:
        for index, (sequence, meta) in enumerate(sorted(sequence_meta.items())):
            internal = f"SEQ{index:07d}"
            internal_meta[internal] = {"sequence": sequence, **meta}
            handle.write(f">{internal}\n{sequence}\n")
    prefix = args.work / "clusters"
    temporary = args.work / "tmp"
    cluster_file = args.work / "clusters_cluster.tsv"
    if not cluster_file.exists():
        subprocess.run([
            str(args.mmseqs), "easy-cluster", str(fasta), str(prefix), str(temporary),
            "--min-seq-id", "0.30", "-c", "0.80", "--cov-mode", "0",
            "--cluster-mode", "0", "-s", "7.5", "--threads", str(args.threads),
        ], check=True)
    cluster_by_member = {}
    with cluster_file.open(encoding="utf-8") as handle:
        for representative, member in csv.reader(handle, delimiter="\t"):
            cluster_by_member[member] = representative
    if set(cluster_by_member) != set(internal_meta):
        raise RuntimeError("MMseqs cluster membership does not cover every unique sequence")

    cluster_members: dict[str, list[str]] = defaultdict(list)
    for member, representative in cluster_by_member.items():
        cluster_members[representative].append(member)
    cluster_info = {}
    candidate_cluster = {}
    reference_cross_split_clusters = 0
    for representative, members in cluster_members.items():
        roles = set().union(*(internal_meta[member]["reference_roles"] for member in members))
        has_train = any(role.endswith("_train") for role in roles)
        has_validation = any(role.endswith("_validation") for role in roles)
        if has_train and has_validation:
            constraint = "conflict"
            reference_cross_split_clusters += 1
        elif has_train:
            constraint = "train"
        elif has_validation:
            constraint = "validation"
        else:
            constraint = "flexible"
        group = "SOLV3P3HG_" + stable_value(internal_meta[representative]["sequence"])[:16]
        rows = []
        for member in members:
            for row in internal_meta[member]["candidates"]:
                item = row | {
                    "p3_homology_group": group,
                    "p3_cluster_representative": representative,
                    "p3_split_constraint": constraint,
                    "p3_reference_roles": ";".join(sorted(roles)),
                    "p3_cluster_unique_sequences": str(len(members)),
                }
                rows.append(item)
                candidate_cluster[row["sequence_id"]] = representative
        cluster_info[representative] = {
            "group": group, "members": members, "roles": roles,
            "constraint": constraint, "candidate_rows": rows,
        }

    # Each complete P3 cluster may be used in only one split and contributes at
    # most one representative.  Solve the scarcest split/bin slot first and
    # prefer split-constrained clusters over flexible clusters.
    selected: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    used_clusters: set[str] = set()
    tasks = {(split, name): quota for split, bins in QUOTAS.items() for name, quota in bins.items()}
    capacity_before_selection = {}
    for split, name in tasks:
        eligible = {
            representative for representative, info in cluster_info.items()
            if info["constraint"] in {split, "flexible"}
            and any(row["length_bin"] == name for row in info["candidate_rows"])
        }
        capacity_before_selection[f"{split}:{name}"] = len(eligible)

    while tasks:
        availability = {}
        for task, needed in tasks.items():
            split, name = task
            choices = []
            for representative, info in cluster_info.items():
                if representative in used_clusters or info["constraint"] not in {split, "flexible"}:
                    continue
                options = [row for row in info["candidate_rows"] if row["length_bin"] == name]
                if options:
                    best = min(options, key=lambda row: (
                        -float(row["sol_score"]), -float(row["fold_plddt_100"]),
                        float(row["fold_mean_pae"]), stable_value(row["sequence_id"]),
                    ))
                    choices.append((representative, info, best))
            availability[task] = choices
            if len(choices) < needed:
                raise RuntimeError(f"insufficient P3 capacity for {split}/{name}: {len(choices)} < {needed}")
        task = min(tasks, key=lambda item: (len(availability[item]) / tasks[item], tasks[item], item))
        needed = tasks.pop(task)
        split, _name = task
        choices = sorted(availability[task], key=lambda item: (
            item[1]["constraint"] == "flexible",
            -float(item[2]["sol_score"]), -float(item[2]["fold_plddt_100"]),
            float(item[2]["fold_mean_pae"]), stable_value("P3", item[2]["sequence_id"]),
        ))
        for representative, _info, row in choices[:needed]:
            selected[split].append(row | {"p3_split": split})
            used_clusters.add(representative)

    train, validation = selected["train"], selected["validation"]
    selected_groups_train = {row["p3_homology_group"] for row in train}
    selected_groups_validation = {row["p3_homology_group"] for row in validation}
    internal_overlap = selected_groups_train & selected_groups_validation
    train_hashes = {row["sequence_sha256"] for row in train}
    validation_hashes = {row["sequence_sha256"] for row in validation}
    exact_overlap = train_hashes & validation_hashes
    cross_attribute_leaks = []
    same_split_overlaps = []
    for split, rows in selected.items():
        opposite = "validation" if split == "train" else "train"
        for row in rows:
            roles = set(filter(None, row["p3_reference_roles"].split(";")))
            bad = sorted(role for role in roles if role.endswith(f"_{opposite}"))
            same = sorted(role for role in roles if role.endswith(f"_{split}"))
            if bad:
                cross_attribute_leaks.append((row, bad))
            if same:
                same_split_overlaps.append((row, same))

    if internal_overlap or exact_overlap or cross_attribute_leaks:
        raise RuntimeError(
            f"P3 leakage audit failed: group={len(internal_overlap)} exact={len(exact_overlap)} "
            f"cross_attribute={len(cross_attribute_leaks)}"
        )
    for split, rows in selected.items():
        observed = Counter(row["length_bin"] for row in rows)
        if any(observed[name] != QUOTAS[split][name] for name in BIN_NAMES):
            raise RuntimeError(f"quota mismatch for {split}: {observed}")

    # Clustering prevents a group from being split, but set-cover membership is
    # not a substitute for a direct pairwise leakage audit.  Search the entire
    # three-attribute Train cohort against the entire three-attribute Validation
    # cohort under the same identity/coverage threshold and reject every hit.
    audit_work = args.work / "pairwise_cross_split_audit"
    audit_work.mkdir(parents=True, exist_ok=True)
    split_sequences: dict[str, dict[str, set[str]]] = {
        "train": defaultdict(set), "validation": defaultdict(set),
    }
    for split, rows in selected.items():
        for row in rows:
            split_sequences[split][row["sequence"]].add(f"sol_{split}")
    for role, rows in reference_rows_by_role.items():
        split = "validation" if role.endswith("_validation") else "train"
        for row in rows:
            split_sequences[split][row["sequence"]].add(role)
    pairwise_exact_overlap = set(split_sequences["train"]) & set(split_sequences["validation"])
    if pairwise_exact_overlap:
        raise RuntimeError(
            f"direct cross-split audit found {len(pairwise_exact_overlap)} exact sequence overlaps"
        )
    audit_ids: dict[str, tuple[str, set[str]]] = {}
    fasta_paths = {}
    for split in ("train", "validation"):
        path = audit_work / f"{split}.fasta"
        fasta_paths[split] = path
        with path.open("w", encoding="ascii") as handle:
            for index, (sequence, roles) in enumerate(sorted(split_sequences[split].items())):
                internal = f"{split.upper()}{index:05d}"
                audit_ids[internal] = (sequence, roles)
                handle.write(f">{internal}\n{sequence}\n")
    raw_hits = audit_work / "cross_split_hits.raw.tsv"
    if raw_hits.exists():
        raw_hits.unlink()
    subprocess.run([
        str(args.mmseqs), "easy-search", str(fasta_paths["train"]),
        str(fasta_paths["validation"]), str(raw_hits), str(audit_work / "tmp"),
        "--min-seq-id", "0.30", "-c", "0.80", "--cov-mode", "0",
        "-s", "7.5", "--max-seqs", "100000", "--threads", str(args.threads),
        "--format-output", "query,target,pident,qcov,tcov,evalue,alnlen",
    ], check=True)
    pairwise_hits = []
    if raw_hits.exists():
        with raw_hits.open(encoding="utf-8") as handle:
            for query, target, identity, qcov, tcov, evalue, alnlen in csv.reader(handle, delimiter="\t"):
                pairwise_hits.append({
                    "query_id": query, "target_id": target,
                    "query_roles": ";".join(sorted(audit_ids[query][1])),
                    "target_roles": ";".join(sorted(audit_ids[target][1])),
                    "identity": identity, "query_coverage": qcov,
                    "target_coverage": tcov, "evalue": evalue,
                    "alignment_length": alnlen,
                })
    pairwise_path = args.output_dir / "P3_PAIRWISE_CROSS_SPLIT_HITS.tsv"
    pairwise_fields = [
        "query_id", "target_id", "query_roles", "target_roles", "identity",
        "query_coverage", "target_coverage", "evalue", "alignment_length",
    ]
    with pairwise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, pairwise_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(pairwise_hits)
    if pairwise_hits:
        raise RuntimeError(
            f"direct pairwise cross-split homology audit found {len(pairwise_hits)} hits"
        )

    output_fields = [
        "dataset", "attribute", "sequence_id", "representative_accession", "cluster_name",
        "sequence_sha256", "original_homology_group", "p3_homology_group",
        "source_stage", "source_partition", "p3_split_constraint", "p3_reference_roles",
        "length_aa", "length_bin", "starts_with_m", "sol_score", "fold_plddt_100",
        "fold_mean_pae", "fold_ptm_01", "global_entropy_bits", "max_single_residue_fraction",
        "max_homopolymer_run", "low_complexity_window_fraction", "unique_residues", "sequence",
    ]
    manifest_rows = []
    for split, rows in selected.items():
        for row in sorted(rows, key=lambda item: (BIN_NAMES.index(item["length_bin"]), int(item["length_aa"]), item["sequence_id"])):
            manifest_rows.append({
                "dataset": "train" if split == "train" else "test",
                "attribute": "sol", "sequence_id": row["sequence_id"],
                "representative_accession": row["representative_accession"],
                "cluster_name": row["cluster_name"], "sequence_sha256": row["sequence_sha256"],
                "original_homology_group": row["homology_group"],
                "p3_homology_group": row["p3_homology_group"],
                "source_stage": row["source_stage"], "source_partition": row["source_partition"],
                "p3_split_constraint": row["p3_split_constraint"],
                "p3_reference_roles": row["p3_reference_roles"],
                "length_aa": row["length_aa"], "length_bin": row["length_bin"],
                "starts_with_m": str(row["sequence"].startswith("M")).lower(),
                "sol_score": row["sol_score"], "fold_plddt_100": row["fold_plddt_100"],
                "fold_mean_pae": row["fold_mean_pae"], "fold_ptm_01": row["fold_ptm_01"],
                "global_entropy_bits": row["global_entropy_bits"],
                "max_single_residue_fraction": row["max_single_residue_fraction"],
                "max_homopolymer_run": row["max_homopolymer_run"],
                "low_complexity_window_fraction": row["low_complexity_window_fraction"],
                "unique_residues": row["unique_residues"], "sequence": row["sequence"],
            })
    manifest_path = args.output_dir / "selected_sequence_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, output_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(manifest_rows)

    for split, rows, name in (
        ("train", train, "train_sol_300.csv"),
        ("validation", validation, "test_sol_100.csv"),
    ):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, ["sequence", "sol"], lineterminator="\n")
            writer.writeheader()
            for row in sorted(rows, key=lambda item: (BIN_NAMES.index(item["length_bin"]), int(item["length_aa"]), item["sequence_id"])):
                writer.writerow({"sequence": row["sequence"], "sol": row["sol_score"]})

    audit_path = args.output_dir / "P3_CROSS_ATTRIBUTE_HOMOLOGY_AUDIT.tsv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "p3_split", "sequence_id", "p3_homology_group", "p3_split_constraint",
            "reference_roles", "same_split_roles", "opposite_split_roles", "classification",
        ]
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for split, rows in selected.items():
            opposite = "validation" if split == "train" else "train"
            for row in sorted(rows, key=lambda item: item["sequence_id"]):
                roles = set(filter(None, row["p3_reference_roles"].split(";")))
                same = sorted(role for role in roles if role.endswith(f"_{split}"))
                bad = sorted(role for role in roles if role.endswith(f"_{opposite}"))
                if bad:
                    classification = "opposite_split_leak"
                elif same:
                    classification = "same_split_allowed"
                else:
                    classification = "no_fold_thermo_homology_at_threshold"
                writer.writerow({
                    "p3_split": row["p3_split"], "sequence_id": row["sequence_id"],
                    "p3_homology_group": row["p3_homology_group"],
                    "p3_split_constraint": row["p3_split_constraint"],
                    "reference_roles": ";".join(sorted(roles)),
                    "same_split_roles": ";".join(same),
                    "opposite_split_roles": ";".join(bad),
                    "classification": classification,
                })

    train_lengths = [int(row["length_aa"]) for row in train]
    validation_lengths = [int(row["length_aa"]) for row in validation]
    moved_from_source_train_to_validation = sum(
        row["source_partition"] == "train" for row in validation
    )
    report = {
        "status": "TRAINING_READY",
        "protocol": "Sol-Designable-V3-Strict70-15-P3-homology-split",
        "thresholds": {"sol_min": 0.70, "plddt_min": 70.0, "pae_max": 15.0},
        "quotas": QUOTAS,
        "raw_strict_candidates": len(raw_candidates),
        "explicit_fragment_candidates_excluded": len(fragment_rows),
        "eligible_candidates": len(candidates),
        "capacity_before_selection": capacity_before_selection,
        "reference_datasets": reference_counts,
        "reference_cross_train_validation_clusters": reference_cross_split_clusters,
        "selection": {"train": summarize(train), "validation": summarize(validation)},
        "train_validation_length_ks": ks_statistic(train_lengths, validation_lengths),
        "leakage_audit": {
            "sol_train_validation_exact_sequence_overlap": len(exact_overlap),
            "sol_train_validation_homology_group_overlap": len(internal_overlap),
            "cross_attribute_train_validation_homology_leaks": len(cross_attribute_leaks),
            "same_split_cross_attribute_homology_overlaps": len(same_split_overlaps),
            "all_attribute_pairwise_cross_split_homology_hits": len(pairwise_hits),
            "all_attribute_pairwise_cross_split_exact_overlaps": len(pairwise_exact_overlap),
            "pairwise_train_unique_sequences": len(split_sequences["train"]),
            "pairwise_validation_unique_sequences": len(split_sequences["validation"]),
        },
        "source_partition_reassignment": {
            "enabled_after_unified_homology_grouping": True,
            "source_train_rows_selected_for_v3_validation": moved_from_source_train_to_validation,
            "reason": "original frozen validation capacity cannot satisfy the fixed V3 length quotas",
        },
        "rules": {
            "complete_homology_group_assigned_once": True,
            "fragment_annotation_excluded": True,
            "threshold_relaxation": False,
            "cropping": False,
            "duplication": False,
            "canonical_sol_v2_modified": False,
            "teacher_training_started": False,
        },
    }
    report_path = args.output_dir / "P3_BUILD_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    status = {
        "status": "TRAINING_READY", "protocol": report["protocol"],
        "train_selected": len(train), "validation_selected": len(validation),
        "cross_split_leaks": 0, "teacher_training_started": False,
        "canonical_sol_v2_modified": False,
    }
    (args.output_dir / "BUILD_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    files = [
        report_path, manifest_path, audit_path,
        pairwise_path,
        args.output_dir / "train_sol_300.csv", args.output_dir / "test_sol_100.csv",
        args.output_dir / "BUILD_STATUS.json",
    ]
    final_audit = {
        "status": "PASS_TRAINING_READY_NOT_STARTED",
        "protocol": report["protocol"],
        "files": {path.name: sha256_file(path) for path in files},
        "train_rows": len(train), "validation_rows": len(validation),
        "all_strict_70_15": all(
            float(row["sol_score"]) >= 0.70 and float(row["fold_plddt_100"]) >= 70
            and float(row["fold_mean_pae"]) <= 15 for row in train + validation
        ),
        "cross_split_leaks": 0,
        "pairwise_cross_split_homology_hits": len(pairwise_hits),
        "canonical_sol_v2_modified": False,
        "teacher_training_started": False,
    }
    (args.output_dir / "FINAL_AUDIT.json").write_text(json.dumps(final_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
