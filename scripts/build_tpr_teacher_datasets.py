#!/usr/bin/env python3
"""Build and atomically publish strict TPR ProLLaMA teacher datasets.

This builder is fail-closed: it consumes frozen scores, derives deterministic
length quotas from available independent homology groups, builds all six files
in staging, audits them together, and only then publishes them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import heapq
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable


AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
BINS = ((50, 149), (150, 249), (250, 349), (350, 512))
BIN_NAMES = ("50-149", "150-249", "250-349", "350-512")
OLD_TRAIN_QUOTAS = (15, 90, 135, 60)
OLD_TEST_QUOTAS = (5, 30, 45, 20)
TARGET_SIZES = {"train": 300, "test": 100}
PROMPT = "Superfamily=<Tetratricopeptide-like helical domain superfamily>"
FAMILY = "G3DSA:1.25.40.10"
SPECS = {
    "plddt": ("fold_score", 85.0, 100.0),
    "sol": ("sol_score", 0.70, 1.0),
    "thermo": ("thermo_score", 0.60, 1.0),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fieldnames: Iterable[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer TSV columns for empty output: {path}")
        fieldnames = rows[0].keys()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_map(paths: list[Path], field: str) -> dict[str, str]:
    scores: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_tsv(path):
            if row.get("status") != "ok":
                continue
            sequence_id, value = row["sequence_id"], row[field]
            if sequence_id in scores and scores[sequence_id] != value:
                raise ValueError(f"Conflicting score for {sequence_id}")
            scores[sequence_id] = value
    return scores


def bin_index(length: int) -> int:
    try:
        return next(i for i, (low, high) in enumerate(BINS) if low <= length <= high)
    except StopIteration as error:
        raise ValueError(f"Length outside supported range: {length}") from error


def stable_key(seed: int, attr: str, split: str, *parts: str) -> str:
    return hashlib.sha256("|".join((str(seed), attr, split, *parts)).encode()).hexdigest()


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def load_base_distribution(path: Path) -> dict:
    lengths = []
    for row in read_tsv(path):
        if "valid_complete" in row and not parse_bool(row["valid_complete"]):
            continue
        text = row.get("length_aa", "")
        length = int(float(text)) if text else len(row.get("sequence", ""))
        if 50 <= length <= 512:
            lengths.append(length)
    if not lengths:
        raise RuntimeError(f"No valid 50-512 aa sequences in Base reference: {path}")
    counts = [sum(low <= length <= high for length in lengths) for low, high in BINS]
    return {
        "path": str(path.resolve()), "sha256": sha256_file(path), "n": len(lengths),
        "counts": counts, "proportions": [count / len(lengths) for count in counts],
    }


def eligible_rows(rows: list[dict[str, str]], attr: str, split: str) -> list[dict[str, str]]:
    field, threshold, _ = SPECS[attr]
    return [row for row in rows if row["split"] == split and float(row[field]) >= threshold]


def capacity(rows: list[dict[str, str]], attr: str, split: str) -> tuple[list[int], list[int]]:
    eligible = eligible_rows(rows, attr, split)
    sequence_counts, group_counts = [], []
    for index in range(len(BINS)):
        in_bin = [row for row in eligible if bin_index(len(row["sequence"])) == index]
        sequence_counts.append(len(in_bin))
        group_counts.append(len({row["homology_group"] for row in in_bin}))
    return sequence_counts, group_counts


def quota_candidates(capacities: list[int], total: int, base_counts: list[int]) -> list[tuple[int, ...]]:
    """Optimize Base-L1, then one-dimensional EMD, using exact integers."""
    base_total = sum(base_counts)
    states: dict[int, tuple[tuple[int, int], list[tuple[int, ...]]]] = {0: ((0, 0), [()])}
    base_cumulative = 0
    for index, cap in enumerate(capacities):
        next_states: dict[int, tuple[tuple[int, int], list[tuple[int, ...]]]] = {}
        base_cumulative += base_counts[index]
        remaining_capacity = sum(capacities[index + 1:])
        for used, (score, prefixes) in states.items():
            lower = max(0, total - used - remaining_capacity)
            upper = min(cap, total - used)
            for quota in range(lower, upper + 1):
                new_used = used + quota
                l1 = score[0] + abs(quota * base_total - total * base_counts[index])
                emd = score[1]
                if index < len(capacities) - 1:
                    emd += abs(new_used * base_total - total * base_cumulative)
                candidate_score = (l1, emd)
                new_prefixes = [prefix + (quota,) for prefix in prefixes]
                existing = next_states.get(new_used)
                if existing is None or candidate_score < existing[0]:
                    next_states[new_used] = (candidate_score, new_prefixes)
                elif candidate_score == existing[0]:
                    next_states[new_used] = (
                        existing[0], sorted(set(existing[1] + new_prefixes))[:10000]
                    )
        states = next_states
    if total not in states:
        raise RuntimeError(
            f"INCOMPLETE target={total}: marginal independent-group capacity is "
            f"{sum(capacities)} across bins {capacities}"
        )
    return sorted(set(states[total][1]))


def optimize_quota_pair(
    train_capacities: list[int], test_capacities: list[int], base_counts: list[int]
) -> tuple[tuple[int, ...], tuple[int, ...], dict]:
    train_options = quota_candidates(train_capacities, 300, base_counts)
    test_options = quota_candidates(test_capacities, 100, base_counts)

    def key(pair):
        train, test = pair
        l1 = sum(abs(train[i] * 100 - test[i] * 300) for i in range(4))
        emd = sum(
            abs(sum(train[:i + 1]) * 100 - sum(test[:i + 1]) * 300)
            for i in range(3)
        )
        return l1, emd, train, test

    train, test = min(
        ((train, test) for train in train_options for test in test_options), key=key
    )
    metadata = {
        "objective_order": [
            "minimum L1 distance to frozen ProLLaMA TPR Base length distribution",
            "minimum one-dimensional earth-mover distance to the Base distribution",
            "minimum Train/Test distribution discrepancy among exact ties",
            "lexicographically smallest integer quota vector",
        ],
        "train_primary_tie_count": len(train_options),
        "test_primary_tie_count": len(test_options),
    }
    return train, test, metadata


def quota_score(quotas: tuple[int, ...], total: int, base_counts: list[int]) -> tuple[int, int]:
    base_total = sum(base_counts)
    l1 = sum(abs(quotas[i] * base_total - total * base_counts[i]) for i in range(4))
    emd = sum(
        abs(sum(quotas[:i + 1]) * base_total - total * sum(base_counts[:i + 1]))
        for i in range(3)
    )
    return l1, emd


def comparability_score(quotas: tuple[int, ...], total: int, reference: tuple[int, ...] | None, reference_total: int) -> tuple[int, int]:
    if reference is None:
        return 0, 0
    l1 = sum(abs(quotas[i] * reference_total - reference[i] * total) for i in range(4))
    emd = sum(
        abs(sum(quotas[:i + 1]) * reference_total - sum(reference[:i + 1]) * total)
        for i in range(3)
    )
    return l1, emd


def nearest_matchable_quota(
    rows: list[dict[str, str]], attr: str, split: str, initial: tuple[int, ...],
    capacities: list[int], base_counts: list[int], seed: int,
    comparison_reference: tuple[int, ...] | None = None, comparison_total: int = 1,
) -> tuple[tuple[int, ...], list[dict[str, str]], dict]:
    """Find the closest Base-matched quota that has a global homology matching.

    The integer quota lattice is searched deterministically from the marginal
    optimum.  Convex L1/EMD objectives ensure outward moves are non-decreasing;
    Train/Test comparability is only used after the two Base-distance terms.
    """
    total = TARGET_SIZES[split]

    def priority(quota: tuple[int, ...]):
        return (
            *quota_score(quota, total, base_counts),
            *comparability_score(quota, total, comparison_reference, comparison_total),
            quota,
        )

    heap = [(priority(initial), initial)]
    seen = {initial}
    attempts = 0
    last_error = ""
    while heap:
        _, quota = heapq.heappop(heap)
        attempts += 1
        try:
            selected = select_with_quotas(rows, attr, split, quota, seed)
            return quota, selected, {
                "marginal_optimum": list(initial),
                "selected_matchable_quota": list(quota),
                "cross_bin_homology_adjusted": quota != initial,
                "quota_vectors_tested": attempts,
            }
        except RuntimeError as error:
            last_error = str(error)
        for source in range(4):
            if quota[source] == 0:
                continue
            for target in range(4):
                if source == target or quota[target] >= capacities[target]:
                    continue
                neighbor = list(quota)
                neighbor[source] -= 1
                neighbor[target] += 1
                candidate = tuple(neighbor)
                if candidate not in seen:
                    seen.add(candidate)
                    heapq.heappush(heap, (priority(candidate), candidate))
    raise RuntimeError(
        f"INCOMPLETE {split}/{attr}: no exact {total}-group allocation exists; "
        f"searched {attempts} quota vectors. Last matching result: {last_error}"
    )


class Dinic:
    def __init__(self, n: int):
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, source: int, target: int, cap: int) -> list[int]:
        forward = [target, len(self.graph[target]), cap]
        reverse = [source, len(self.graph[source]), 0]
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for target, _, cap in self.graph[node]:
                    if cap and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return flow
            cursor = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, reverse, cap = edge
                    if cap and level[target] == level[node] + 1:
                        sent = send(target, min(amount, cap))
                        if sent:
                            edge[2] -= sent
                            self.graph[target][reverse][2] += sent
                            return sent
                    cursor[node] += 1
                return 0

            while True:
                sent = send(source, 10**9)
                if not sent:
                    break
                flow += sent


def select_with_quotas(
    rows: list[dict[str, str]], attr: str, split: str, quotas: tuple[int, ...], seed: int
) -> list[dict[str, str]]:
    representatives: dict[tuple[str, int], dict[str, str]] = {}
    for row in eligible_rows(rows, attr, split):
        index = bin_index(len(row["sequence"]))
        key = row["homology_group"], index
        old = representatives.get(key)
        if old is None or stable_key(seed, attr, split, row["sequence_id"], row["sequence"]) < stable_key(
            seed, attr, split, old["sequence_id"], old["sequence"]
        ):
            representatives[key] = row
    group_bins: dict[str, set[int]] = defaultdict(set)
    for group, index in representatives:
        group_bins[group].add(index)
    groups = sorted(group_bins, key=lambda group: stable_key(seed, attr, split, group))
    source, group_start = 0, 1
    bin_start = group_start + len(groups)
    sink = bin_start + 4
    network = Dinic(sink + 1)
    tracked = []
    for offset, group in enumerate(groups):
        node = group_start + offset
        network.add_edge(source, node, 1)
        for index in sorted(group_bins[group]):
            tracked.append((group, index, network.add_edge(node, bin_start + index, 1)))
    for index, quota in enumerate(quotas):
        network.add_edge(bin_start + index, sink, quota)
    achieved, target = network.max_flow(source, sink), sum(quotas)
    if achieved != target:
        raise RuntimeError(
            f"INCOMPLETE {split}/{attr}: quotas {list(quotas)} need {target} unique groups; "
            f"cross-bin homology matching supplies {achieved}; exact deficit {target - achieved}"
        )
    selected = [representatives[(group, index)] for group, index, edge in tracked if edge[2] == 0]
    selected.sort(key=lambda row: (
        bin_index(len(row["sequence"])),
        stable_key(seed, attr, split, row["homology_group"], row["sequence_id"]),
    ))
    actual = [sum(bin_index(len(row["sequence"])) == i for row in selected) for i in range(4)]
    if actual != list(quotas):
        raise AssertionError(f"Flow extraction mismatch: expected {quotas}, got {actual}")
    return selected


def parse_record_sequence(record: dict) -> str:
    output = str(record.get("output", ""))
    return output[5:-1] if output.startswith("Seq=<") and output.endswith(">") else output


def dataset_metrics(records: list[dict], attr: str, sequence_groups: dict[str, str], status: str) -> dict:
    sequences = [parse_record_sequence(record) for record in records]
    divisor = SPECS[attr][2]
    values = [float(record[attr]) * divisor for record in records if attr in record]
    groups = {sequence_groups[seq] for seq in sequences if seq in sequence_groups}
    counts = [sum(low <= len(seq) <= high for seq in sequences) for low, high in BINS]
    return {
        "status": status, "n": len(records), "length_counts": counts,
        "length_proportions": [count / len(records) if records else None for count in counts],
        "target_mean": sum(values) / len(values) if values else None,
        "unique_homology_groups": len(groups),
        "unmapped_homology_sequences": sum(seq not in sequence_groups for seq in sequences),
    }


def previous_metrics(destination: Path, rows: list[dict[str, str]]) -> dict:
    sequence_groups = {row["sequence"]: row["homology_group"] for row in rows}
    paths = {
        ("train", "plddt"): destination / "fold/train_plddt_300.json",
        ("test", "plddt"): destination / "fold/test_plddt_100.json",
        ("train", "sol"): destination / "sol/train_sol_300.json",
        ("test", "sol"): destination / "sol/test_sol_100.json",
        ("train", "thermo"): destination / "thermo/train_thermo_300.json",
        ("test", "thermo"): destination / "thermo/test_thermo_100.json",
    }
    result = {}
    for key, path in paths.items():
        if path.is_file():
            result["_".join(key)] = dataset_metrics(
                json.loads(path.read_text()), key[1], sequence_groups, "previously_published"
            )
    return result


def audit_all(final: dict[tuple[str, str], list[dict[str, str]]]) -> dict:
    datasets, failures = {}, []
    split_groups, split_sequences = {"train": set(), "test": set()}, {"train": set(), "test": set()}
    for (split, attr), chosen in sorted(final.items()):
        field, threshold, _ = SPECS[attr]
        sequences = [row["sequence"] for row in chosen]
        groups = [row["homology_group"] for row in chosen]
        expected = TARGET_SIZES[split]
        checks = {
            "size_exact": len(chosen) == expected,
            "unique_sequences": len(set(sequences)) == expected,
            "unique_homology_groups": len(set(groups)) == expected,
            "thresholds": all(float(row[field]) >= threshold for row in chosen),
            "family": all(row["cath_superfamily"] == FAMILY and row["family_evidence"] for row in chosen),
            "alphabet_and_length": all(50 <= len(row["sequence"]) <= 512 and set(row["sequence"]) <= AA for row in chosen),
            "frozen_split": all(row["split"] == split for row in chosen),
        }
        if not all(checks.values()):
            failures.append({"dataset": f"{split}_{attr}", "checks": checks})
        datasets[f"{split}_{attr}"] = checks
        split_groups[split].update(groups)
        split_sequences[split].update(sequences)
    group_overlap = split_groups["train"] & split_groups["test"]
    sequence_overlap = split_sequences["train"] & split_sequences["test"]
    if group_overlap or sequence_overlap:
        failures.append({
            "dataset": "cross_split", "homology_overlap": len(group_overlap),
            "sequence_overlap": len(sequence_overlap),
        })
    return {
        "status": "PASS" if not failures else "FAIL", "family": FAMILY, "prompt": PROMPT,
        "datasets": datasets,
        "cross_attribute_train_test_homology_overlap": len(group_overlap),
        "cross_attribute_train_test_sequence_overlap": len(sequence_overlap),
        "failures": failures,
    }


def atomic_publish(staging: Path, destination: Path, provenance: Path) -> None:
    mapping = {
        "train_plddt_300.json": destination / "fold/train_plddt_300.json",
        "test_plddt_100.json": destination / "fold/test_plddt_100.json",
        "train_sol_300.json": destination / "sol/train_sol_300.json",
        "test_sol_100.json": destination / "sol/test_sol_100.json",
        "train_thermo_300.json": destination / "thermo/train_thermo_300.json",
        "test_thermo_100.json": destination / "thermo/test_thermo_100.json",
        "conditional_teacher_manifest.tsv": provenance / "conditional_teacher_manifest.tsv",
        "conditional_teacher_audit.json": provenance / "conditional_teacher_audit.json",
        "capacity_matrix.tsv": provenance / "conditional_teacher_capacity_matrix.tsv",
        "quota_optimization.tsv": provenance / "conditional_teacher_quota_optimization.tsv",
        "selection_comparison.tsv": provenance / "conditional_teacher_selection_comparison.tsv",
        "build_report.json": provenance / "conditional_teacher_build_report.json",
    }
    missing = [str(staging / source) for source in mapping if not (staging / source).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot publish; staging files missing: {missing}")
    token = str(os.getpid())
    prepared, backups = {}, {}
    try:
        for source_name, target in mapping.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".new.{token}")
            shutil.copy2(staging / source_name, temporary)
            prepared[target] = temporary
        for target in mapping.values():
            if target.exists():
                backup = target.with_name(target.name + f".rollback.{token}")
                shutil.copy2(target, backup)
                backups[target] = backup
        for target, temporary in prepared.items():
            os.replace(temporary, target)
    except Exception:
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        for path in list(prepared.values()) + list(backups.values()):
            path.unlink(missing_ok=True)


def write_failure(staging: Path, report: dict, message: str) -> None:
    report.update({"status": "INCOMPLETE", "failure": message, "published": False})
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (staging / "conditional_teacher_audit.json").write_text(json.dumps(
        {"status": "FAIL", "reason": message, "published": False}, indent=2
    ) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--base-length-reference", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("data/conditional"))
    parser.add_argument("--provenance", type=Path, default=Path("data/provenance"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    args.staging.mkdir(parents=True, exist_ok=True)
    base = load_base_distribution(args.base_length_reference)
    rows = read_tsv(args.candidates)
    fold_paths = sorted(args.score_root.glob("tpr_esmfold_shard_*.tsv"))
    sol_paths = sorted(args.score_root.glob("tpr_*sol.tsv"))
    thermo_paths = sorted(args.score_root.glob("tpr_*thermo.tsv"))
    fold = score_map(fold_paths, "fold_plddt_100")
    sol = score_map(sol_paths, "solubility_scaled_01")
    thermo = score_map(thermo_paths, "thermophilicity_score_01")
    for row in rows:
        sequence = row["sequence"]
        if not (50 <= len(sequence) <= 512 and set(sequence) <= AA):
            raise ValueError(f"Invalid sequence: {row['sequence_id']}")
        if row.get("cath_superfamily") != FAMILY or not row.get("family_evidence"):
            raise ValueError(f"Missing direct family evidence: {row['sequence_id']}")
        if not row.get("homology_group") or not row.get("split"):
            raise ValueError(f"Missing homology/frozen split assignment: {row['sequence_id']}")
        try:
            row["fold_score"] = fold[row["sequence_id"]]
            row["sol_score"] = sol[row["sequence_id"]]
            row["thermo_score"] = thermo[row["sequence_id"]]
        except KeyError as error:
            raise RuntimeError(f"INCOMPLETE scoring for {row['sequence_id']}: {error}") from error

    score_files = fold_paths + sol_paths + thermo_paths
    report = {
        "status": "BUILDING", "thresholds": {key: spec[1] for key, spec in SPECS.items()},
        "selection_seed": args.seed, "family": FAMILY, "prompt": PROMPT,
        "target_sizes": TARGET_SIZES,
        "old_fixed_quotas": {
            "train": dict(zip(BIN_NAMES, OLD_TRAIN_QUOTAS)),
            "test": dict(zip(BIN_NAMES, OLD_TEST_QUOTAS)),
        },
        "base_length_reference": base,
        "candidate_pool": {
            "path": str(args.candidates.resolve()), "sha256": sha256_file(args.candidates),
            "n": len(rows),
            "source_database_counts": dict(sorted(Counter(
                row.get("source_database", "unknown") or "unknown" for row in rows
            ).items())),
            "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
            "score_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in score_files
            ],
        },
    }

    capacity_rows, capacities = [], {}
    for split in ("train", "test"):
        for attr in SPECS:
            sequences, groups = capacity(rows, attr, split)
            capacities[(split, attr)] = groups
            for index, name in enumerate(BIN_NAMES):
                capacity_rows.append({
                    "split": split, "attribute": attr, "length_bin": name,
                    "qualified_sequences": sequences[index],
                    "independent_homology_groups": groups[index],
                    "threshold": SPECS[attr][1],
                })
    write_tsv(args.staging / "capacity_matrix.tsv", capacity_rows)
    report["capacity_matrix"] = capacity_rows
    previous = previous_metrics(args.destination, rows)
    report["previously_published_metrics"] = previous

    quotas, quota_metadata, selected_cache = {}, {}, {}
    try:
        for attr in SPECS:
            train, test, metadata = optimize_quota_pair(
                capacities[("train", attr)], capacities[("test", attr)], base["counts"]
            )
            train, train_selected, train_matching = nearest_matchable_quota(
                rows, attr, "train", train, capacities[("train", attr)],
                base["counts"], args.seed,
            )
            test, test_selected, test_matching = nearest_matchable_quota(
                rows, attr, "test", test, capacities[("test", attr)],
                base["counts"], args.seed,
                comparison_reference=train, comparison_total=300,
            )
            quotas[("train", attr)], quotas[("test", attr)] = train, test
            selected_cache[("train", attr)] = train_selected
            selected_cache[("test", attr)] = test_selected
            metadata["homology_matching"] = {
                "train": train_matching, "test": test_matching,
            }
            quota_metadata[attr] = metadata
    except RuntimeError as error:
        write_failure(args.staging, report, str(error))
        raise

    quota_rows, support_notes = [], []
    for attr in SPECS:
        train, test = quotas[("train", attr)], quotas[("test", attr)]
        for split, optimized, old in (
            ("train", train, OLD_TRAIN_QUOTAS), ("test", test, OLD_TEST_QUOTAS)
        ):
            total, cap = TARGET_SIZES[split], capacities[(split, attr)]
            for index, name in enumerate(BIN_NAMES):
                expected = base["proportions"][index] * total
                reasons = []
                if cap[index] == 0:
                    reasons.append("no_eligible_homology_groups")
                if cap[index] < expected:
                    reasons.append("capacity_below_base_target")
                if optimized[index] > expected and not reasons:
                    reasons.append("redistributed_to_reach_exact_dataset_size")
                matching = quota_metadata[attr]["homology_matching"][split]
                if matching["cross_bin_homology_adjusted"] and optimized[index] != matching["marginal_optimum"][index]:
                    reasons.append("adjusted_for_cross_bin_homology_matching")
                quota_rows.append({
                    "split": split, "attribute": attr, "length_bin": name,
                    "old_fixed_quota": old[index], "base_expected_count": f"{expected:.6f}",
                    "available_independent_groups": cap[index],
                    "optimized_quota": optimized[index], "reason": ";".join(reasons),
                })
        support_notes.append({
            "attribute": attr, "train_quotas": list(train), "test_quotas": list(test),
            "absolute_proportion_differences": [abs(a / 300 - b / 100) for a, b in zip(train, test)],
            "reason": "Differences remain where independent-group support constrains Base-matched quotas",
        })
    write_tsv(args.staging / "quota_optimization.tsv", quota_rows)
    report["quota_optimizer"] = quota_metadata
    report["optimized_quotas"] = {
        f"{split}_{attr}": dict(zip(BIN_NAMES, value)) for (split, attr), value in quotas.items()
    }
    report["train_test_support_notes"] = support_notes

    final, manifest = {}, []
    try:
        for split in ("train", "test"):
            for attr, (field, _, divisor) in SPECS.items():
                chosen = selected_cache[(split, attr)]
                final[(split, attr)] = chosen
                records = [{
                    "instruction": "[Generate by superfamily]", "input": PROMPT,
                    "output": f"Seq=<{row['sequence']}>", attr: float(row[field]) / divisor,
                } for row in chosen]
                name = f"{split}_{attr}_{TARGET_SIZES[split]}.json"
                (args.staging / name).write_text(json.dumps(records, indent=2) + "\n")
                for row in chosen:
                    manifest.append({
                        "sequence_id": row["sequence_id"], "sequence": row["sequence"],
                        "split": split, "attribute": attr, "homology_group": row["homology_group"],
                        "length_aa": len(row["sequence"]),
                        "length_bin": BIN_NAMES[bin_index(len(row["sequence"]))],
                        "cath_superfamily": FAMILY, "tpr_coverage_union": row.get("tpr_coverage_union", ""),
                        "fold_score": row["fold_score"], "sol_score": row["sol_score"],
                        "thermo_score": row["thermo_score"], "family_evidence": row["family_evidence"],
                        "source_database": row.get("source_database", ""),
                    })
                report.setdefault("optimized_dataset_metrics", {})[f"{split}_{attr}"] = dataset_metrics(
                    records, attr, {row["sequence"]: row["homology_group"] for row in chosen},
                    "optimized_staged",
                )
    except RuntimeError as error:
        write_failure(args.staging, report, str(error))
        raise

    comparison_rows = []
    for phase, metrics_map in (
        ("before", previous), ("after", report["optimized_dataset_metrics"])
    ):
        for dataset, metrics in sorted(metrics_map.items()):
            for index, name in enumerate(BIN_NAMES):
                comparison_rows.append({
                    "phase": phase, "dataset": dataset, "status": metrics["status"],
                    "n": metrics["n"], "length_bin": name,
                    "length_count": metrics["length_counts"][index],
                    "length_proportion": metrics["length_proportions"][index],
                    "target_attribute_mean": metrics["target_mean"],
                    "unique_homology_groups": metrics["unique_homology_groups"],
                    "unmapped_homology_sequences": metrics["unmapped_homology_sequences"],
                })
    write_tsv(args.staging / "selection_comparison.tsv", comparison_rows)
    write_tsv(args.staging / "conditional_teacher_manifest.tsv", manifest)

    audit = audit_all(final)
    audit.update({
        "selection_seed": args.seed, "thresholds": {key: spec[1] for key, spec in SPECS.items()},
        "candidate_pool_sha256": report["candidate_pool"]["sha256"],
        "base_length_reference": base, "published": False,
    })
    (args.staging / "conditional_teacher_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        report.update({"status": "INCOMPLETE", "failure": audit["failures"], "published": False})
        (args.staging / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(f"Full six-dataset audit failed: {audit['failures']}")

    report.update({"status": "PASS", "published": bool(args.publish)})
    (args.staging / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.publish:
        audit["published"] = True
        (args.staging / "conditional_teacher_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        atomic_publish(args.staging, args.destination, args.provenance)
    print(json.dumps({
        "status": report["status"], "published": bool(args.publish),
        "optimized_quotas": report["optimized_quotas"], "audit": audit["status"],
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
