#!/usr/bin/env python3
"""Stage length-matched, complexity-audited ProtGPT2 teacher datasets.

This builder is deliberately non-publishing.  It keeps the frozen train and
validation partitions, exact property thresholds, and one sequence per
homology group.  Validation length quotas are optimized against the observed
valid-complete ProtGPT2 Base distribution; training uses exactly three times
each validation-bin quota.  This makes the 300:100 split comparable by design.

The CSVs remain raw amino-acid strings.  Pretraining-compatible FASTA wrapping
is a model-input protocol and is applied only by ``train_teacher.py`` so the
same source sequence is never duplicated or rewritten in provenance records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable


ARCHIVE_PREFIX = "ProDiPA_frozen_dataset_2026-11-02"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
BINS = ((50, 149), (150, 249), (250, 349), (350, 512))
BIN_NAMES = ("50-149", "150-249", "250-349", "350-512")
SPECS = {
    "plddt": ("fold_score", 85.0, "pLDDT", 100.0),
    "sol": ("sol_score", 0.70, "sol", 1.0),
    "thermo": ("thermo_score", 0.60, "thermo", 1.0),
}
COMPLEXITY_GATE = {
    "min_global_entropy_bits": 2.5,
    "max_single_residue_fraction": 0.35,
    "max_homopolymer_run": 12,
    "max_low_complexity_window_fraction": 0.25,
    "min_unique_residues": 8,
    "window_aa": 32,
    "window_entropy_bits": 2.5,
    "window_max_single_residue_fraction": 0.5,
}
SOL_STRUCTURE_QUALIFIED = {"min_fold_score": 50.0, "max_fold_mean_pae": 25.0}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_key(seed: int, *parts: str) -> str:
    return sha256_bytes((str(seed) + "|" + "|".join(parts)).encode())


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_release(archive: Path, name: str) -> list[dict[str, str]]:
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile(f"{ARCHIVE_PREFIX}/{name}")
        if member is None:
            raise FileNotFoundError(name)
        with gzip.GzipFile(fileobj=member) as compressed:
            return list(csv.DictReader(io.TextIOWrapper(compressed), delimiter="\t"))


def attach_fold_scores(candidate_path: Path, score_paths: list[Path], split: str) -> list[dict[str, str]]:
    candidates = {row["sequence_id"]: row for row in read_tsv(candidate_path)}
    scores: dict[str, dict[str, str]] = {}
    for path in score_paths:
        if not path.is_file():
            continue
        for row in read_tsv(path):
            if row.get("status") != "ok":
                continue
            previous = scores.setdefault(row["sequence_id"], row)
            if previous["fold_plddt_100"] != row["fold_plddt_100"]:
                raise ValueError(f"Conflicting Fold score for {row['sequence_id']}")
    attached = []
    for sequence_id, score in scores.items():
        row = candidates.get(sequence_id)
        if row is None or row["split"] != split:
            raise ValueError(f"Fold score is outside frozen {split} candidates: {sequence_id}")
        attached.append(row | {
            "fold_score": score["fold_plddt_100"],
            "fold_ptm_01": score.get("fold_ptm_01", ""),
            "fold_mean_pae": score.get("fold_mean_pae", ""),
            "fold_status": "ok",
        })
    return attached


def length_bin(length: int) -> int:
    for index, (low, high) in enumerate(BINS):
        if low <= length <= high:
            return index
    raise ValueError(f"Length outside 50-512 aa: {length}")


def entropy_bits(sequence: str) -> float:
    counts = Counter(sequence)
    total = len(sequence)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def complexity_metrics(sequence: str) -> dict[str, float | int | bool]:
    counts = Counter(sequence)
    total = len(sequence)
    run = max_run = 1
    for left, right in zip(sequence, sequence[1:]):
        run = run + 1 if left == right else 1
        max_run = max(max_run, run)
    window_flags = []
    width = int(COMPLEXITY_GATE["window_aa"])
    for start in range(max(1, total - width + 1)):
        window = sequence[start:start + width]
        window_counts = Counter(window)
        window_flags.append(
            entropy_bits(window) < COMPLEXITY_GATE["window_entropy_bits"]
            or max(window_counts.values()) / len(window)
            > COMPLEXITY_GATE["window_max_single_residue_fraction"]
        )
    metrics: dict[str, float | int | bool] = {
        "global_entropy_bits": entropy_bits(sequence),
        "max_single_residue_fraction": max(counts.values()) / total,
        "max_homopolymer_run": max_run,
        "low_complexity_window_fraction": sum(window_flags) / len(window_flags),
        "unique_residues": len(counts),
    }
    metrics["passes_complexity_gate"] = bool(
        metrics["global_entropy_bits"] >= COMPLEXITY_GATE["min_global_entropy_bits"]
        and metrics["max_single_residue_fraction"] <= COMPLEXITY_GATE["max_single_residue_fraction"]
        and metrics["max_homopolymer_run"] <= COMPLEXITY_GATE["max_homopolymer_run"]
        and metrics["low_complexity_window_fraction"] <= COMPLEXITY_GATE["max_low_complexity_window_fraction"]
        and metrics["unique_residues"] >= COMPLEXITY_GATE["min_unique_residues"]
    )
    return metrics


def safe_float(row: dict[str, str], field: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return default


def structure_qualified(row: dict[str, str]) -> bool:
    return (
        safe_float(row, "fold_score", -math.inf) >= SOL_STRUCTURE_QUALIFIED["min_fold_score"]
        and safe_float(row, "fold_mean_pae", math.inf) <= SOL_STRUCTURE_QUALIFIED["max_fold_mean_pae"]
    )


def eligible(rows: Iterable[dict[str, str]], attribute: str) -> list[dict[str, str]]:
    field, threshold, _, _ = SPECS[attribute]
    output = []
    for row in rows:
        sequence = row["sequence"].strip().upper()
        if not (50 <= len(sequence) <= 512) or not set(sequence) <= AA:
            continue
        if safe_float(row, field, -math.inf) < threshold:
            continue
        metrics = complexity_metrics(sequence)
        if not metrics["passes_complexity_gate"]:
            continue
        output.append(row | {"sequence": sequence, "_complexity": metrics})
    return output


def group_representatives(rows: list[dict[str, str]], attribute: str, seed: int) -> list[dict[str, str]]:
    field = SPECS[attribute][0]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["homology_group"], []).append(row)
    result = []
    for group, members in grouped.items():
        if attribute == "sol":
            key: Callable[[dict[str, str]], tuple] = lambda row: (
                not structure_qualified(row),
                -safe_float(row, field),
                -safe_float(row, "fold_score", -math.inf),
                safe_float(row, "fold_mean_pae", math.inf),
                stable_key(seed, attribute, group, row["sequence_id"]),
            )
        else:
            key = lambda row: (
                -safe_float(row, field),
                stable_key(seed, attribute, group, row["sequence_id"]),
            )
        result.append(min(members, key=key))
    return result


def base_length_reference(path: Path) -> dict[str, object]:
    rows = read_tsv(path)
    lengths = []
    for row in rows:
        valid = row.get("valid_complete", "").strip().lower() in {"true", "1"}
        length = int(row["length_aa"])
        if valid and 50 <= length <= 512:
            lengths.append(length)
    if not lengths:
        raise RuntimeError(f"No valid-complete 50-512 aa Base records in {path}")
    counts = [sum(low <= length <= high for length in lengths) for low, high in BINS]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "n": len(lengths),
        "counts": counts,
        "proportions": [count / len(lengths) for count in counts],
        "lengths": lengths,
    }


def optimize_validation_quotas(
    train_capacities: list[int], validation_capacities: list[int], base_counts: list[int]
) -> tuple[int, int, int, int]:
    """Find a 100-record validation vector feasible for an exact 3:1 train pair."""

    caps = [min(validation_capacities[i], train_capacities[i] // 3) for i in range(4)]
    base_total = sum(base_counts)
    best: tuple[tuple[int, int, tuple[int, ...]], tuple[int, ...]] | None = None
    for a in range(min(caps[0], 100) + 1):
        for b in range(min(caps[1], 100 - a) + 1):
            for c in range(min(caps[2], 100 - a - b) + 1):
                d = 100 - a - b - c
                if d < 0 or d > caps[3]:
                    continue
                quota = (a, b, c, d)
                l1 = sum(abs(quota[i] * base_total - 100 * base_counts[i]) for i in range(4))
                emd = sum(
                    abs(sum(quota[:i + 1]) * base_total - 100 * sum(base_counts[:i + 1]))
                    for i in range(3)
                )
                candidate = ((l1, emd, quota), quota)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    if best is None:
        raise RuntimeError(f"Cannot build exact 300:100 pair from capacities train={train_capacities}, validation={validation_capacities}")
    return best[1]  # type: ignore[return-value]


def empirical_quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def select_bin(
    candidates: list[dict[str, str]], attribute: str, quota: int,
    target_lengths: list[int], seed: int,
) -> list[dict[str, str]]:
    """Cover the complete positive-score range while matching Base lengths.

    Candidates are ordered by the target property and divided into ``quota``
    rank blocks.  One row is selected from every block.  Thus near-threshold,
    middle and very high positives are represented without post-hoc top-score
    cherry-picking.  For Sol, structurally qualified rows are preferred inside
    each narrow score block before length distance and frozen structure scores.
    """

    if quota == 0:
        return []
    if len(candidates) < quota:
        raise RuntimeError(f"Need {quota} rows but only {len(candidates)} candidates remain")
    field = SPECS[attribute][0]
    ordered = sorted(candidates, key=lambda row: (
        safe_float(row, field), stable_key(seed, attribute, row["homology_group"], row["sequence_id"])
    ))
    selected = []
    for slot in range(quota):
        start = math.floor(slot * len(ordered) / quota)
        stop = math.floor((slot + 1) * len(ordered) / quota)
        block = ordered[start:max(start + 1, stop)]
        target = target_lengths[slot]
        if attribute == "sol":
            key = lambda row: (
                not structure_qualified(row),
                abs(len(row["sequence"]) - target),
                -safe_float(row, "fold_score", -math.inf),
                safe_float(row, "fold_mean_pae", math.inf),
                stable_key(seed, attribute, str(slot), row["sequence_id"]),
            )
        else:
            key = lambda row: (
                abs(len(row["sequence"]) - target),
                stable_key(seed, attribute, str(slot), row["sequence_id"]),
            )
        selected.append(min(block, key=key))
    if len({row["homology_group"] for row in selected}) != quota:
        raise AssertionError("Homology representative selection was not unique")
    return selected


def select_dataset(
    representatives: list[dict[str, str]], attribute: str, quotas: tuple[int, ...],
    base_lengths: list[int], seed: int,
) -> list[dict[str, str]]:
    selected = []
    for index, quota in enumerate(quotas):
        rows = [row for row in representatives if length_bin(len(row["sequence"])) == index]
        reference = [length for length in base_lengths if length_bin(length) == index]
        targets = [empirical_quantile(reference, (slot + 0.5) / quota) for slot in range(quota)] if quota else []
        selected.extend(select_bin(rows, attribute, quota, targets, seed))
    return selected


def select_sol_training_matched_to_validation(
    representatives: list[dict[str, str]], quotas: tuple[int, ...],
    selected_validation: list[dict[str, str]], seed: int,
) -> list[dict[str, str]]:
    """Match both length and coarse structure-quality composition at 3:1."""

    selected = []
    for index, quota in enumerate(quotas):
        validation_bin = [
            row for row in selected_validation
            if length_bin(len(row["sequence"])) == index
        ]
        before = len(selected)
        for qualified in (True, False):
            validation_part = [
                row for row in validation_bin
                if structure_qualified(row) == qualified
            ]
            part_quota = 3 * len(validation_part)
            candidates = [
                row for row in representatives
                if length_bin(len(row["sequence"])) == index
                and structure_qualified(row) == qualified
            ]
            targets = sorted(
                len(row["sequence"]) for row in validation_part for _ in range(3)
            )
            selected.extend(select_bin(candidates, "sol", part_quota, targets, seed))
        if len(selected) - before != quota:
            raise AssertionError("Sol structure-matched selection did not fill its length quota")
    return selected


def ks_statistic(left: list[int], right: list[int]) -> float:
    values = sorted(set(left + right))
    return max(
        abs(sum(item <= value for item in left) / len(left) - sum(item <= value for item in right) / len(right))
        for value in values
    )


def summarize(rows: list[dict[str, str]], attribute: str) -> dict[str, object]:
    field = SPECS[attribute][0]
    lengths = [len(row["sequence"]) for row in rows]
    values = [safe_float(row, field) for row in rows]
    summary: dict[str, object] = {
        "n": len(rows),
        "length_mean": sum(lengths) / len(lengths),
        "length_median": sorted(lengths)[len(lengths) // 2],
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_bins": [sum(low <= length <= high for length in lengths) for low, high in BINS],
        "attribute_mean_source_scale": sum(values) / len(values),
        "attribute_min_source_scale": min(values),
        "attribute_max_source_scale": max(values),
        "starts_with_m": sum(row["sequence"].startswith("M") for row in rows),
        "complexity_gate_failures": sum(not row["_complexity"]["passes_complexity_gate"] for row in rows),
        "sol_structure_qualified": sum(structure_qualified(row) for row in rows) if attribute == "sol" else None,
    }
    if attribute == "sol":
        fold_scores = [safe_float(row, "fold_score") for row in rows]
        pae_scores = [safe_float(row, "fold_mean_pae") for row in rows]
        summary.update({
            "fold_score_mean": sum(fold_scores) / len(fold_scores),
            "fold_mean_pae_mean": sum(pae_scores) / len(pae_scores),
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, str]], attribute: str) -> None:
    field, _, output_field, divisor = SPECS[attribute]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ["sequence", output_field], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"sequence": row["sequence"], output_field: f"{safe_float(row, field) / divisor:.9g}"})


def write_manifest(path: Path, datasets: dict[tuple[str, str], list[dict[str, str]]]) -> None:
    fields = [
        "dataset", "attribute", "sequence_id", "representative_accession", "cluster_name",
        "sequence_sha256", "homology_group", "source_split", "length_aa", "length_bin",
        "starts_with_m", "global_entropy_bits", "max_single_residue_fraction",
        "max_homopolymer_run", "low_complexity_window_fraction", "unique_residues",
        "fold_score", "fold_mean_pae", "fold_ptm_01", "sol_score", "thermo_score",
        "sol_structure_qualified",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for (split, attribute), rows in sorted(datasets.items()):
            for row in rows:
                metrics = row["_complexity"]
                writer.writerow({
                    "dataset": split, "attribute": attribute,
                    "sequence_id": row["sequence_id"],
                    "representative_accession": row.get("representative_accession", ""),
                    "cluster_name": row.get("cluster_name", ""),
                    "sequence_sha256": sha256_bytes(row["sequence"].encode()),
                    "homology_group": row["homology_group"], "source_split": row["split"],
                    "length_aa": len(row["sequence"]),
                    "length_bin": BIN_NAMES[length_bin(len(row["sequence"]))],
                    "starts_with_m": str(row["sequence"].startswith("M")).lower(),
                    "global_entropy_bits": f"{metrics['global_entropy_bits']:.9g}",
                    "max_single_residue_fraction": f"{metrics['max_single_residue_fraction']:.9g}",
                    "max_homopolymer_run": metrics["max_homopolymer_run"],
                    "low_complexity_window_fraction": f"{metrics['low_complexity_window_fraction']:.9g}",
                    "unique_residues": metrics["unique_residues"],
                    "fold_score": row.get("fold_score", ""), "fold_mean_pae": row.get("fold_mean_pae", ""),
                    "fold_ptm_01": row.get("fold_ptm_01", ""), "sol_score": row.get("sol_score", ""),
                    "thermo_score": row.get("thermo_score", ""),
                    "sol_structure_qualified": str(structure_qualified(row)).lower(),
                })


def write_tokenization_audit(staging: Path, tokenizer_path: Path) -> None:
    """Verify the exact FASTA60 text round-trips and fits the token budget."""

    from transformers import AutoTokenizer
    from prodipa.unconditional.teacher_sequence_protocol import (
        PROTGPT2_FASTA60_V1,
        encode_teacher_sequence,
        format_teacher_sequence,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    report: dict[str, object] = {
        "status": "PASS",
        "protocol": PROTGPT2_FASTA60_V1,
        "tokenizer": str(tokenizer_path.resolve()),
        "eos_token_id": tokenizer.eos_token_id,
        "datasets": {},
    }
    for path in sorted(staging.glob("*.csv")):
        rows = read_csv(path)
        token_lengths, linebreaks = [], []
        mismatches = 0
        for row in rows:
            sequence = row["sequence"]
            text = format_teacher_sequence(sequence, PROTGPT2_FASTA60_V1)
            token_ids = encode_teacher_sequence(tokenizer, sequence, PROTGPT2_FASTA60_V1)
            token_lengths.append(len(token_ids))
            linebreaks.append(text.count("\n"))
            decoded = tokenizer.decode(
                token_ids[1:-1], clean_up_tokenization_spaces=False,
            )
            mismatches += decoded != text
        report["datasets"][path.name] = {  # type: ignore[index]
            "n": len(rows),
            "min_tokens_including_bos_eos": min(token_lengths),
            "mean_tokens_including_bos_eos": sum(token_lengths) / len(token_lengths),
            "max_tokens_including_bos_eos": max(token_lengths),
            "over_512_tokens": sum(length > 512 for length in token_lengths),
            "roundtrip_mismatch": mismatches,
            "min_text_newlines": min(linebreaks),
            "max_text_newlines": max(linebreaks),
        }
        if mismatches or any(length > 512 for length in token_lengths):
            raise RuntimeError(f"FASTA60 tokenization audit failed for {path}")
    (staging / "TOKENIZATION_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-archive", type=Path, default=Path("releases/zenodo/ProDiPA_frozen_dataset_2026-11-02.tar.gz"))
    parser.add_argument("--rebuild-root", type=Path, default=Path("data/rebuild/teacher_data_threshold_85_070_060"))
    parser.add_argument("--base-generation", type=Path, default=Path("outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers/generation/unconditional/base.tsv"))
    parser.add_argument("--staging", type=Path, default=Path("data/rebuild/unconditional_teacher_balanced"))
    parser.add_argument("--tokenizer", type=Path, default=Path("__PRODIPA_MODELS__/ProtGPT2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = read_release(args.release_archive, "train_25000_complete.tsv.gz")
    validation = read_release(args.release_archive, "validation_1000_complete.tsv.gz")
    score_root = args.rebuild_root / "scores"
    train += attach_fold_scores(
        args.rebuild_root / "unconditional_train_short_unscored.tsv",
        sorted(score_root.glob("unconditional_train_short_esmfold_shard_*.tsv")), "train",
    )
    validation += attach_fold_scores(
        args.rebuild_root / "unconditional_validation_unscored.tsv",
        sorted(score_root.glob("unconditional_validation_esmfold_shard_*.tsv")), "validation",
    )
    base = base_length_reference(args.base_generation)
    args.staging.mkdir(parents=True, exist_ok=True)
    datasets: dict[tuple[str, str], list[dict[str, str]]] = {}
    report: dict[str, object] = {
        "status": "STAGED_NOT_PUBLISHED", "seed": args.seed,
        "selection_protocol": "base_length_matched_3to1_complexity_audited_v1",
        "teacher_sequence_protocol": "protgpt2_fasta60_v1",
        "base_length_reference": {key: value for key, value in base.items() if key != "lengths"},
        "thresholds": {attribute: spec[1] for attribute, spec in SPECS.items()},
        "complexity_gate": COMPLEXITY_GATE,
        "sol_structure_qualified_definition": SOL_STRUCTURE_QUALIFIED,
        "datasets": {}, "quotas": {}, "candidate_capacities": {},
        "validation_is_checkpoint_selection_not_blind_test": True,
    }
    for attribute in SPECS:
        train_reps = group_representatives(eligible(train, attribute), attribute, args.seed)
        validation_reps = group_representatives(eligible(validation, attribute), attribute, args.seed)
        train_caps = [sum(length_bin(len(row["sequence"])) == i for row in train_reps) for i in range(4)]
        validation_caps = [sum(length_bin(len(row["sequence"])) == i for row in validation_reps) for i in range(4)]
        test_quota = optimize_validation_quotas(train_caps, validation_caps, base["counts"])  # type: ignore[arg-type]
        train_quota = tuple(3 * value for value in test_quota)
        selected_test = select_dataset(validation_reps, attribute, test_quota, base["lengths"], args.seed)  # type: ignore[arg-type]
        if attribute == "sol":
            selected_train = select_sol_training_matched_to_validation(
                train_reps, train_quota, selected_test, args.seed,
            )
        else:
            matched_lengths = [
                len(row["sequence"]) for row in selected_test for _ in range(3)
            ]
            selected_train = select_dataset(
                train_reps, attribute, train_quota, matched_lengths, args.seed,
            )
        datasets[("train", attribute)] = selected_train
        datasets[("test", attribute)] = selected_test
        report["candidate_capacities"][attribute] = {"train": train_caps, "test": validation_caps}  # type: ignore[index]
        report["quotas"][attribute] = {"train": list(train_quota), "test": list(test_quota)}  # type: ignore[index]
        train_summary = summarize(selected_train, attribute)
        test_summary = summarize(selected_test, attribute)
        train_summary["length_ks_vs_test"] = ks_statistic(
            [len(row["sequence"]) for row in selected_train],
            [len(row["sequence"]) for row in selected_test],
        )
        report["datasets"][f"train_{attribute}"] = train_summary  # type: ignore[index]
        report["datasets"][f"test_{attribute}"] = test_summary  # type: ignore[index]

    train_groups = {row["homology_group"] for (split, _), rows in datasets.items() if split == "train" for row in rows}
    test_groups = {row["homology_group"] for (split, _), rows in datasets.items() if split == "test" for row in rows}
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Cross-attribute train/test homology leakage: {sorted(overlap)[:10]}")
    report["cross_attribute_train_test_homology_overlap"] = 0

    for (split, attribute), rows in datasets.items():
        count = 300 if split == "train" else 100
        write_csv(args.staging / f"{split}_{attribute}_{count}.csv", rows, attribute)
    write_manifest(args.staging / "selected_sequence_manifest.tsv", datasets)
    (args.staging / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_tokenization_audit(args.staging, args.tokenizer)
    audit = {
        "status": "PASS_STAGED_NOT_PUBLISHED",
        "files": {
            path.name: sha256_file(path)
            for path in sorted(args.staging.iterdir())
            if path.is_file() and path.name not in {"FINAL_AUDIT.json", "build.log"}
        },
        "canonical_data_unchanged": True,
        "cross_attribute_train_test_homology_overlap": 0,
    }
    (args.staging / "FINAL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
