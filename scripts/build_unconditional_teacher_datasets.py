#!/usr/bin/env python3
"""Build the strict 85/.70/.60 ProtGPT2 teacher datasets atomically.

The builder keeps the original immutable split assignment, selects at most one
sequence per homology group and attribute, and refuses to publish any partial
revision.  Newly computed ESMFold scores are accepted only for sequences that
already belong to the frozen train or validation partition.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path


ARCHIVE_PREFIX = "ProDiPA_frozen_dataset_2026-11-02"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
BINS = ((50, 149), (150, 249), (250, 349), (350, 512))
BIN_NAMES = ("50-149", "150-249", "250-349", "350-512")
TRAIN_QUOTAS = (15, 90, 135, 60)
TEST_QUOTAS = (5, 30, 45, 20)
QUOTAS = {
    ("train", "plddt"): (1, 104, 135, 60),
    ("train", "sol"): TRAIN_QUOTAS,
    ("train", "thermo"): TRAIN_QUOTAS,
    ("test", "plddt"): (0, 32, 47, 21),
    ("test", "sol"): (24, 50, 20, 6),
    ("test", "thermo"): (30, 35, 19, 16),
}
SPECS = {
    "plddt": ("fold_score", 85.0, "pLDDT", 100.0),
    "sol": ("sol_score", 0.70, "sol", 1.0),
    "thermo": ("thermo_score", 0.60, "thermo", 1.0),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_release(archive: Path, name: str) -> list[dict[str, str]]:
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile(f"{ARCHIVE_PREFIX}/{name}")
        if member is None:
            raise FileNotFoundError(name)
        with gzip.GzipFile(fileobj=member) as compressed:
            return list(csv.DictReader(io.TextIOWrapper(compressed), delimiter="\t"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_key(seed: int, attribute: str, split: str, group: str, sequence: str) -> str:
    payload = f"{seed}|{attribute}|{split}|{group}|{sequence}".encode()
    return sha256_bytes(payload)


def length_bin(length: int) -> int:
    for index, (low, high) in enumerate(BINS):
        if low <= length <= high:
            return index
    raise ValueError(f"Sequence length outside 50-512: {length}")


def attach_fold_scores(
    candidate_path: Path, score_paths: list[Path], expected_split: str,
) -> list[dict[str, str]]:
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
                raise ValueError(f"Conflicting ESMFold score for {row['sequence_id']}")
    rows = []
    for sequence_id, score in scores.items():
        source = candidates.get(sequence_id)
        if source is None:
            raise ValueError(f"Score not present in frozen candidate table: {sequence_id}")
        if source["split"] != expected_split:
            raise ValueError(f"Unexpected split for {sequence_id}: {source['split']}")
        rows.append(source | {
            "fold_score": score["fold_plddt_100"],
            "fold_ptm_01": score.get("fold_ptm_01", ""),
            "fold_mean_pae": score.get("fold_mean_pae", ""),
            "fold_status": "ok",
        })
    return rows


def validate_source(rows: list[dict[str, str]], split: str) -> None:
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for row in rows:
        sequence = row["sequence"].strip().upper()
        if row["sequence_id"] in seen_ids or sequence in seen_sequences:
            raise ValueError(f"Duplicate source sequence/id in {split}: {row['sequence_id']}")
        if row["split"] != split:
            raise ValueError(f"Source split mismatch: {row['sequence_id']} / {row['split']}")
        if not (50 <= len(sequence) <= 512) or not set(sequence) <= AA:
            raise ValueError(f"Invalid sequence: {row['sequence_id']}")
        if not row.get("homology_group"):
            raise ValueError(f"Blank homology group: {row['sequence_id']}")
        seen_ids.add(row["sequence_id"])
        seen_sequences.add(sequence)


def select(
    rows: list[dict[str, str]], attribute: str, split: str,
    quotas: tuple[int, int, int, int], seed: int,
) -> list[dict[str, str]]:
    score_field, threshold, _, _ = SPECS[attribute]
    positive = []
    for row in rows:
        try:
            score = float(row.get(score_field, ""))
        except ValueError:
            continue
        if score >= threshold:
            positive.append(row)
    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()
    for bin_index, quota in enumerate(quotas):
        candidates = [row for row in positive if length_bin(len(row["sequence"])) == bin_index]
        candidates.sort(key=lambda row: stable_key(
            seed, attribute, split, row["homology_group"], row["sequence"],
        ))
        by_group: dict[str, dict[str, str]] = {}
        for row in candidates:
            if row["homology_group"] not in used_groups:
                by_group.setdefault(row["homology_group"], row)
        chosen = list(by_group.values())[:quota]
        if len(chosen) != quota:
            raise RuntimeError(
                f"INCOMPLETE {split}/{attribute}/{BIN_NAMES[bin_index]}: "
                f"need {quota}, available groups {len(by_group)}"
            )
        selected.extend(chosen)
        used_groups.update(row["homology_group"] for row in chosen)
    return selected


def write_csv(path: Path, rows: list[dict[str, str]], attribute: str) -> None:
    score_field, _, output_field, divisor = SPECS[attribute]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["sequence", output_field], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sequence": row["sequence"],
                output_field: f"{float(row[score_field]) / divisor:.9g}",
            })


def atomic_publish(staging: Path, destination: Path) -> None:
    mapping = {
        "train_plddt_300.csv": destination / "fold/train_plddt_300.csv",
        "test_plddt_100.csv": destination / "fold/test_plddt_100.csv",
        "train_sol_300.csv": destination / "sol/train_sol_300.csv",
        "test_sol_100.csv": destination / "sol/test_sol_100.csv",
        "train_thermo_300.csv": destination / "thermo/train_thermo_300.csv",
        "test_thermo_100.csv": destination / "thermo/test_thermo_100.csv",
        "selected_sequence_manifest.tsv": destination / "selected_sequence_manifest.tsv",
        "build_report.json": destination / "build_report.json",
        "FINAL_AUDIT.json": destination / "FINAL_AUDIT.json",
    }
    for source_name, target in mapping.items():
        source = staging / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".new")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-archive", type=Path,
                        default=Path("releases/zenodo/ProDiPA_frozen_dataset_2026-11-02.tar.gz"))
    parser.add_argument("--rebuild-root", type=Path,
                        default=Path("data/rebuild/teacher_data_threshold_85_070_060"))
    parser.add_argument("--staging", type=Path,
                        default=Path("data/rebuild/teacher_data_threshold_85_070_060/unconditional_staging"))
    parser.add_argument("--destination", type=Path, default=Path("data/unconditional"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    train = read_release(args.release_archive, "train_25000_complete.tsv.gz")
    validation = read_release(args.release_archive, "validation_1000_complete.tsv.gz")
    score_root = args.rebuild_root / "scores"
    train_fold_extra = attach_fold_scores(
        args.rebuild_root / "unconditional_train_short_unscored.tsv",
        sorted(score_root.glob("unconditional_train_short_esmfold_shard_*.tsv")), "train",
    )
    validation_fold_extra = attach_fold_scores(
        args.rebuild_root / "unconditional_validation_unscored.tsv",
        sorted(score_root.glob("unconditional_validation_esmfold_shard_*.tsv")), "validation",
    )
    train_all = train + train_fold_extra
    validation_all = validation + validation_fold_extra
    validate_source(train_all, "train")
    validate_source(validation_all, "validation")

    args.staging.mkdir(parents=True, exist_ok=True)
    final: dict[tuple[str, str], list[dict[str, str]]] = {}
    report: dict = {
        "status": "PASS", "seed": args.seed,
        "thresholds": {key: value[1] for key, value in SPECS.items()},
        "nominal_train_length_quotas": dict(zip(BIN_NAMES, TRAIN_QUOTAS)),
        "nominal_test_length_quotas": dict(zip(BIN_NAMES, TEST_QUOTAS)),
        "audited_adaptive_quotas": {
            f"{split}_{attribute}": dict(zip(BIN_NAMES, quotas))
            for (split, attribute), quotas in QUOTAS.items()
        },
        "datasets": {},
        "validation_is_checkpoint_selection_not_blind_test": True,
    }
    for split, source in (("train", train_all), ("test", validation_all)):
        for attribute in SPECS:
            quotas = QUOTAS[(split, attribute)]
            rows = select(source, attribute, split, quotas, args.seed)
            final[(split, attribute)] = rows
            name = f"{split}_{attribute}_{300 if split == 'train' else 100}.csv"
            write_csv(args.staging / name, rows, attribute)
            report["datasets"][f"{split}_{attribute}"] = {
                "n": len(rows),
                "length_bins": [sum(length_bin(len(row["sequence"])) == i for row in rows) for i in range(4)],
                "independent_homology_groups": len({row["homology_group"] for row in rows}),
            }

    train_groups = {row["homology_group"] for (split, _), rows in final.items() if split == "train" for row in rows}
    test_groups = {row["homology_group"] for (split, _), rows in final.items() if split == "test" for row in rows}
    if train_groups & test_groups:
        raise RuntimeError("Cross-attribute train/test homology leakage")

    manifest_rows = []
    for (split, attribute), rows in final.items():
        for row in rows:
            manifest_rows.append({
                "dataset": split, "attribute": attribute,
                "sequence_id": row["sequence_id"],
                "sequence_sha256": sha256_bytes(row["sequence"].encode()),
                "homology_group": row["homology_group"],
                "source_split": row["split"],
            })
    manifest_path = args.staging / "selected_sequence_manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(manifest_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(manifest_rows)
    report["cross_attribute_train_test_homology_overlap"] = 0
    (args.staging / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    audit = {
        "status": "PASS", "files": {},
        "cross_attribute_train_test_homology_overlap": 0,
    }
    for path in sorted(args.staging.iterdir()):
        if path.is_file() and path.name != "FINAL_AUDIT.json":
            audit["files"][path.name] = sha256_bytes(path.read_bytes())
    (args.staging / "FINAL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    if args.publish:
        atomic_publish(args.staging, args.destination)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
