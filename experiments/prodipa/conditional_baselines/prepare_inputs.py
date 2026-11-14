#!/usr/bin/env python3
"""Prepare frozen TPR inputs required by Conditional comparison methods."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import tempfile
from collections import Counter
from pathlib import Path


AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
FAMILY = "G3DSA:1.25.40.10"
FAMILY_NAME = "Tetratricopeptide-like helical domain superfamily"
INSTRUCTION = "[Generate by superfamily]"
SCORE_SPECS = {
    "fold": ("fold_plddt_100", 70.0, "plddt"),
    "sol": ("solubility_scaled_01", 0.50, "sol"),
    "thermo": ("thermophilicity_score_01", 0.40, "thermo"),
}
POSITIVE_FILES = {
    "fold": "data/conditional/fold/train_plddt_300.json",
    "sol": "data/conditional/sol/train_sol_300.json",
    "thermo": "data/conditional/thermo/train_thermo_300.json",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        delete=False, prefix=f".{path.name}.",
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sequence_from_record(row: dict[str, object]) -> str:
    sequence = str(row.get("sequence", ""))
    if not sequence:
        sequence = str(row.get("output", "")).split("Seq=<", 1)[-1].split(">", 1)[0]
    return "".join(sequence.split()).upper()


def length_bin(length: int) -> str:
    if 50 <= length <= 149:
        return "50-149"
    if length <= 249:
        return "150-249"
    if length <= 349:
        return "250-349"
    if length <= 512:
        return "350-512"
    raise ValueError(f"sequence length is outside 50-512: {length}")


def score_map(paths: list[Path], column: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for path in paths:
        for row in read_tsv(path):
            if row.get("status") == "ok" and row.get(column):
                sequence_id = row["sequence_id"]
                value = float(row[column])
                if sequence_id in values and values[sequence_id] != value:
                    raise ValueError(f"conflicting score for {sequence_id}: {column}")
                values[sequence_id] = value
    return values


def stable_key(seed: int, purpose: str, row: dict[str, str]) -> str:
    value = f"{seed}|{purpose}|{row['homology_group']}|{row['sequence_id']}"
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--msa-depth", type=int, default=15)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/evaluation/conditional/requested_baselines"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    rebuild = root / "data/rebuild/teacher_data_threshold_85_070_060"
    candidates_path = rebuild / "tpr_candidates_clustered.tsv"
    candidates = read_tsv(candidates_path)

    fold = score_map(sorted((rebuild / "scores").glob("tpr_esmfold_shard_*.tsv")), "fold_plddt_100")
    sol = score_map(sorted((rebuild / "scores").glob("tpr_*sol.tsv")), "solubility_scaled_01")
    thermo = score_map(sorted((rebuild / "scores").glob("tpr_*thermo.tsv")), "thermophilicity_score_01")
    score_maps = {"fold": fold, "sol": sol, "thermo": thermo}

    manifest = read_tsv(root / "data/provenance/conditional_teacher_manifest.tsv")
    test_groups = {row["homology_group"] for row in manifest if row["split"] == "test"}
    train_groups = {row["homology_group"] for row in manifest if row["split"] == "train"}
    if train_groups & test_groups:
        raise ValueError("published Conditional manifest has cross-split homology overlap")

    by_sequence = {row["sequence_id"]: row for row in candidates}
    complete: list[dict[str, str]] = []
    for row in candidates:
        sequence_id = row["sequence_id"]
        sequence = row["sequence"]
        if (
            row.get("cath_superfamily") == FAMILY
            and row.get("split") in {"train", "test"}
            and row.get("homology_group")
            and 50 <= len(sequence) <= 512
            and set(sequence) <= AA
            and all(sequence_id in mapping for mapping in score_maps.values())
        ):
            complete.append(row)

    negative_dir = output / "aspo_negative_pool"
    homology_rows: list[dict[str, object]] = []
    negative_summary: dict[str, object] = {}
    for prop, (_, threshold, output_score_key) in SCORE_SPECS.items():
        positive_path = root / POSITIVE_FILES[prop]
        positives = json.loads(positive_path.read_text(encoding="utf-8"))
        positive_bins = Counter(length_bin(len(sequence_from_record(row))) for row in positives)
        if sum(positive_bins.values()) != 300:
            raise ValueError(f"{prop} positive set is not exactly 300 sequences")

        eligible = [
            row for row in complete
            if row["split"] == "train"
            and row["homology_group"] not in test_groups
            and score_maps[prop][row["sequence_id"]] <= threshold
        ]
        unique_groups: dict[str, dict[str, str]] = {}
        for row in sorted(eligible, key=lambda item: stable_key(args.seed, prop, item)):
            unique_groups.setdefault(row["homology_group"], row)
        selected: list[dict[str, str]] = []
        for bin_name in ("50-149", "150-249", "250-349", "350-512"):
            available = [
                row for row in unique_groups.values()
                if length_bin(len(row["sequence"])) == bin_name
            ]
            needed = positive_bins[bin_name]
            if len(available) < needed:
                raise RuntimeError(
                    f"{prop} negative capacity {bin_name}: need {needed}, found {len(available)}"
                )
            selected.extend(available[:needed])
        if len(selected) != 300 or len({row["homology_group"] for row in selected}) != 300:
            raise AssertionError(f"invalid {prop} negative selection")

        records = []
        for row in selected:
            sequence_id = row["sequence_id"]
            record = {
                "sequence_id": sequence_id,
                "sequence": row["sequence"],
                "instruction": INSTRUCTION,
                "input": f"Superfamily=<{FAMILY_NAME}>",
                "output": f"Seq=<{row['sequence']}>",
                "label": 0,
                "homology_group": row["homology_group"],
                "fold_score": fold[sequence_id],
                "sol_score": sol[sequence_id],
                "thermo_score": thermo[sequence_id],
            }
            record[output_score_key] = (
                score_maps[prop][sequence_id] / 100.0
                if output_score_key == "plddt" else score_maps[prop][sequence_id]
            )
            records.append(record)
            homology_rows.append({
                "property": prop,
                "role": "negative",
                "sequence_id": sequence_id,
                "homology_group": row["homology_group"],
                "split": "train",
                "length_aa": len(row["sequence"]),
                "length_bin": length_bin(len(row["sequence"])),
            })
        write_json(negative_dir / f"{prop}_low.json", records)
        negative_summary[prop] = {
            "threshold_max": threshold,
            "count": len(records),
            "unique_homology_groups": len({row["homology_group"] for row in selected}),
            "length_bins": dict(sorted(Counter(
                length_bin(len(row["sequence"])) for row in selected
            ).items())),
            "positive_length_bins": dict(sorted(positive_bins.items())),
        }

    write_tsv(
        negative_dir / "homology_manifest.tsv", homology_rows,
        ["property", "role", "sequence_id", "homology_group", "split", "length_aa", "length_bin"],
    )

    # Family-prompted models receive only frozen training homologs. Test groups never
    # appear in a context panel, and every panel contains distinct homology groups.
    context_pool: dict[str, dict[str, str]] = {}
    for row in manifest:
        if row["split"] != "train" or row["homology_group"] in test_groups:
            continue
        candidate = by_sequence.get(row["sequence_id"])
        sequence = row["sequence"] if candidate is None else candidate["sequence"]
        context_pool.setdefault(row["homology_group"], {
            "sequence_id": row["sequence_id"],
            "sequence": sequence,
            "homology_group": row["homology_group"],
        })
    pool = sorted(context_pool.values(), key=lambda row: stable_key(args.seed, "msa_pool", row))
    if len(pool) < args.msa_depth:
        raise RuntimeError(f"need {args.msa_depth} independent TPR contexts, found {len(pool)}")
    panels: list[dict[str, object]] = []
    for request_index in range(args.requests):
        rng = random.Random(args.seed + request_index)
        chosen = rng.sample(pool, args.msa_depth)
        panels.append({
            "request_id": f"protgpt3_msa_tpr_{request_index:04d}",
            "seed": args.seed + request_index,
            "mode": "no_gap",
            "family_id": FAMILY,
            "family_name": FAMILY_NAME,
            "context_sequence_ids": [row["sequence_id"] for row in chosen],
            "context_homology_groups": [row["homology_group"] for row in chosen],
            "context_sequences": [row["sequence"] for row in chosen],
        })
    write_jsonl(output / "protgpt3_msa_contexts.jsonl", panels)

    report = {
        "status": "COMPLETE",
        "selection_seed": args.seed,
        "family_id": FAMILY,
        "family_name": FAMILY_NAME,
        "candidate_source": str(candidates_path.resolve()),
        "complete_scored_candidates": len(complete),
        "negative_pools": negative_summary,
        "protgpt3_msa": {
            "requests": len(panels),
            "context_depth": args.msa_depth,
            "eligible_independent_train_groups": len(pool),
            "test_groups_excluded": len(test_groups),
        },
    }
    write_json(output / "INPUT_AUDIT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
