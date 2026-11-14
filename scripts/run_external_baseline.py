#!/usr/bin/env python3
"""Minimal same-dataset external baseline utilities (currently NSGA-II)."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


AA = "ACDEFGHIKLMNPQRSTVWY"
OBJECTIVES = ("fold_plddt_100", "solubility_scaled_01", "thermophilicity_score_01")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def dominates(left: dict, right: dict) -> bool:
    gains = [float(left[key]) - float(right[key]) for key in OBJECTIVES]
    return all(gain >= 0 for gain in gains) and any(gain > 0 for gain in gains)


def nondominated_fronts(rows: list[dict]) -> list[list[int]]:
    dominates_set = [[] for _ in rows]
    dominated_count = [0] * len(rows)
    fronts = [[]]
    for i, left in enumerate(rows):
        for j, right in enumerate(rows):
            if i == j:
                continue
            if dominates(left, right):
                dominates_set[i].append(j)
            elif dominates(right, left):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)
    cursor = 0
    while cursor < len(fronts) and fronts[cursor]:
        following = []
        for i in fronts[cursor]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    following.append(j)
        if following:
            fronts.append(following)
        cursor += 1
    return fronts


def crowding(rows: list[dict], front: list[int]) -> dict[int, float]:
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for key in OBJECTIVES:
        ordered = sorted(front, key=lambda index: float(rows[index][key]))
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        low, high = float(rows[ordered[0]][key]), float(rows[ordered[-1]][key])
        if high == low:
            continue
        for position in range(1, len(ordered) - 1):
            before = float(rows[ordered[position - 1]][key])
            after = float(rows[ordered[position + 1]][key])
            distance[ordered[position]] += (after - before) / (high - low)
    return distance


def select_population(rows: list[dict], size: int) -> list[dict]:
    selected = []
    for rank, front in enumerate(nondominated_fronts(rows)):
        distances = crowding(rows, front)
        for index in front:
            rows[index]["nsga_rank"] = rank
            rows[index]["crowding_distance"] = distances[index]
        ordered = sorted(front, key=lambda index: distances[index], reverse=True)
        selected.extend(rows[index] for index in ordered[: size - len(selected)])
        if len(selected) == size:
            break
    return selected


def mutate(sequence: str, rng: random.Random) -> tuple[str, str]:
    count = rng.choices((1, 2, 3), weights=(0.70, 0.25, 0.05), k=1)[0]
    positions = rng.sample(range(len(sequence)), count)
    residues = list(sequence)
    changes = []
    for position in positions:
        choices = AA.replace(residues[position], "")
        replacement = rng.choice(choices)
        changes.append(f"{residues[position]}{position + 1}{replacement}")
        residues[position] = replacement
    return "".join(residues), ";".join(changes)


def make_offspring(population: list[dict], generation: int, size: int, seed: int) -> list[dict]:
    rng = random.Random(seed + generation * 1009)
    children = []
    seen = {row["sequence"] for row in population}
    while len(children) < size:
        parent = rng.choice(population)
        sequence, mutations = mutate(parent["sequence"], rng)
        if sequence in seen:
            continue
        seen.add(sequence)
        children.append({
            "sequence_id": f"nsga_s{seed}_g{generation}_{len(children):04d}", "method": "nsga_ii",
            "seed": seed,
            "generation": generation, "parent_id": parent["sequence_id"],
            "mutations": mutations, "sequence": sequence, "length": len(sequence),
            "valid": "true", "termination_status": "not_applicable_search_candidate",
        })
    return children


def initialize(args) -> None:
    metadata = {row["sequence_id"]: row for row in read_tsv(args.metadata)}
    rows = []
    for score in read_tsv(args.scores):
        if score["sequence_id"] not in metadata or not all(score.get(key) for key in OBJECTIVES):
            continue
        meta = metadata[score["sequence_id"]]
        rows.append({
            "sequence_id": score["sequence_id"], "method": "nsga_ii", "seed": args.seed,
            "generation": -1, "parent_id": "", "mutations": "",
            "sequence": meta["sequence"], "length": meta["length"],
            "valid": "true", "termination_status": "natural_seed",
            **{key: score[key] for key in OBJECTIVES},
            "nsga_rank": "", "crowding_distance": "",
        })
    rng = random.Random(args.seed)
    population = rng.sample(rows, args.population_size)
    write_tsv(args.population_out, population)
    write_tsv(args.candidates_out, make_offspring(population, 0, args.population_size, args.seed))


def scored_candidates(args) -> list[dict]:
    candidates = {row["sequence_id"]: row for row in read_tsv(args.candidates)}
    fold = {row["sequence_id"]: row for row in read_tsv(args.esmfold)}
    sol = {row["sequence_id"]: row for row in read_tsv(args.protein_sol)}
    thermo = {row["sequence_id"]: row for row in read_tsv(args.temberture)}
    missing = {
        name: sorted(candidates.keys() - scores.keys())
        for name, scores in (("fold", fold), ("sol", sol), ("thermo", thermo))
        if candidates.keys() - scores.keys()
    }
    if missing:
        raise ValueError(f"NSGA-II candidate score coverage mismatch: {missing}")
    rows = []
    for sequence_id, row in candidates.items():
        rows.append({
            **row, "fold_plddt_100": fold[sequence_id]["fold_plddt_100"],
            "solubility_scaled_01": sol[sequence_id]["solubility_scaled_01"],
            "thermophilicity_score_01": thermo[sequence_id]["thermophilicity_score_01"],
            "nsga_rank": "", "crowding_distance": "",
        })
    return rows


def advance(args) -> None:
    previous = read_tsv(args.population)
    scored = scored_candidates(args)
    population = select_population(previous + scored, args.population_size)
    write_tsv(args.population_out, population)
    archive = (read_tsv(args.archive) if args.archive.exists() else []) + scored
    write_tsv(args.archive, archive)
    if args.next_candidates_out:
        write_tsv(
            args.next_candidates_out,
            make_offspring(population, args.generation + 1, args.population_size, args.seed),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--population-size", type=int, default=128)
    common.add_argument("--seed", type=int, default=42)

    init = subparsers.add_parser("nsga-initialize", parents=[common])
    init.add_argument("--scores", type=Path, default=Path("data/properties/train_property_scores.tsv"))
    init.add_argument("--metadata", type=Path, default=Path("data/processed/uniref_split/split_metadata.tsv"))
    init.add_argument("--population-out", type=Path, required=True)
    init.add_argument("--candidates-out", type=Path, required=True)
    init.set_defaults(func=initialize)

    advance_parser = subparsers.add_parser("nsga-advance", parents=[common])
    advance_parser.add_argument("--generation", type=int, required=True)
    advance_parser.add_argument("--population", type=Path, required=True)
    advance_parser.add_argument("--candidates", type=Path, required=True)
    advance_parser.add_argument("--esmfold", type=Path, required=True)
    advance_parser.add_argument("--protein-sol", type=Path, required=True)
    advance_parser.add_argument("--temberture", type=Path, required=True)
    advance_parser.add_argument("--population-out", type=Path, required=True)
    advance_parser.add_argument("--archive", type=Path, required=True)
    advance_parser.add_argument("--next-candidates-out", type=Path)
    advance_parser.set_defaults(func=advance)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
