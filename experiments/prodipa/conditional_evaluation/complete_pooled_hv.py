#!/usr/bin/env python3
"""Build the pooled Conditional main table and exact five-objective HV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np


MAIN_FIELDS = ["Model", "PPL", "ProTrek %", "pLDDT", "pAE", "Sol", "Thermo", "HV"]
HV_FIELDS = ["ppl", "protrek_percent", "plddt", "sol", "thermo"]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def fmt(value: float) -> str:
    return f"{value:.8g}"


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        delete=False, prefix=f".{path.name}.",
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def nondominated(points: np.ndarray) -> np.ndarray:
    points = np.unique(points, axis=0)
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        if keep[index] and np.any(
            np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        ):
            keep[index] = False
    return points[keep]


def exact_hv(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    points = points[np.all(points > 0, axis=1)]
    if len(points) == 0:
        return 0.0
    points = nondominated(points)
    if points.shape[1] == 1:
        return float(points[:, 0].max())
    volume = 0.0
    previous = 0.0
    for boundary in np.unique(points[:, -1]):
        active = points[points[:, -1] >= boundary, :-1]
        volume += (float(boundary) - previous) * exact_hv(active)
        previous = float(boundary)
    return volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--novelty-u", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-sequence-scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-models", required=True)
    parser.add_argument("--requests-per-model", type=int, default=512)
    parser.add_argument(
        "--normalization-scope-label",
        default="all complete generated sequences pooled across the configured Conditional cohort",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for path in args.sequence_scores:
        values = read(path)
        if not values:
            raise ValueError(f"empty sequence-score table: {path}")
        for field in values[0]:
            if field not in fields:
                fields.append(field)
        rows.extend(values)
    if "novelty_u_percent" not in fields:
        fields.append("novelty_u_percent")
    novelty = {row["sequence_id"]: row["novelty_percent"] for row in read(args.novelty_u)}
    for row in rows:
        row["novelty_u_percent"] = novelty.get(row.get("sequence_id", ""), "")

    expected = [item for item in args.expected_models.split(",") if item]
    observed = {row["model"] for row in rows}
    if observed != set(expected):
        raise ValueError(f"model mismatch: observed={sorted(observed)}, expected={expected}")
    for model in expected:
        count = sum(row["model"] == model for row in rows)
        if count != args.requests_per_model:
            raise ValueError(f"{model} has {count} requests, expected {args.requests_per_model}")

    complete_rows: list[dict[str, str]] = []
    oriented: list[list[float]] = []
    for row in rows:
        if row.get("valid_complete", "").lower() != "true":
            continue
        values = [number(row.get(field)) for field in HV_FIELDS]
        if any(value is None for value in values):
            continue
        ppl, protrek, plddt, sol, thermo = (float(value) for value in values)
        complete_rows.append(row)
        oriented.append([-math.log1p(ppl), protrek, plddt, sol, thermo])
    if not complete_rows:
        raise ValueError("no rows have all five Conditional HV objectives")

    oriented_array = np.asarray(oriented, dtype=float)
    lower = oriented_array.min(axis=0)
    upper = oriented_array.max(axis=0)
    span = upper - lower
    normalized = np.divide(
        oriented_array - lower, span, out=np.ones_like(oriented_array), where=span != 0
    )
    hv_by_model: dict[str, float] = {}
    frontier_counts: dict[str, int] = {}
    for model in expected:
        mask = np.asarray([row["model"] == model for row in complete_rows], dtype=bool)
        points = normalized[mask]
        if len(points) == 0:
            raise ValueError(f"{model} has no complete HV points")
        frontier = nondominated(points)
        frontier_counts[model] = len(frontier)
        hv_by_model[model] = exact_hv(frontier)

    main_rows = []
    for model in expected:
        model_rows = [
            row for row in rows
            if row["model"] == model and row.get("valid_complete", "").lower() == "true"
        ]
        keys = ("ppl", "protrek_percent", "plddt", "pae", "sol", "thermo")
        values = {
            key: [value for row in model_rows if (value := number(row.get(key))) is not None]
            for key in keys
        }
        if any(not values[key] for key in keys):
            raise ValueError(f"{model} lacks a required Conditional main-table metric")
        main_rows.append({
            "Model": model,
            "PPL": fmt(statistics.fmean(values["ppl"])),
            "ProTrek %": fmt(statistics.fmean(values["protrek_percent"])),
            "pLDDT": fmt(statistics.fmean(values["plddt"])),
            "pAE": fmt(statistics.fmean(values["pae"])),
            "Sol": fmt(statistics.fmean(values["sol"])),
            "Thermo": fmt(statistics.fmean(values["thermo"])),
            "HV": fmt(hv_by_model[model]),
        })

    write_tsv(args.output_sequence_scores, rows, fields)
    write_tsv(args.output_table, main_rows, MAIN_FIELDS)
    write_json(args.manifest, {
        "definition_source": "frozen pooled min-max Conditional HV protocol",
        "objectives": [
            {"field": "ppl", "transform": "-log(1 + PPL)", "direction": "maximize"},
            {"field": "protrek_percent", "transform": "identity", "direction": "maximize"},
            {"field": "plddt", "transform": "identity", "direction": "maximize"},
            {"field": "sol", "transform": "identity", "direction": "maximize"},
            {"field": "thermo", "transform": "identity", "direction": "maximize"},
        ],
        "excluded_from_hv": ["Novelty-U", "Novelty-T", "pAE", "Strict JSR"],
        "normalization_scope": args.normalization_scope_label,
        "reference_point": [0, 0, 0, 0, 0],
        "oriented_min": lower.tolist(),
        "oriented_max": upper.tolist(),
        "complete_points": len(complete_rows),
        "complete_points_by_model": {
            model: sum(row["model"] == model for row in complete_rows) for model in expected
        },
        "nondominated_points_by_model": frontier_counts,
        "hypervolume_by_model": hv_by_model,
        "novelty_u_file": str(args.novelty_u.resolve()),
        "sequence_score_files": [str(path.resolve()) for path in args.sequence_scores],
    })
    print(f"wrote pooled Conditional main table: {args.output_table}")


if __name__ == "__main__":
    main()
