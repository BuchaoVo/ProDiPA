#!/usr/bin/env python3
"""Build the final pooled Unconditional table with UniProtKB novelty and exact HV."""

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


MAIN_FIELDS = [
    "Model", "PPL", "Novelty-U", "Novelty-T", "pLDDT", "%>70",
    "pAE", "%<10", "Sol", "Thermo", "HV",
]
HV_OBJECTIVES = ["ppl", "novelty_u_percent", "plddt", "sol", "thermo"]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.8g}"


def atomic_write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
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


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}.",
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


def exact_anchored_hypervolume(points: np.ndarray) -> float:
    """Exact union volume of boxes [0, point] by recursive dimension slicing."""
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
        volume += (float(boundary) - previous) * exact_anchored_hypervolume(active)
        previous = float(boundary)
    return volume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--novelty-u", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-sequence-scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path,
                        help="Reuse predeclared normalization bounds and reference; never refit")
    parser.add_argument(
        "--expected-models",
        default="base,fold_teacher,sol_teacher,thermo_teacher,progen2_base,dplm_650m",
    )
    parser.add_argument(
        "--normalization-scope-label",
        default="all complete generated sequences pooled across the configured Unconditional cohort",
    )
    parser.add_argument("--requests-per-model", type=int, default=512)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for path in args.sequence_scores:
        values = read_tsv(path)
        if not values:
            raise ValueError(f"empty sequence-score table: {path}")
        # Native teacher results and external baseline results share all
        # evaluation fields, while baseline rows additionally retain method
        # provenance columns.  Preserve the ordered union instead of rejecting
        # scientifically compatible tables solely because of those extras.
        for field in values[0]:
            if field not in fields:
                fields.append(field)
        rows.extend(values)
    if not fields:
        raise ValueError("no sequence-score fields found")

    novelty_rows = read_tsv(args.novelty_u)
    novelty = {row["sequence_id"]: row["novelty_percent"] for row in novelty_rows}
    for row in rows:
        if row.get("sequence_id") in novelty:
            row["novelty_u_percent"] = novelty[row["sequence_id"]]

    expected = args.expected_models.split(",")
    observed = sorted({row["model"] for row in rows})
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(f"model mismatch: missing={missing}, extra={extra}")
    for model in expected:
        count = sum(row["model"] == model for row in rows)
        if count != args.requests_per_model:
            raise ValueError(
                f"{model} has {count} requests, expected {args.requests_per_model}"
            )

    complete_rows = []
    oriented = []
    for row in rows:
        if row.get("valid_complete", "").lower() != "true":
            continue
        values = [number(row.get(key)) for key in HV_OBJECTIVES]
        if any(value is None for value in values):
            continue
        ppl, novelty_u, plddt, sol, thermo = (float(value) for value in values)
        complete_rows.append(row)
        oriented.append([-math.log1p(ppl), novelty_u, plddt, sol, thermo])
    if not complete_rows:
        raise ValueError("no rows have all five pooled-HV objectives")

    oriented_array = np.asarray(oriented, dtype=float)
    lower = oriented_array.min(axis=0)
    upper = oriented_array.max(axis=0)
    reference = np.zeros(5)
    if args.frozen_manifest:
        frozen = json.loads(args.frozen_manifest.read_text())
        if [o['field'] for o in frozen['objectives']] != HV_OBJECTIVES:
            raise ValueError('Frozen HV objective mismatch')
        lower = np.asarray(frozen['oriented_min'], dtype=float)
        upper = np.asarray(frozen['oriented_max'], dtype=float)
        reference = np.asarray(frozen['reference_point'], dtype=float)
        if any(v.shape != (5,) or not np.isfinite(v).all() for v in (lower, upper, reference)) or np.any(upper < lower):
            raise ValueError('Invalid frozen HV scale')
    span = upper - lower
    normalized = np.divide(
        oriented_array - lower,
        span,
        out=np.ones_like(oriented_array),
        where=span != 0,
    )
    normalized -= reference

    hv_by_model: dict[str, float] = {}
    nondominated_counts: dict[str, int] = {}
    for model in expected:
        model_points = normalized[
            np.asarray([row["model"] == model for row in complete_rows], dtype=bool)
        ]
        if len(model_points) == 0:
            raise ValueError(f"{model} has no complete HV points")
        frontier = nondominated(model_points)
        nondominated_counts[model] = len(frontier)
        hv_by_model[model] = exact_anchored_hypervolume(frontier)

    main_rows = []
    for model in expected:
        model_rows = [
            row for row in rows
            if row["model"] == model and row.get("valid_complete", "").lower() == "true"
        ]
        values = {
            key: [value for row in model_rows if (value := number(row.get(key))) is not None]
            for key in ("ppl", "novelty_u_percent", "novelty_t_percent", "plddt", "pae", "sol", "thermo")
        }
        required = ("ppl", "novelty_u_percent", "novelty_t_percent", "plddt", "pae", "sol", "thermo")
        if any(not values[key] for key in required):
            raise ValueError(f"{model} lacks a required main-table metric")
        main_rows.append({
            "Model": model,
            "PPL": fmt(statistics.fmean(values["ppl"])),
            "Novelty-U": fmt(statistics.fmean(values["novelty_u_percent"])),
            "Novelty-T": fmt(statistics.fmean(values["novelty_t_percent"])),
            "pLDDT": fmt(statistics.fmean(values["plddt"])),
            "%>70": fmt(100 * sum(value > 70 for value in values["plddt"]) / len(values["plddt"])),
            "pAE": fmt(statistics.fmean(values["pae"])),
            "%<10": fmt(100 * sum(value < 10 for value in values["pae"]) / len(values["pae"])),
            "Sol": fmt(statistics.fmean(values["sol"])),
            "Thermo": fmt(statistics.fmean(values["thermo"])),
            "HV": fmt(hv_by_model[model]),
        })

    atomic_write_tsv(args.output_sequence_scores, rows, fields)
    atomic_write_tsv(args.output_table, main_rows, MAIN_FIELDS)
    atomic_write_json(args.manifest, {
        "definition_source": "frozen pooled min-max HV protocol",
        "objectives": [
            {"field": "ppl", "transform": "-log(1 + PPL)", "direction": "maximize"},
            {"field": "novelty_u_percent", "transform": "identity", "direction": "maximize"},
            {"field": "plddt", "transform": "identity", "direction": "maximize"},
            {"field": "sol", "transform": "identity", "direction": "maximize"},
            {"field": "thermo", "transform": "identity", "direction": "maximize"},
        ],
        "excluded_from_hv": ["Novelty-T", "pAE", "%>70", "%<10"],
        "normalization_scope": (f"frozen scale from {args.frozen_manifest}" if args.frozen_manifest else args.normalization_scope_label),
        "reference_point": reference.tolist(),
        "oriented_min": lower.tolist(),
        "oriented_max": upper.tolist(),
        "complete_points": len(complete_rows),
        "complete_points_by_model": {
            model: sum(row["model"] == model for row in complete_rows) for model in expected
        },
        "nondominated_points_by_model": nondominated_counts,
        "hypervolume_by_model": hv_by_model,
        "novelty_u_file": str(args.novelty_u.resolve()),
        "sequence_score_files": [str(path.resolve()) for path in args.sequence_scores],
    })
    print(f"wrote pooled Unconditional main table: {args.output_table}")


if __name__ == "__main__":
    main()
