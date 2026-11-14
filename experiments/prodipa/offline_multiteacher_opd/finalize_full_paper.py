#!/usr/bin/env python3
"""Create frozen-scale paper tables and the full paired-run conclusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


FIELDS = [
    "Group",
    "Model",
    "PPL",
    "Novelty-U",
    "Novelty-T",
    "ProTrek %",
    "pLDDT",
    "pAE",
    "Sol",
    "Thermo",
    "HV",
    "HV Ret. (%)",
    "Train GPU-h",
    "HV-AUC",
    "HQ Disc.",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def number(value: object) -> float | None:
    try:
        result = float(value) if value not in (None, "", "--") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def fmt(value: float | None, digits: int = 8) -> str:
    return "--" if value is None else f"{value:.{digits}g}"


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


def novelty_map(path: Path) -> dict[str, float]:
    return {
        row["sequence_id"]: float(row["novelty_percent"])
        for row in read_tsv(path)
        if number(row.get("novelty_percent")) is not None
    }


def add_novelty(rows: list[dict[str, str]], values: dict[str, float]) -> None:
    for row in rows:
        if row.get("sequence_id") in values:
            row["novelty_u_percent"] = str(values[row["sequence_id"]])


def mean_metric(rows: list[dict[str, str]], field: str) -> float | None:
    values = [value for row in rows if (value := number(row.get(field))) is not None]
    return statistics.fmean(values) if values else None


def percentage_matching(
    rows: list[dict[str, str]], field: str, predicate
) -> float | None:
    values = [value for row in rows if (value := number(row.get(field))) is not None]
    return 100.0 * sum(predicate(value) for value in values) / len(values) if values else None


def summarize_rows(
    rows: list[dict[str, str]],
    manifest: dict,
    paradigm: str,
) -> dict[str, float | int | None]:
    complete = [row for row in rows if row.get("valid_complete", "").lower() == "true"]
    if not complete:
        raise ValueError(f"{paradigm} method has no complete evaluation sequences")
    hv_rows: list[list[float]] = []
    for row in complete:
        if paradigm == "conditional":
            raw = [
                number(row.get("ppl")),
                number(row.get("protrek_percent")),
                number(row.get("plddt")),
                number(row.get("sol")),
                number(row.get("thermo")),
            ]
        else:
            raw = [
                number(row.get("ppl")),
                number(row.get("novelty_u_percent")),
                number(row.get("plddt")),
                number(row.get("sol")),
                number(row.get("thermo")),
            ]
        if any(value is None for value in raw):
            continue
        ppl, second, plddt, sol, thermo = (float(value) for value in raw)
        hv_rows.append([-math.log1p(ppl), second, plddt, sol, thermo])
    if not hv_rows:
        raise ValueError(f"{paradigm} method has no complete frozen-HV points")
    lower = np.asarray(manifest["oriented_min"], dtype=float)
    upper = np.asarray(manifest["oriented_max"], dtype=float)
    oriented = np.asarray(hv_rows, dtype=float)
    normalized = np.divide(
        oriented - lower,
        upper - lower,
        out=np.ones_like(oriented),
        where=(upper - lower) != 0,
    )
    return {
        "requests": len(rows),
        "complete": len(complete),
        "PPL": mean_metric(complete, "ppl"),
        "Novelty-U": mean_metric(complete, "novelty_u_percent"),
        "Novelty-T": mean_metric(complete, "novelty_t_percent"),
        "ProTrek %": mean_metric(complete, "protrek_percent"),
        "pLDDT": mean_metric(complete, "plddt"),
        "%>70": percentage_matching(complete, "plddt", lambda value: value > 70.0),
        "pAE": mean_metric(complete, "pae"),
        "%<10": percentage_matching(complete, "pae", lambda value: value < 10.0),
        "Sol": mean_metric(complete, "sol"),
        "Thermo": mean_metric(complete, "thermo"),
        "HV": exact_hv(normalized),
        "frontier_points": len(nondominated(normalized)),
    }


def allocated_gpu_hours(paths: list[Path]) -> float:
    seconds = 0.0
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "COMPLETE":
            raise ValueError(f"incomplete timing record: {path}")
        seconds += float(payload["allocated_gpu_seconds"])
    return seconds / 3600.0


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def table_row(
    group: str,
    model: str,
    metrics: dict[str, float | int | None],
    retention: float | None = None,
    gpu_hours: float | None = None,
) -> dict[str, str]:
    row = {field: "--" for field in FIELDS}
    row.update({"Group": group, "Model": model, "HV-AUC": "--", "HQ Disc.": "--"})
    for field in ("PPL", "Novelty-U", "Novelty-T", "ProTrek %", "pLDDT", "pAE", "Sol", "Thermo", "HV"):
        row[field] = fmt(number(metrics.get(field)))
    row["HV Ret. (%)"] = (fmt(retention, 6) if retention is not None else
                           "N/A" if model in {"Reference θ0", "Online OPD (paired)", "ProDiPA (Offline)"} else "--")
    row["Train GPU-h"] = fmt(gpu_hours, 6)
    return row


def canonical_rows(root: Path, paradigm: str) -> list[dict[str, str]]:
    main_path = root / "outputs/evaluation" / paradigm / "results" / f"{paradigm}_prodipa_main.tsv"
    sequence_path = root / "outputs/evaluation" / paradigm / "results" / f"{paradigm}_sequence_scores.tsv"
    main = read_tsv(main_path)
    sequences = read_tsv(sequence_path)
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in sequences:
        by_model.setdefault(row["model"], []).append(row)
    rows: list[dict[str, str]] = []
    backbone = {"base", "fold_teacher", "sol_teacher", "thermo_teacher"}
    historical_student = {
        "conditional": "prollama_multiteacher_student",
        "unconditional": "protgpt2_multiteacher_student",
    }[paradigm]
    paired_models = {"reference_theta0", "online_opd_paired", "prodipa_offline"}
    for item in main:
        model = item["Model"].strip()
        if model in paired_models:
            continue
        if model in backbone:
            group = "Backbone Model and Attribute Teachers"
        elif model == historical_student:
            group = "OPD and Offline Distillation Baselines"
        else:
            group = "External Protein Design Baselines"
        complete = [
            row for row in by_model.get(model, [])
            if row.get("valid_complete", "").lower() == "true"
        ]
        metrics = {
            "PPL": number(item.get("PPL")),
            "Novelty-U": mean_metric(complete, "novelty_u_percent"),
            "Novelty-T": mean_metric(complete, "novelty_t_percent"),
            "ProTrek %": number(item.get("ProTrek %")),
            "pLDDT": number(item.get("pLDDT")),
            "pAE": number(item.get("pAE")),
            "Sol": number(item.get("Sol")),
            "Thermo": number(item.get("Thermo")),
            "HV": number(item.get("HV")),
        }
        rows.append(table_row(group, model, metrics))
    return rows


def merge_into_canonical_main(
    root: Path,
    paradigm: str,
    metrics: dict[str, dict[str, float | int | None]],
) -> None:
    path = root / "outputs/evaluation" / paradigm / "results" / f"{paradigm}_prodipa_main.tsv"
    existing = read_tsv(path)
    model_ids = ("reference_theta0", "online_opd_paired", "prodipa_offline")
    existing = [row for row in existing if row.get("Model") not in model_ids]
    if paradigm == "conditional":
        fields = ["Model", "PPL", "ProTrek %", "pLDDT", "pAE", "Sol", "Thermo", "HV"]
    else:
        fields = [
            "Model", "PPL", "Novelty-U", "Novelty-T", "pLDDT", "%>70",
            "pAE", "%<10", "Sol", "Thermo", "HV",
        ]
    for model in model_ids:
        values = metrics[model]
        row = {field: "--" for field in fields}
        row["Model"] = model
        for field in fields[1:]:
            row[field] = fmt(number(values.get(field)))
        existing.append(row)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "--") for field in fields} for row in existing)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment_root.resolve()
    root = experiment.parents[2]
    paper = root / "outputs/evaluation/paper"
    paper.mkdir(parents=True, exist_ok=True)

    manifests = {
        paradigm: json.loads(
            (root / "outputs/evaluation" / paradigm / "results" / f"{paradigm}_hv_manifest.json").read_text()
        )
        for paradigm in ("conditional", "unconditional")
    }
    novelty = {
        paradigm: novelty_map(experiment / "evaluation/novelty" / f"{paradigm}_novelty_u.tsv")
        for paradigm in ("conditional", "unconditional")
    }
    score_paths = {
        "conditional": {
            "reference_theta0": root / "outputs/experiments/offline_multiteacher_opd/evaluation/methods/reference_theta0/results/sequence_scores.tsv",
            "online_opd_paired": experiment / "evaluation/methods/conditional/online_opd_paired/results/sequence_scores.tsv",
            "prodipa_offline": experiment / "evaluation/methods/conditional/prodipa_offline/results/sequence_scores.tsv",
        },
        "unconditional": {
            model: experiment / "evaluation/methods/unconditional" / model / "results/sequence_scores.tsv"
            for model in ("reference_theta0", "online_opd_paired", "prodipa_offline")
        },
    }

    metrics: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for paradigm in ("conditional", "unconditional"):
        metrics[paradigm] = {}
        for model, path in score_paths[paradigm].items():
            rows = read_tsv(path)
            add_novelty(rows, novelty[paradigm])
            metrics[paradigm][model] = summarize_rows(rows, manifests[paradigm], paradigm)

    cache_paths = {
        "conditional": [
            root / "outputs/experiments/offline_multiteacher_opd/cache/shard_0.json",
            root / "outputs/experiments/offline_multiteacher_opd/cache/shard_1.json",
        ],
        "unconditional": [
            experiment / "unconditional/cache/shard_0.json",
            experiment / "unconditional/cache/shard_1.json",
        ],
    }
    costs: dict[str, dict[str, float]] = {}
    for paradigm in ("conditional", "unconditional"):
        training_root = experiment / paradigm / "training"
        online = allocated_gpu_hours([
            training_root / "online/timings/rank_0.json",
            training_root / "online/timings/rank_1.json",
        ])
        offline_training = allocated_gpu_hours([
            training_root / "offline/timings/rank_0.json",
            training_root / "offline/timings/rank_1.json",
        ])
        cache = allocated_gpu_hours(cache_paths[paradigm])
        offline = cache + offline_training
        costs[paradigm] = {
            "online_gpu_hours": online,
            "offline_cache_gpu_hours": cache,
            "offline_student_gpu_hours": offline_training,
            "offline_gpu_hours": offline,
            "speedup": online / offline,
            "compute_saving_percent": 100.0 * (1.0 - offline / online),
        }

    comparison: dict[str, dict[str, float | str | None]] = {}
    for paradigm in ("conditional", "unconditional"):
        ref = float(metrics[paradigm]["reference_theta0"]["HV"])
        online = float(metrics[paradigm]["online_opd_paired"]["HV"])
        offline = float(metrics[paradigm]["prodipa_offline"]["HV"])
        gain = online - ref
        retention = 100.0 * (offline - ref) / gain if gain > 1e-6 else None
        retention_interpretation = (
            "positive_online_gain"
            if gain > 1e-6
            else "undefined_negative_online_gain"
            if gain < -1e-6
            else "undefined_near_zero_online_gain"
        )
        comparison[paradigm] = {
            "hv_reference": ref,
            "hv_online": online,
            "hv_offline": offline,
            "hv_offline_minus_online": offline - online,
            "hv_online_gain": gain,
            "hv_retention_percent": retention,
            "hv_retention_interpretation": retention_interpretation,
        }

    display = {
        "reference_theta0": "Reference θ0",
        "online_opd_paired": "Online OPD (paired)",
        "prodipa_offline": "ProDiPA (Offline)",
    }
    for paradigm in ("conditional", "unconditional"):
        rows = canonical_rows(root, paradigm)
        comp = comparison[paradigm]
        ref_retention = 0.0 if comp["hv_retention_percent"] is not None else None
        online_retention = 100.0 if comp["hv_retention_percent"] is not None else None
        rows.extend([
            table_row(
                "OPD and Offline Distillation Baselines",
                display["reference_theta0"],
                metrics[paradigm]["reference_theta0"],
                retention=ref_retention,
            ),
            table_row(
                "OPD and Offline Distillation Baselines",
                display["online_opd_paired"],
                metrics[paradigm]["online_opd_paired"],
                retention=online_retention,
                gpu_hours=costs[paradigm]["online_gpu_hours"],
            ),
            table_row(
                "OPD and Offline Distillation Baselines",
                "Semi-OPD",
                {},
            ),
            table_row(
                "Our Method",
                display["prodipa_offline"],
                metrics[paradigm]["prodipa_offline"],
                retention=number(comp["hv_retention_percent"]),
                gpu_hours=costs[paradigm]["offline_gpu_hours"],
            ),
        ])
        group_order = {
            "Backbone Model and Attribute Teachers": 0,
            "External Protein Design Baselines": 1,
            "OPD and Offline Distillation Baselines": 2,
            "Our Method": 3,
        }
        rows.sort(key=lambda row: group_order[row["Group"]])
        write_tsv(paper / f"{paradigm}_main.tsv", rows)
        merge_into_canonical_main(root, paradigm, metrics[paradigm])

    accounting = {
        "scope": "paired continuation from a shared recoverable checkpoint; excludes common pre-checkpoint training",
        "conditional": {**costs["conditional"], **comparison["conditional"]},
        "unconditional": {**costs["unconditional"], **comparison["unconditional"]},
        "evaluation_cost": "recorded separately under the experiment evaluation/timings directory and excluded from training GPU-hours",
    }
    (paper / "offline_full_run_metrics.json").write_text(
        json.dumps({"metrics": metrics, "accounting": accounting}, indent=2) + "\n"
    )

    lines = [
        "# Offline Multi-Teacher OPD — full paired experiment",
        "",
        "Both paradigms use a common recoverable checkpoint for each Online/Offline pair. "
        "The reported GPU-hours are paired continuation costs, not end-to-end costs from the base model. "
        "Evaluation generation and scorer costs are recorded separately and are excluded from training cost.",
        "",
        "| Paradigm | Method | HV | HV Retention (%) | GPU-hours | PPL | Novelty-U | pLDDT | pAE | Sol | Thermo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for paradigm in ("conditional", "unconditional"):
        comp = comparison[paradigm]
        values = [
            ("reference_theta0", 0.0 if comp["hv_retention_percent"] is not None else None, None),
            ("online_opd_paired", 100.0 if comp["hv_retention_percent"] is not None else None, costs[paradigm]["online_gpu_hours"]),
            ("prodipa_offline", number(comp["hv_retention_percent"]), costs[paradigm]["offline_gpu_hours"]),
        ]
        for model, retention, gpu_hours in values:
            m = metrics[paradigm][model]
            lines.append(
                f"| {paradigm.title()} | {display[model]} | {fmt(number(m['HV']))} | "
                f"{fmt(retention, 6) if retention is not None else 'N/A'} | {fmt(gpu_hours, 6)} | {fmt(number(m['PPL']))} | "
                f"{fmt(number(m['Novelty-U']))} | {fmt(number(m['pLDDT']))} | "
                f"{fmt(number(m['pAE']))} | {fmt(number(m['Sol']))} | {fmt(number(m['Thermo']))} |"
            )
    lines.extend(["", "## Compute and protocol", ""])
    for paradigm, updates, start in (("conditional", 64, 192), ("unconditional", 56, 200)):
        c = costs[paradigm]
        comp = comparison[paradigm]
        retention_text = fmt(number(comp["hv_retention_percent"]), 6) if comp["hv_retention_percent"] is not None else "N/A"
        lines.extend([
            f"- **{paradigm.title()}**: formal endpoint 256, continuation {start}→256 "
            f"({updates} student updates); fixed cache 512 trajectories; evaluation 512 independent free-generation requests per method.",
            f"  Online={c['online_gpu_hours']:.6g} GPU-h; Offline={c['offline_gpu_hours']:.6g} GPU-h "
            f"(cache={c['offline_cache_gpu_hours']:.6g}, student={c['offline_student_gpu_hours']:.6g}); "
            f"speedup={c['speedup']:.6g}×; saving={c['compute_saving_percent']:.6g}%; "
            f"HV retention={retention_text}; Offline−Online HV={float(comp['hv_offline_minus_online']):.8g}.",
        ])
        if float(comp["hv_online_gain"]) < -1e-6:
            lines.append(
                "  Because Online−Reference HV is negative, HV Retention is N/A. "
                "The negative-denominator ratio is not interpreted as retained positive gain."
            )
    lines.extend([
        "",
        "The Conditional HV uses the frozen canonical five-objective scale "
        "[-log(1+PPL), ProTrek %, pLDDT, Sol, Thermo]. Novelty-U is reported separately and is not ProTrek. "
        "The Unconditional HV uses the frozen canonical scale [-log(1+PPL), Novelty-U, pLDDT, Sol, Thermo]. "
        "No min-max range was refit using the paired methods.",
        "",
        "## Conclusion",
        "",
    ])
    conclusions = []
    for paradigm in ("conditional", "unconditional"):
        retention = number(comparison[paradigm]["hv_retention_percent"])
        online_gain = float(comparison[paradigm]["hv_online_gain"])
        if retention is None:
            conclusions.append(
                f"{paradigm.title()}: Online did not provide a meaningful positive HV gain over θ0; HV Retention is N/A."
            )
        elif online_gain < 0.0:
            conclusions.append(
                f"{paradigm.title()}: the signed ΔHV ratio is {retention:.6g}%, but both Online and "
                "Offline underperform θ0, so this is not positive-gain retention."
            )
        elif retention >= 90.0 and costs[paradigm]["speedup"] > 1.0:
            conclusions.append(
                f"{paradigm.title()}: the fixed cache retained the main HV gain while reducing paired continuation compute."
            )
        else:
            conclusions.append(
                f"{paradigm.title()}: the fixed cache did not jointly meet the 90% HV-retention and >1× speedup criterion."
            )
    lines.extend([f"- {item}" for item in conclusions])
    lines.append("")
    lines.append(
        "Core answer: fixed trajectories are sufficient for effective lower-cost distillation only in paradigms "
        "that meet both the measured retention and compute criteria above; the two rows provide the direct empirical test."
    )
    (paper / "offline_full_run_conclusion.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
