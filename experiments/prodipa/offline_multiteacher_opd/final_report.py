#!/usr/bin/env python3
"""Summarize the paired Offline/Online OPD experiment from measured artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


MODEL_ORDER = ("reference_theta0", "online_opd", "offline_opd")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sum_rank_seconds(root: Path) -> tuple[float, list[dict]]:
    records = [read_json(path) for path in sorted((root / "timings").glob("rank_*.json"))]
    if len(records) != 2 or any(item.get("status") != "COMPLETE" for item in records):
        raise ValueError(f"incomplete two-rank timing evidence: {root}")
    return sum(float(item["allocated_gpu_seconds"]) for item in records), records


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}g}"


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    results = root / "results"
    table_rows = {row["Model"]: row for row in read_tsv(results / "conditional_prodipa_main.tsv")}
    if set(table_rows) != set(MODEL_ORDER):
        raise ValueError(f"paired HV table has wrong models: {sorted(table_rows)}")
    hv = {model: float(table_rows[model]["HV"]) for model in MODEL_ORDER}
    sequence_rows = read_tsv(results / "conditional_sequence_scores.tsv")
    novelty_u = {}
    for model in MODEL_ORDER:
        values = [
            float(row["novelty_u_percent"])
            for row in sequence_rows
            if row["model"] == model
            and row.get("valid_complete", "").lower() == "true"
            and row.get("novelty_u_percent") not in (None, "")
        ]
        if not values:
            raise ValueError(f"Novelty-U is missing for {model}")
        novelty_u[model] = statistics.fmean(values)
    online_gain = hv["online_opd"] - hv["reference_theta0"]
    offline_gain = hv["offline_opd"] - hv["reference_theta0"]
    retention = 100.0 * offline_gain / online_gain if online_gain > 1e-8 else None

    online_seconds, online_ranks = sum_rank_seconds(root / "training/online")
    offline_student_seconds, offline_ranks = sum_rank_seconds(root / "training/offline")
    online_updates = {int(item["student_updates"]) for item in online_ranks}
    offline_updates = {int(item["student_updates"]) for item in offline_ranks}
    if online_updates != {64} or offline_updates != {64}:
        raise ValueError(
            f"paired update budget mismatch: online={online_updates}, offline={offline_updates}"
        )
    cache_reports = [read_json(path) for path in sorted((root / "cache").glob("shard_*.json"))]
    if len(cache_reports) != 2 or any(item.get("status") != "COMPLETE" for item in cache_reports):
        raise ValueError("cache timing evidence is incomplete")
    cache_seconds = sum(float(item["allocated_gpu_seconds"]) for item in cache_reports)
    online_gpu_hours = online_seconds / 3600.0
    offline_gpu_hours = (cache_seconds + offline_student_seconds) / 3600.0
    speedup = online_gpu_hours / offline_gpu_hours
    saving = 100.0 * (1.0 - offline_gpu_hours / online_gpu_hours)
    cache_manifest = read_json(root / "cache/CACHE_MANIFEST.json")
    consistency = read_json(root / "cache/CONSISTENCY_AUDIT.json")

    metric_rows = []
    for model in MODEL_ORDER:
        row = table_rows[model]
        if model == "reference_theta0":
            retention_value = "0" if retention is not None else "N/A"
            gpu_hours = ""
        elif model == "online_opd":
            retention_value = "100" if retention is not None else "N/A"
            gpu_hours = fmt(online_gpu_hours)
        else:
            retention_value = fmt(retention)
            gpu_hours = fmt(offline_gpu_hours)
        metric_rows.append({
            "Metric model": model,
            "HV": row["HV"],
            "HV Retention (%)": retention_value,
            "GPU-hours": gpu_hours,
            "PPL": row["PPL"],
            "Novelty-U": fmt(novelty_u[model]),
            "ProTrek %": row["ProTrek %"],
            "pLDDT": row["pLDDT"],
            "pAE": row["pAE"],
            "Sol": row["Sol"],
            "Thermo": row["Thermo"],
        })
    fields = list(metric_rows[0])
    write_tsv(results / "core_metrics.tsv", metric_rows, fields)

    accounting = {
        "status": "COMPLETE",
        "definition": "sum of actual per-rank allocated GPU wall seconds; evaluation excluded",
        "online": {
            "gpu_seconds": online_seconds,
            "gpu_hours": online_gpu_hours,
            "rank_records": online_ranks,
        },
        "offline": {
            "cache_gpu_seconds": cache_seconds,
            "student_gpu_seconds": offline_student_seconds,
            "total_gpu_seconds": cache_seconds + offline_student_seconds,
            "gpu_hours": offline_gpu_hours,
            "cache_rank_records": cache_reports,
            "student_rank_records": offline_ranks,
        },
        "speedup": speedup,
        "compute_saving_percent": saving,
        "evaluation": read_json(root / "evaluation/EVALUATION_TIMING.json"),
    }
    (results / "compute_accounting.json").write_text(json.dumps(accounting, indent=2) + "\n")
    conclusion = (
        "Offline retained the online HV improvement with lower measured distillation cost."
        if retention is not None and retention >= 90 and saving > 0
        else "The first core experiment does not establish both strong HV retention and lower distillation cost."
    )
    retention_text = fmt(retention)
    report = f"""# Offline Multi-Teacher OPD — First Core Validation

## Result

| Metric | Reference | Online OPD | Offline OPD |
| --- | ---: | ---: | ---: |
| HV | {table_rows['reference_theta0']['HV']} | {table_rows['online_opd']['HV']} | {table_rows['offline_opd']['HV']} |
| HV Retention (%) | {'0' if retention is not None else 'N/A'} | {'100' if retention is not None else 'N/A'} | {retention_text} |
| GPU-hours | — | {fmt(online_gpu_hours)} | {fmt(offline_gpu_hours)} |
| PPL | {table_rows['reference_theta0']['PPL']} | {table_rows['online_opd']['PPL']} | {table_rows['offline_opd']['PPL']} |
| Novelty-U | {fmt(novelty_u['reference_theta0'])} | {fmt(novelty_u['online_opd'])} | {fmt(novelty_u['offline_opd'])} |
| ProTrek % | {table_rows['reference_theta0']['ProTrek %']} | {table_rows['online_opd']['ProTrek %']} | {table_rows['offline_opd']['ProTrek %']} |
| pLDDT | {table_rows['reference_theta0']['pLDDT']} | {table_rows['online_opd']['pLDDT']} | {table_rows['offline_opd']['pLDDT']} |
| pAE | {table_rows['reference_theta0']['pAE']} | {table_rows['online_opd']['pAE']} | {table_rows['offline_opd']['pAE']} |
| Sol | {table_rows['reference_theta0']['Sol']} | {table_rows['online_opd']['Sol']} | {table_rows['offline_opd']['Sol']} |
| Thermo | {table_rows['reference_theta0']['Thermo']} | {table_rows['online_opd']['Thermo']} | {table_rows['offline_opd']['Thermo']} |

HV values use one pooled three-model min-max normalization and the same zero reference point.
The absolute Offline–Online HV difference is {fmt(hv['offline_opd'] - hv['online_opd'])}.
{'Online HV improved over theta-zero, so Retention is defined.' if retention is not None else 'Online HV did not clearly improve over theta-zero, so Retention is N/A.'}

## Compute

- Online distillation: {fmt(online_gpu_hours)} GPU-h.
- Offline cache construction plus student training: {fmt(offline_gpu_hours)} GPU-h.
- Measured speedup: {fmt(speedup)}×.
- Compute saving: {fmt(saving)}%.
- Evaluation compute is recorded separately and excluded from both values.

Both methods used {next(iter(online_updates))} measured student updates. Online used 2,048 changing-policy trajectories.
Offline cached {cache_manifest['sample_count']} theta-zero trajectories and reused them for
{cache_manifest['reuse_epochs']} epochs. Each model used 512 independent evaluation requests.

## Consistency gate

The cache/live audit status is **{consistency['status']}**. Maximum full-vocabulary target
log-probability difference: {consistency['max_abs_target_logprob_diff']}; maximum loss
difference on the same batch: {consistency['max_absolute_loss_diff']}.

## Core conclusion

{conclusion}
"""
    (results / "FINAL_REPORT.md").write_text(report)
    print(results / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
