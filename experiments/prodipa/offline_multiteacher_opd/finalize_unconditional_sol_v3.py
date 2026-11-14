#!/usr/bin/env python3
"""Publish the Sol-V3-teacher Unconditional Innovation-I paired experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from finalize_full_paper import (
    FIELDS,
    add_novelty,
    allocated_gpu_hours,
    novelty_map,
    number,
    read_tsv,
    summarize_rows,
)


MODEL_IDS = ("reference_theta0", "online_opd_paired", "prodipa_offline")
PAPER_NAMES = {
    "reference_theta0": "Reference θ0",
    "online_opd_paired": "Online OPD (paired)",
    "prodipa_offline": "ProDiPA (Offline)",
}


def fmt(value: float | None, digits: int = 8) -> str:
    return "--" if value is None else f"{value:.{digits}g}"


def atomic_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "--") for field in fields} for row in rows)
        temporary = handle.name
    os.replace(temporary, path)


def canonical_row(model: str, values: dict[str, float | int | None]) -> dict[str, str]:
    fields = [
        "Model", "PPL", "Novelty-U", "Novelty-T", "pLDDT", "%>70",
        "pAE", "%<10", "Sol", "Thermo", "HV",
    ]
    row = {field: "--" for field in fields}
    row["Model"] = model
    for field in fields[1:]:
        row[field] = fmt(number(values.get(field)))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    experiment = args.experiment_root.resolve()

    manifest_path = experiment / "FROZEN_HV_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing preflight-frozen HV manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    novelty_path = experiment / "evaluation/novelty/unconditional_novelty_u.tsv"
    novelty = novelty_map(novelty_path)

    metrics: dict[str, dict[str, float | int | None]] = {}
    for model in MODEL_IDS:
        score_path = experiment / "evaluation/methods" / model / "results/sequence_scores.tsv"
        rows = read_tsv(score_path)
        add_novelty(rows, novelty)
        metrics[model] = summarize_rows(rows, manifest, "unconditional")

    train_root = experiment / "training"
    online_gpu_h = allocated_gpu_hours([
        train_root / "online/timings/rank_0.json",
        train_root / "online/timings/rank_1.json",
    ])
    offline_student_gpu_h = allocated_gpu_hours([
        train_root / "offline/timings/rank_0.json",
        train_root / "offline/timings/rank_1.json",
    ])
    cache_gpu_h = allocated_gpu_hours([
        experiment / "cache/shard_0.json",
        experiment / "cache/shard_1.json",
    ])
    offline_gpu_h = cache_gpu_h + offline_student_gpu_h
    speedup = online_gpu_h / offline_gpu_h
    saving = 100.0 * (1.0 - offline_gpu_h / online_gpu_h)

    reference_hv = float(metrics["reference_theta0"]["HV"])
    online_hv = float(metrics["online_opd_paired"]["HV"])
    offline_hv = float(metrics["prodipa_offline"]["HV"])
    online_gain = online_hv - reference_hv
    retention = (
        100.0 * (offline_hv - reference_hv) / online_gain
        if online_gain > 1e-6 else None
    )

    result_fields = [
        "Method", "Requests", "Valid complete", "PPL", "Novelty-U", "Novelty-T",
        "pLDDT", "pAE", "Sol", "Thermo", "HV", "HV Ret. (%)", "Train GPU-h",
    ]
    result_rows: list[dict[str, str]] = []
    for model in MODEL_IDS:
        values = metrics[model]
        gpu_h = None
        if model == "online_opd_paired":
            gpu_h = online_gpu_h
        elif model == "prodipa_offline":
            gpu_h = offline_gpu_h
        row = {
            "Method": PAPER_NAMES[model],
            "Requests": str(values["requests"]),
            "Valid complete": str(values["complete"]),
            "HV Ret. (%)": (
                "0" if model == "reference_theta0" and retention is not None else
                "100" if model == "online_opd_paired" and retention is not None else
                fmt(retention, 6) if model == "prodipa_offline" and retention is not None else
                "N/A"
            ),
            "Train GPU-h": fmt(gpu_h, 6),
        }
        for field in ("PPL", "Novelty-U", "Novelty-T", "pLDDT", "pAE", "Sol", "Thermo", "HV"):
            row[field] = fmt(number(values.get(field)))
        result_rows.append(row)
    atomic_tsv(experiment / "results/core_metrics.tsv", result_fields, result_rows)

    canonical_path = root / "outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv"
    canonical = read_tsv(canonical_path)
    canonical = [row for row in canonical if row.get("Model", "").strip() not in MODEL_IDS]
    canonical.extend(canonical_row(model, metrics[model]) for model in MODEL_IDS)
    canonical_fields = [
        "Model", "PPL", "Novelty-U", "Novelty-T", "pLDDT", "%>70",
        "pAE", "%<10", "Sol", "Thermo", "HV",
    ]
    atomic_tsv(canonical_path, canonical_fields, canonical)

    paper_path = root / "outputs/evaluation/paper/unconditional_main.tsv"
    paper_fields, paper_rows = None, None
    with paper_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        paper_fields = list(reader.fieldnames or FIELDS)
        paper_rows = list(reader)
    by_name = {PAPER_NAMES[model]: model for model in MODEL_IDS}
    seen: set[str] = set()
    for row in paper_rows:
        model = by_name.get(row.get("Model", ""))
        if model is None:
            continue
        seen.add(model)
        values = metrics[model]
        for field in ("PPL", "Novelty-U", "Novelty-T", "pLDDT", "pAE", "Sol", "Thermo", "HV"):
            row[field] = fmt(number(values.get(field)))
        row["ProTrek %"] = "--"
        row["HV Ret. (%)"] = (
            "0" if model == "reference_theta0" and retention is not None else
            "100" if model == "online_opd_paired" and retention is not None else
            fmt(retention, 6) if model == "prodipa_offline" and retention is not None else
            "N/A"
        )
        row["Train GPU-h"] = (
            fmt(online_gpu_h, 6) if model == "online_opd_paired" else
            fmt(offline_gpu_h, 6) if model == "prodipa_offline" else "--"
        )
        row["HV-AUC"] = "--"
        row["HQ Disc."] = "--"
    if seen != set(MODEL_IDS):
        raise ValueError(f"paper table is missing paired rows: {set(MODEL_IDS) - seen}")
    atomic_tsv(paper_path, paper_fields, paper_rows)

    report = experiment / "results/FINAL_REPORT.md"
    report.write_text(
        "# Unconditional Offline Multi-Teacher OPD — Sol-V3 teacher paired run\n\n"
        "This is a paired continuation from the same recoverable step-200 checkpoint. "
        "GPU-hours exclude the shared warm start and formal evaluation.\n\n"
        "| Method | HV | HV Ret. (%) | GPU-h | PPL | Novelty-U | pLDDT | pAE | Sol | Thermo |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['Method']} | {row['HV']} | {row['HV Ret. (%)']} | {row['Train GPU-h']} | "
            f"{row['PPL']} | {row['Novelty-U']} | {row['pLDDT']} | {row['pAE']} | "
            f"{row['Sol']} | {row['Thermo']} |"
            for row in result_rows
        )
        + "\n\n"
        f"- Online continuation: {online_gpu_h:.6g} GPU-h.\n"
        f"- Offline total: {offline_gpu_h:.6g} GPU-h "
        f"(cache {cache_gpu_h:.6g}, student {offline_student_gpu_h:.6g}).\n"
        f"- Speedup: {speedup:.6g}×; compute saving: {saving:.6g}%.\n"
        f"- Offline−Online HV: {offline_hv - online_hv:.8g}.\n"
        + (
            f"- HV retention: {retention:.6g}%.\n"
            if retention is not None else
            "- HV retention: N/A because Online did not improve HV over Reference θ0.\n"
        ),
        encoding="utf-8",
    )
    paper_report = root / "outputs/evaluation/paper/offline_unconditional_sol_v3_conclusion.md"
    paper_report.write_text(report.read_text(), encoding="utf-8")

    summary = {
        "status": "COMPLETE",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "hv_manifest": str(manifest_path),
        "hv_manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest(),
        "metrics": metrics,
        "compute": {
            "online_gpu_hours": online_gpu_h,
            "offline_cache_gpu_hours": cache_gpu_h,
            "offline_student_gpu_hours": offline_student_gpu_h,
            "offline_gpu_hours": offline_gpu_h,
            "speedup": speedup,
            "saving_percent": saving,
            "hv_retention_percent": retention,
        },
    }
    (experiment / "results/FINAL_RESULTS.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
