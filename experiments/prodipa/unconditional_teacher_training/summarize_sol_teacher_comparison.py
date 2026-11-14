#!/usr/bin/env python3
"""Summarize the independent old/new Sol Prefix-teacher comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
    except ValueError:
        return None
    return result if result is not None and math.isfinite(result) else None


def q(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--sol", type=Path, required=True)
    parser.add_argument("--thermo", type=Path, required=True)
    parser.add_argument("--endpoint-eos", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    structure = {row["sequence_id"]: row for row in read_tsv(args.structure)}
    sol = {row["sequence_id"]: row for row in read_tsv(args.sol)}
    thermo = {row["sequence_id"]: row for row in read_tsv(args.thermo)}
    endpoint_rows = read_tsv(args.endpoint_eos)
    endpoint_native = {
        row["model"]: row for row in endpoint_rows if row["native_protocol"].lower() == "true"
    }

    result_rows = []
    evidence_rows = []
    for model in ("old_sol_teacher", "new_sol_teacher"):
        requests = read_tsv(args.generation_dir / f"{model}.tsv")
        valid = [row for row in requests if row["valid_complete"].lower() == "true"]
        natural = [row for row in requests if row["natural_eos_reached"].lower() == "true"]
        forced = [row for row in requests if row["length_forced_eos"].lower() == "true"]
        lengths_all = [float(row["length_aa"]) for row in requests]
        lengths_valid = [float(row["length_aa"]) for row in valid]
        metrics = {"plddt": [], "pae": [], "sol": [], "thermo": []}
        for row in valid:
            sid = "sha256:" + __import__("hashlib").sha256(row["sequence"].encode("ascii")).hexdigest()
            values = {
                "plddt": finite(structure.get(sid, {}).get("fold_plddt_100")),
                "pae": finite(structure.get(sid, {}).get("fold_mean_pae")),
                "sol": finite(sol.get(sid, {}).get("solubility_scaled_01")),
                "thermo": finite(thermo.get(sid, {}).get("thermophilicity_score_01")),
            }
            for key, value in values.items():
                if value is not None:
                    metrics[key].append(value)
            evidence_rows.append({
                "model": model,
                "sample_id": row["sample_id"],
                "sequence_id": sid,
                "length_aa": row["length_aa"],
                "natural_eos": row["natural_eos_reached"],
                "forced_eos": row["length_forced_eos"],
                **{key: fmt(value) for key, value in values.items()},
            })
        endpoint = endpoint_native[model]
        result_rows.append({
            "Model": model,
            "Requests": len(requests),
            "Natural EOS": len(natural),
            "Natural EOS rate": len(natural) / len(requests),
            "Forced/censored": len(forced),
            "Forced/censored rate": len(forced) / len(requests),
            "Valid complete 50-512 aa": len(valid),
            "Valid complete rate": len(valid) / len(requests),
            "Empty": sum(int(row["length_aa"]) == 0 for row in requests),
            "Under 50 aa": sum(int(row["length_aa"]) < 50 for row in requests),
            "All length mean": statistics.fmean(lengths_all),
            "Valid length q05": q(lengths_valid, 0.05),
            "Valid length q25": q(lengths_valid, 0.25),
            "Valid length median": q(lengths_valid, 0.5),
            "Valid length q75": q(lengths_valid, 0.75),
            "Valid length q95": q(lengths_valid, 0.95),
            "Endpoint EOS probability": float(endpoint["eos_probability_mean"]),
            "Endpoint EOS median rank": float(endpoint["eos_rank_median"]),
            "Endpoint EOS reachable rate": float(endpoint["eos_reachable_rate"]),
            "Sol": statistics.fmean(metrics["sol"]) if metrics["sol"] else None,
            "pLDDT": statistics.fmean(metrics["plddt"]) if metrics["plddt"] else None,
            "pAE": statistics.fmean(metrics["pae"]) if metrics["pae"] else None,
            "Thermo": statistics.fmean(metrics["thermo"]) if metrics["thermo"] else None,
            "Scored N": min(len(values) for values in metrics.values()),
        })

    comparison_path = args.out_dir / "sol_teacher_comparison.tsv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, result_rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: fmt(value) for key, value in row.items()} for row in result_rows)
    with (args.out_dir / "scored_sequences.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, evidence_rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence_rows)

    old, new = result_rows
    deltas = {
        key: new[key] - old[key]
        for key in ("Natural EOS rate", "Valid complete rate", "Endpoint EOS reachable rate", "Sol", "pLDDT", "pAE", "Thermo")
        if new[key] is not None and old[key] is not None
    }
    payload = {
        "status": "COMPLETE",
        "models": result_rows,
        "new_minus_old": deltas,
        "protocol": {
            "requests_per_model": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 200,
            "repetition_penalty": 1.2,
            "max_new_tokens": 512,
            "generation_max_aa": 512,
            "sequence_format": "legacy",
            "forced_terminal_counts_as_complete": False,
        },
    }
    (args.out_dir / "COMPARISON_RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = [
        "# Balanced Sol Prefix teacher comparison",
        "",
        "Independent generation uses identical seeds and the frozen Legacy sampling settings. "
        "The 512-aa processor is a safety censor; forced endpoints are excluded from valid-complete counts.",
        "",
        "| Model | Natural EOS | Forced | Valid 50–512 aa | Endpoint EOS reachable | Sol | pLDDT | pAE | Thermo |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        report.append(
            f"| {row['Model']} | {row['Natural EOS']}/{row['Requests']} "
            f"({100*row['Natural EOS rate']:.2f}%) | {row['Forced/censored']} "
            f"({100*row['Forced/censored rate']:.2f}%) | "
            f"{row['Valid complete 50-512 aa']}/{row['Requests']} "
            f"({100*row['Valid complete rate']:.2f}%) | "
            f"{100*row['Endpoint EOS reachable rate']:.2f}% | "
            f"{row['Sol']:.4f} | {row['pLDDT']:.3f} | {row['pAE']:.3f} | {row['Thermo']:.4f} |"
        )
    report.extend([
        "",
        "Endpoint reachability is measured at the true endpoint of the same held-out balanced Sol set. "
        "Each model's native training protocol is used for the headline comparison; the cross-protocol "
        "matrix remains available in `endpoint_eos_summary.tsv`.",
    ])
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
