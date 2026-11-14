#!/usr/bin/env python3
"""Create canonical main tables and ProDiPA supplementary evaluation tables."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def optional_index(path: Path | None, key: str = "sequence_id") -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    return {row[key]: row for row in read(path)}


def number(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.8g}"


def write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_difference(a: list[float], b: list[float], seed: int, iterations: int = 2000) -> tuple[float, float, float]:
    rng = random.Random(seed)
    observed = statistics.fmean(a) - statistics.fmean(b)
    samples = []
    for _ in range(iterations):
        aa = [a[rng.randrange(len(a))] for _ in a]
        bb = [b[rng.randrange(len(b))] for _ in b]
        samples.append(statistics.fmean(aa) - statistics.fmean(bb))
    samples.sort()
    return observed, samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--sol", type=Path, required=True)
    parser.add_argument("--thermo", type=Path, required=True)
    parser.add_argument("--ppl", type=Path, required=True)
    parser.add_argument("--novelty-t-unconditional", type=Path)
    parser.add_argument("--novelty-t-conditional", type=Path)
    parser.add_argument("--novelty-u", type=Path)
    parser.add_argument("--protrek", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--fold-threshold", type=float, default=85.0)
    parser.add_argument("--sol-threshold", type=float, default=0.7)
    parser.add_argument("--thermo-threshold", type=float, default=0.6)
    args = parser.parse_args()

    requests = read(args.requests)
    structure = optional_index(args.structure)
    sol = optional_index(args.sol)
    thermo = optional_index(args.thermo)
    ppl = optional_index(args.ppl)
    novelty_t = {
        "unconditional": optional_index(args.novelty_t_unconditional),
        "conditional": optional_index(args.novelty_t_conditional),
    }
    novelty_u = optional_index(args.novelty_u)
    protrek = optional_index(args.protrek)

    scored = []
    for request in requests:
        sid = request["sequence_id"]
        row = dict(request)
        row.update({
            "ppl": ppl.get(sid, {}).get("ppl", ""),
            "novelty_u_percent": novelty_u.get(sid, {}).get("novelty_percent", ""),
            "novelty_t_percent": novelty_t[request["paradigm"]].get(sid, {}).get("novelty_percent", ""),
            "plddt": structure.get(sid, {}).get("fold_plddt_100", ""),
            "pae": structure.get(sid, {}).get("fold_mean_pae", ""),
            "ptm": structure.get(sid, {}).get("fold_ptm_01", ""),
            "sol": sol.get(sid, {}).get("solubility_scaled_01", ""),
            "thermo": thermo.get(sid, {}).get("thermophilicity_score_01", ""),
            "protrek_percent": protrek.get(sid, {}).get("protrek_percent", ""),
        })
        scored.append(row)
    score_fields = list(scored[0])
    write(args.out_root / "sequence_scores.tsv", scored, score_fields)

    groups = {}
    for row in scored:
        groups.setdefault((row["paradigm"], row["model"]), []).append(row)

    main_rows = {"unconditional": [], "conditional": []}
    supplement = []
    completeness = []
    metric_keys = ("ppl", "novelty_u_percent", "novelty_t_percent", "plddt", "pae", "sol", "thermo", "protrek_percent")
    for (paradigm, model), rows in sorted(groups.items()):
        valid = [row for row in rows if row["valid_complete"].lower() == "true"]
        values = {key: [x for row in valid if (x := number(row[key])) is not None] for key in metric_keys}
        fold70 = [x for x in values["plddt"] if x > 70]
        pae10 = [x for x in values["pae"] if x < 10]
        joint = [
            row for row in rows
            if row["valid_complete"].lower() == "true"
            and (number(row["plddt"]) or -math.inf) >= args.fold_threshold
            and (number(row["sol"]) or -math.inf) >= args.sol_threshold
            and (number(row["thermo"]) or -math.inf) >= args.thermo_threshold
        ]
        base = {
            "Model": model,
            "PPL": fmt(mean(values["ppl"])),
            "pLDDT": fmt(mean(values["plddt"])),
            "pAE": fmt(mean(values["pae"])),
            "Sol": fmt(mean(values["sol"])),
            "Thermo": fmt(mean(values["thermo"])),
            "HV": "",
        }
        if paradigm == "unconditional":
            base.update({
                "Novelty-U": fmt(mean(values["novelty_u_percent"])),
                "Novelty-T": fmt(mean(values["novelty_t_percent"])),
                "%>70": fmt(100 * len(fold70) / len(values["plddt"]) if values["plddt"] else None),
                "%<10": fmt(100 * len(pae10) / len(values["pae"]) if values["pae"] else None),
            })
        else:
            base["ProTrek %"] = fmt(mean(values["protrek_percent"]))
        main_rows[paradigm].append(base)
        lengths = [int(row["length_aa"]) for row in valid]
        supplement.append({
            "Paradigm": paradigm,
            "Model": model,
            "Requests": len(rows),
            "Valid Rate": fmt(len(valid) / len(rows)),
            "EOS Rate": fmt(sum(row["eos_reached"].lower() == "true" for row in rows) / len(rows)),
            "Strict JSR": fmt(len(joint) / len(rows)),
            "Mean Length": fmt(mean(lengths)),
            "Median Length": fmt(float(statistics.median(lengths)) if lengths else None),
        })
        completeness.append({
            "Paradigm": paradigm, "Model": model, "Requests": len(rows), "Valid complete": len(valid),
            **{f"N {key}": len(values[key]) for key in metric_keys},
        })

    uncond_fields = ["Model", "PPL", "Novelty-U", "Novelty-T", "pLDDT", "%>70", "pAE", "%<10", "Sol", "Thermo", "HV"]
    cond_fields = ["Model", "PPL", "ProTrek %", "pLDDT", "pAE", "Sol", "Thermo", "HV"]
    write(args.out_root / "unconditional_prodipa_main.tsv", main_rows["unconditional"], uncond_fields)
    write(args.out_root / "conditional_prodipa_main.tsv", main_rows["conditional"], cond_fields)
    write(args.out_root / "supplementary_metrics.tsv", supplement, list(supplement[0]))
    write(args.out_root / "scoring_completeness.tsv", completeness, list(completeness[0]))

    contrast_rows = []
    for paradigm in ("unconditional", "conditional"):
        if (paradigm, "base") not in groups:
            continue
        base_rows = groups[(paradigm, "base")]
        for model in ("fold_teacher", "sol_teacher", "thermo_teacher"):
            if (paradigm, model) not in groups:
                continue
            teacher_rows = groups[(paradigm, model)]
            for metric in ("plddt", "sol", "thermo"):
                a = [x for row in teacher_rows if (x := number(row[metric])) is not None]
                b = [x for row in base_rows if (x := number(row[metric])) is not None]
                if not a or not b:
                    continue
                observed, low, high = bootstrap_difference(a, b, seed=42 + len(contrast_rows))
                contrast_rows.append({
                    "Paradigm": paradigm, "Teacher": model, "Metric": metric,
                    "Gain": fmt(observed), "CI low": fmt(low), "CI high": fmt(high),
                    "Teacher N": len(a), "Base N": len(b), "Bootstrap iterations": 2000,
                })
    write(args.out_root / "teacher_vs_base_bootstrap.tsv", contrast_rows, list(contrast_rows[0]) if contrast_rows else ["Paradigm"])

    missing = []
    if not novelty_u:
        missing.append("Novelty-U: frozen full UniProtKB reference is unavailable")
    if not protrek:
        missing.append("ProTrek: local official ProTrek model/runtime is unavailable")
    missing.append("HV: deferred to pooled cross-model completion after frozen UniProtKB Novelty-U is available")
    status = (
        "# Formal teacher evaluation status\n\n"
        f"Generated requests represented: {len(requests)}.\n\n"
        "The two main tables use successfully scored, valid-complete sequences for continuous metrics. "
        "Strict JSR, Valid Rate, and EOS Rate use all requests as denominator.\n\n"
        "## Pending official dependencies or definitions\n\n"
        + "\n".join(f"- {item}" for item in missing) + "\n"
    )
    (args.out_root / "EVALUATION_STATUS.md").write_text(status, encoding="utf-8")


if __name__ == "__main__":
    main()
