#!/usr/bin/env python3
"""Publish the user-selected main-table HV objectives without rerunning scorers."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from moocore import hypervolume

from common import ROOT, dump, read, rows, sha, tsv
from selection import eligible


PAPER = ROOT / "outputs/evaluation/paper"
METHODS = ("fixed", "periodic", "adaptive")
DISPLAY = {
    "reference": "Reference θ0",
    "online": "Online OPD (paired)",
    "fixed": "Fixed Offline OPD",
    "periodic": "Periodic Cache OPD",
    "adaptive": "ProDiPA (Adaptive Cache)",
}
IDS = {
    "reference": "reference_theta0",
    "online": "online_opd_paired",
    "fixed": "prodipa_offline",
    "periodic": "periodic_cache_opd",
    "adaptive": "adaptive_cache_opd",
}
ALIASES = {label: IDS[method] for method, label in DISPLAY.items()}
ALIASES["ProDiPA (Offline)"] = IDS["fixed"]
ROOTS = {
    "conditional": ROOT / "outputs/experiments/adaptive_cache",
    "unconditional": ROOT / "outputs/experiments/adaptive_cache_unconditional_sol_v3",
}
SCORE_TABLES = {
    paradigm: ROOT / f"outputs/evaluation/{paradigm}/results/{paradigm}_sequence_scores.tsv"
    for paradigm in ROOTS
}
TABLES = {
    paradigm: [
        PAPER / f"{paradigm}_main.tsv",
        ROOT / f"outputs/evaluation/{paradigm}/results/{paradigm}_prodipa_main.tsv",
    ]
    for paradigm in ROOTS
}
PAE_MAX = 31.75


def make_specs() -> dict[str, dict]:
    conditional_source = ROOT / "outputs/evaluation/conditional/results/conditional_hv_manifest.json"
    unconditional_source = ROOTS["unconditional"] / "protocol/unconditional_hv.json"
    conditional = read(conditional_source)
    unconditional = read(unconditional_source)

    # Preserve every previously frozen shared-objective range. pAE uses the
    # fixed ESMFold output range rather than a cohort-dependent fitted range.
    specs = {
        "conditional": {
            "definition_source": "User-selected canonical Conditional objectives",
            "objectives": [
                {"field": "ppl", "transform": "-log(1 + PPL)", "direction": "maximize"},
                {"field": "protrek_percent", "transform": "identity", "direction": "maximize"},
                {"field": "plddt", "transform": "identity", "direction": "maximize"},
                {"field": "pae", "transform": "negate", "direction": "maximize"},
                {"field": "sol", "transform": "identity", "direction": "maximize"},
                {"field": "thermo", "transform": "identity", "direction": "maximize"},
            ],
            "oriented_min": conditional["oriented_min"][:3] + [-PAE_MAX] + conditional["oriented_min"][3:],
            "oriented_max": conditional["oriented_max"][:3] + [0.0] + conditional["oriented_max"][3:],
            "reference_point": [0.0] * 6,
            "excluded_from_hv": ["Novelty-U", "Novelty-T", "%>70", "%<10"],
            "scale_policy": "Reuse frozen PPL/ProTrek/pLDDT/Sol/Thermo ranges; add fixed ESMFold pAE range 0–31.75 A; no cohort refit or clipping",
            "inherited_manifest": str(conditional_source),
            "inherited_manifest_sha256": sha(conditional_source),
            "label": "frozen_6d_ppl_protrek_plddt_pae_sol_thermo",
        },
        "unconditional": {
            "definition_source": "User-selected canonical Unconditional objectives",
            "objectives": [
                {"field": "ppl", "transform": "-log(1 + PPL)", "direction": "maximize"},
                {"field": "plddt", "transform": "identity", "direction": "maximize"},
                {"field": "pae", "transform": "negate", "direction": "maximize"},
                {"field": "sol", "transform": "identity", "direction": "maximize"},
                {"field": "thermo", "transform": "identity", "direction": "maximize"},
            ],
            "oriented_min": [unconditional["oriented_min"][0], unconditional["oriented_min"][2],
                             -PAE_MAX, *unconditional["oriented_min"][3:]],
            "oriented_max": [unconditional["oriented_max"][0], unconditional["oriented_max"][2],
                             0.0, *unconditional["oriented_max"][3:]],
            "reference_point": [0.0] * 5,
            "excluded_from_hv": ["Novelty-U", "Novelty-T", "ProTrek %", "%>70", "%<10"],
            "scale_policy": "Reuse frozen PPL/pLDDT/Sol/Thermo ranges; add fixed ESMFold pAE range 0–31.75 A; no cohort refit or clipping",
            "inherited_manifest": str(unconditional_source),
            "inherited_manifest_sha256": sha(unconditional_source),
            "label": "frozen_5d_ppl_plddt_pae_sol_thermo",
        },
    }
    for spec in specs.values():
        spec.update(
            pAE_range_angstrom=[0.0, PAE_MAX],
            invalid_policy="Invalid requests remain in request counts and contribute no HV box",
            algorithm="moocore exact hypervolume, maximise=True",
        )
    return specs


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def fmt(value):
    return "--" if value is None else f"{value:.9g}"


def point(record, spec):
    values = []
    for objective in spec["objectives"]:
        value = number(record.get(objective["field"]))
        if value is None:
            return None
        if objective["field"] == "ppl":
            value = -math.log1p(value)
        elif objective["transform"] == "negate":
            value = -value
        values.append(value)
    lower = np.asarray(spec["oriented_min"], dtype=float)
    upper = np.asarray(spec["oriented_max"], dtype=float)
    return (np.asarray(values, dtype=float) - lower) / (upper - lower)


def exact_hv(points, dimensions):
    points = np.asarray(points, dtype=float).reshape(-1, dimensions)
    points = points[np.all(points > 0, axis=1)]
    if not len(points):
        return 0.0
    return float(hypervolume(points, ref=np.zeros(dimensions), maximise=True))


def cohort_hv(records, spec):
    points = [p for record in records
              if str(record.get("valid_complete", "")).lower() == "true"
              and (p := point(record, spec)) is not None]
    return exact_hv(points, len(spec["objectives"]))


def archive_stats(records, spec, gate):
    unique = {record["sequence_id"]: record for record in records if eligible(record, gate)}
    points = [p for record in unique.values() if (p := point(record, spec)) is not None]
    array = np.asarray(points, dtype=float).reshape(-1, len(spec["objectives"]))
    nondominated = 0
    for candidate in array:
        dominated = np.any(np.all(array >= candidate, axis=1) & np.any(array > candidate, axis=1))
        nondominated += int(not dominated)
    return exact_hv(array, len(spec["objectives"])), nondominated, len(unique)


def normalized_auc(curve, stop):
    x = np.asarray([row["training_gpu_hours"] for row in curve], dtype=float)
    y = np.asarray([row["trusted_archive_hv"] for row in curve], dtype=float)
    if x[0] != 0 or np.any(np.diff(x) <= 0) or stop > x[-1]:
        raise ValueError("Invalid cost curve support")
    support = np.unique(np.append(x[x <= stop], stop))
    return float(np.trapz(np.interp(support, x, y), support) / stop)


def load_cohorts(paradigm):
    records = rows(SCORE_TABLES[paradigm])
    cohorts = {}
    for record in records:
        cohorts.setdefault(record["model"], []).append(record)
    root = ROOTS[paradigm]
    for method, model in IDS.items():
        node = 0 if method == "reference" else 4
        cohorts[model] = rows(root / f"evaluation/{paradigm}/{method}/node_{node}/scores/sequence_scores.tsv")
    return cohorts


def validate_cohorts(paradigm, cohorts, spec):
    wanted = set()
    for path in TABLES[paradigm]:
        for row in rows(path):
            model = ALIASES.get(row["Model"], row["Model"])
            if model != "Semi-OPD":
                wanted.add(model)
    missing = sorted(wanted - set(cohorts))
    if missing:
        raise ValueError(f"{paradigm}: missing sequence evidence for {missing}")
    for model in wanted:
        records = cohorts[model]
        if len(records) != 512 or len({row["sample_id"] for row in records}) != 512:
            raise ValueError(f"{paradigm}/{model}: expected 512 distinct requests")
        for record in records:
            if str(record.get("valid_complete", "")).lower() == "true" and point(record, spec) is None:
                raise ValueError(f"{paradigm}/{model}: missing requested objective evidence")


def compute_paradigm(paradigm, spec):
    root = ROOTS[paradigm]
    protocol = read(root / "PROTOCOL.json")
    gate = protocol["evaluation_gate"]
    cohorts = load_cohorts(paradigm)
    validate_cohorts(paradigm, cohorts, spec)
    results = {
        model: {
            "HV": cohort_hv(records, spec),
            "requests": len(records),
            "complete": sum(str(row.get("valid_complete", "")).lower() == "true" for row in records),
        }
        for model, records in cohorts.items()
    }

    reference = results[IDS["reference"]]["HV"]
    gain = results[IDS["online"]]["HV"] - reference
    threshold = max(protocol["meaningful_online_gain_absolute_min"],
                    abs(reference) * protocol["meaningful_online_gain_relative_min"])
    meaningful = gain > threshold
    for model in IDS.values():
        results[model]["hv_retention_percent"] = (
            100 * (results[model]["HV"] - reference) / gain if meaningful else None
        )

    curves = []
    for method in METHODS:
        archive = list(cohorts[IDS["reference"]])
        hv, count, qualified = archive_stats(archive, spec, gate)
        node_zero = read(root / f"{paradigm}/{method}/node_0.json")
        curve = [{
            "paradigm": paradigm, "method": method, "node": "0",
            "global_step": protocol["nodes"][paradigm][0], "training_gpu_hours": 0.0,
            "trusted_archive_hv": hv, "hq_archive_size": count,
            "qualified_unique": qualified, "cumulative_requests": 512,
        }]
        curve.append(dict(curve[0], node="cache_ready",
                          training_gpu_hours=float(node_zero["training_gpu_hours"])))
        for node in range(1, 5):
            archive.extend(rows(root / f"evaluation/{paradigm}/{method}/node_{node}/scores/sequence_scores.tsv"))
            hv, count, qualified = archive_stats(archive, spec, gate)
            node_info = read(root / f"{paradigm}/{method}/node_{node}.json")
            curve.append({
                "paradigm": paradigm, "method": method, "node": str(node),
                "global_step": node_info["global_step"],
                "training_gpu_hours": float(node_info["training_gpu_hours"]),
                "trusted_archive_hv": hv, "hq_archive_size": count,
                "qualified_unique": qualified, "cumulative_requests": (node + 1) * 512,
            })
        final_cost = curve[-1]["training_gpu_hours"]
        results[IDS[method]].update(
            training_gpu_hours=final_cost,
            hq_unique_final_nd=count,
            hq_discovery_per_100_gpu_hours=100 * count / final_cost,
        )
        curves.extend(curve)

    final_costs = [results[IDS[method]]["training_gpu_hours"] for method in METHODS]
    common_cost = min(final_costs)
    for method in METHODS:
        curve = [row for row in curves if row["method"] == method]
        result = results[IDS[method]]
        result["hv_auc"] = normalized_auc(curve, common_cost)
        result["common_cost_gpu_hours"] = common_cost
        result["hq_at_common_cost"] = next(
            row["hq_archive_size"] for row in reversed(curve)
            if row["training_gpu_hours"] <= common_cost
        )
        result["hq_discovery_common_cost"] = 100 * result["hq_at_common_cost"] / common_cost

    return {
        "reference_hv": reference,
        "online_hv": results[IDS["online"]]["HV"],
        "online_hv_gain": gain,
        "meaningful_gain_threshold": threshold,
        "retention_applicable": meaningful,
        "results": results,
        "curves": curves,
        "cohorts": cohorts,
    }


def publish_tables(paradigm, computed, spec):
    results = computed["results"]
    for path in TABLES[paradigm]:
        table = rows(path)
        fields = list(table[0])
        for row in table:
            model = ALIASES.get(row["Model"], row["Model"])
            result = results.get(model)
            if not result:
                continue
            row["HV"] = fmt(result["HV"])
            row["HV Comparable"] = spec["label"]
            if model in IDS.values():
                if computed["retention_applicable"]:
                    if model == IDS["reference"]:
                        row["HV Ret. (%)"] = "0"
                    elif model == IDS["online"]:
                        row["HV Ret. (%)"] = "100"
                    else:
                        row["HV Ret. (%)"] = fmt(result["hv_retention_percent"])
                else:
                    row["HV Ret. (%)"] = "--"
                if model in (IDS[method] for method in METHODS):
                    row["HV-AUC"] = fmt(result["hv_auc"])
                    row["HQ Disc."] = fmt(result["hq_discovery_per_100_gpu_hours"])
            else:
                row["HV Ret. (%)"] = "--"
                row["HV-AUC"] = "--"
                row["HQ Disc."] = "--"
        tsv(path, table, fields)


def publish_report(all_results, specs):
    lines = [
        "# Main-table HV under the requested evaluation objectives", "",
        "No generation, scoring, or training was rerun. Existing complete sequence-level evidence was re-aggregated.", "",
        "| Paradigm | Method | HV | HV Ret. (%) | HV-AUC | HQ Disc. | Train GPU-h |", 
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for paradigm, computed in all_results.items():
        for method in ("reference", "online", *METHODS):
            result = computed["results"][IDS[method]]
            retention = None
            if computed["retention_applicable"]:
                retention = 0.0 if method == "reference" else 100.0 if method == "online" else result["hv_retention_percent"]
            lines.append(
                f"| {paradigm} | {DISPLAY[method]} | {fmt(result['HV'])} | {fmt(retention)} | "
                f"{fmt(result.get('hv_auc'))} | {fmt(result.get('hq_discovery_per_100_gpu_hours'))} | "
                f"{fmt(result.get('training_gpu_hours'))} |"
            )
    lines += ["", "## Definitions", ""]
    for paradigm, spec in specs.items():
        objectives = ", ".join(
            ("PPL↓" if obj["field"] == "ppl" else "pAE↓" if obj["field"] == "pae" else obj["field"] + "↑")
            for obj in spec["objectives"]
        )
        computed = all_results[paradigm]
        lines.append(
            f"- {paradigm}: {objectives}. Online−Reference={fmt(computed['online_hv_gain'])}; "
            f"positive-gain threshold={fmt(computed['meaningful_gain_threshold'])}; "
            f"Retention applicable={computed['retention_applicable']}."
        )
    lines += ["", "HV Retention is reported only when paired Online OPD has a meaningful positive gain over its shared Reference θ0.", ""]
    (PAPER / "main_hv_requested_objectives.md").write_text("\n".join(lines))


def main():
    started = time.time()
    specs = make_specs()
    target_files = [path for paths in TABLES.values() for path in paths]
    target_files += [PAPER / "adaptive_cache_hv_cost.tsv", PAPER / "adaptive_cache_metrics.json",
                     PAPER / "conditional_hv_definition.json"]
    backup = PAPER / "main_hv_before_requested_objectives.json"
    if not backup.exists():
        dump(backup, {
            "created_at_unix": started,
            "reason": "State before requested PPL/structure/property HV publication",
            "files": {str(path): path.read_text() for path in target_files if path.exists()},
        })

    all_results = {}
    all_curves = []
    for paradigm, spec in specs.items():
        print(f"Computing {paradigm} requested-objective HV", flush=True)
        computed = compute_paradigm(paradigm, spec)
        all_results[paradigm] = computed
        all_curves.extend(computed["curves"])

    # Every calculation and validation has succeeded before publication starts.
    for paradigm, computed in all_results.items():
        publish_tables(paradigm, computed, specs[paradigm])
        dump(PAPER / f"{paradigm}_hv_definition.json", specs[paradigm])

    tsv(PAPER / "adaptive_cache_hv_cost.tsv", all_curves)
    serializable = {
        paradigm: {key: value for key, value in computed.items() if key not in {"curves", "cohorts"}}
        for paradigm, computed in all_results.items()
    }
    dump(PAPER / "adaptive_cache_metrics.json", {
        "protocol": {
            "definition": "requested main-table objectives",
            "conditional_hv_definition": str(PAPER / "conditional_hv_definition.json"),
            "unconditional_hv_definition": str(PAPER / "unconditional_hv_definition.json"),
            "retention_rule": "Online-Reference > max(1e-6, 1% * abs(Reference))",
            "training_and_scoring_rerun": False,
        },
        "results": serializable,
        "published_paradigms": ["conditional", "unconditional"],
    })
    publish_report(all_results, specs)
    dump(PAPER / "main_hv_recalculation.json", {
        "state": "COMPLETE",
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "definitions": {paradigm: spec for paradigm, spec in specs.items()},
        "results": serializable,
        "sources_sha256": {
            str(path): sha(path)
            for path in [*SCORE_TABLES.values(),
                         *[ROOTS[p] / "PROTOCOL.json" for p in ROOTS]]
        },
        "updated_tables": [str(path) for path in target_files[:4]],
        "latex_generated": False,
    })
    print("Published both main tables and dependent HV columns", flush=True)


if __name__ == "__main__":
    main()
