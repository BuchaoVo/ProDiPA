#!/usr/bin/env python3
"""Publish measured cache comparisons using frozen evaluation scales."""
from __future__ import annotations

import math
import argparse
import sys
import statistics

import numpy as np

from common import *
from selection import eligible
sys.path.insert(0, str(OLD))
from finalize_full_paper import exact_hv as recursive_exact_hv, FIELDS

PAPER = ROOT / "outputs/evaluation/paper"
DISPLAY = {"reference": "Reference θ0", "online": "Online OPD (paired)",
           "fixed": "Fixed Offline OPD", "periodic": "Periodic Cache OPD", "adaptive": "ProDiPA (Adaptive Cache)"}
IDS = {"reference": "reference_theta0", "online": "online_opd_paired", "fixed": "prodipa_offline",
       "periodic": "periodic_cache_opd", "adaptive": "adaptive_cache_opd"}
METRICS = {"PPL":"ppl", "Novelty-U":"novelty_u_percent", "Novelty-T":"novelty_t_percent", "ProTrek %":"protrek_percent",
           "pLDDT":"plddt", "pAE":"pae", "Sol":"sol", "Thermo":"thermo"}


def exact_hv(points):
    """Keep historical 5D computations unchanged; use compiled HV for 4D/6D."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 2 and points.shape[1] in (4, 6):
        from moocore import hypervolume
        points = points[np.all(points > 0, axis=1)]
        return float(hypervolume(points, ref=np.zeros(points.shape[1]), maximise=True)) if len(points) else 0.0
    return recursive_exact_hv(points)


def num(x):
    try:
        value = float(x)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def fmt(x):
    return "N/A" if x is None else f"{x:.9g}"


def point(r, paradigm, manifest):
    if paradigm == "conditional" and len(manifest["reference_point"]) in (4, 6):
        values = [num(r.get(o["field"])) for o in manifest["objectives"]]
        if any(x is None for x in values):
            return None
        values = [-v if o["transform"] == "negate" else v
                  for v, o in zip(values, manifest["objectives"])]
        lo, hi = np.asarray(manifest["oriented_min"]), np.asarray(manifest["oriented_max"])
        return np.divide(np.asarray(values)-lo, hi-lo, out=np.ones(len(lo)), where=hi!=lo) - np.asarray(manifest["reference_point"])
    keys = ["ppl", "protrek_percent" if paradigm == "conditional" else "novelty_u_percent", "plddt", "sol", "thermo"]
    values = [num(r.get(k)) for k in keys]
    if any(x is None for x in values):
        return None
    values[0] = -math.log1p(values[0])
    lo, hi = np.asarray(manifest["oriented_min"]), np.asarray(manifest["oriented_max"])
    return np.divide(np.asarray(values)-lo, hi-lo, out=np.ones(len(lo)), where=hi!=lo) - np.asarray(manifest["reference_point"])


def hv(records, paradigm, manifest):
    points = [p for r in records if str(r.get("valid_complete")).lower() == "true" and (p:=point(r, paradigm, manifest)) is not None]
    return exact_hv(np.asarray(points).reshape(-1,len(manifest["reference_point"])))


def summary(records, paradigm, manifest):
    valid = [r for r in records if str(r.get("valid_complete")).lower() == "true"]
    result = {"requests":len(records), "complete":len(valid), "HV":hv(records, paradigm, manifest)}
    for label, key in METRICS.items():
        values = [v for r in valid if (v:=num(r.get(key))) is not None]
        result[label] = statistics.fmean(values) if values else None
    for label, key, pred in [("%>70", "plddt", lambda x:x>70), ("%<10", "pae", lambda x:x<10)]:
        values = [v for r in valid if (v:=num(r.get(key))) is not None]
        result[label] = 100*sum(pred(x) for x in values)/len(values) if values else None
    return result


def archive_stats(records, paradigm, manifest, gate):
    unique = {r["sequence_id"]:r for r in records if eligible(r, gate)}
    valid = [(r,p) for r in unique.values() if (p:=point(r,paradigm,manifest)) is not None]
    points = np.asarray([p for _,p in valid]).reshape(-1,len(manifest["reference_point"]))
    count = sum(not np.any(np.all(points>=p,axis=1)&np.any(points>p,axis=1)) for p in points)
    return exact_hv(points), int(count), len(unique)


def normalized_auc(curve, stop):
    x = np.asarray([r["training_gpu_hours"] for r in curve])
    y = np.asarray([r["trusted_archive_hv"] for r in curve])
    assert x[0] == 0 and np.all(np.diff(x)>0) and stop <= x[-1]
    support = np.unique(np.append(x[x<=stop], stop))
    interpolated = np.interp(support, x, y)
    return float(np.trapz(interpolated, support)/stop)


def historical_rows(paradigm, manifest):
    old = rows(PAPER/f"{paradigm}_main.tsv")
    excluded = set(DISPLAY.values()) | {"ProDiPA (Offline)"}
    old = [r for r in old if r["Model"] not in excluded and r["Model"] not in IDS.values()]
    known = {r["Model"] for r in old}
    canonical = ROOT/"outputs/evaluation"/paradigm/"results"/f"{paradigm}_prodipa_main.tsv"
    for row in rows(canonical):
        if row["Model"] not in known and row["Model"] not in IDS.values() and row["Model"] not in excluded:
            old.append(dict(row, Group="External Protein Design Baselines"))
            known.add(row["Model"])
    config = read(OUT/"protocol"/f"{paradigm}_evaluation.json")
    evidence = {}
    novelty = {}
    novelty_path = ROOT/"outputs/evaluation"/paradigm/"scores/novelty_u.tsv"
    if novelty_path.exists():
        novelty = {r["sequence_id"]:r["novelty_percent"] for r in rows(novelty_path)}
    current = read(ROOT/"configs/evaluation"/f"{paradigm}.json")
    groups = list(config["score_groups"])
    # Newly finished baselines may be absent from the frozen model inventory.
    # Reuse their scores only under the same scoring protocol, never their HV scale.
    protocol_keys = ("requests_per_model", "generation_protocol", "references", "scorers", "supplementary_thresholds")
    if all(current.get(k) == config.get(k) for k in protocol_keys):
        seen = {g["evaluation_root"] for g in groups}
        groups.extend(g for g in current["score_groups"] if g["evaluation_root"] not in seen)
    for group in groups:
        directory = ROOT/group["evaluation_root"]
        source = directory/"results/sequence_scores.tsv"
        if not source.exists():
            source = directory/"results"/f"{paradigm}_sequence_scores.tsv"
        if not source.exists():
            continue
        protrek_file = directory/"scores/protrek.tsv"
        protrek = {r["sequence_id"]:r["protrek_percent"] for r in rows(protrek_file)} if protrek_file.exists() else {}
        for r in rows(source):
            if r["sequence_id"] in novelty:
                r["novelty_u_percent"] = novelty[r["sequence_id"]]
            if r["sequence_id"] in protrek:
                r["protrek_percent"] = protrek[r["sequence_id"]]
            evidence.setdefault(r["model"], {})[r["request_id"]] = r
    for r in old:
        r["HV Ret. (%)"] = "--"
        if r["Model"] == "Semi-OPD":
            r["HV"] = "--"
            r["HV Comparable"] = "not_run"
            continue
        data = list(evidence.get(r["Model"], {}).values())
        complete = [s for s in data if s.get("valid_complete", "").lower()=="true"]
        comparable = bool(complete) and all(point(s,paradigm,manifest) is not None for s in complete)
        r["HV"] = fmt(hv(data,paradigm,manifest)) if comparable else "N/A"
        r["HV Comparable"] = ({4: "posthoc_four_objective_scale", 6: "posthoc_six_objective_scale"}.get(len(manifest["reference_point"]), "frozen_scale")) if comparable else "unavailable_sequence_evidence"
    return old


def latex(path, records, fields):
    def escape(text):
        replacements={"&":r"\&", "%":r"\%", "_":r"\_", "#":r"\#", "θ":r"$\theta$"}
        return "".join(replacements.get(c,c) for c in str(text))
    lines = [r"\begin{tabular}{" + "l"*len(fields) + "}", r"\hline", " & ".join(map(escape, fields))+r" \\",r"\hline"]
    lines += [" & ".join(escape(r.get(k,"--")) for k in fields)+r" \\" for r in records]
    lines += [r"\hline",r"\end{tabular}"]
    path.write_text("\n".join(lines)+"\n")


def validate_inputs(paradigms, protocol):
    """Fail before publication, without starting any generation or scoring jobs."""
    for paradigm in paradigms:
        if (OUT/paradigm/"PAUSED.json").exists():
            raise RuntimeError(f"{paradigm} is paused; select only a completed, active paradigm")
        if not (OUT/paradigm/"EVALUATION_COMPLETE.json").exists():
            raise RuntimeError(f"{paradigm} evaluation is not complete")
        nodes = [("reference", 0), ("online", 4)] + [(m,n) for m in METHODS for n in range(1,5)]
        for method, node in nodes:
            directory = OUT/"evaluation"/paradigm/method/f"node_{node}/scores"
            if not (directory/"COMPLETE.json").exists():
                raise RuntimeError(f"Missing score completion marker: {directory}")
            records = rows(directory/"sequence_scores.tsv")
            assert len(records) == protocol["evaluation_requests"], directory
            assert len({r["sample_id"] for r in records}) == len(records), directory
            for r in records:
                if str(r.get("valid_complete")).lower() == "true":
                    keys = [v for k,v in METRICS.items() if paradigm == "conditional" or k != "ProTrek %"]
                    assert all(num(r.get(k)) is not None for k in keys), (directory, r["sample_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paradigm", choices=[*STARTS, "both"], default="both")
    args = parser.parse_args(argv)
    paradigms = list(STARTS) if args.paradigm == "both" else [args.paradigm]
    protocol = read(OUT/"PROTOCOL.json")
    validate_inputs(paradigms, protocol)
    publication_status = OUT/"REPORT_STATUS.json"
    dump(publication_status, dict(state="RUNNING", paradigms=paradigms, started_at_unix=time.time()))
    all_curves, report = [], {}
    for paradigm in paradigms:
        print(f"Summarizing {paradigm} reference and paired online", flush=True)
        override = PAPER / "conditional_hv_definition.json"
        manifest = read(override if paradigm == "conditional" and override.exists()
                        else OUT/"protocol"/f"{paradigm}_hv.json")
        reference = rows(OUT/"evaluation"/paradigm/"reference/node_0/scores/sequence_scores.tsv")
        online = rows(OUT/"evaluation"/paradigm/"online/node_4/scores/sequence_scores.tsv")
        assert len(reference)==len(online)==512
        ref_summary = summary(reference,paradigm,manifest)
        on_summary = summary(online,paradigm,manifest)
        gain = on_summary["HV"] - ref_summary["HV"]
        meaningful = gain > max(protocol["meaningful_online_gain_absolute_min"],
                                abs(ref_summary["HV"])*protocol["meaningful_online_gain_relative_min"])
        online_cost = sum(read(ONLINE_MODELS[paradigm]/"timings"/f"rank_{i}.json")["allocated_gpu_seconds"] for i in range(2))/3600
        report[paradigm] = {"reference":ref_summary, "online":dict(on_summary, training_gpu_hours=online_cost),
                            "online_hv_gain":gain, "retention_applicable":meaningful}
        curves = {}
        for method in METHODS:
            print(f"Summarizing {paradigm}/{method}: independent archive and cost curve", flush=True)
            archive = list(reference)
            h, count, qualified = archive_stats(archive,paradigm,manifest,protocol["evaluation_gate"])
            curve = [dict(paradigm=paradigm,method=method,node=0,global_step=STARTS[paradigm],training_gpu_hours=0.0,
                          trusted_archive_hv=h,hq_archive_size=count,qualified_unique=qualified,cumulative_requests=512)]
            initial_cost_file = OUT/paradigm/method/"node_0.json"
            preparation_cost=(read(initial_cost_file)["training_gpu_hours"] if initial_cost_file.exists()
                              else cost(paradigm,method,0))
            curve.append(dict(curve[0], node="cache_ready",training_gpu_hours=preparation_cost))
            for node in range(1,5):
                records = rows(OUT/"evaluation"/paradigm/method/f"node_{node}/scores/sequence_scores.tsv")
                assert len(records)==512
                archive.extend(records)
                h,count,qualified=archive_stats(archive,paradigm,manifest,protocol["evaluation_gate"])
                node_info=read(OUT/paradigm/method/f"node_{node}.json")
                curve.append(dict(paradigm=paradigm,method=method,node=node,global_step=node_info["global_step"],
                                  training_gpu_hours=node_info["training_gpu_hours"],trusted_archive_hv=h,
                                  hq_archive_size=count,qualified_unique=qualified,cumulative_requests=(node+1)*512))
            curves[method]=curve
            measured=summary(records,paradigm,manifest)
            measured.update(training_gpu_hours=curve[-1]["training_gpu_hours"],hq_unique_final_nd=count,
                            hq_discovery_per_100_gpu_hours=100*count/curve[-1]["training_gpu_hours"],
                            hv_retention_percent=100*(measured["HV"]-ref_summary["HV"])/gain if meaningful else None,
                            hv_minus_online=measured["HV"]-on_summary["HV"],
                            evaluation_gpu_hours=cost(paradigm,method,4,"evaluation"),refreshes=0,new_candidate_requests=0,teacher_scoring_tokens=0)
            for node in range(1,4):
                directory=OUT/paradigm/method/f"refresh_{node}"
                if (directory/"selection.json").exists():
                    decision=read(directory/"selection.json")
                    measured["refreshes"]+=int(decision["refreshed"])
                    measured["new_candidate_requests"]+=decision["requested"]
                if (directory/"TEACHER_COMPLETE.json").exists():
                    measured["teacher_scoring_tokens"]+=read(directory/"TEACHER_COMPLETE.json")["teacher_scoring_tokens"]
            report[paradigm][method]=measured
        cstar=min(curve[-1]["training_gpu_hours"] for curve in curves.values())
        for method,curve in curves.items():
            result=report[paradigm][method]
            result["hv_auc"]=normalized_auc(curve,cstar)
            result["common_cost_gpu_hours"]=cstar
            result["hq_at_common_cost"]=next(r["hq_archive_size"] for r in reversed(curve) if r["training_gpu_hours"]<=cstar)
            result["hq_discovery_common_cost"]=100*result["hq_at_common_cost"]/cstar
            all_curves.extend(curve)
        table=historical_rows(paradigm,manifest)
        for method in ("reference","online",*METHODS):
            m=report[paradigm][method]
            r={k:"--" for k in FIELDS}
            r.update(Group="Our Method" if method=="adaptive" else "OPD and Offline Distillation Baselines", Model=DISPLAY[method])
            r.update({k:fmt(m.get(k)) for k in (*METRICS,"HV")})
            r["HV Ret. (%)"]=fmt((0.0 if method=="reference" else 100.0 if method=="online" else m["hv_retention_percent"]) if meaningful else None)
            r["Train GPU-h"]=fmt(m.get("training_gpu_hours"))
            r["HV-AUC"]=fmt(m.get("hv_auc"))
            r["HQ Disc."]=fmt(m.get("hq_discovery_per_100_gpu_hours"))
            r["HV Comparable"]={4: "posthoc_four_objective_scale", 6: "posthoc_six_objective_scale"}.get(len(manifest["reference_point"]), "frozen_scale")
            table.append(r)
        groups={"Backbone Model and Attribute Teachers":0,"External Protein Design Baselines":1,
                "OPD and Offline Distillation Baselines":2,"Our Method":3}
        table.sort(key=lambda r:groups.get(r["Group"],9))
        fields=FIELDS+["HV Comparable"]
        tsv(PAPER/f"{paradigm}_main.tsv",[{k:r.get(k,"--") for k in fields} for r in table],fields)
        canonical=ROOT/"outputs/evaluation"/paradigm/"results"/f"{paradigm}_prodipa_main.tsv"
        existing=rows(canonical)
        canonical_fields=list(dict.fromkeys(list(existing[0])+["Novelty-U","Novelty-T","HV Ret. (%)","Train GPU-h","HV-AUC","HQ Disc.","HV Comparable"]))
        replacements={IDS[m]:next(r for r in table if r["Model"]==DISPLAY[m]) for m in IDS}
        historical={r["Model"]:r for r in table}
        updated=[]
        for r in existing:
            if r["Model"] in replacements:
                continue
            if r["Model"] in historical:
                r["HV"]=historical[r["Model"]]["HV"]
                r["HV Comparable"]=historical[r["Model"]]["HV Comparable"]
                for field in ("Novelty-U", "Novelty-T"):
                    r[field] = historical[r["Model"]].get(field, "--")
            updated.append(r)
        for model,r in replacements.items():
            method=next(m for m,mid in IDS.items() if mid==model)
            merged=dict(r,Model=model)
            for field in canonical_fields:
                if field in report[paradigm][method]:
                    merged[field]=fmt(report[paradigm][method][field])
            updated.append(merged)
        tsv(canonical,[{k:r.get(k,"--") for k in canonical_fields} for r in updated],canonical_fields)
        print(f"Published {paradigm} paper and canonical TSV tables", flush=True)
    tsv(PAPER/"adaptive_cache_hv_cost.tsv",all_curves)
    omitted = [p for p in STARTS if p not in paradigms]
    dump(PAPER/"adaptive_cache_metrics.json",dict(protocol=protocol,results=report,
         published_paradigms=paradigms, omitted_paradigms=omitted))
    lines=["# Quality-Aware Adaptive Cache: " + ", ".join(paradigms) + " measured comparison", "",
           "Published scope: " + ", ".join(paradigms) + ". " + ("Not published in this report: " + ", ".join(omitted) + "; existing tables and pause markers are unchanged." if omitted else "Both paradigms are complete."), "",
           "Costs are paired continuation GPU-hours. All methods include initial cache construction; Adaptive includes every training oracle and refresh allocation. Common warm-start training is excluded. Formal evaluation GPU-hours are recorded separately.","",
           "| Paradigm | Method | Final HV | HV-AUC | Unique HQ ND | HQ Disc./100 GPU-h | Train GPU-h | Refreshes | HQ Disc. at common cost |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in paradigms:
        for m in METHODS:
            r=report[p][m]
            lines.append(f"| {p} | {DISPLAY[m]} | {fmt(r['HV'])} | {fmt(r['hv_auc'])} | {r['hq_unique_final_nd']} | {fmt(r['hq_discovery_per_100_gpu_hours'])} | {fmt(r['training_gpu_hours'])} | {r['refreshes']} | {fmt(r['hq_discovery_common_cost'])} |")
    lines += ["", "## Protocol and definitions", "",f"Frozen protocol: `{OUT/'PROTOCOL.json'}`.",
              "Conditional: global step 192→256 (64 continuation updates); Unconditional: 200→256 (56). These are the existing formal endpoints, not pilot budgets. Fixed was replayed to retain intermediate checkpoints. The 512-slot cache keeps 128 anchors and allows at most 96 replacements at each of three boundaries; Periodic/Adaptive request 96 candidates per boundary without retries for validity.",
              "Each of five independent checkpoints has 512 requests, including invalid outputs (2,560 cumulative per method); the initial 512-request evaluation is shared. Evaluation seeds are disjoint from cache seeds. Online reuses the completed paired weights and is evaluated with the new frozen final seeds; no online retraining.",
              protocol["novelty_definition"],
              f"HQ gates (frozen before training): {json.dumps(protocol['evaluation_gate'])}. HQ Disc. counts sequence-deduplicated, final nondominated qualified independent evaluation sequences; it is computational prediction quality, not wet-lab validation. Training selections are excluded.",
              "HQ Disc. is a standardized rate (100 times observed HQ count divided by measured training GPU-hours), not an observed 100-GPU-hour experiment or a prediction of linear discovery growth. Raw unique HQ counts are shown alongside it.",
              "Training selection ΔHV remains [pLDDT/100, Sol, Thermo] against zero, plus frozen initial-cache quartile cells and 3-mer Jaccard clusters (threshold 0.5). Unconditional evaluation retains [-log(1+PPL), Novelty-U, pLDDT, Sol, Thermo]. Conditional evaluation uses the objectives and scale explicitly recorded in paper/conditional_hv_definition.json when present: four-objective [pLDDT, -pAE, Sol, Thermo] or six-objective with Novelty-U and Novelty-T; otherwise it retains the original five-objective protocol. The same published scale is used for final HV, archive HV-AUC and nondominance; the original training protocol is not modified. HV values of different dimensions are not directly comparable. Historical rows without sufficient sequence evidence are N/A and excluded from HV ranking.",
              "HV-AUC uses trapezoidal interpolation on the shared supported cost interval [0,min(final training costs)], normalized by its width. The initial reference archive is at cost zero, with a plateau during initial preparation. Common-cost HQ discovery uses the last observed archive at or before that budget, without extrapolation. Final and common-cost HQ rates are both reported.",
              "HV Retention is N/A unless Online−Reference exceeds max(1e-6,1% of Reference HV). Negative online gain never receives positive-gain retention.","","## Adaptive versus Periodic",""]
    for p in paradigms:
        a,b=report[p]["adaptive"],report[p]["periodic"]
        for key in ("HV","hv_auc","hq_discovery_per_100_gpu_hours","training_gpu_hours"):
            delta=a[key]-b[key]
            relative=100*delta/abs(b[key]) if b[key] else None
            lines.append(f"- {p} {key}: Adaptive−Periodic={fmt(delta)}; relative={fmt(relative)}%.")
        lines.append(f"- {p}: Online−Reference HV={fmt(report[p]['online_hv_gain'])}; Retention applicable={report[p]['retention_applicable']}. Common compute budget={fmt(a['common_cost_gpu_hours'])} GPU-h.")
        preparation = next(r["training_gpu_hours"] for r in all_curves
                           if r["paradigm"] == p and r["method"] == "adaptive" and r["node"] == "cache_ready")
        if preparation > a["common_cost_gpu_hours"]:
            lines.append(f"- {p}: Adaptive preparation alone costs {fmt(preparation)} GPU-h, exceeding the common budget {fmt(a['common_cost_gpu_hours'])} GPU-h. Its common-cost archive therefore contains only the shared reference evaluation; no post-update Adaptive checkpoint is observed within that budget.")
        for m in METHODS:
            r=report[p][m]
            lines.append(f"- {p}/{m}: candidate requests={r['new_candidate_requests']}; refresh teacher input tokens summed across three teachers={r['teacher_scoring_tokens']}; evaluation GPU-h={fmt(r['evaluation_gpu_hours'])}.")
    answers=[]
    for p in paradigms:
        a,b=report[p]["adaptive"],report[p]["periodic"]
        win=a["hv_auc"]>b["hv_auc"] and a["hq_discovery_common_cost"]>b["hq_discovery_common_cost"]
        answers.append(f"{p}: " + ("quality-aware selection improves both common-cost discovery measures" if win else "the measured common-cost measures do not jointly establish that quality gains cover oracle cost"))
    lines += ["", "Oracle-cost conclusion: " + "; ".join(answers) + ". No additional ablations or parameter searches were run.",""]
    (PAPER/"adaptive_cache_core_results.md").write_text("\n".join(lines))
    dump(publication_status, dict(state="COMPLETE", paradigms=paradigms,
         omitted_paradigms=omitted, finished_at_unix=time.time(), latex_generated=False))
    print("Report complete; no training or scoring jobs launched", flush=True)


if __name__ == "__main__":
    main()
