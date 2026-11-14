"""Quality gates, exact Pareto contribution and frozen coverage tie breaks."""
from collections import Counter
import math
import random
import sys

import numpy as np

from common import OLD, dump, read, rows
sys.path.insert(0, str(OLD))
from finalize_full_paper import exact_hv


def eligible(row, gate):
    try:
        seq = row["sequence"]
        counts = Counter(seq)
        triples = Counter(seq[i:i+3] for i in range(len(seq)-2))
        return (str(row["valid_complete"]).lower() == "true" and
                float(row["plddt"]) >= gate["plddt_min"] and float(row["pae"]) <= gate["pae_max"] and
                float(row["sol"]) >= gate["sol_min"] and float(row["thermo"]) >= gate["thermo_min"] and
                float(row["novelty_u_percent"]) > gate["novelty_u_min_exclusive"] and
                float(row["novelty_t_percent"]) > gate["novelty_t_min_exclusive"] and
                max(counts.values()) / len(seq) <= gate["max_residue_fraction"] and
                max(triples.values()) / (len(seq)-2) <= gate["max_trimer_fraction"])
    except (KeyError, ValueError, ZeroDivisionError):
        return False


def vector(r):
    return [float(r["plddt"]) / 100, float(r["sol"]), float(r["thermo"])]


def kmers(seq):
    return {seq[i:i+3] for i in range(len(seq)-2)}


def similarity(a, b):
    return len(a & b) / max(1, len(a | b))


def cell(r, bins):
    return tuple(int(np.searchsorted(bins[k], float(r[k]), side="right")) for k in ("plddt", "sol", "thermo"))


def select(directory, state, mode, protocol, seed):
    all_candidates = rows(directory / "candidates.tsv")
    rng = random.Random(seed)
    slots = [i for i in range(512) if i not in protocol["anchors"]]
    if mode == "periodic":
        candidates = [r for r in all_candidates if r["valid_complete"].lower() == "true"]
        rng.shuffle(candidates)
        accepted = candidates[:96]
        rng.shuffle(slots)
    else:
        scored = rows(directory / "scores/sequence_scores.tsv")
        existing_ids = {r["sequence_id"] for r in state["archive"]}
        candidates = list({r["sequence_id"]: r for r in scored
                           if eligible(r, protocol["training_gate"]) and r["sequence_id"] not in existing_ids}.values())
        archive = [r for r in state["archive"] if eligible(r, protocol["training_gate"])]
        bins = state["bins"]
        occupied = Counter(cell(r, bins) for r in archive)
        clusters = [kmers(r["sequence"]) for r in archive]
        points = np.asarray([vector(r) for r in archive], dtype=float).reshape(-1, 3)
        baseline = exact_hv(points)
        ranked = []
        for r in candidates:
            contribution = exact_hv(np.vstack([points, vector(r)])) - baseline
            novelty = 1 - max((similarity(kmers(r["sequence"]), c) for c in clusters), default=0)
            low_coverage = occupied[cell(r, bins)] < protocol["coverage_max_occupancy"]
            new_cluster = novelty >= 1 - protocol["cluster_jaccard_threshold"]
            if contribution > protocol["delta_hv_epsilon"] or low_coverage or new_cluster:
                r = dict(r, delta_hv=contribution, coverage_new=low_coverage,
                         diversity=novelty, cluster_new=new_cluster)
                ranked.append(r)
        # Reserve a quarter of capacity for coverage/sequence diversity; all must pass gates.
        diverse = sorted(ranked, key=lambda r: (r["coverage_new"], r["cluster_new"], r["diversity"], r["delta_hv"]), reverse=True)
        reserve = diverse[:min(24, math.ceil(len(diverse) / 4))]
        reserved = {r["sequence_id"] for r in reserve}
        remaining = [r for r in ranked if r["sequence_id"] not in reserved]
        remaining.sort(key=lambda r: (r["delta_hv"], r["coverage_new"], r["diversity"]), reverse=True)
        accepted = remaining[:96-len(reserve)] + reserve
        occupancy = Counter(cell(r, bins) for r in state["slots"] if eligible(r, protocol["training_gate"]))
        def replace_value(i):
            r = state["slots"][i]
            if not eligible(r, protocol["training_gate"]):
                return (0, 0, 0)
            return (1, -occupancy[cell(r, bins)], sum(vector(r)))
        slots.sort(key=replace_value)
    assignments = [{"slot": i, "candidate": int(r["sample_id"])} for i, r in zip(slots, accepted)]
    result = dict(mode=mode, requested=96, legal=sum(r["valid_complete"].lower() == "true" for r in all_candidates),
                  accepted=[int(r["sample_id"]) for r in accepted], assignments=assignments,
                  accepted_rows=accepted, refreshed=bool(accepted))
    dump(directory / "selection.json", result)
    return result
