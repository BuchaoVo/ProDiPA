#!/usr/bin/env python3
"""Execute the full Fixed/Periodic/Adaptive comparison and independent evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time

import numpy as np

from common import *
from scoring import score
from selection import select


def weight(path):
    return next(p for p in (path / "adapter_model.safetensors", path / "adapter_model.bin") if p.is_file())


def freeze():
    if (OUT / "PROTOCOL.json").exists():
        return read(OUT / "PROTOCOL.json")
    gate = dict(plddt_min=70.0, pae_max=10.0, sol_min=0.7, thermo_min=0.6,
                novelty_u_min_exclusive=0.0, novelty_t_min_exclusive=0.0,
                max_residue_fraction=0.4, max_trimer_fraction=0.2)
    protocol = dict(version=1, frozen_at_unix=time.time(), final_global_step=256,
                    starts=STARTS, nodes={p: [s + (256-s)*i//4 for i in range(5)] for p, s in STARTS.items()},
                    methods=list(METHODS), cache_size=512, anchors=sorted(random.Random(2718).sample(range(512),128)),
                    replace_limit=96, candidate_requests=96, train_seed=42,
                    candidate_seed_base=900001, evaluation_seed_base=700000001, evaluation_requests=512,
                    evaluation_seed_rule="700000001 + node * 10000000 + sample_id * 1009",
                    training_gate=gate, evaluation_gate=dict(gate),
                    gate_source="predeclared structure floors pLDDT>=70/pAE<=10 and existing Sol>=0.7/Thermo>=0.6; no evaluation calibration",
                    training_hv=dict(objectives=["plddt", "sol", "thermo"], lower=[0,0,0], upper=[100,1,1], reference=[0,0,0]),
                    delta_hv_epsilon=1e-8, coverage_max_occupancy=2, cluster_jaccard_threshold=0.5,
                    coverage_bins="initial training-cache attribute quartiles frozen before first refresh",
                    diversity_reserved_capacity=24, initial_cache_gpu_hours={}, evidence={},
                    meaningful_online_gain_relative_min=0.01, meaningful_online_gain_absolute_min=1e-6,
                    accounting_scope="paired continuation; shared pre-checkpoint training excluded; initial cache charged once to each method",
                    fixed_replay_reason="quarter-point checkpoints absent in the completed Fixed run",
                    novelty_definition="Novelty-U = 100*(1-max returned MMseqs2 alignment identity) against UniProtKB_2026_03, no_hit=100; Novelty-T uses the frozen teacher train union. Neither is ProTrek.",
                    hq_definition="valid complete independent requests; unique sequence SHA256; gates then nondominance on the same frozen evaluation HV objectives; cumulative across five nodes",
                    cost_curve="normalized trapezoidal AUC of cumulative trusted archive HV, common min endpoint, interpolation only; reference archive at cost zero, charged cache preparation plateau until first student checkpoint")
    for paradigm in STARTS:
        state = read(THETAS[paradigm] / "trainer_state.json")
        assert state["global_step"] == STARTS[paradigm] and state["max_steps"] == 256
        manifest = read(CACHES[paradigm] / "CACHE_MANIFEST.json")
        assert manifest["status"] == "COMPLETE" and manifest["sample_count"] == 512
        assert sha(weight(THETAS[paradigm])) == manifest["theta_zero"]["adapter_weights_sha256"]
        for teacher in manifest["teachers"]:
            assert sha(weight(Path(teacher["adapter_path"]))) == teacher["adapter_weights_sha256"]
        for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
            assert (THETAS[paradigm] / name).is_file()
        costs = [read(CACHES[paradigm] / f"shard_{i}.json")["allocated_gpu_seconds"] for i in range(2)]
        protocol["initial_cache_gpu_hours"][paradigm] = sum(costs) / 3600
        protocol["evidence"][paradigm] = dict(cache_manifest_sha256=sha(CACHES[paradigm] / "CACHE_MANIFEST.json"),
                theta_weights_sha256=sha(weight(THETAS[paradigm])), optimizer_sha256=sha(THETAS[paradigm]/"optimizer.pt"),
                scheduler_sha256=sha(THETAS[paradigm]/"scheduler.pt"))
        for record in range(512):
            assert (CACHES[paradigm] / "records" / f"{record:06d}.pt").is_file()
        evaluation = ROOT / "configs/evaluation" / f"{paradigm}.json"
        dump(OUT / "protocol" / f"{paradigm}_evaluation.json", read(evaluation))
        hv = ROOT / "outputs/evaluation" / paradigm / "results" / f"{paradigm}_hv_manifest.json"
        dump(OUT / "protocol" / f"{paradigm}_hv.json", read(hv))
        protocol["evidence"][paradigm]["hv_manifest_sha256"] = sha(hv)
    dump(OUT / "PROTOCOL.json", protocol)
    return protocol


def initialize(paradigm, method, protocol):
    directory = OUT / paradigm / method
    cache = directory / "cache"
    if not (cache / "CACHE_MANIFEST.json").exists():
        (cache / "records").mkdir(parents=True, exist_ok=True)
        for index in range(512):
            target = cache / "records" / f"{index:06d}.pt"
            if not target.exists():
                target.symlink_to(CACHES[paradigm] / "records" / target.name)
        dump(cache / "CACHE_MANIFEST.json", read(CACHES[paradigm] / "CACHE_MANIFEST.json"))
    initial = OUT / paradigm / "initial_cache"
    job(f"{paradigm}_decode_initial", [PY, HERE / "cache_worker.py", "decode", "--paradigm", paradigm,
        "--directory", initial], paradigm, "shared", "preparation", 0)
    if not (directory / "cache_state.json").exists():
        entries = rows(initial / "candidates.tsv")
        if method == "adaptive":
            entries = score(initial / "candidates.tsv", initial / "scores", paradigm, method, "training", 0)
        bins = {}
        for key in ("plddt", "sol", "thermo"):
            values = [float(r[key]) for r in entries if r.get(key) not in (None, "")]
            bins[key] = list(np.quantile(values, [.25,.5,.75])) if values else []
        state = dict(slots=entries, archive=entries, bins=bins, applied_nodes=[])
        dump(directory / "cache_state.json", state)
    if not (directory / "node_0.json").exists():
        dump(directory / "node_0.json", dict(training_gpu_hours=cost(paradigm, method, 0),
                                            checkpoint=str(THETAS[paradigm])))
    return directory


def train_segment(paradigm, method, directory, node, protocol):
    stop = protocol["nodes"][paradigm][node]
    previous = protocol["nodes"][paradigm][node-1]
    training = directory / "training"
    resume = THETAS[paradigm] if node == 1 else training / f"checkpoint-{previous}"
    script = OLD / ("train.py" if paradigm == "conditional" else "train_unconditional.py")
    port = "29745" if paradigm == "conditional" else "29746"
    command = [TORCHRUN, "--nproc_per_node", "2", "--master_port", port, script,
        "--mode", "offline", "--theta-zero", THETAS[paradigm], "--resume-checkpoint", resume,
        "--cache-dir", directory / "cache", "--output-dir", training, "--seed", "42",
        "--offline-sample-count", "512", "--offline-epochs", "16", "--initial-global-step", str(STARTS[paradigm]),
        "--target-global-step", "256", "--stop-global-step", str(stop)]
    command += (["--config", CONFIG, "--resume-optimizer-state"] if paradigm == "conditional"
                else ["--base-model", BASES[paradigm], "--teacher-config", TEACHERS,
                      "--sequence-format", "legacy", "--max-length-aa", "512"])
    job(f"{paradigm}_{method}_train_{node}", command, paradigm, method, "training", node,
        gpus=2, free_mib=32000 if paradigm == "conditional" else 14000)
    state = read(training / f"checkpoint-{stop}/trainer_state.json")
    assert state["global_step"] == stop and state["max_steps"] == 256
    if not (directory / f"node_{node}.json").exists():
        dump(directory / f"node_{node}.json", dict(global_step=stop, training_gpu_hours=cost(paradigm, method, node),
                                                  checkpoint=str(training / f"checkpoint-{stop}")))


def refresh(paradigm, method, directory, node, protocol):
    state_path = directory / "cache_state.json"
    state = read(state_path)
    if node in state["applied_nodes"]:
        return
    refresh_dir = directory / f"refresh_{node}"
    step = protocol["nodes"][paradigm][node]
    seed = protocol["candidate_seed_base"] + node * 10000
    job(f"{paradigm}_{method}_rollout_{node}", [PY, HERE / "cache_worker.py", "rollout", "--paradigm", paradigm,
        "--directory", refresh_dir, "--adapter", directory / "training" / f"checkpoint-{step}", "--seed", str(seed)],
        paradigm, method, "training", node, 1, 22000 if paradigm == "conditional" else 8000)
    if method == "adaptive":
        score(refresh_dir / "candidates.tsv", refresh_dir / "scores", paradigm, method, "training", node)
    selection = select(refresh_dir, state, method, protocol, seed)
    if selection["accepted"]:
        job(f"{paradigm}_{method}_teacher_{node}", [PY, HERE / "cache_worker.py", "teacher", "--paradigm", paradigm,
            "--directory", refresh_dir], paradigm, method, "training", node, 1, 22000 if paradigm == "conditional" else 10000)
    accepted = {int(r["sample_id"]): r for r in selection["accepted_rows"]}
    for replacement in selection["assignments"]:
        slot, candidate = replacement["slot"], replacement["candidate"]
        assert slot not in protocol["anchors"]
        path = directory / "cache/records" / f"{slot:06d}.pt"
        temp = path.with_suffix(".link")
        if temp.is_symlink():
            temp.unlink()
        temp.symlink_to(refresh_dir / "records" / f"{candidate:06d}.pt")
        os.replace(temp, path)
        state["slots"][slot] = accepted[candidate]
    state["archive"].extend(rows(refresh_dir / "scores/sequence_scores.tsv") if method == "adaptive" else selection["accepted_rows"])
    state["applied_nodes"].append(node)
    dump(state_path, state)
    for slot in protocol["anchors"]:
        assert (directory / "cache/records" / f"{slot:06d}.pt").resolve() == (CACHES[paradigm] / "records" / f"{slot:06d}.pt").resolve()


def generate(paradigm, method, node, adapter, protocol):
    directory = OUT / "evaluation" / paradigm / method / f"node_{node}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "generation.tsv"
    if not target.exists():
        def shard(index):
            job(f"{paradigm}_{method}_eval_{node}_generation_{index}", [PY, ROOT / "experiments/prodipa/conditional_evaluation/generate_model.py",
                "--paradigm", paradigm, "--model-name", method, "--base-model", BASES[paradigm], "--adapter", adapter,
                "--out", directory / f"shard_{index}.tsv", "--num-sequences", "512", "--sample-id-start", str(index*128),
                "--sample-id-stop", str((index+1)*128), "--base-seed", str(protocol["evaluation_seed_base"]+node*10000000),
                "--temperature", "0.7", "--top-p", "0.9", "--top-k", "200", "--repetition-penalty", "1.2", "--max-new-tokens", "512",
                *(["--sequence-format", "legacy", "--generation-max-aa", "512"]
                  if paradigm == "unconditional" else [])],
                paradigm, method, "evaluation", node, 1, 18000 if paradigm == "conditional" else 8000)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(shard, range(4)))
        records = [r for i in range(4) for r in rows(directory / f"shard_{i}.tsv")]
        assert len(records) == 512 and len({r["sample_id"] for r in records}) == 512
        tsv(target, records)
    score(target, directory / "scores", paradigm, method, "evaluation", node)


def train_paradigm(paradigm, protocol):
    try:
        for method in METHODS:
            status(f"{paradigm}/{method}/prepare")
            directory = initialize(paradigm, method, protocol)
            for node in range(1, 5):
                status(f"{paradigm}/{method}/training/{node}")
                train_segment(paradigm, method, directory, node, protocol)
                if method != "fixed" and node < 4:
                    status(f"{paradigm}/{method}/refresh/{node}")
                    refresh(paradigm, method, directory, node, protocol)
        dump(OUT / paradigm / "TRAINING_COMPLETE.json", {"status": "COMPLETE", "completed_at": time.time()})
        status(f"{paradigm}/training_complete", "COMPLETE")
    except BaseException as error:
        status(f"{paradigm}/interrupted", "FAILED", error=repr(error))
        raise


_training_completion_lock = threading.Lock()


def run_paradigm(paradigm, protocol):
    """Start independent evaluation as soon as this paradigm finishes training."""
    train_paradigm(paradigm, protocol)
    with _training_completion_lock:
        if all((OUT / p / "TRAINING_COMPLETE.json").exists() for p in STARTS):
            marker = OUT / "TRAINING_COMPLETE.json"
            if not marker.exists():
                dump(marker, {"status": "COMPLETE", "completed_at": time.time()})
    try:
        status(f"{paradigm}/reference/evaluation")
        generate(paradigm, "reference", 0, THETAS[paradigm], protocol)
        for method in METHODS:
            for node in range(1, 5):
                status(f"{paradigm}/{method}/evaluation/{node}")
                step = protocol["nodes"][paradigm][node]
                generate(paradigm, method, node, OUT / paradigm / method / "training" / f"checkpoint-{step}", protocol)
        status(f"{paradigm}/online/evaluation")
        generate(paradigm, "online", 4, ONLINE_MODELS[paradigm], protocol)
        dump(OUT / paradigm / "EVALUATION_COMPLETE.json", {"status": "COMPLETE", "completed_at": time.time()})
        status(f"{paradigm}/evaluation_complete", "COMPLETE")
    except BaseException as error:
        status(f"{paradigm}/evaluation_interrupted", "FAILED", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / ".pipeline.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        protocol = freeze()
        if args.freeze_only:
            status("protocol_frozen", "READY")
            return
        active = []
        paused = []
        for paradigm in STARTS:
            marker = OUT / paradigm / "PAUSED.json"
            if marker.exists():
                paused.append(paradigm)
                payload = dict(state="PAUSED", stage=f"{paradigm}/paused_by_user",
                               updated_at_unix=time.time(), reason=read(marker).get("reason"))
                dump(OUT / paradigm / "STATUS.json", payload)
                # Preserve paused state in aggregate updates from the active thread.
                import common
                common._training_states[paradigm] = payload
            else:
                active.append(paradigm)
        # Each paradigm owns separate caches, scores and job records. GPU locks
        # limit simultaneous work without changing either training protocol.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="paradigm") as pool:
            futures = [pool.submit(run_paradigm, paradigm, protocol) for paradigm in active]
            for future in futures:
                future.result()
        if paused:
            status("active_paradigms_complete", "PARTIAL", paused_paradigms=paused,
                   completed_paradigms=active, note="Combined publication withheld; paused model lineage must not be reused")
            return
        status("paper_tables")
        report_scope = "both" if len(STARTS) > 1 else next(iter(STARTS))
        job("paper_tables", [PY, HERE / "report.py", "--paradigm", report_scope],
            report_scope, "all", "report", 4)
        status("complete", "COMPLETE")
    except BaseException as error:
        status("interrupted", "FAILED", error=repr(error))
        raise


if __name__ == "__main__":
    main()
