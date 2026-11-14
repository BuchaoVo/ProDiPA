#!/usr/bin/env python3
"""Annotate ProDiPA sequences with local Hugging Face ESMFold.

Existing scoring calls retain batch=1 and recycling=4. Opt-in length-bucketed
batches use --batch-size 8 --max-tokens-per-batch 3200. --chunk-size 0 disables
chunking. --auto-gpus reserves eligible GPUs through the existing scoring locks.
Use --benchmark with a separate --out to compare batches, TF32 and recycling
on the same fixed cohort; --benchmark-resume skips completed settings.
No PDB is produced. TSV output preserves the established evaluation interface.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForProteinFolding


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_sequences(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        if handle.read(1) != ">":
            return read_rows(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                rows.append({"sequence_id": line[1:].split()[0], "sequence": ""})
            elif line:
                rows[-1]["sequence"] += line
    return rows


def make_batches(rows, batch_size, max_tokens):
    """Bound padded tokens (batch size * longest length), without truncation."""
    batch = []
    for row in sorted(rows, key=lambda r: len(r["sequence"])):
        length = len(row["sequence"])
        if batch and (len(batch) >= batch_size or (len(batch) + 1) * length > max_tokens):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


def eligible_gpus(min_free_gib, max_util, max_gpus):
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    permitted = None if visible is None else set(visible.split(","))
    candidates = []
    for line in result.stdout.splitlines():
        index, uuid, memory, util = [x.strip() for x in line.split(",")]
        if permitted is not None and index not in permitted and uuid not in permitted:
            continue
        if float(memory) >= min_free_gib * 1024 and float(util) <= max_util:
            candidates.append(index)
    return candidates[:max_gpus]


def reserve_gpus(args, limit):
    """Use the same advisory GPU locks as the existing ProDiPA scoring queues."""
    first = eligible_gpus(args.min_free_memory_gib, args.max_gpu_utilization, limit)
    time.sleep(2)
    second = eligible_gpus(args.min_free_memory_gib, args.max_gpu_utilization, limit)
    selected, locks = [], []
    for gpu in second:
        if gpu not in first:
            continue
        uuid = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"],
                              check=True, text=True, capture_output=True).stdout.strip()
        acquired = []
        try:
            for suffix in (".shared_scoring.lock", ".lock"):
                handle = open(f"/tmp/prodipa_gpu_{uuid}{suffix}", "a")
                acquired.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            for handle in acquired:
                handle.close()
            continue
        selected.append(uuid)
        locks.extend(acquired)
    return selected, locks


def tensor_mean(value) -> float:
    if value is None:
        return math.nan
    return float(value.detach().float().mean().cpu())


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "sequence_id", "fold_plddt_100", "fold_ptm_01", "fold_mean_pae",
        "inference_time_s", "status", "error_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def extract_metrics(outputs, lengths):
    from transformers.models.esm.openfold_utils.loss import compute_tm

    results = []
    for index, length in enumerate(lengths):
        # Preserve the historical all-atom pLDDT mean, excluding padded residues.
        plddt = tensor_mean(outputs.plddt[index, :length])
        if plddt <= 1.0:
            plddt *= 100.0
        # HF compute_tm reduces across the entire batch; recompute per protein.
        ptm = tensor_mean(compute_tm(outputs.ptm_logits[index, :length, :length],
                                     max_bin=31, no_bins=outputs.ptm_logits.shape[-1]))
        pae = tensor_mean(outputs.predicted_aligned_error[index, :length, :length])
        if not all(math.isfinite(x) for x in (plddt, ptm, pae)):
            raise ValueError("nonfinite ESMFold confidence metric")
        results.append({"fold_plddt_100": f"{plddt:.9g}", "fold_ptm_01": f"{ptm:.9g}",
                        "fold_mean_pae": f"{pae:.9g}"})
    return results


def load_model(args):
    patch_compute_tm()
    if args.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = EsmForProteinFolding.from_pretrained(
        args.model, low_cpu_mem_usage=True, local_files_only=True,
    )
    model.trunk.set_chunk_size(args.chunk_size or None)
    model.eval().to(device)
    return tokenizer, model, device


def infer_batch(batch, args, tokenizer, model, device):
    encoded = tokenizer([r["sequence"] for r in batch], padding=True,
                        return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        outputs = model(encoded["input_ids"], attention_mask=encoded["attention_mask"],
                        num_recycles=args.num_recycles)
        return extract_metrics(outputs, encoded["attention_mask"].sum(-1).tolist())


def run_batch(batch, args, tokenizer, model, device):
    cap = getattr(args, "_effective_batch_size", args.batch_size)
    if len(batch) > cap:
        return [row for start in range(0, len(batch), cap)
                for row in run_batch(batch[start:start + cap], args, tokenizer, model, device)]
    started = time.perf_counter()
    metrics = None
    error_reason = ""
    oom = False
    for attempt in range(args.max_retries + 1):
        try:
            metrics = infer_batch(batch, args, tokenizer, model, device)
            break
        except Exception as error:
            error_reason = repr(error)
            oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        # Outside except: release traceback references before empty_cache/retry.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if oom and len(batch) > 1:
            break
        if attempt < args.max_retries:
            time.sleep(min(2 ** attempt, 4))
    if metrics is None and len(batch) > 1:
        midpoint = len(batch) // 2
        if oom:
            args._effective_batch_size = midpoint
        print(f"retry batch={len(batch)} split={midpoint} oom={oom}", flush=True)
        results = run_batch(batch[:midpoint], args, tokenizer, model, device)
        results += run_batch(batch[midpoint:], args, tokenizer, model, device)
        # Include failed attempts in total inference cost.
        total = time.perf_counter() - started
        for result in results:
            result["inference_time_s"] = f"{total / len(results):.6f}"
        return results
    elapsed = time.perf_counter() - started
    return [dict(sequence_id=row["sequence_id"],
                 fold_plddt_100="", fold_ptm_01="", fold_mean_pae="",
                 inference_time_s=f"{elapsed / len(batch):.6f}",
                 status="ok" if metrics is not None else "failed",
                 error_reason="" if metrics is not None else error_reason) | (metrics[i] if metrics is not None else {})
            for i, row in enumerate(batch)]


_WORKER = None


def init_worker(args, gpu):
    global _WORKER
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    args.device = "cuda:0"
    _WORKER = (args, *load_model(args))


def worker_batch(batch):
    return run_batch(batch, *_WORKER)


def parallel_batches(batches, args, gpus):
    """One resident model per GPU; assign next batch to the first free worker."""
    pools = [ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"),
                                 initializer=init_worker, initargs=(args, gpu)) for gpu in gpus]
    pending = {}
    iterator = iter(batches)
    try:
        for pool in pools:
            batch = next(iterator, None)
            if batch:
                pending[pool.submit(worker_batch, batch)] = pool
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pool = pending.pop(future)
                yield future.result()
                batch = next(iterator, None)
                if batch:
                    pending[pool.submit(worker_batch, batch)] = pool
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)


def protocol(args):
    import transformers
    return {"model": str(args.model.resolve()), "num_recycles": args.num_recycles,
            "chunk_size": args.chunk_size, "tf32": args.tf32,
            "effective_matmul_tf32": torch.backends.cuda.matmul.allow_tf32 if args.tf32 is None else args.tf32,
            "effective_cudnn_tf32": torch.backends.cudnn.allow_tf32 if args.tf32 is None else args.tf32,
            "batch_size": args.batch_size, "max_tokens_per_batch": args.max_tokens_per_batch,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "plddt_definition": "historical_all_atom_mean_over_unpadded_residues",
            "ptm_definition": "per_sequence_unpadded_full_pair_logits",
            "inference_time_definition": "batch wall time divided by batch size; includes retries, excludes model load",
            "amp": False, "save_pdb": False,
            "trunk_passes": args.num_recycles + 1,
            "command": sys.argv,
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}


def run_benchmark(selected, args):
    """Isolated paired benchmark; never writes into formal score outputs."""
    import copy
    import statistics
    from scipy.stats import spearmanr

    cohort = sorted([r for r in selected if r.get("valid_complete", "true").lower() == "true"
                     and 200 <= len(r["sequence"]) <= 400], key=lambda r: (len(r["sequence"]), r["sequence_id"]))
    count = min(32, len(cohort))
    if count < 4:
        raise ValueError("benchmark needs at least four valid 200–400 aa sequences")
    cohort = [cohort[i * (len(cohort) - 1) // (count - 1)] for i in range(count)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_lock = args.out.with_suffix(args.out.suffix + ".lock").open("a")
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.auto_gpus:
        gpus, gpu_locks = reserve_gpus(args, 1)
        if not gpus:
            raise RuntimeError("Benchmark deferred: no GPU satisfies memory/utilization limits in two polls")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
        args.device = "cuda:0"
    previous = None
    if args.benchmark_resume:
        previous = json.loads(args.out.with_suffix(".benchmark.json").read_text())
        if previous["sequence_ids"] != [r["sequence_id"] for r in cohort]:
            raise ValueError("benchmark resume cohort differs")
        current = protocol(args)
        for key in ("model", "input_sha256", "torch", "transformers", "max_tokens_per_batch"):
            if previous["protocol"][key] != current[key]:
                raise ValueError(f"benchmark resume protocol differs: {key}")
    elif args.out.exists() or args.out.with_suffix(".benchmark.json").exists():
        raise FileExistsError("benchmark output exists; preserve it and use an unused output filename")
    if not args.device.startswith("cuda"):
        raise ValueError("throughput benchmark requires CUDA")
    benchmark_started = time.perf_counter()
    args.tf32 = False
    tokenizer, model, device = load_model(args)
    settings = [("reference", 1, 4, args.chunk_size, False)]
    settings += [(f"batch_{size}", size, 4, 0, False) for size in (4, 8, 12, 16)]
    settings += [("tf32", 8, 4, 0, True), ("fast_recycle1", 8, 1, 0, True)]
    summary = read_rows(args.out) if previous and args.out.exists() else []
    reference_path = args.out.with_name(args.out.stem + "_reference_scores.tsv")
    reference = {r["sequence_id"]: r for r in read_rows(reference_path)} if previous and reference_path.exists() else None
    if args.benchmark_confirm_best:
        if not reference:
            raise ValueError("confirmation requires --benchmark-resume and completed reference scores")
        candidates = [r for r in summary if r["status"] == "ok" and int(r["num_recycles"]) == 4]
        best = max(candidates, key=lambda r: float(r["seq_per_second"]))
        settings = [(best["setting"] + "_confirmation", int(best["requested_batch_size"]),
                     4, int(best["chunk_size"]), str(best["tf32"]).lower() == "true")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    details = previous or {"protocol": protocol(args), "gpu": torch.cuda.get_device_name(device),
               "sequence_ids": [r["sequence_id"] for r in cohort],
               "lengths": [len(r["sequence"]) for r in cohort], "runs": []}
    if previous and "current_invocation_wall_seconds" in previous:
        details.setdefault("previous_invocations", []).append({
            key: value for key, value in previous.items() if key.startswith("current_invocation_")
        })
    args.out.with_suffix(".benchmark.json").write_text(json.dumps(details, indent=2) + "\n")
    for name, size, recycles, chunk, tf32 in settings:
        if any(row["setting"] == name for row in summary):
            continue
        config = copy.copy(args)
        config.batch_size, config.num_recycles, config.chunk_size, config.tf32 = size, recycles, chunk, tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        model.trunk.set_chunk_size(chunk or None)
        batches = list(make_batches(cohort, size, args.max_tokens_per_batch))
        predictions, times = {}, []
        failure = ""
        torch.cuda.reset_peak_memory_stats(device)
        try:
            infer_batch(batches[0], config, tokenizer, model, device)  # excluded warm-up
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for repeat in range(args.benchmark_repeats):
                started = time.perf_counter()
                for batch in batches:
                    metrics = infer_batch(batch, config, tokenizer, model, device)
                    if repeat == 0:
                        predictions.update({r["sequence_id"]: m for r, m in zip(batch, metrics)})
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - started)
                print(f"benchmark={name} repeat={repeat + 1}/{args.benchmark_repeats} seconds={times[-1]:.3f}", flush=True)
        except Exception as error:
            failure = repr(error)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        gc.collect()
        torch.cuda.empty_cache()
        row = {"setting": name, "requested_batch_size": size,
               "actual_max_batch_size": max(map(len, batches)), "num_recycles": recycles,
               "chunk_size": chunk, "tf32": tf32, "sequences": count,
               "timed_repeats": len(times),
               "median_seconds": statistics.median(times) if len(times) == args.benchmark_repeats else "",
               "seq_per_second": count / statistics.median(times) if len(times) == args.benchmark_repeats else "",
               "peak_allocated_gib": peak,
               "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
               "status": "failed" if failure else "ok", "error": failure}
        if not failure:
            scored = [dict(sequence_id=sid, **values, status="ok", error_reason="", inference_time_s="")
                      for sid, values in predictions.items()]
            write_rows(args.out.with_name(args.out.stem + f"_{name}_scores.tsv"), scored)
            if name == "reference":
                reference = predictions
            if reference:
                for metric in ("fold_plddt_100", "fold_ptm_01", "fold_mean_pae"):
                    x = [float(reference[r["sequence_id"]][metric]) for r in cohort]
                    y = [float(predictions[r["sequence_id"]][metric]) for r in cohort]
                    row[metric + "_max_abs_diff"] = max(abs(a - b) for a, b in zip(x, y))
                    row[metric + "_mean_abs_diff"] = statistics.mean(abs(a - b) for a, b in zip(x, y))
                    row[metric + "_spearman"] = float(spearmanr(x, y).statistic)
        summary.append(row)
        details["runs"].append({**row, "repeat_seconds": times})
        details["current_invocation_wall_seconds"] = time.perf_counter() - benchmark_started
        details["current_invocation_gpu_hours"] = details["current_invocation_wall_seconds"] / 3600
        details["current_invocation_command"] = sys.argv
        fields = list(dict.fromkeys(key for record in summary for key in record))
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(summary)
        args.out.with_suffix(".benchmark.json").write_text(json.dumps(details, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if name == "reference" and failure:
            raise RuntimeError(f"reference benchmark failed: {failure}")


def patch_compute_tm() -> None:
    """Keep pLDDT/PAE when this transformers version cannot compute pTM."""
    import transformers.models.esm.modeling_esmfold as modeling
    import transformers.models.esm.openfold_utils.loss as loss

    original = modeling.compute_tm

    def safe_compute_tm(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except IndexError:
            return args[0].new_tensor(float("nan"))

    modeling.compute_tm = safe_compute_tm
    loss.compute_tm = safe_compute_tm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/properties/pilot_1000.tsv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("__PRODIPA_MODELS__/esmfold_v1_hf"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=64, help="0 disables chunking")
    parser.add_argument("--num-recycles", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--reuse", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens-per-batch", type=int, default=3200,
                        help="Maximum padded residues per batch; an oversized singleton is never truncated")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--auto-gpus", action="store_true")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--min-free-memory-gib", type=float, default=65)
    parser.add_argument("--max-gpu-utilization", type=float, default=20)
    parser.add_argument("--benchmark", action="store_true",
                        help="Use the same fixed input for batch/TF32/recycle comparisons; separate outputs only")
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--benchmark-resume", action="store_true")
    parser.add_argument("--benchmark-confirm-best", action="store_true",
                        help="Resume and repeat the fastest completed recycling=4 setting")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if min(args.batch_size, args.max_tokens_per_batch, args.max_gpus) < 1 or args.max_retries < 0:
        raise ValueError("batch/token/GPU limits must be positive; retries must be nonnegative")
    if args.benchmark_repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    selected = [
        row for index, row in enumerate(read_sequences(args.input))
        if index % args.num_shards == args.shard_index
    ]
    if args.limit:
        selected = selected[:args.limit]
    if len({r["sequence_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate sequence_id in input; use the existing unique-sequence preparation step")

    if args.benchmark:
        run_benchmark(selected, args)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output_lock = args.out.with_suffix(args.out.suffix + ".lock").open("a")
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata_path = args.out.with_suffix(args.out.suffix + ".metadata.json")
    current_protocol = protocol(args)
    for source in [args.out, *args.reuse]:
        if not source.exists():
            continue
        metadata = source.with_suffix(source.suffix + ".metadata.json")
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            for key in ("model", "num_recycles", "chunk_size", "tf32", "plddt_definition", "ptm_definition",
                        "input_sha256", "torch", "transformers", "effective_matmul_tf32", "effective_cudnn_tf32"):
                if previous.get(key) != current_protocol[key]:
                    raise ValueError(f"incompatible resume/reuse {source}: {key}")
        elif args.batch_size != 1 or args.tf32 is not None or args.num_recycles != 4:
            raise ValueError(f"cannot mix unversioned scores with a changed inference protocol: {source}")

    selected_ids = {row["sequence_id"] for row in selected}
    completed_by_id = {}
    sources = ([args.out] if args.out.exists() else []) + [path for path in args.reuse if path.exists()]
    for source in sources:
        # Failed rows are retryable; keeping them in `done` silently makes a
        # transient shared-GPU OOM permanent on resume.
        for row in read_rows(source):
            if row["sequence_id"] in selected_ids and row["status"] in {"ok", "skipped"}:
                completed_by_id.setdefault(row["sequence_id"], row)
    completed = [completed_by_id[row["sequence_id"]] for row in selected if row["sequence_id"] in completed_by_id]
    done = set(completed_by_id)
    if len(done) == len(selected):
        write_rows(args.out, completed)
        print(f"shard={args.shard_index} reused={len(done)} pending=0", flush=True)
        return

    pending_rows = []
    for row in selected:
        if row["sequence_id"] in done:
            continue
        if row.get("valid_complete", "true").lower() != "true":
            completed.append(dict(sequence_id=row["sequence_id"], status="skipped",
                                  error_reason="invalid_or_incomplete_generation", inference_time_s="0"))
            continue
        pending_rows.append(row)
    gpus = None
    if args.auto_gpus and pending_rows:
        gpus, gpu_locks = reserve_gpus(args, args.max_gpus)
        if not gpus:
            raise RuntimeError("No eligible GPUs in two polls; existing evaluation jobs were not interrupted")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(current_protocol, indent=2) + "\n")
    write_rows(args.out, completed)
    started = time.perf_counter()
    batches = list(make_batches(pending_rows, args.batch_size, args.max_tokens_per_batch))
    if gpus:
        results = parallel_batches(batches, args, gpus)
    elif batches:
        tokenizer, model, device = load_model(args)
        results = (run_batch(batch, args, tokenizer, model, device) for batch in batches)
    else:
        results = iter(())
    for batch_result in results:
        completed.extend(batch_result)
        write_rows(args.out, completed)
        print(f"shard={args.shard_index} completed={len(completed)}/{len(selected)} "
              f"batch={len(batch_result)} failed={sum(r['status'] == 'failed' for r in batch_result)}", flush=True)
    order = {row["sequence_id"]: i for i, row in enumerate(selected)}
    write_rows(args.out, sorted(completed, key=lambda row: order[row["sequence_id"]]))
    seconds = time.perf_counter() - started
    print(f"wall_seconds_including_load={seconds:.3f} new_sequences={len(pending_rows)}", flush=True)


if __name__ == "__main__":
    main()
