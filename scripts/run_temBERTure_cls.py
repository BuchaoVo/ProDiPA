#!/usr/bin/env python3
"""Run the bundled TemBERTure classification or melting-temperature adapters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import torch

BASE_MODEL = "Rostlab/prot_bert_bfd"


def restore_base(repo: Path, revision: str | None) -> None:
    """Recover only the official PyTorch base and tokenizer, never the adapters."""
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(endpoint="https://huggingface.co").model_info(
        BASE_MODEL, revision=revision or "main", files_metadata=True,
    )
    files = ["config.json", "pytorch_model.bin", "vocab.txt",
             "tokenizer_config.json", "special_tokens_map.json"]
    started = time.perf_counter()
    snapshot = Path(snapshot_download(
        BASE_MODEL, revision=info.sha, allow_patterns=files,
        endpoint="https://huggingface.co", max_workers=2,
    ))
    fingerprints = {}
    for path in [snapshot / name for name in files] + sorted(
        (repo / "temBERTure/temBERTure_CLS").rglob("*")
    ):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        fingerprints[str(path)] = digest.hexdigest()
    for name in files:
        if not (snapshot / name).is_file():
            raise RuntimeError(f"Missing required official base file: {name}")
    weight = next(f for f in info.siblings if f.rfilename == "pytorch_model.bin")
    if weight.lfs and fingerprints[str(snapshot / "pytorch_model.bin")] != weight.lfs.sha256:
        raise RuntimeError("Downloaded base does not match official LFS SHA256")
    metadata = dict(model=BASE_MODEL, revision=info.sha, snapshot=str(snapshot),
                    source="https://huggingface.co/Rostlab/prot_bert_bfd",
                    historical_base_revision="UNKNOWN",
                    historical_equivalence="NOT_ESTABLISHED; fixed-sequence comparison required",
                    recovered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    download_wall_seconds=time.perf_counter()-started, sha256=fingerprints)
    destination = repo / "base_recovery.json"
    if destination.exists() and json.loads(destination.read_text())["revision"] != info.sha:
        raise RuntimeError("Refusing to replace an already frozen recovery revision")
    destination.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/properties/pilot_1000.tsv"))
    parser.add_argument("--repo", type=Path, default=Path("tools/TemBERTure"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/properties/pilot_1000_temBERTure.tsv"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=("cls", "tm"), default="cls")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-summary", type=Path)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--base-revision", help="Official base commit; defaults to recorded recovery revision")
    parser.add_argument("--restore-base", action="store_true", help="Download the official base on CPU and record provenance; no inference")
    parser.add_argument("--compare-history", action="store_true", help="Compare fixed CLS inputs containing historical_thermo_score; never change labels")
    args = parser.parse_args()
    if args.compare_history and args.task != "cls":
        parser.error("Historical CLS comparison cannot use Tm")

    if args.restore_base:
        restore_base(args.repo, args.base_revision)
        return
    recovery = args.repo / "base_recovery.json"
    base_model_source = BASE_MODEL
    if recovery.exists():
        provenance = json.loads(recovery.read_text())
        if provenance["model"] != BASE_MODEL:
            raise ValueError("Recovery metadata must identify the original ProtBERT-BFD")
        if args.base_revision and args.base_revision != provenance["revision"]:
            raise ValueError("Requested revision differs from the frozen recovery revision")
        args.base_revision = provenance["revision"]
        snapshot = Path(provenance["snapshot"])
        required = (snapshot / "config.json", snapshot / "pytorch_model.bin", snapshot / "vocab.txt")
        if all(path.exists() for path in required):
            # Older adapter-transformers releases probe the Hub for model.safetensors even
            # when the recorded revision is fully cached.  Pointing them at the immutable
            # snapshot avoids that network lookup without changing model weights.
            base_model_source = str(snapshot)

    sys.path.insert(0, str(args.repo / "temBERTure"))
    from temBERTure import TemBERTure  # noqa: PLC0415

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[:args.limit]
    score_field = "thermophilicity_score_01" if args.task == "cls" else "melting_temperature_c"
    reused = {}
    if args.reuse:
        requested_ids = {row["sequence_id"] for row in rows}
        reused = {
            row["sequence_id"]: float(row[score_field])
            for row in read_rows(args.reuse)
            if row["status"] == "ok" and row["sequence_id"] in requested_ids
        }
    eligible = [row for row in rows if row.get("valid_complete", "true").lower() == "true"]
    eligible_ids = {row["sequence_id"] for row in eligible}
    pending = [row for row in eligible if row["sequence_id"] not in reused]
    started = time.perf_counter()
    new_scores: dict[str, float] = {}
    if pending:
        adapters = ([args.repo / "temBERTure" / "temBERTure_CLS"] if args.task == "cls" else
                    [args.repo / "temBERTure" / "temBERTure_TM" / f"replica{i}" for i in range(1, 4)])
        replica_scores: list[dict[str, float]] = []
        for adapter in adapters:
            predictor = TemBERTure(
                adapter_path=f"{adapter}/", model_name=base_model_source,
                batch_size=args.batch_size, device=args.device,
                task="classification" if args.task == "cls" else "regression",
                revision=None if base_model_source != BASE_MODEL else args.base_revision,
            )
            model = predictor.model.to(args.device)
            model.eval()
            scores_for_replica = {}
            with torch.inference_mode():
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start:start + args.batch_size]
                    spaced = [" ".join(row["sequence"]) for row in batch]
                    encoded = predictor.tokenizer(
                        spaced, truncation=True, padding=True, max_length=512,
                        return_tensors="pt",
                    ).to(args.device)
                    logits = model(**encoded).logits.reshape(-1)
                    batch_scores = (torch.sigmoid(logits) if args.task == "cls" else logits).float().cpu().tolist()
                    scores_for_replica.update(
                        (row["sequence_id"], score) for row, score in zip(batch, batch_scores)
                    )
            replica_scores.append(scores_for_replica)
            del model, predictor
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        new_scores = {
            row["sequence_id"]: sum(scores[row["sequence_id"]] for scores in replica_scores) / len(replica_scores)
            for row in pending
        }
    elapsed = time.perf_counter() - started
    scores = {**reused, **new_scores}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = (["sequence_id", "thermophilicity_score_01", "predicted_class"] if args.task == "cls" else
              ["sequence_id", "melting_temperature_c", "replica_count"])
    fields += ["input_truncated", "status", "error_reason"]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        if not all(row["sequence_id"] in scores for row in eligible):
            raise RuntimeError(f"expected {len(eligible)} valid-generation scores, got {len(scores)}")
        for row in rows:
            score = scores.get(row["sequence_id"])
            input_length = int(row.get("length", row.get("length_aa", len(row.get("sequence", "")))))
            output = {
                "sequence_id": row["sequence_id"],
                "input_truncated": str(input_length > 510).lower(),
                "status": "ok" if score is not None and math.isfinite(score) else (
                    "failed" if row["sequence_id"] in eligible_ids else "skipped"
                ),
                "error_reason": "" if score is not None and math.isfinite(score) else (
                    "nonfinite_score" if row["sequence_id"] in eligible_ids else "invalid_or_incomplete_generation"
                ),
            }
            if args.task == "cls":
                output.update({
                    "thermophilicity_score_01": f"{score:.9g}" if score is not None and math.isfinite(score) else "",
                    "predicted_class": ("Thermophilic" if score is not None and score > 0.5
                                        else "Non-thermophilic" if score is not None else ""),
                })
            else:
                output.update({
                    "melting_temperature_c": f"{score:.9g}" if score is not None and math.isfinite(score) else "",
                    "replica_count": 3 if score is not None else "",
                })
            writer.writerow(output)

    print(f"sequences={len(rows)} reused={len(reused)} valid={len(scores)} elapsed_s={elapsed:.3f}")
    if args.compare_history:
        comparisons = []
        for row in rows:
            old = float(row["historical_thermo_score"])
            new = scores[row["sequence_id"]]
            if not math.isfinite(old) or not math.isfinite(new):
                raise RuntimeError("Nonfinite historical consistency score")
            comparisons.append(dict(sequence_id=row["sequence_id"], length=row["length"],
                historical_score=old, restored_score=new, absolute_difference=abs(old-new),
                agrees_at_1e_minus_5=abs(old-new) <= 1e-5,
                base_revision=args.base_revision or "UNKNOWN", historical_base_revision="UNKNOWN"))
        with args.out.with_suffix(".comparison.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(comparisons[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(comparisons)
        print("historical_check_n=",len(comparisons),"max_abs_diff=",
              max(r["absolute_difference"] for r in comparisons),
              "historical_revision=UNKNOWN; sample agreement does not prove provenance identity")
        maximum = max(r["absolute_difference"] for r in comparisons)
        mean = sum(r["absolute_difference"] for r in comparisons)/len(comparisons)
        agreement = all(r["agrees_at_1e_minus_5"] for r in comparisons)
        args.out.with_suffix(".md").write_text(
            "# TemBERTureCLS recovery consistency\n\n"
            f"Official base: `{BASE_MODEL}` at `{args.base_revision}`.\n\n"
            f"Fixed historical sequences: {len(comparisons)}; "
            f"maximum absolute difference: {maximum:.9g}; mean: {mean:.9g}. "
            f"All agree at absolute tolerance 1e-5: {agreement}.\n\n"
            "Selection: first three stable sequence IDs in each predeclared length bin "
            "50–150, 151–250, 251–350, 351–512 aa from the original 15K. "
            "Selection preceded recovered inference; no historical labels were changed.\n\n"
            "The historical base revision is UNKNOWN. Agreement on this small panel is "
            "not proof of identical historical provenance or equivalence on all sequences. "
            "The existing CLS adapter/head, uppercase spaced residues, max_length=512 "
            "(including special tokens), model.eval(), float32 inference and sigmoid "
            "of the single thermophilic-class logit were retained. No Tm scoring was run.\n\n"
            f"Inference wall time (including model load): {elapsed:.3f} s. "
            "Allocated GPU wall time and exact command are in the unified execution ledger.\n\n"
            "These are classification proxy scores, not melting temperatures or laboratory phenotypes.\n"
        )
        if recovery.exists():
            provenance = json.loads(recovery.read_text())
            provenance["historical_score_check"] = dict(
                n=len(comparisons), max_absolute_difference=maximum,
                mean_absolute_difference=mean, absolute_tolerance=1e-5,
                sample_agreement=agreement, comparison_file=str(args.out.with_suffix(".comparison.tsv")))
            provenance["historical_equivalence"] = (
                "SAMPLE_SCORES_AGREE; historical base revision remains UNKNOWN" if agreement else
                "SAMPLE_SCORES_DISAGREE; do not claim historical numerical equivalence")
            recovery.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.run_summary:
        args.run_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.run_summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                ["predictor", "sequences", "valid", "wall_time_s", "base_model", "base_revision",
                 "transformers_version", "adapters_version", "torch_version", "batch_size",
                 "max_length", "input_format", "score_definition"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow({
                "predictor": "TemBERTureCLS" if args.task == "cls" else "TemBERTureTm_3replica_mean",
                "sequences": len(rows),
                "valid": len(scores),
                "wall_time_s": f"{elapsed:.6f}",
                "base_model": BASE_MODEL,
                "base_revision": args.base_revision or "UNPINNED_HISTORICAL",
                "transformers_version": importlib.metadata.version("transformers"),
                "adapters_version": importlib.metadata.version("adapters"),
                "torch_version": torch.__version__,
                "batch_size": args.batch_size,
                "max_length": 512,
                "input_format": "space-separated amino acids; unchanged case",
                "score_definition": "sigmoid(single CLS logit), thermophilic probability" if args.task == "cls" else "mean of three Tm regressors",
            })


if __name__ == "__main__":
    main()
