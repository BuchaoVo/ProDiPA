#!/usr/bin/env python3
"""Wait for configured scores, then publish pooled Novelty-U and HV results."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def status(path: Path, state: str, stage: str, **extra: object) -> None:
    atomic_json(path, {
        "state": state,
        "stage": stage,
        **extra,
        "updated_at": datetime.now().astimezone().isoformat(),
    })


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run(command: list[str], log) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def tsv_ids(path: Path, field: str) -> set[str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if field not in (reader.fieldnames or []):
            return None
        values = [row[field] for row in reader if row.get(field)]
    return set(values) if values and len(values) == len(set(values)) else None


def novelty_cache_valid(
    combined: Path,
    novelty: Path,
    manifest: Path,
    reference_db: Path,
    reference_label: str,
    mmseqs_version: str,
) -> bool:
    combined_ids = tsv_ids(combined, "sequence_id")
    novelty_ids = tsv_ids(novelty, "sequence_id")
    if combined_ids is None or novelty_ids != combined_ids or not manifest.is_file():
        return False
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("reference_kind") == "mmseqs_database"
        and metadata.get("reference_label") == reference_label
        and Path(metadata.get("reference_path", "")).resolve() == reference_db.resolve()
        and metadata.get("queries") == len(combined_ids)
        and metadata.get("query_input_sha256") == hashlib.sha256(combined.read_bytes()).hexdigest()
        and metadata.get("mmseqs_version") == mmseqs_version
    )


def merge_group_tables(
    paths: list[Path], output: Path, expected_models: list[str]
) -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            current_fields = list(reader.fieldnames or [])
            for field in current_fields:
                if field not in fields:
                    fields.append(field)
            rows.extend(reader)
    if not fields:
        raise ValueError("no rows available for cohort table")
    by_model = {row["Model"]: row for row in rows}
    if set(by_model) != set(expected_models) or len(rows) != len(by_model):
        raise ValueError(
            f"cohort table model mismatch: observed={sorted(by_model)}, "
            f"expected={expected_models}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=output.parent,
        delete=False, prefix=f".{output.name}."
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(by_model[model] for model in expected_models)
        temporary = handle.name
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/evaluation/unconditional.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--once", action="store_true", help="Check readiness once and exit if blocked")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = resolve(root, str(args.config))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve(root, config["output_root"])
    for directory in ("prepared", "scores", "results", "logs", "tmp"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    status_path = output / "STATUS.json"
    lock_handle = (output / ".completion.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)

    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config.get("schema_version") != 1 or config.get("evaluation_name") != "unconditional":
        raise ValueError("unsupported Unconditional evaluation config")
    resources = config.get("resources", {})
    threads = args.threads or int(resources.get("mmseqs_threads", 32))
    split_memory_limit = resources.get("mmseqs_split_memory_limit")
    groups = config["score_groups"]
    expected_models: list[str] = []
    score_files: list[Path] = []
    unique_files: list[Path] = []
    supplementary_files: list[Path] = []
    completeness_files: list[Path] = []
    for group in groups:
        expected_models.extend(group["models"])
        evaluation_root = resolve(root, group["evaluation_root"])
        score_files.append(evaluation_root / "results/sequence_scores.tsv")
        unique_files.append(evaluation_root / "prepared/unique_valid_sequences.tsv")
        supplementary_files.append(evaluation_root / "results/supplementary_metrics.tsv")
        completeness_files.append(evaluation_root / "results/scoring_completeness.tsv")
    if len(expected_models) != len(set(expected_models)):
        raise ValueError("configured model identifiers are not unique")

    reference_db = resolve(root, config["references"]["novelty_u_database"])
    ready_marker = resolve(root, config["references"]["novelty_u_ready_marker"])
    required = (
        score_files + unique_files + supplementary_files + completeness_files
        + [reference_db, Path(str(reference_db) + ".dbtype"), ready_marker]
    )
    while True:
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if not missing:
            break
        status(status_path, "WAITING", "wait_for_inputs", missing=missing, config_sha256=config_hash)
        if args.once:
            print(json.dumps({"state": "WAITING", "missing": missing}, indent=2))
            return
        print(f"waiting for {len(missing)} required inputs", flush=True)
        time.sleep(args.poll_seconds)

    python = os.environ.get(
        "PRODIPA_PYTHON", "__PRODIPA_ENVS__/ProDiPA/bin/python"
    )
    helpers = root / "experiments/prodipa/external_baseline_evaluation"
    scorer = root / "experiments/prodipa/conditional_evaluation"
    combined = output / "prepared/cohort_unique_valid_sequences.tsv"
    novelty = output / "scores/novelty_u.tsv"
    main_table = output / "results/unconditional_prodipa_main.tsv"
    all_scores = output / "results/unconditional_sequence_scores.tsv"
    hv_manifest = output / "results/unconditional_hv_manifest.json"
    log_path = output / "logs/cohort_completion.log"

    try:
        with log_path.open("a", encoding="utf-8") as log:
            status(status_path, "RUNNING", "combine_sequences", config_sha256=config_hash)
            run([
                python, str(helpers / "combine_unique_sequences.py"), "--input",
                *(str(path) for path in unique_files), "--output", str(combined),
            ], log)

            status(status_path, "RUNNING", "novelty_u", config_sha256=config_hash)
            novelty_manifest = novelty.with_suffix(".manifest.json")
            reference_label = config["references"]["novelty_u_label"]
            mmseqs_version = config["scorers"]["novelty"]["mmseqs_commit"]
            if novelty_cache_valid(
                combined, novelty, novelty_manifest, reference_db, reference_label,
                mmseqs_version,
            ):
                print(f"reusing complete Novelty-U table: {novelty}", file=log, flush=True)
            else:
                novelty_command = [
                    python, str(scorer / "score_mmseqs_novelty.py"),
                    "--input", str(combined), "--reference-db", str(reference_db),
                    "--reference-label", reference_label,
                    "--mmseqs", str(root / "tools/mmseqs/bin/mmseqs"),
                    "--threads", str(threads), "--tmp-dir", str(output / "tmp"),
                    "--out", str(novelty),
                ]
                if split_memory_limit:
                    novelty_command.extend(["--split-memory-limit", str(split_memory_limit)])
                run(novelty_command, log)

            status(status_path, "RUNNING", "pooled_hv", config_sha256=config_hash)
            run([
                python, str(helpers / "complete_unconditional_novelty_hv.py"),
                "--sequence-scores", *(str(path) for path in score_files),
                "--novelty-u", str(novelty), "--output-table", str(main_table),
                "--output-sequence-scores", str(all_scores),
                "--manifest", str(hv_manifest),
                "--expected-models", ",".join(expected_models),
                "--requests-per-model", str(config["requests_per_model"]),
                "--normalization-scope-label",
                "all complete generated sequences pooled across the configured Unconditional cohort",
                *(["--frozen-manifest", str(resolve(root, config["hypervolume"]["frozen_manifest"]))]
                  if config["hypervolume"].get("frozen_manifest") else []),
            ], log)

            status(status_path, "RUNNING", "merge_supplementary", config_sha256=config_hash)
            merge_group_tables(
                supplementary_files,
                output / "results/unconditional_supplementary_metrics.tsv",
                expected_models,
            )
            merge_group_tables(
                completeness_files,
                output / "results/unconditional_scoring_completeness.tsv",
                expected_models,
            )

            status(status_path, "RUNNING", "audit_results", config_sha256=config_hash)
            run([
                python,
                str(root / "experiments/prodipa/unconditional_evaluation/audit_results.py"),
                "--config", str(config_path), "--project-root", str(root),
            ], log)

        compatibility = config.get("compatibility_main_table")
        if compatibility:
            target = resolve(root, compatibility)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            shutil.copyfile(main_table, temporary)
            os.replace(temporary, target)
        atomic_json(output / "COMPLETE.json", {
            "status": "COMPLETE",
            "config": str(config_path),
            "config_sha256": config_hash,
            "models": expected_models,
            "main_table": str(main_table),
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        atomic_json(output / "EVALUATION_PROTOCOL.json", {
            "schema_version": config["schema_version"],
            "config": str(config_path),
            "config_sha256": config_hash,
            "requests_per_model": config["requests_per_model"],
            "generation_protocol": config["generation_protocol"],
            "references": config["references"],
            "scorers": config["scorers"],
            "resources": config["resources"],
            "hypervolume": config["hypervolume"],
            "supplementary_thresholds": config["supplementary_thresholds"],
            "models": expected_models,
        })
        status(status_path, "COMPLETE", "complete", config_sha256=config_hash)
    except BaseException:
        status(status_path, "FAILED", "cohort_completion", config_sha256=config_hash)
        raise


if __name__ == "__main__":
    main()
