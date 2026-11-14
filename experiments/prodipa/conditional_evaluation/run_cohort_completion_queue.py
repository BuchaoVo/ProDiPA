#!/usr/bin/env python3
"""Wait for Conditional method scores, then publish pooled novelty and HV."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
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
        "state": state, "stage": stage, **extra,
        "updated_at": datetime.now().astimezone().isoformat(),
    })


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run(command: list[str], log) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def ids(path: Path, field: str) -> set[str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    rows = read(path)
    values = [row.get(field, "") for row in rows]
    return set(values) if values and all(values) and len(values) == len(set(values)) else None


def novelty_cache_valid(
    combined: Path, novelty: Path, manifest: Path, database: Path,
    label: str, mmseqs_version: str,
) -> bool:
    combined_ids = ids(combined, "sequence_id")
    if combined_ids is None or ids(novelty, "sequence_id") != combined_ids or not manifest.is_file():
        return False
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("reference_kind") == "mmseqs_database"
        and metadata.get("reference_label") == label
        and Path(metadata.get("reference_path", "")).resolve() == database.resolve()
        and metadata.get("queries") == len(combined_ids)
        and metadata.get("mmseqs_version") == mmseqs_version
    )


def missing_extensions(score_files: list[Path], required: list[str]) -> dict[str, int]:
    missing: dict[str, int] = {}
    for path in score_files:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        rows = read(path)
        valid = [row for row in rows if row.get("valid_complete", "").lower() == "true"]
        for field in required:
            count = sum(row.get(field, "") == "" for row in valid)
            if count:
                missing[f"{path}:{field}"] = count
    return missing


def merge_tables(paths: list[Path], output: Path, expected: list[str]) -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
            rows.extend(reader)
    by_model = {row["Model"]: row for row in rows}
    if len(rows) != len(by_model) or set(by_model) != set(expected):
        raise ValueError(f"cohort table model mismatch: observed={sorted(by_model)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=output.parent,
        delete=False, prefix=f".{output.name}.",
    ) as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(by_model[model] for model in expected)
        temporary = handle.name
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/conditional.json"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = resolve(root, str(args.config))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("evaluation_name") != "conditional":
        raise ValueError("unsupported Conditional evaluation config")
    output = resolve(root, config["output_root"])
    for directory in ("prepared", "scores", "results", "logs", "tmp"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    lock = (output / ".completion.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    status_path = output / "STATUS.json"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

    expected: list[str] = []
    score_files: list[Path] = []
    unique_files: list[Path] = []
    supplementary_files: list[Path] = []
    completeness_files: list[Path] = []
    for group in config["score_groups"]:
        expected.extend(group["models"])
        group_root = resolve(root, group["evaluation_root"])
        score_files.append(group_root / "results/sequence_scores.tsv")
        unique_files.append(group_root / "prepared/unique_valid_sequences.tsv")
        supplementary_files.append(group_root / "results/supplementary_metrics.tsv")
        completeness_files.append(group_root / "results/scoring_completeness.tsv")
    if len(expected) != len(set(expected)):
        raise ValueError("configured model identifiers are not unique")

    database = resolve(root, config["references"]["novelty_u_database"])
    ready = resolve(root, config["references"]["novelty_u_ready_marker"])
    required_paths = score_files + unique_files + supplementary_files + completeness_files + [
        database, Path(str(database) + ".dbtype"), ready,
    ]
    required_extensions = [
        field for field, enabled in config.get("required_extensions", {}).items() if enabled
    ]
    while True:
        missing = [str(path) for path in required_paths if not path.is_file() or path.stat().st_size == 0]
        extension_gaps = missing_extensions(score_files, required_extensions)
        if not missing and not extension_gaps:
            break
        status(
            status_path, "WAITING", "wait_for_inputs", missing=missing,
            missing_extension_counts=extension_gaps, config_sha256=config_hash,
        )
        if args.once:
            print(json.dumps({"state": "WAITING", "missing": missing, "extensions": extension_gaps}, indent=2))
            return
        time.sleep(args.poll_seconds)

    python = os.environ.get("PRODIPA_PYTHON", "__PRODIPA_ENVS__/ProDiPA/bin/python")
    helpers = root / "experiments/prodipa/external_baseline_evaluation"
    scorer = root / "experiments/prodipa/conditional_evaluation"
    combined = output / "prepared/cohort_unique_valid_sequences.tsv"
    novelty = output / "scores/novelty_u.tsv"
    main_table = output / "results/conditional_prodipa_main.tsv"
    all_scores = output / "results/conditional_sequence_scores.tsv"
    manifest = output / "results/conditional_hv_manifest.json"
    log_path = output / "logs/cohort_completion.log"
    resources = config["resources"]
    threads = args.threads or int(resources.get("mmseqs_threads", 32))

    try:
        with log_path.open("a", encoding="utf-8") as log:
            status(status_path, "RUNNING", "combine_sequences", config_sha256=config_hash)
            run([
                python, str(helpers / "combine_unique_sequences.py"), "--input",
                *(str(path) for path in unique_files), "--output", str(combined),
            ], log)

            status(status_path, "RUNNING", "novelty_u", config_sha256=config_hash)
            novelty_manifest = novelty.with_suffix(".manifest.json")
            label = config["references"]["novelty_u_label"]
            mmseqs_version = config["scorers"]["novelty"]["mmseqs_commit"]
            if not novelty_cache_valid(combined, novelty, novelty_manifest, database, label, mmseqs_version):
                command = [
                    python, str(scorer / "score_mmseqs_novelty.py"),
                    "--input", str(combined), "--reference-db", str(database),
                    "--reference-label", label, "--mmseqs", str(root / "tools/mmseqs/bin/mmseqs"),
                    "--threads", str(threads), "--tmp-dir", str(output / "tmp"), "--out", str(novelty),
                ]
                if resources.get("mmseqs_split_memory_limit"):
                    command.extend(["--split-memory-limit", str(resources["mmseqs_split_memory_limit"])])
                run(command, log)

            status(status_path, "RUNNING", "pooled_hv", config_sha256=config_hash)
            run([
                python, str(scorer / "complete_pooled_hv.py"),
                "--sequence-scores", *(str(path) for path in score_files),
                "--novelty-u", str(novelty), "--output-table", str(main_table),
                "--output-sequence-scores", str(all_scores), "--manifest", str(manifest),
                "--expected-models", ",".join(expected),
                "--requests-per-model", str(config["requests_per_model"]),
            ], log)

            status(status_path, "RUNNING", "merge_supplementary", config_sha256=config_hash)
            merge_tables(
                supplementary_files,
                output / "results/conditional_supplementary_metrics.tsv", expected,
            )
            merge_tables(
                completeness_files,
                output / "results/conditional_scoring_completeness.tsv", expected,
            )
            status(status_path, "RUNNING", "audit_results", config_sha256=config_hash)
            run([
                python, str(scorer / "audit_results.py"), "--config", str(config_path),
                "--project-root", str(root),
            ], log)

        (output / "results/EVALUATION_STATUS.md").write_text(
            "# Conditional evaluation status\n\n"
            "Status: COMPLETE.\n\n"
            f"Models: {', '.join(expected)}.\n\n"
            f"Requests: {config['requests_per_model']} per model, "
            f"{config['requests_per_model'] * len(expected)} total.\n\n"
            "All configured main-table and supplementary metrics are complete; "
            "the pooled cohort HV and fail-closed result audit have finished.\n",
            encoding="utf-8",
        )

        atomic_json(output / "COMPLETE.json", {
            "status": "COMPLETE", "config": str(config_path),
            "config_sha256": config_hash, "models": expected,
            "main_table": str(main_table),
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        atomic_json(output / "EVALUATION_PROTOCOL.json", {
            key: config[key] for key in (
                "schema_version", "evaluation_name", "requests_per_model", "condition",
                "generation_protocol", "references", "required_extensions", "scorers",
                "resources", "hypervolume", "supplementary_thresholds",
            )
        } | {"config": str(config_path), "config_sha256": config_hash, "models": expected})
        status(status_path, "COMPLETE", "complete", config_sha256=config_hash)
    except BaseException:
        status(status_path, "FAILED", "cohort_completion", config_sha256=config_hash)
        raise


if __name__ == "__main__":
    main()
