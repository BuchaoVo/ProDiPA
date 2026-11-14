#!/usr/bin/env python3
"""Publish the retrained Sol-V3-era Prefix teachers and reset the old student."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/experiments/unconditional_teacher_sol_v3_fasta60_v1"
SOURCE_EVAL = SOURCE / "evaluation/full_metrics"
SOURCE_COHORT = SOURCE / "evaluation/cohort"
TARGET_TEACHERS = ROOT / "outputs/unconditional/protgpt2"
TARGET_EVAL = ROOT / "outputs/evaluation/unconditional/methods/protgpt2_base_and_prefix_teachers"
TARGET_STUDENT = ROOT / "outputs/unconditional/protgpt2_multiteacher_student"
TARGET_STUDENT_EVAL = ROOT / "outputs/evaluation/unconditional/methods/protgpt2_multiteacher_student"
ATTRIBUTES = ("fold", "sol", "thermo")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merged_model_table(
    old_path: Path, new_path: Path, model_field: str
) -> tuple[list[str], list[dict[str, str]]]:
    old_fields, old_rows = read_tsv(old_path)
    new_fields, new_rows = read_tsv(new_path)
    if old_fields != new_fields:
        raise ValueError(f"schema mismatch: {old_path} vs {new_path}")
    rows = [row for row in old_rows if row.get(model_field) == "base"] + new_rows
    expected = ("base", "fold_teacher", "sol_teacher", "thermo_teacher")
    if {row.get(model_field) for row in rows} != set(expected):
        raise ValueError(f"model merge failed for {old_path.name}")
    order = {name: i for i, name in enumerate(expected)}
    rows.sort(key=lambda row: (order.get(row.get(model_field, ""), 99), row.get("sample_id", "")))
    return old_fields, rows


def replace_dir(staged: Path, target: Path) -> None:
    displaced = target.with_name(f".{target.name}.replaced.{os.getpid()}")
    if displaced.exists():
        shutil.rmtree(displaced)
    if target.exists():
        os.replace(target, displaced)
    try:
        os.replace(staged, target)
    except BaseException:
        if displaced.exists() and not target.exists():
            os.replace(displaced, target)
        raise
    if displaced.exists():
        shutil.rmtree(displaced)


def main() -> None:
    if not (SOURCE / "TRAINING_COMPLETE.json").is_file():
        raise FileNotFoundError("teacher training is incomplete")
    if not (SOURCE / "evaluation/EVALUATION_COMPLETE.json").is_file():
        raise FileNotFoundError("teacher evaluation is incomplete")
    if not (SOURCE_COHORT / "COMPLETE.json").is_file():
        raise FileNotFoundError("teacher Novelty-U/HV cohort is incomplete")
    for attribute in ATTRIBUTES:
        adapter = SOURCE / "teachers" / attribute / "best_adapter/adapter_model.safetensors"
        if not adapter.is_file():
            raise FileNotFoundError(adapter)

    teacher_stage = TARGET_TEACHERS.with_name(f".{TARGET_TEACHERS.name}.next.{os.getpid()}")
    eval_stage = TARGET_EVAL.with_name(f".{TARGET_EVAL.name}.next.{os.getpid()}")
    for path in (teacher_stage, eval_stage):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    protocol = {
        "version": 1,
        "name": "protgpt2_fasta60_v1",
        "leading_newline": True,
        "line_width_aa": 60,
        "trailing_newline_before_eos": True,
        "generation_sequence_format": "legacy",
        "generation_max_aa": 512,
        "forced_terminal_is_complete": False,
    }
    hashes: dict[str, str] = {}
    for attribute in ATTRIBUTES:
        source = SOURCE / "teachers" / attribute
        target = teacher_stage / attribute
        target.mkdir()
        shutil.copytree(source / "best_adapter", target / "best_adapter")
        for name in (
            "DONE.json", "preflight.json", "run_metadata.json", "train_result.json",
            "training_history.tsv",
        ):
            if (source / name).is_file():
                shutil.copy2(source / name, target / name)
        config = SOURCE / "teachers/configs" / f"{attribute}.yaml"
        shutil.copy2(config, target / "training_config.yaml")
        (target / "best_adapter/TEACHER_SEQUENCE_PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
        )
        hashes[attribute] = sha256(target / "best_adapter/adapter_model.safetensors")

    # Keep the existing Base evidence and replace all three teacher cohorts.
    (eval_stage / "generation/unconditional").mkdir(parents=True)
    shutil.copy2(
        TARGET_EVAL / "generation/unconditional/base.tsv",
        eval_stage / "generation/unconditional/base.tsv",
    )
    old_base_meta = TARGET_EVAL / "generation/unconditional/base.metadata.json"
    if old_base_meta.is_file():
        shutil.copy2(old_base_meta, eval_stage / "generation/unconditional/base.metadata.json")
    for model in ("fold_teacher", "sol_teacher", "thermo_teacher"):
        for suffix in (".tsv", ".metadata.json"):
            source = SOURCE_EVAL / "generation/unconditional" / f"{model}{suffix}"
            if source.is_file():
                shutil.copy2(source, eval_stage / "generation/unconditional" / source.name)

    table_specs = (
        ("prepared/all_requests.tsv", "model"),
        ("results/sequence_scores.tsv", "model"),
        ("results/supplementary_metrics.tsv", "Model"),
        ("results/scoring_completeness.tsv", "Model"),
        ("results/unconditional_prodipa_main.tsv", "Model"),
    )
    for relative, model_field in table_specs:
        fields, rows = merged_model_table(TARGET_EVAL / relative, SOURCE_EVAL / relative, model_field)
        write_tsv(eval_stage / relative, fields, rows)

    _, merged_requests = read_tsv(eval_stage / "prepared/all_requests.tsv")
    base_valid = {
        row["sequence_id"]: {
            "sequence_id": row["sequence_id"], "sequence": row["sequence"],
            "length": row["length_aa"], "valid_complete": "true",
        }
        for row in merged_requests
        if row.get("model") == "base" and row.get("valid_complete", "").lower() == "true"
    }
    unique_fields, new_unique = read_tsv(SOURCE_EVAL / "prepared/unique_valid_sequences.tsv")
    by_sequence = dict(base_valid)
    for row in new_unique:
        previous = by_sequence.get(row["sequence_id"])
        if previous is not None and previous["sequence"] != row["sequence"]:
            raise ValueError(f"sequence hash collision: {row['sequence_id']}")
        by_sequence[row["sequence_id"]] = row
    write_tsv(
        eval_stage / "prepared/unique_valid_sequences.tsv", unique_fields,
        [by_sequence[key] for key in sorted(by_sequence)],
    )

    base_ids = set(base_valid)
    score_names = ("protgpt2_ppl.tsv", "novelty_t.tsv", "protein_sol.tsv", "temberture.tsv", "esmfold.tsv")
    for name in score_names:
        old_fields, old_rows = read_tsv(TARGET_EVAL / "scores" / name)
        new_fields, new_rows = read_tsv(SOURCE_EVAL / "scores" / name)
        if old_fields != new_fields:
            raise ValueError(f"score schema mismatch: {name}")
        by_id = {row["sequence_id"]: row for row in old_rows if row["sequence_id"] in base_ids}
        for row in new_rows:
            by_id[row["sequence_id"]] = row
        write_tsv(eval_stage / "scores" / name, old_fields, [by_id[key] for key in sorted(by_id)])

    for directory in ("logs", "prepared/references"):
        source = SOURCE_EVAL / directory
        if source.is_dir():
            shutil.copytree(source, eval_stage / directory, dirs_exist_ok=True)
    (eval_stage / "SCORING_STATUS.json").write_text(
        json.dumps({"state": "COMPLETE", "paradigm": "unconditional", "stage": "published"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (eval_stage / "TEACHER_EVALUATION_COMPLETE.json").write_text(
        json.dumps({
            "status": "COMPLETE", "models": ["base", "fold_teacher", "sol_teacher", "thermo_teacher"],
            "source": str(SOURCE_EVAL), "published_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n", encoding="utf-8",
    )

    _, metrics = read_tsv(SOURCE_COHORT / "results/unconditional_prodipa_main.tsv")
    release = {
        "status": "PUBLISHED_SOL_V3_TEACHERS_STUDENT_RETRAIN_REQUIRED",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "data_protocol": "Sol-Designable-V3-Strict70-15",
        "teacher_sequence_protocol": protocol,
        "teacher_adapter_sha256": hashes,
        "teacher_metrics_frozen_hv": metrics,
        "source_experiment": str(SOURCE),
        "replacement_policy": "canonical Prefix teachers and teacher evaluation replaced atomically; old student removed",
    }
    (teacher_stage / "CANONICAL_RELEASE.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )

    replace_dir(teacher_stage, TARGET_TEACHERS)
    replace_dir(eval_stage, TARGET_EVAL)
    for target in (TARGET_STUDENT, TARGET_STUDENT_EVAL):
        if target.exists():
            shutil.rmtree(target)
    report = SOURCE / "CANONICAL_PROMOTION.json"
    report.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
