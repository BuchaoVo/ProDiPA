#!/usr/bin/env python3
"""Fail-closed audit of the canonical Unconditional evaluation publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def finite(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{row.get('Model')}: missing/non-numeric {field}: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{row.get('Model')}: non-finite {field}: {raw!r}")
    return value


def require_models(rows: list[dict[str, str]], expected: list[str], path: Path) -> None:
    observed = [row.get("Model", "") for row in rows]
    if Counter(observed) != Counter(expected):
        raise ValueError(f"{path}: models={observed}, expected={expected}")


def require_main_models(
    rows: list[dict[str, str]], expected: list[str], allowed_supplemental: list[str], path: Path
) -> list[str]:
    observed = [row.get("Model", "") for row in rows]
    duplicates = [model for model, count in Counter(observed).items() if count != 1]
    missing = sorted(set(expected) - set(observed))
    unknown = sorted(set(observed) - set(expected) - set(allowed_supplemental))
    if duplicates or missing or unknown:
        raise ValueError(
            f"{path}: duplicate={duplicates}, missing={missing}, unknown={unknown}, "
            f"expected={expected}, allowed_supplemental={allowed_supplemental}"
        )
    return [model for model in observed if model in allowed_supplemental]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve(root, config["output_root"])
    result_root = output / "results"
    expected = [model for group in config["score_groups"] for model in group["models"]]
    allowed_supplemental = list(config.get("supplemental_main_table_models", []))
    if len(expected) != len(set(expected)):
        raise ValueError("configured model identifiers are not unique")
    if set(expected) & set(allowed_supplemental):
        raise ValueError("supplemental main-table models overlap the configured cohort")

    main_path = result_root / "unconditional_prodipa_main.tsv"
    fields, main_rows = read_tsv(main_path)
    if fields != config["main_table_fields"]:
        raise ValueError(f"main-table schema={fields}, expected={config['main_table_fields']}")
    supplemental_observed = require_main_models(
        main_rows, expected, allowed_supplemental, main_path
    )
    for row in main_rows:
        values = {field: finite(row, field) for field in fields if field != "Model"}
        if values["PPL"] <= 0 or values["pAE"] < 0:
            raise ValueError(f"{row['Model']}: invalid PPL/pAE")
        for field in ("Novelty-U", "Novelty-T", "pLDDT", "%>70", "%<10"):
            if not 0 <= values[field] <= 100:
                raise ValueError(f"{row['Model']}: {field} outside [0,100]")
        for field in ("Sol", "Thermo", "HV"):
            if not 0 <= values[field] <= 1:
                raise ValueError(f"{row['Model']}: {field} outside [0,1]")

    score_path = result_root / "unconditional_sequence_scores.tsv"
    _, score_rows = read_tsv(score_path)
    request_counts = Counter(row.get("model", "") for row in score_rows)
    expected_count = int(config["requests_per_model"])
    if request_counts != Counter({model: expected_count for model in expected}):
        raise ValueError(f"sequence-score request counts are invalid: {dict(request_counts)}")
    if {row.get("paradigm") for row in score_rows} != {"unconditional"}:
        raise ValueError("sequence-score paradigm is not uniformly unconditional")

    valid_ids = {
        row["sequence_id"] for row in score_rows
        if row.get("valid_complete", "").lower() == "true" and row.get("sequence_id")
    }
    _, novelty_rows = read_tsv(output / "scores/novelty_u.tsv")
    novelty_ids = [row.get("sequence_id", "") for row in novelty_rows]
    if len(novelty_ids) != len(set(novelty_ids)) or set(novelty_ids) != valid_ids:
        raise ValueError("Novelty-U sequence IDs do not exactly match valid cohort sequence IDs")

    hv = json.loads((result_root / "unconditional_hv_manifest.json").read_text(encoding="utf-8"))
    objective_fields = [item.get("field") for item in hv.get("objectives", [])]
    expected_objectives = ["ppl", "novelty_u_percent", "plddt", "sol", "thermo"]
    if objective_fields != expected_objectives:
        raise ValueError(f"HV objective fields={objective_fields}, expected={expected_objectives}")
    if set(hv.get("hypervolume_by_model", {})) != set(expected):
        raise ValueError("HV model set does not match configured cohort")
    if hv.get("reference_point") != config["hypervolume"]["reference_point"]:
        raise ValueError("HV reference point does not match config")

    for name in ("unconditional_supplementary_metrics.tsv", "unconditional_scoring_completeness.tsv"):
        path = result_root / name
        _, rows = read_tsv(path)
        require_models(rows, expected, path)

    audit = {
        "status": "PASS",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "models": expected,
        "supplemental_main_table_models": supplemental_observed,
        "requests_per_model": expected_count,
        "total_requests": len(score_rows),
        "unique_valid_sequences": len(valid_ids),
        "main_table": str(main_path),
        "audited_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(output / "RESULT_AUDIT.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
