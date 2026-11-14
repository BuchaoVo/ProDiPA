import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_batch.py"
if not SCRIPT.exists():
    pytest.skip(
        "retired build_batch compatibility script is not part of the maintained homology workflow",
        allow_module_level=True,
    )
spec = importlib.util.spec_from_file_location("build_batch", SCRIPT)
build_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_batch)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], text=True, capture_output=True)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metadata_row(accession, family, uniprot, verified="true"):
    return [accession, "Swiss-Prot_2099_01", family, f"Family {family}", verified, uniprot, "", "synthetic"]


def test_stage_deduplicates_with_provenance_and_quarantines_conflict(tmp_path):
    seq_a = "A" * 60
    seq_b = "C" * 60
    fasta = write(tmp_path / "raw.fasta", f">a\n{seq_a}\n>b\n{seq_a}\n>c\n{seq_b}\n>d\n{'X' * 60}\n")
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(build_batch.META_FIELDS)
        writer.writerow(metadata_row("a", "F1", "U1"))
        writer.writerow(metadata_row("b", "F2", "U2"))
        writer.writerow(metadata_row("c", "F3", "U3"))
        writer.writerow(metadata_row("d", "F4", "U4"))
    out = tmp_path / "staged"
    result = run("stage", "--fasta", fasta, "--metadata", metadata, "--out", out)
    assert result.returncode == 0, result.stderr
    assert len(csv_rows(out / "staged.csv")) == 1
    assert len(csv_rows(out / "quarantine.csv")) == 2
    assert csv_rows(out / "exclusions.csv")[0]["reason"] == "noncanonical_residue"


def staged_row(letter, family, uniprot, verified="true"):
    sequence = letter * 60
    seq_hash = build_batch.sha256_text(sequence)
    return {
        "seq_sha256": seq_hash,
        "sequence": sequence,
        "length": "60",
        "accessions": json.dumps([f"A_{letter}"]),
        "source_releases": json.dumps(["R1"]),
        "family_ids": json.dumps([family]),
        "family_names": json.dumps([family]),
        "verified_family_ids": json.dumps([family]) if verified == "true" else "[]",
        "family_verified": verified,
        "uniprot_ids": json.dumps([uniprot]),
        "pdb_ids": "[]",
        "sources": json.dumps(["synthetic"]),
    }


def test_split_unions_all_threshold_edges_and_reserves_verified_family(tmp_path):
    rows = [
        staged_row("A", "F1", "U1"),
        staged_row("C", "F1", "U2"),
        staged_row("D", "F2", "U3"),
        staged_row("E", "HELD", "U4"),
        staged_row("F", "F3", "U5"),
    ]
    staged = tmp_path / "staged.csv"
    build_batch.write_csv(staged, rows)
    cluster = write(tmp_path / "cluster.tsv", "")
    all_hits = write(
        tmp_path / "hits.tsv",
        f"{rows[0]['seq_sha256']}\t{rows[1]['seq_sha256']}\t30\t0.8\t0.8\n",
    )
    out = tmp_path / "split"
    result = run(
        "split", "--staged", staged, "--cluster-tsv", cluster,
        "--all-hits-tsv", all_hits, "--heldout-family", "HELD",
        "--val-fraction", "0.2", "--test-h-fraction", "0.2", "--out", out,
    )
    assert result.returncode == 0, result.stderr
    manifest = {row["seq_sha256"]: row for row in csv_rows(out / "split_manifest.csv")}
    assert manifest[rows[0]["seq_sha256"]]["component_id"] == manifest[rows[1]["seq_sha256"]]["component_id"]
    assert manifest[rows[0]["seq_sha256"]]["split"] == manifest[rows[1]["seq_sha256"]]["split"]
    assert manifest[rows[3]["seq_sha256"]]["split"] == "test_f"


def test_split_unions_optional_parent_identity(tmp_path):
    rows = [staged_row("A", "F1", "U1"), staged_row("C", "F1", "U2"), staged_row("D", "HELD", "U3")]
    rows[0]["parent_ids"] = json.dumps(["PARENT-1"])
    rows[1]["parent_ids"] = json.dumps(["PARENT-1"])
    rows[2]["parent_ids"] = json.dumps(["PARENT-2"])
    staged = tmp_path / "staged.csv"
    build_batch.write_csv(staged, rows)
    empty_cluster = write(tmp_path / "cluster.tsv", "")
    empty_hits = write(tmp_path / "hits.tsv", "")
    out = tmp_path / "split"
    result = run(
        "split", "--staged", staged, "--cluster-tsv", empty_cluster,
        "--all-hits-tsv", empty_hits, "--heldout-family", "HELD", "--out", out,
    )
    assert result.returncode == 0, result.stderr
    manifest = {row["seq_sha256"]: row for row in csv_rows(out / "split_manifest.csv")}
    assert manifest[rows[0]["seq_sha256"]]["component_id"] == manifest[rows[1]["seq_sha256"]]["component_id"]


def test_independent_audit_fails_on_cross_split_hit(tmp_path):
    left, right = staged_row("A", "F1", "U1"), staged_row("C", "F2", "U2")
    manifest = [{**left, "component_id": left["seq_sha256"], "split": "train"}, {**right, "component_id": right["seq_sha256"], "split": "test_h"}]
    manifest_path = tmp_path / "manifest.csv"
    build_batch.write_csv(manifest_path, manifest)
    hits = write(tmp_path / "hits.tsv", f"{left['seq_sha256']}\t{right['seq_sha256']}\t0.31\t0.9\t0.9\n")
    out = tmp_path / "audit"
    result = run("audit", "--manifest", manifest_path, "--hits-tsv", hits, "--out", out)
    assert result.returncode != 0
    report = json.loads((out / "independent_sequence_audit.json").read_text())
    assert report["status"] == "FAIL"
    assert report["cross_split_hits"] == 1


def test_export_requires_exact_full_cohort_scores_and_keeps_eval_shared(tmp_path):
    rows = []
    for letter, split in zip("ACDE", build_batch.SPLITS):
        row = staged_row(letter, "F1" if split != "test_f" else "HELD", f"U{letter}")
        rows.append({**row, "component_id": row["seq_sha256"], "split": split})
    manifest = tmp_path / "manifest.csv"
    build_batch.write_csv(manifest, rows)
    scores = tmp_path / "scores.csv"
    build_batch.write_csv(
        scores,
        [
            {"seq_sha256": row["seq_sha256"], "fold_plddt_100": 90, "solubility_01": 0.9, "thermostability_01": 0.9}
            for row in rows
        ],
        build_batch.SCORE_FIELDS,
    )
    out = tmp_path / "export"
    result = run("export", "--manifest", manifest, "--scores", scores, "--teacher-cap", "5", "--out", out)
    assert result.returncode == 0, result.stderr
    assert sum(1 for _ in (out / "foldability_teacher_train.jsonl").open()) == 1
    for split in ("val", "test_h", "test_f"):
        exported = csv_rows(out / f"{split}_shared_three_oracle.csv")
        assert len(exported) == 1
        assert set(build_batch.SCORE_FIELDS[1:]).issubset(exported[0])

    incomplete = tmp_path / "incomplete.csv"
    build_batch.write_csv(incomplete, csv_rows(scores)[:-1], build_batch.SCORE_FIELDS)
    failed = run("export", "--manifest", manifest, "--scores", incomplete, "--out", tmp_path / "bad")
    assert failed.returncode != 0


def test_full_vocab_poe_advantage_and_eos():
    sys.path.insert(0, str(ROOT / "prodipa_e0"))
    from poe import full_vocab_poe, scalar_advantage

    logits = np.array([[[[3.0, 0.0, 1.0]]], [[[1.0, 0.0, 3.0]]]])
    poe = full_vocab_poe(logits)
    np.testing.assert_allclose(np.exp(poe).sum(axis=-1), 1.0, atol=1e-6)
    eos_id = np.array([[2]])
    advantage = scalar_advantage(np.zeros((1, 1, 3)), poe, eos_id)
    assert advantage.shape == (1, 1)
    assert np.isfinite(advantage).all()


def test_parity_checker_rejects_different_prefixes(tmp_path):
    logits = np.array([[[[1.0, 2.0]]], [[[2.0, 1.0]]]], dtype=np.float32)
    shared = {
        "token_ids": np.array([[1]]),
        "response_mask": np.array([[True]]),
        "response_start": np.array(2),
        "condition_id": np.array("family-a"),
        "observed_eos": np.array(True),
        "eos_position": np.array(0),
        "truncated": np.array(False),
        "teacher_versions": np.array(["fold@a", "sol@b"]),
        "teacher_weights": np.array([0.5, 0.5], dtype=np.float32),
        "teacher_temperatures": np.array([1.0, 1.0], dtype=np.float32),
        "tokenizer_hash": np.array("tok"),
        "prompt_hash": np.array("prompt"),
    }
    forward = tmp_path / "forward.npz"
    cache = tmp_path / "cache.npz"
    np.savez(forward, prefix_input_ids=np.array([[1, 2]]), teacher_logits=logits, **shared)
    np.savez(cache, prefix_input_ids=np.array([[1, 3]]), poe_scalar_logprob=np.array([[-0.6931472]]), **shared)
    result = subprocess.run(
        [sys.executable, str(ROOT / "prodipa_e0" / "check_teacher_cache_parity.py"), "--forward", str(forward), "--cache", str(cache)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert '"status": "FAIL"' in result.stdout


def test_parity_checker_compares_full_vocab_scalar_alpha(tmp_path):
    logits = np.array([[[[1.0, 2.0]]], [[[2.0, 1.0]]]], dtype=np.float32)
    shared = {
        "prefix_input_ids": np.array([[1, 2]]),
        "token_ids": np.array([[1]]),
        "response_mask": np.array([[True]]),
        "response_start": np.array(2),
        "condition_id": np.array("family-a"),
        "observed_eos": np.array(True),
        "eos_position": np.array(0),
        "truncated": np.array(False),
        "teacher_versions": np.array(["fold@a", "sol@b"]),
        "teacher_weights": np.array([0.5, 0.5], dtype=np.float32),
        "teacher_temperatures": np.array([1.0, 1.0], dtype=np.float32),
        "tokenizer_hash": np.array("tok"),
        "prompt_hash": np.array("prompt"),
    }
    forward = tmp_path / "forward.npz"
    cache = tmp_path / "cache.npz"
    np.savez(forward, teacher_logits=logits, **shared)
    np.savez(cache, poe_scalar_logprob=np.array([[-np.log(2)]], dtype=np.float32), **shared)
    result = subprocess.run(
        [sys.executable, str(ROOT / "prodipa_e0" / "check_teacher_cache_parity.py"), "--forward", str(forward), "--cache", str(cache)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"status": "PASS"' in result.stdout
