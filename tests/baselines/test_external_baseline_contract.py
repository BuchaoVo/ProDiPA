from __future__ import annotations

import json

from prodipa.baselines.audit_generation import audit_rows
from prodipa.baselines.common import build_generation_record, normalize_sequence, sha256_text
from prodipa.baselines.run_progen2 import _split_completion


def make_record(request_id: str, sequence: str = "A" * 50):
    return build_generation_record(
        request_id=request_id,
        benchmark="unconditional",
        model="test_model",
        seed=42,
        condition="",
        raw_output=sequence,
        sequence=sequence,
        terminated=True,
        termination_reason="test_complete",
        generation_method="test",
        generation_config_id="test_v1",
    )


def test_generation_record_has_stable_hash_and_validity():
    row = make_record("test_000001")
    assert row["valid"] is True
    assert row["length_aa"] == 50
    assert row["sequence_sha256"] == sha256_text("A" * 50)


def test_invalid_symbols_are_not_silently_removed():
    row = make_record("test_000001", "A" * 49 + "X")
    assert row["valid"] is False
    assert row["sequence"].endswith("X")


def test_generation_audit_fails_duplicate_request_ids():
    rows = [make_record("test_000001"), make_record("test_000001", "C" * 50)]
    report = audit_rows(
        rows,
        expected_requests=2,
        benchmark="unconditional",
        model="test_model",
        request_prefix="test_",
        min_length=50,
        max_length=512,
    )
    assert report["status"] == "failed"
    assert report["duplicate_request_ids"] == ["test_000001"]


def test_duplicate_sequences_are_reported_but_retained():
    rows = [make_record("test_000001"), make_record("test_000002")]
    report = audit_rows(
        rows,
        expected_requests=2,
        benchmark="unconditional",
        model="test_model",
        request_prefix="test_",
        min_length=50,
        max_length=512,
    )
    assert report["status"] == "passed"
    assert report["duplicate_sequence_instances"] == 1


def test_progen_control_and_terminal_are_not_amino_acids():
    sequence, terminated, reason = _split_completion("1ACDE2TRAILING", "1")
    assert sequence == "ACDE"
    assert terminated is True
    assert reason == "terminal_2"


def test_normalization_only_removes_whitespace():
    assert normalize_sequence(" a c\nd ") == "ACD"
    assert normalize_sequence("ACDX") == "ACDX"
