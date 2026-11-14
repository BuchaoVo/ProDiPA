"""Confidence extraction and recovery must preserve per-protein semantics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "esmfold_scorer", Path(__file__).resolve().parents[2] / "scripts/run_esmfold_properties.py"
)
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


def test_batches_bound_padding_and_keep_long_sequences():
    rows = [{"sequence_id": str(n), "sequence": "A" * n} for n in (400, 201, 900, 200, 202)]
    batches = list(scorer.make_batches(rows, 8, 800))
    assert sum(map(len, batches)) == len(rows)
    assert [len(r["sequence"]) for b in batches for r in b] == [200, 201, 202, 400, 900]
    assert all(len(b) == 1 or len(b) * max(len(r["sequence"]) for r in b) <= 800 for b in batches)


@pytest.mark.parametrize("requested,expected", [("cuda", "cuda:2"), ("cuda:0", "cuda:0"), ("cpu", "cpu")])
def test_load_model_normalizes_unindexed_cuda(monkeypatch, requested, expected):
    from unittest.mock import MagicMock
    selected = []
    monkeypatch.setattr(scorer, "patch_compute_tm", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value = model
    monkeypatch.setattr(scorer.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(scorer.EsmForProteinFolding, "from_pretrained", lambda *a, **kw: model)
    _, _, device = scorer.load_model(SimpleNamespace(device=requested, tf32=None, model="local", chunk_size=32))
    assert str(device) == expected
    assert selected == ([] if requested == "cpu" else [device])


def test_scores_exclude_padding_and_ptm_is_per_protein():
    from transformers.models.esm.openfold_utils.loss import compute_tm
    torch.manual_seed(42)
    logits = torch.randn(2, 7, 7, 64)
    outputs = SimpleNamespace(plddt=torch.full((2, 7, 37), 0.7), ptm_logits=logits,
                              predicted_aligned_error=torch.full((2, 7, 7), 4.0))
    outputs.plddt[0, 3:] = 100
    outputs.predicted_aligned_error[0, 3:, :] = 1000
    outputs.predicted_aligned_error[0, :, 3:] = 1000
    results = scorer.extract_metrics(outputs, [3, 7])
    assert float(results[0]["fold_plddt_100"]) == pytest.approx(70)
    assert float(results[0]["fold_mean_pae"]) == pytest.approx(4)
    for i, length in enumerate((3, 7)):
        expected = float(compute_tm(logits[i, :length, :length], max_bin=31, no_bins=64))
        assert float(results[i]["fold_ptm_01"]) == pytest.approx(expected)


def test_oom_shrinks_subsequent_batches_without_losing_ids(monkeypatch):
    attempted = []

    def infer(batch, *args):
        attempted.append(len(batch))
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return [{"fold_plddt_100": "80", "fold_ptm_01": "0.7", "fold_mean_pae": "5"} for _ in batch]

    monkeypatch.setattr(scorer, "infer_batch", infer)
    args = SimpleNamespace(max_retries=0, batch_size=8)
    rows = [{"sequence_id": str(i)} for i in range(8)]
    results = scorer.run_batch(rows, args, None, None, torch.device("cpu"))
    assert [r["sequence_id"] for r in results] == [str(i) for i in range(8)]
    assert all(r["status"] == "ok" for r in results)
    attempted.clear()
    scorer.run_batch(rows, args, None, None, torch.device("cpu"))
    assert max(attempted) <= 2


def test_atomic_results_and_fasta(tmp_path):
    path = tmp_path / "scores.tsv"
    rows = [{"sequence_id": "a", "status": "failed", "error_reason": "OOM"}]
    scorer.write_rows(path, rows)
    assert scorer.read_rows(path)[0]["status"] == "failed"
    assert not list(tmp_path.glob("*.tmp"))
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">a description\nACD\nEF\n>b\nGH\n")
    assert scorer.read_sequences(fasta) == [
        {"sequence_id": "a", "sequence": "ACDEF"}, {"sequence_id": "b", "sequence": "GH"}]


def test_gpu_selection_respects_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(scorer.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="0, GPU-A, 80000, 0\n1, GPU-B, 70000, 10\n2, GPU-C, 70000, 95\n"))
    assert scorer.eligible_gpus(65, 20, 4) == ["1"]


def test_failed_single_sequence_is_preserved(monkeypatch):
    def infer(*args):
        raise ValueError("invalid input")
    monkeypatch.setattr(scorer, "infer_batch", infer)
    args = SimpleNamespace(max_retries=0, batch_size=1)
    result = scorer.run_batch([{"sequence_id": "bad"}], args, None, None, torch.device("cpu"))
    assert result[0]["status"] == "failed"
    assert result[0]["sequence_id"] == "bad"
    assert "invalid input" in result[0]["error_reason"]


def test_dynamic_queue_delivers_each_batch_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(scorer, "ProcessPoolExecutor", lambda **kw: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(scorer, "worker_batch", lambda batch: batch)
    batches = [[i] for i in range(11)]
    result = list(scorer.parallel_batches(batches, SimpleNamespace(), ["0", "1"]))
    assert sorted(item for batch in result for item in batch) == list(range(11))
