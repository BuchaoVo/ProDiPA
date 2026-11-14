import pytest

from prodipa.data.dataset import training_records


def valid_record(**changes):
    row = {
        "seq_sha256": "abc",
        "accessions": ["P1"],
        "uniprot_ids": ["P1"],
        "source_release": "R1",
        "dataset_version": "prodipa_v1",
        "split_version": "split_v1",
        "split": "train",
    }
    row.update(changes)
    return row


def test_num_09_loader_rejects_test_and_missing_provenance():
    assert training_records([valid_record()])
    with pytest.raises(ValueError, match="Train"):
        training_records([valid_record(split="test_h")])
    with pytest.raises(ValueError, match="provenance"):
        training_records([valid_record(accessions=[])])

