import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "build_tpr_teacher_datasets.py"
SPEC = importlib.util.spec_from_file_location("build_tpr_teacher_datasets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_zero_capacity_bin_is_never_filled():
    train, test, _ = MODULE.optimize_quota_pair(
        [0, 192, 309, 328], [0, 52, 94, 123], [7, 406, 99, 0]
    )
    assert train == (0, 192, 108, 0)
    assert test == (0, 52, 48, 0)
    assert sum(train) == 300
    assert sum(test) == 100


def test_selection_uses_one_sequence_per_homology_group():
    rows = [
        {
            "sequence_id": "a1", "sequence": "A" * 100, "homology_group": "g1",
            "split": "test", "fold_score": "90", "sol_score": "0.8", "thermo_score": "0.7",
        },
        {
            "sequence_id": "a2", "sequence": "A" * 200, "homology_group": "g1",
            "split": "test", "fold_score": "90", "sol_score": "0.8", "thermo_score": "0.7",
        },
        {
            "sequence_id": "b", "sequence": "C" * 100, "homology_group": "g2",
            "split": "test", "fold_score": "90", "sol_score": "0.8", "thermo_score": "0.7",
        },
    ]
    selected = MODULE.select_with_quotas(rows, "sol", "test", (1, 1, 0, 0), 42)
    assert len(selected) == 2
    assert len({row["homology_group"] for row in selected}) == 2

