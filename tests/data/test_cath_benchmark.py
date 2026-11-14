import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "prepare_cath_benchmark.py"
SPEC = importlib.util.spec_from_file_location("prepare_cath_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chain_prefix_and_missing_backbone():
    assert MODULE.chain_prefix("1abc.A") == "1abcA"
    coords = {atom: [[0.0, 1.0, 2.0], [0.0, float("nan"), 2.0]] for atom in ("N", "CA", "C", "O")}
    assert MODULE.missing_backbone(coords, 2) == (True, 1)
