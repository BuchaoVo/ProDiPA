import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "split_uniref_main.py"
SPEC = importlib.util.spec_from_file_location("split_uniref_main", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_component_split_never_breaks_union_find_groups():
    identifiers = [f"S{i}" for i in range(20)]
    union_find = MODULE.UnionFind(identifiers)
    union_find.union("S0", "S1")
    union_find.union("S1", "S2")
    split, component = MODULE.component_split(identifiers, union_find, seed=42)
    assert split["S0"] == split["S1"] == split["S2"]
    assert component["S0"] == component["S1"] == component["S2"]
    assert set(split.values()) == {"train", "validation", "test_h"}


def test_cross_edges_detects_only_split_violations():
    edges = [
        {"sequence_id_a": "A", "sequence_id_b": "B"},
        {"sequence_id_a": "B", "sequence_id_b": "C"},
    ]
    split = {"A": "train", "B": "train", "C": "test_h"}
    assert MODULE.cross_edges(edges, split) == [edges[1]]
