import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_homology.py"
SPEC = importlib.util.spec_from_file_location("audit_homology", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unique_sequences_preserves_duplicate_provenance():
    records, metadata = MODULE.unique_sequences(
        [{"sequence": "AAA", "id": "a"}, {"sequence": "AAA", "id": "b"}],
        "X",
    )
    assert records == [("X000001", "AAA")]
    assert [row["id"] for row in metadata["X000001"]] == ["a", "b"]


def test_source_provenance_supports_main_schema():
    provenance = MODULE.source_provenance({
        "representative_accession": "P12345",
        "uniref_cluster": "UniRef50_P12345",
    })
    assert provenance == {
        "accession": "P12345",
        "cluster": "UniRef50_P12345",
        "condition": "",
        "interpro": "",
    }
