import gzip
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "build_uniref_main.py"
SPEC = importlib.util.spec_from_file_location("build_uniref_main", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stream_qc_and_reservoir_are_deterministic(tmp_path):
    records = [
        ("UniRef50_P00001 Alpha n=2 Tax=Bacteria RepID=A_TEST", "A" * 50),
        ("UniRef50_P00002 Beta n=1 Tax=Homo sapiens RepID=B_HUMAN", "C" * 51),
        ("UniRef50_P00003 Too short n=1 Tax=Viruses RepID=C_TEST", "D" * 49),
        ("UniRef50_P00004 Ambiguous n=1 Tax=Archaea RepID=D_TEST", "E" * 49 + "X"),
        ("UniRef50_P00005 Gamma n=3 Tax=Eukaryota RepID=E_TEST", "F" * 52),
    ]
    source = tmp_path / "fixture.fasta.gz"
    with gzip.open(source, "wt", encoding="ascii") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")

    first, counts = MODULE.sample_records(source, target=2, seed=42)
    second, second_counts = MODULE.sample_records(source, target=2, seed=42)
    assert [row["sequence"] for row in first] == [row["sequence"] for row in second]
    assert counts == second_counts
    assert counts["input_records"] == 5
    assert counts["qc_passing_records"] == 3
    assert counts["excluded_length_outside_50_512"] == 1
    assert counts["excluded_noncanonical_residue"] == 1
    assert len(first) == 2
    assert all(set(row["sequence"]) <= MODULE.ALPHABET for row in first)


def test_uniref_header_parser_preserves_provenance():
    parsed = MODULE.parse_header(
        "UniRef50_P99999 Cytochrome c n=33 Tax=Eukaryota TaxID=2759 RepID=CYC_HUMAN"
    )
    assert parsed["cluster"] == "UniRef50_P99999"
    assert parsed["representative_accession"] == "P99999"
    assert parsed["members"] == "33"
    assert parsed["taxon_id"] == "2759"
    assert parsed["rep_id"] == "CYC_HUMAN"


def test_uniref_header_parser_allows_readme_format_without_taxid():
    parsed = MODULE.parse_header(
        "UniRef50_P99999 Cytochrome c n=33 Tax=Eukaryota RepID=CYC_HUMAN"
    )
    assert parsed["taxon_id"] is None
