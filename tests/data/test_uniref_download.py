import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "download_uniref50.py"
SPEC = importlib.util.spec_from_file_location("download_uniref50", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_byte_ranges_cover_input_exactly_once():
    ranges = MODULE.byte_ranges(23, 5)
    observed = [value for start, end in ranges for value in range(start, end + 1)]
    assert observed == list(range(23))
    assert ranges == [(0, 4), (5, 9), (10, 14), (15, 19), (20, 22)]
