import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "summarize_teacher_data_quality.py"
SPEC = importlib.util.spec_from_file_location("summarize_teacher_data_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_quality_summary_helpers():
    assert MODULE.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert MODULE.effective_count({"a": 2, "b": 2}) == 2
    assert MODULE.effective_count({}) == 0
    assert MODULE.length_bin(100) == "50_100"
    assert MODULE.length_bin(101) == "101_150"


def test_instruction_row_uses_frozen_template_shape():
    template = {"instruction": "generate", "input": "high", "output": "Seq=<{sequence}>"}
    assert MODULE.instruction_row("ACDE", template) == {
        "instruction": "generate",
        "input": "high",
        "output": "Seq=<ACDE>",
    }
