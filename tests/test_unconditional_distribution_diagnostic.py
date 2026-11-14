from pathlib import Path
import runpy
import torch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    'experiments/prodipa/unconditional_student_training/diagnose_distributions.py'))


class Tokenizer:
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i != 0)


def test_natural_eos_is_observed_and_not_protein():
    rows = MODULE['positions'](Tokenizer(), [65, 67, 10, 0], 0, False)
    assert [r['region'] for r in rows] == ['protein', 'protein', 'whitespace', 'eos']
    assert rows[-1]['aa_before'] == 2
    assert all(r['observed_target'] for r in rows)


def test_token_budget_end_is_not_a_real_eos_target():
    rows = MODULE['positions'](Tokenizer(), [65, 67], 0, True)
    assert rows[-1]['region'] == 'budget_endpoint'
    assert rows[-1]['observed_target'] is False
    assert rows[-1]['target_id'] == ''
    assert rows[-1]['aa_before'] == 2


def test_long_context_binning_counts_aa_not_whitespace():
    rows = MODULE['positions'](Tokenizer(), [65] * 512 + [10, 67], 0, True)
    assert rows[-2]['aa_before'] == 512
    assert rows[-2]['aa_bin'] == '512+'
    assert rows[-1]['aa_before'] == 513


def test_event_choice_is_separated_from_residue_content():
    teacher=torch.tensor([[.04,.04,.01,.9,.01]]).log()
    student=torch.tensor([[.4,.4,.01,.18,.01]]).log()
    result=MODULE['decompose']([teacher]*3+[student],teacher,[1/3]*3,[[0,1],[2],[3],[4]])
    assert abs(result['residue_student_poe_js'][0])<1e-6
    assert result['boundary_student_poe_js'][0]>.1
    assert abs(result['student_residue_mass_prob'][0]-.8)<1e-6
