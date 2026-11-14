"""Numerical and token-position checks for read-only distribution diagnostics."""
from pathlib import Path
import runpy

import torch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    'experiments/prodipa/conditional_multiteacher_student/diagnose_distributions.py'))


def measure(logits):
    return MODULE['measures'](logits, torch.tensor([0, 1]), [1.] * 4,
                              [1/3] * 3, {'eos': 2, 'close': [1]})


def test_identical_distributions_have_no_conflict():
    x = torch.tensor([[4., 1., -2.], [1., 3., 0.]])
    result = measure([x] * 4)
    for key in ('raw_teacher_js', 'distill_student_poe_js', 'confident_teacher_conflict',
                'teacher_top1_disagreement'):
        assert torch.allclose(torch.tensor(result[key]), torch.zeros(2), atol=1e-6)


def test_confident_opposition_and_js_bounds():
    x = torch.tensor([[8., 0., -3.], [0., 8., -3.]])
    opposite = torch.tensor([[0., 8., -3.], [8., 0., -3.]])
    result = measure([x, opposite, x, x])
    assert result['confident_teacher_conflict'] == [1., 1.]
    assert all(0 <= n <= 1.099 for n in result['raw_teacher_js'])
    assert all(torch.isfinite(torch.tensor(v)).all() for v in result.values())


def test_near_ties_are_not_confident_conflict():
    x = torch.tensor([[.02, .01, 0.], [.02, .01, 0.]])
    y = x.flip(-1)
    result = measure([x, y, x, x])
    assert result['teacher_top1_disagreement'] == [1., 1.]
    assert result['confident_teacher_conflict'] == [0., 0.]


def test_regions_keep_wrapper_and_eos_out_of_protein():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ''.join(chr(i) for i in ids if i != 0)
    text = 'Seq=<ACD>'
    ids = [ord(c) for c in text] + [0]
    rows = MODULE['locations'](Tokenizer(), ids, text, 0)
    assert [r['target_text'] for r in rows if r['region'] == 'protein'] == ['A','C','D']
    assert rows[-2]['region'] == 'closing_or_mixed'
    assert rows[-1]['region'] == 'eos'
    assert [r['aa_before'] for r in rows[5:8]] == [0,1,2]
