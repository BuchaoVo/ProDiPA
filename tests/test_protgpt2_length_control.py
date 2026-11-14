"""CPU regression tests using the actual local ProtGPT2 tokenizer (no training)."""
import unittest

import torch
from transformers import AutoTokenizer

from prodipa.unconditional.length_control import ProteinLengthLogitsProcessor
from prodipa.unconditional.trainer import UnconditionalProDiPATrainer


class LengthControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tok = AutoTokenizer.from_pretrained('__PRODIPA_MODELS__/ProtGPT2', local_files_only=True)
        cls.eos = cls.tok.eos_token_id
        cls.a = cls.tok.encode('A', add_special_tokens=False)[0]
        assert cls.tok.decode([cls.a]) == 'A'

    def processor(self, cap=512):
        return ProteinLengthLogitsProcessor(self.tok, cap)

    def test_token_lengths_match_real_decoding(self):
        proc = self.processor()
        canonical = set('ACDEFGHIKLMNPQRSTVWY')
        checked = 0
        multi = 0
        for i in range(len(self.tok)):
            text = ''.join(self.tok.decode([i], skip_special_tokens=True).split())
            if text and set(text) <= canonical:
                self.assertEqual(proc.lengths[i].item(), len(text))
                checked += 1
                multi += len(text) > 1
        self.assertGreater(checked, 1000)
        self.assertGreater(multi, 1000)

    def test_masks_crossing_token_before_sampling(self):
        proc = self.processor()
        ids = torch.tensor([[self.eos] + [self.a] * 511])
        scores = proc(ids, torch.zeros(1, len(self.tok)))
        self.assertTrue(torch.isneginf(scores[0, proc.lengths > 1]).all())
        self.assertTrue(torch.isfinite(scores[0, self.a]))
        self.assertTrue(torch.isfinite(scores[0, self.eos]))
        self.assertEqual(proc.forced_flags(), [False])

    def test_natural_and_forced_eos_in_one_batch(self):
        proc = self.processor(2)
        ids = torch.tensor([[self.eos, self.a, self.a], [self.eos, self.a, self.eos]])
        scores = proc(ids, torch.zeros(2, len(self.tok)))
        self.assertEqual(proc.forced_flags(), [True, False])
        self.assertEqual(torch.isfinite(scores).sum().item(), 2)
        self.assertTrue(torch.isfinite(scores[:, self.eos]).all())

    def test_whitespace_does_not_use_residue_budget(self):
        proc = self.processor(2)
        newline = self.tok.encode('\n', add_special_tokens=False)
        ids = torch.tensor([[self.eos, self.a, *newline]])
        scores = proc(ids, torch.zeros(1, len(self.tok)))
        self.assertTrue(torch.isfinite(scores[0, self.a]))
        self.assertEqual(proc.forced_flags(), [False])

    def test_repeated_generation_never_exceeds_cap(self):
        proc = self.processor()
        ids = torch.tensor([[self.eos]])
        canonical = torch.tensor([bool(s := self.tok.decode([i])) and set(s) <= set('ACDEFGHIKLMNPQRSTVWY')
                                  for i in range(len(self.tok))])
        scores = proc.lengths.float().masked_fill(~canonical, -float('inf'))
        scores[self.eos] = -100
        for _ in range(513):
            next_id = proc(ids, scores[None, :].clone()).argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            sequence = ''.join(self.tok.decode(ids[0, 1:], skip_special_tokens=True).split())
            self.assertLessEqual(len(sequence), 512)
            if next_id.item() == self.eos:
                break
        self.assertEqual(len(sequence), 512)
        self.assertEqual(proc.forced_flags(), [True])

    def test_training_masks_only_artificial_terminal_position(self):
        trainer = object.__new__(UnconditionalProDiPATrainer)
        trainer._tokenizer = self.tok
        trainer.max_length_aa = 2
        trainer._length_processor = self.processor(2)
        trainer._length_processor.forced_eos = torch.tensor([True, False, False])
        trainer._buffer_train_metrics = lambda _: None
        generated = torch.tensor([[self.eos, self.a, self.a, self.eos],
                                  [self.eos, self.a, self.eos, self.eos],
                                  [self.eos, self.a, self.a, self.a]])
        prompt = generated[:, :1]
        _, _, labels, _ = trainer._canonicalize_rollouts(generated, prompt)
        self.assertEqual(labels[0, 3].item(), -100)
        self.assertEqual(labels[1, 2].item(), self.eos)
        self.assertEqual(labels[2, 4].item(), -100)
        self.assertEqual(labels[0, 1].item(), self.a)
        trainer.max_length_aa = 0
        trainer._length_processor = None
        _, _, historical, _ = trainer._canonicalize_rollouts(generated, prompt)
        self.assertEqual(historical[0, 3].item(), self.eos)
        self.assertEqual(historical[2, 4].item(), -100)


if __name__ == '__main__':
    unittest.main()
