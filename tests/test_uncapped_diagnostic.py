"""CPU checks for diagnostic-only length eligibility and repetition statistics."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('uncapped_diagnostic',ROOT/'experiments/prodipa/unconditional_student_training/diagnose_uncapped_generation.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DiagnosticTests(unittest.TestCase):
    def row(self,sequence='A'*600,eos='true',truncated='false'):
        return dict(sequence=sequence,eos_reached=eos,truncated=truncated,valid_complete='false')

    def test_long_natural_sequence_is_not_rejected_by_formal_length_gate(self):
        r=m.annotate(self.row())
        self.assertTrue(r['diagnostic_structure_eligible'])
        self.assertEqual(r['length_aa'],600)
        self.assertEqual(r['valid_complete'],'false')

    def test_budget_censoring_not_scored_as_complete(self):
        r=m.annotate(self.row(eos='false',truncated='true'))
        self.assertFalse(r['diagnostic_structure_eligible'])
        self.assertEqual(m.summarize([r])['token_budget_censored'],1)

    def test_invalid_and_short_sequences_remain_in_request_denominator(self):
        records=[m.annotate(self.row(sequence=s)) for s in ['AXX','A'*20,'']]
        self.assertFalse(any(r['diagnostic_structure_eligible'] for r in records))
        self.assertEqual(m.summarize(records)['requests'],3)

    def test_repetition_definition(self):
        r=m.repetition('AAAAAA')
        self.assertEqual(r['max_homopolymer_run'],6)
        self.assertEqual(r['aa_entropy_bits'],0)
        self.assertEqual(r['repeated_3mer_fraction'],.75)
        self.assertEqual(m.repetition('ACDEFG')['repeated_3mer_fraction'],0)


if __name__=='__main__':unittest.main()
