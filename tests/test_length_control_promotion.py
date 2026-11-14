"""Promotion gates and publication transactions: CPU-only synthetic fixtures."""
import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('length_promotion', ROOT/'experiments/prodipa/unconditional_student_training/evaluate_length_control.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def records(model):
    return [dict(model=model, paradigm='unconditional', sample_id=str(i), seed=str(42+i*1009),
                 sequence='A'*60, sequence_id='sha256:fixture', length_aa='60', generation_max_aa='512',
                 temperature='.7', top_p='.9', top_k='200', repetition_penalty='1.2', max_new_tokens='512',
                 valid_complete='true', valid_sequence='true', eos_reached='true', truncated='false',
                 length_forced_eos='false', natural_eos_reached='true',
                 ppl='10', novelty_u_percent='50', novelty_t_percent='50', plddt='80', pae='5', sol='.8', thermo='.8') for i in range(512)]


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.protocol={'gates':{'absolute_hv_epsilon':1e-6,'student_min_relative_hv_gain':.01,
                               'teacher_nonregression':{'fold':'pLDDT','sol':'Sol','thermo':'Thermo'}}}
        self.metrics={name:dict(HV=.5,pLDDT=70,Sol=.7,Thermo=.7,complete=400)
                      for name in ['base','fold','sol','thermo','opd','old_fold','old_sol','old_thermo','old_opd']}
        self.metrics['base'].update(HV=.4,pLDDT=60,Sol=.6,Thermo=.6)
        self.metrics['opd']['HV']=.6

    def test_gates_require_all_models_and_meaningful_gain(self):
        self.assertTrue(m.gates(self.metrics,self.protocol)['pass_all'])
        self.metrics['opd']['HV']=.504
        self.assertFalse(m.gates(self.metrics,self.protocol)['pass_all'])
        self.metrics['opd']['HV']=.6
        self.metrics['fold']['complete']=399
        self.assertFalse(m.gates(self.metrics,self.protocol)['pass_all'])

    def test_quality_regression_prevents_replacement(self):
        self.metrics['opd']['Sol']=.69
        result=m.gates(self.metrics,self.protocol)
        self.assertIn('opd_Sol_nonregression', result['failed'])

    def test_historical_old_opd_cannot_authorize_promotion(self):
        self.metrics['old_opd']['comparison_scope']='historical_unmatched_generation'
        result=m.gates(self.metrics,self.protocol)
        self.assertFalse(result['pass_all'])
        self.assertIsNone(result['checks']['opd_hv_gain_vs_old'])
        self.assertIn('old_opd_matched_protocol_available',result['failed'])

    def test_historical_reuse_preserves_uncapped_requests(self):
        with tempfile.TemporaryDirectory(prefix='prodipa_historical_test_') as tmp:
            p=Path(tmp)
            rs=records(m.IDS['opd'])
            rs[0].update(sequence='A'*900,length_aa='900',valid_complete='false')
            for r in rs:
                r.pop('generation_max_aa')
                r.pop('length_forced_eos')
                r.pop('natural_eos_reached')
            m.tsv(p/'canonical/results/unconditional_sequence_scores.tsv',rs)
            manifest=m.read(ROOT/'outputs/evaluation/unconditional/results/unconditional_hv_manifest.json')
            with patch.multiple(m,CANONICAL=p/'canonical',RUN=p/'run',EVAL=p/'eval'):
                reused,metrics=m.reuse_historical_old_opd(manifest)
                self.assertEqual(len(reused),512)
                self.assertEqual(metrics['complete'],511)
                self.assertEqual(metrics['comparison_scope'],'historical_unmatched_generation')
                self.assertEqual(m.read(m.RUN/'HISTORICAL_OLD_OPD_REUSE.json')['max_observed_aa'],900)

    def test_invalid_complete_forced_eos_is_rejected(self):
        rs=records('base');m.validate(rs)
        rs[0]['length_forced_eos']='true'
        with self.assertRaises(ValueError):m.validate(rs)
        rs[0]['valid_complete']='false';m.validate(rs)

    def test_failed_gate_has_no_publication_side_effect(self):
        with self.assertRaises(ValueError):m.promote({},self.protocol,{'pass_all':False})

    def test_cohort_rebuild_retains_frozen_hv_scale(self):
        with tempfile.TemporaryDirectory(prefix='prodipa_frozen_hv_test_') as tmp:
            p=Path(tmp)
            rs=records('base')
            manifest=m.read(ROOT/'outputs/evaluation/unconditional/results/unconditional_hv_manifest.json')
            m.dump(p/'frozen.json',manifest);m.tsv(p/'scores.tsv',rs)
            m.tsv(p/'novelty.tsv',[{'sequence_id':'sha256:fixture','novelty_percent':'50'}])
            subprocess.run([m.PY,str(ROOT/'experiments/prodipa/external_baseline_evaluation/complete_unconditional_novelty_hv.py'),
                '--sequence-scores',str(p/'scores.tsv'),'--novelty-u',str(p/'novelty.tsv'),
                '--output-table',str(p/'main.tsv'),'--output-sequence-scores',str(p/'all.tsv'),
                '--manifest',str(p/'hv.json'),'--expected-models','base','--frozen-manifest',str(p/'frozen.json')],check=True)
            self.assertEqual(m.read(p/'hv.json')['oriented_min'],manifest['oriented_min'])
            self.assertAlmostEqual(float(m.rows(p/'main.tsv')[0]['HV']),m.hv(rs,'unconditional',manifest),places=7)

    def fixture(self, root):
        config=m.read(ROOT/'configs/evaluation/unconditional.json')
        config['score_groups']=[{'name':'old','models':list(m.IDS.values()),'evaluation_root':'old'},
                                {'name':'external','models':['external_fixture'],'evaluation_root':'external'}]
        config['output_root']='outputs/evaluation/unconditional'
        canonical=root/config['output_root']; run=root/'candidate';ev=run/'evaluation'
        paper=root/'outputs/evaluation/paper/unconditional_main.tsv';cp=root/'configs/evaluation/unconditional.json'
        manifest=m.read(ROOT/'outputs/evaluation/unconditional/results/unconditional_hv_manifest.json')
        m.dump(ev/'protocol/unconditional_evaluation.json',config)
        m.dump(ev/'protocol/unconditional_hv.json',manifest)
        m.dump(cp,config)
        dest=root/'experiments/prodipa/unconditional_evaluation/audit_results.py'
        dest.parent.mkdir(parents=True);shutil.copy2(ROOT/'experiments/prodipa/unconditional_evaluation/audit_results.py',dest)
        overrides=dict(ROOT=root,RUN=run,EVAL=ev,CONFIG=cp,CANONICAL=canonical,PAPER=paper)
        with patch.multiple(m,**overrides):
            rs=[r for model in list(m.IDS.values())+['external_fixture'] for r in records(model)]
            mains,supp,comp=m.export_group(canonical,rs,manifest)
            m.tsv(canonical/'results/unconditional_sequence_scores.tsv',rs)
            m.tsv(canonical/'results/unconditional_supplementary_metrics.tsv',supp)
            m.tsv(canonical/'results/unconditional_scoring_completeness.tsv',comp)
            m.dump(canonical/'results/unconditional_hv_manifest.json',manifest)
            m.tsv(canonical/'scores/novelty_u.tsv',[{'sequence_id':'sha256:fixture','novelty_percent':'50'}])
            m.tsv(paper,[dict(Group='Existing group',**r,**{'HV Ret. (%)':'N/A'}) for r in mains])
        return overrides

    def test_publication_updates_evidence_and_preserves_external(self):
        with tempfile.TemporaryDirectory(prefix='prodipa_promotion_test_') as tmp:
            overrides=self.fixture(Path(tmp))
            with patch.multiple(m,**overrides):
                original=m.sha(m.CONFIG)
                m.promote({k:records(k) for k in m.IDS},self.protocol,{'pass_all':True})
                self.assertTrue((m.RUN/'PROMOTION_COMPLETE.json').exists())
                self.assertNotEqual(m.sha(m.CONFIG),original)
                self.assertEqual(m.read(m.CANONICAL/'RESULT_AUDIT.json')['status'],'PASS')
                self.assertIn('external_fixture',{r['Model'] for r in m.rows(m.PAPER)})
                self.assertEqual(m.read(m.CONFIG)['hypervolume']['frozen_manifest'],str(m.EVAL/'protocol/unconditional_hv.json'))
                self.assertIn('natural_eos_reached',m.rows(m.CANONICAL/'results/unconditional_sequence_scores.tsv')[0])

    def test_failed_live_audit_rolls_back(self):
        with tempfile.TemporaryDirectory(prefix='prodipa_promotion_rollback_test_') as tmp:
            overrides=self.fixture(Path(tmp))
            with patch.multiple(m,**overrides):
                originals={p:m.sha(p) for p in [m.CONFIG,m.PAPER,m.CANONICAL/'results/unconditional_sequence_scores.tsv']}
                actual_run=subprocess.run
                calls=[]
                def fail_second(*args,**kwargs):
                    calls.append(1)
                    if len(calls)==2:raise RuntimeError('synthetic live audit failure')
                    return actual_run(*args,**kwargs)
                with patch.object(m.subprocess,'run',side_effect=fail_second):
                    with self.assertRaises(RuntimeError):
                        m.promote({k:records(k) for k in m.IDS},self.protocol,{'pass_all':True})
                self.assertFalse((m.RUN/'PROMOTION_COMPLETE.json').exists())
                self.assertEqual({p:m.sha(p) for p in originals},originals)


if __name__=='__main__':unittest.main()
