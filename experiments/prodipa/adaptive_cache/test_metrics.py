"""Small deterministic checks of accounting math and cache safety; no model runs."""
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import common
import run as pipeline
import report
from common import dump, tsv, read, rows
from report import normalized_auc, archive_stats
from selection import eligible, select


class MetricsTest(unittest.TestCase):
    def test_report_rejects_paused_paradigm(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            dump(output / "unconditional/PAUSED.json", {"state": "PAUSED_BY_USER"})
            with patch.object(report, "OUT", output):
                with self.assertRaisesRegex(RuntimeError, "paused"):
                    report.validate_inputs(["unconditional"], {})

    def test_conditional_report_does_not_require_unconditional(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            dump(output / "unconditional/PAUSED.json", {"state": "PAUSED_BY_USER"})
            dump(output / "conditional/EVALUATION_COMPLETE.json", {"status": "COMPLETE"})
            nodes = [("reference", 0), ("online", 4)] + [(m,n) for m in common.METHODS for n in range(1,5)]
            for method, node in nodes:
                directory = output / f"evaluation/conditional/{method}/node_{node}/scores"
                dump(directory / "COMPLETE.json", {"requests": 1})
                tsv(directory / "sequence_scores.tsv", [self.row])
            with patch.object(report, "OUT", output):
                report.validate_inputs(["conditional"], {"evaluation_requests": 1})
                (output / "evaluation/conditional/adaptive/node_4/scores/COMPLETE.json").unlink()
                with self.assertRaisesRegex(RuntimeError, "Missing score"):
                    report.validate_inputs(["conditional"], {"evaluation_requests": 1})

    def test_evaluation_does_not_wait_for_other_paradigm(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            protocol = {"nodes": {"conditional": [192, 208, 224, 240, 256]}}
            calls = []

            def train(paradigm, protocol):
                calls.append("training")
                dump(output / paradigm / "TRAINING_COMPLETE.json", {"status": "COMPLETE"})

            with patch.object(pipeline, "OUT", output), \
                    patch.object(pipeline, "train_paradigm", side_effect=train), \
                    patch.object(pipeline, "status"), \
                    patch.object(pipeline, "generate", side_effect=lambda *args: calls.append(args[1:3])):
                pipeline.run_paradigm("conditional", protocol)
            self.assertEqual(calls[0], "training")
            self.assertEqual(calls[1], ("reference", 0))
            self.assertEqual(calls[-1], ("online", 4))
            self.assertEqual(len(calls), 15)
            self.assertTrue((output / "conditional/EVALUATION_COMPLETE.json").exists())
            self.assertFalse((output / "TRAINING_COMPLETE.json").exists())
            self.assertFalse((output / "unconditional").exists())

    def test_aligned_table_headers_and_values(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "aligned.tsv"
            path.write_text(" Model \t HV \n base \t 0.25 \n")
            self.assertEqual(rows(path), [{"Model": "base", "HV": "0.25"}])

    def test_parallel_status_keeps_both_paradigms(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(common, "OUT", Path(d)), patch.object(common, "_training_states", {}):
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="paradigm") as pool:
                    list(pool.map(lambda p: common.status(p + "/training/1"), common.STARTS))
                result = read(Path(d) / "STATUS.json")
                self.assertEqual(set(result["paradigms"]), set(common.STARTS))
                for paradigm in common.STARTS:
                    self.assertEqual(read(Path(d) / paradigm / "STATUS.json")["state"], "RUNNING")

    def setUp(self):
        self.gate = dict(plddt_min=70,pae_max=10,sol_min=.7,thermo_min=.6,
                         novelty_u_min_exclusive=0,novelty_t_min_exclusive=0,
                         max_residue_fraction=.4,max_trimer_fraction=.2)
        self.row = dict(sample_id=0,sequence_id="one",sequence="ACDEFGHIKLMNPQRSTVWY"*4,
                        valid_complete="true",plddt=80,pae=5,sol=.8,thermo=.8,
                        novelty_u_percent=50,novelty_t_percent=50,ppl=10,protrek_percent=10)

    def test_gates_fail_closed(self):
        self.assertTrue(eligible(self.row,self.gate))
        for changes in (dict(pae=20),dict(sol=.5),dict(novelty_u_percent=0),dict(valid_complete="false"),dict(sequence="A"*100)):
            self.assertFalse(eligible(dict(self.row,**changes),self.gate))

    def test_common_cost_interpolation(self):
        curve=[dict(training_gpu_hours=x,trusted_archive_hv=y) for x,y in [(0,0),(2,2),(4,4)]]
        self.assertAlmostEqual(normalized_auc(curve,3),1.5)
        with self.assertRaises(AssertionError):
            normalized_auc(curve,5)

    def test_dedup_and_final_dominance(self):
        manifest=dict(oriented_min=[-5,0,0,0,0],oriented_max=[0,100,100,1,1],reference_point=[0]*5)
        weaker=dict(self.row,sequence_id="two",plddt=75)
        hv,nd,qualified=archive_stats([self.row,self.row,weaker],"unconditional",manifest,self.gate)
        self.assertGreater(hv,0)
        self.assertEqual((nd,qualified),(1,2))

    def test_periodic_no_quality_ranking_and_anchors_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            records=[dict(self.row,sample_id=i,sol=.1) for i in range(96)]
            tsv(directory/"candidates.tsv",records)
            result=select(directory,{},"periodic",dict(anchors=list(range(128))),17)
            self.assertEqual(len(result["accepted"]),96)
            self.assertTrue(all(a["slot"]>=128 for a in result["assignments"]))

    def test_adaptive_rejects_ineligible_without_refresh(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d)
            bad=dict(self.row,sol=.1)
            tsv(directory/"candidates.tsv",[bad])
            tsv(directory/"scores/sequence_scores.tsv",[bad])
            state=dict(archive=[],slots=[bad]*512,bins={k:[] for k in ("plddt","sol","thermo")})
            protocol=dict(anchors=list(range(128)),training_gate=self.gate,coverage_max_occupancy=2,
                          cluster_jaccard_threshold=.5,delta_hv_epsilon=1e-8)
            result=select(directory,state,"adaptive",protocol,17)
            self.assertFalse(result["refreshed"])
            self.assertEqual(result["assignments"],[])


if __name__=="__main__":
    unittest.main()
