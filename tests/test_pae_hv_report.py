"""Deterministic checks of the supplementary pAE HV; no model runs."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/prodipa/adaptive_cache'))
from report_pae_hv import point, valid_points, exact_hv, six_oriented, SIX_FIELDS, project_four_spec
from report import point as main_point, hv as main_hv, recursive_exact_hv


class PaeHVTest(unittest.TestCase):
    def setUp(self):
        self.spec=dict(oriented_min=[0,0,0,0,-31.75],oriented_max=[100,100,1,1,0],reference_point=[0]*5)
        self.row=dict(protrek_percent=50,novelty_u_percent=80,plddt=80,sol=.7,thermo=.6,
                      pae=6.35,ppl=100,valid_complete='true')

    def test_pae_direction_and_scale(self):
        p=point(self.row,'conditional',self.spec)
        np.testing.assert_allclose(p,[.5,.8,.7,.6,.8])
        self.assertGreater(point(dict(self.row,pae=3),'conditional',self.spec)[-1],p[-1])

    def test_ppl_has_no_effect(self):
        np.testing.assert_array_equal(point(self.row,'conditional',self.spec),
                                      point(dict(self.row,ppl=''),'conditional',self.spec))

    def test_paradigm_specific_second_objective(self):
        self.assertEqual(point(self.row,'unconditional',self.spec)[0],.8)

    def test_invalid_and_missing(self):
        self.assertEqual(valid_points([dict(self.row,valid_complete='false')],'conditional',self.spec).shape,(0,5))
        for value in ('',float('nan'),32,-1):
            with self.assertRaises(ValueError):
                point(dict(self.row,pae=value),'conditional',self.spec)

    def test_union_not_sum_and_no_reference_volume(self):
        p=point(self.row,'conditional',self.spec)
        self.assertAlmostEqual(exact_hv(np.array([p,p*.5])),np.prod(p))
        self.assertEqual(exact_hv(np.array([point(dict(self.row,pae=31.75),'conditional',self.spec)])),0)


class ConditionalSixHVTest(unittest.TestCase):
    def setUp(self):
        self.spec = dict(objectives=[dict(field=k, transform='negate' if k=='pae' else 'identity')
                                     for k in SIX_FIELDS],
                         oriented_min=[0,0,0,-31.75,0,0], oriented_max=[100,100,100,0,1,1],
                         reference_point=[0]*6)
        self.row = dict(novelty_u_percent=80,novelty_t_percent=60,plddt=80,pae=6.35,
                        sol=.7,thermo=.6,valid_complete='true')

    def test_two_novelty_dimensions_and_pae_direction(self):
        p = main_point(self.row,'conditional',self.spec)
        np.testing.assert_allclose(p,[.8,.6,.8,.8,.7,.6])
        for key,index in [('novelty_u_percent',0),('novelty_t_percent',1)]:
            self.assertGreater(main_point(dict(self.row,**{key:90}),'conditional',self.spec)[index],p[index])
        self.assertGreater(main_point(dict(self.row,pae=3),'conditional',self.spec)[3],p[3])

    def test_ppl_and_protrek_do_not_enter_new_hv(self):
        np.testing.assert_array_equal(main_point(self.row,'conditional',self.spec),
            main_point(dict(self.row,ppl=1e9,protrek_percent=-50),'conditional',self.spec))

    def test_exact_backend_matches_recursive_implementation(self):
        rng = np.random.default_rng(42)
        for dimensions in (4,6):
            for n in (1,2,5,10):
                points = rng.uniform(.01,1,(n,dimensions))
                self.assertAlmostEqual(exact_hv(points),recursive_exact_hv(points),places=12)

    def test_four_objective_projection_preserves_scales(self):
        prior = dict(self.spec, raw_directions=['maximize']*3+['minimize']+['maximize']*2)
        projected = project_four_spec(prior)
        p6 = main_point(self.row,'conditional',prior)
        p4 = main_point(self.row,'conditional',projected)
        np.testing.assert_array_equal(p4,p6[2:])
        self.assertEqual(projected['reference_point'],[0]*4)
        self.assertGreater(main_point(dict(self.row,pae=3),'conditional',projected)[1],p4[1])
        without_novelty = {k:v for k,v in self.row.items() if not k.startswith('novelty')}
        np.testing.assert_array_equal(main_point(without_novelty,'conditional',projected),p4)
        self.assertAlmostEqual(main_hv([without_novelty],'conditional',projected),np.prod(p4))

    def test_invalid_not_contributing_duplicate_not_double_counted(self):
        row = self.row
        expected = np.prod(main_point(row,'conditional',self.spec))
        self.assertAlmostEqual(main_hv([row,row],'conditional',self.spec),expected)
        self.assertEqual(main_hv([dict(row,valid_complete='false')],'conditional',self.spec),0)

    def test_missing_objectives_cannot_be_imputed(self):
        for key in SIX_FIELDS:
            with self.assertRaises(ValueError):
                six_oriented(dict(self.row,**{key:''}))


if __name__=='__main__':
    unittest.main()
