import unittest
import numpy as np
from src.camera_partial_alignment import partial_angles,align,contiguous


def sequence(scale=1,offset=0,missing_arms=False):
    t=np.arange(25)*.1
    x=np.column_stack([.5+.25*np.sin(t*3+j*.7)+.08*np.cos(t*7+j) for j in range(6)])
    return [{'time_sec':float(offset+scale*time),'feature':[None if missing_arms and j<2 else float(v) for j,v in enumerate(row)]}
            for time,row in zip(t,x)]


class PartialAlignmentTests(unittest.TestCase):
    def test_missing_wrist_preserves_leg_angles(self):
        k=np.arange(34,dtype=float).reshape(17,2)**1.2
        c=np.ones(17);c[9]=.1
        f=partial_angles({'keypoints':k.tolist(),'keypoint_confidences':c.tolist()})
        self.assertIsNone(f[0]);self.assertTrue(all(v is not None for v in f[2:]))
    def test_unknown_not_zero(self):self.assertEqual(partial_angles({}),[None]*6)
    def test_static_unknown(self):
        r=[{'time_sec':i*.1,'feature':[.5]*6} for i in range(25)]
        self.assertEqual(align(r,r)['status'],'unknown')
    def test_partial_double_speed_time(self):
        r=align(sequence(missing_arms=True),sequence(scale=2,offset=4,missing_arms=True))
        self.assertEqual(r['status'],'relative_slowdown_candidate',r)
        self.assertAlmostEqual(r['best_alignment']['stretch'],2)
        self.assertFalse(r['identity_verified']);self.assertIsNone(r['absolute_speed_factor'])
    def test_normal_not_slow(self):
        self.assertNotEqual(align(sequence(),sequence(offset=3))['status'],'relative_slowdown_candidate')
    def test_reversed_not_slow(self):
        r=sequence();q=sequence(scale=2,offset=3)
        for out,original in zip(q,r[::-1]):out['feature']=original['feature']
        self.assertEqual(align(r,q)['status'],'unknown')
    def test_shuffled_not_slow(self):
        r=sequence();q=sequence(scale=2)
        order=np.random.default_rng(42).permutation(len(q))
        for i in range(len(q)):q[i]['feature']=r[order[i]]['feature']
        self.assertNotEqual(align(r,q)['status'],'relative_slowdown_candidate')
    def test_time_order(self):
        self.assertEqual(align(sequence()[::-1],sequence())['reason'],'invalid_time_order')
    def test_gap_not_bridged(self):
        r=sequence();r=r[:8]+r[19:]
        self.assertEqual(align(r,sequence(scale=2))['reason'],'insufficient_contiguous_partial_pose')
    def test_missing_lower_body(self):
        r=sequence()
        for row in r:row['feature'][2:]=[None]*4
        self.assertEqual(align(r,sequence())['status'],'unknown')
    def test_json_finite(self):
        import json
        json.dumps(align(sequence(),sequence(scale=2)),allow_nan=False)
    def test_no_input_mutation(self):
        import copy
        r=sequence();before=copy.deepcopy(r);align(r,r);self.assertEqual(r,before)


if __name__=='__main__':unittest.main()
