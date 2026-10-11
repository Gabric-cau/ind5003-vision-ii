import unittest
import numpy as np
from src.kicking_foot_redetection import undo_mirror, transform_points, fuse_evidence, approach_candidates, foot_evidence, followthrough_evidence, camera_scoped_approach


class RedetectionTests(unittest.TestCase):
    def test_double_reflection_identity(self):
        pts=np.arange(34,dtype=float).reshape(17,2); conf=np.arange(17)/17
        p,c=undo_mirror(pts,conf,100);p,c=undo_mirror(p,c,100)
        np.testing.assert_allclose(p,pts);np.testing.assert_allclose(c,conf)

    def test_reflection_swaps_anatomical_channels(self):
        p=np.zeros((17,2));p[15]=[20,7];p[16]=[80,9]
        q,c=undo_mirror(p,np.ones(17),101)
        np.testing.assert_allclose(q[15],[20,9]);np.testing.assert_allclose(q[16],[80,7])

    def test_affine(self):
        np.testing.assert_allclose(transform_points([[1,2]],[[1,0,5],[0,1,3],[0,0,1]]),[[6,5]])

    def test_disagreement_abstains(self):
        a=dict(motion_candidate='right',contact_candidate='right');b=dict(motion_candidate='left',contact_candidate='left')
        self.assertEqual(fuse_evidence(a,b)['candidate'],'unknown')

    def test_one_weak_observation_abstains(self):
        a=dict(motion_candidate='right',contact_candidate='unknown');b=dict(motion_candidate='unknown',contact_candidate='unknown')
        self.assertEqual(fuse_evidence(a,b)['candidate'],'unknown')

    def test_consistent_components(self):
        a=dict(motion_candidate='right',contact_candidate='right');b=dict(motion_candidate='right',contact_candidate='unknown')
        self.assertEqual(fuse_evidence(a,b)['candidate'],'right')

    def test_no_samples_abstains(self):
        self.assertEqual(foot_evidence([],np.zeros(2),10,12)['motion_candidate'],'unknown')
        self.assertEqual(approach_candidates([],np.zeros(2),[100,0],10)['screen_x_candidate'],'unknown')

    def test_screen_direction_not_anatomical_proof(self):
        k=np.zeros((17,2));k[15]=[-20,0];k[16]=[-20,0]
        ss=[dict(frame=f,keypoints=k,confidence=np.ones(17)) for f in [1,2,3]]
        r=approach_candidates(ss,[0,0],[100,0],10)
        self.assertEqual(r['screen_x_candidate'],'right');self.assertFalse(r['contact_observed_by_this_rule'])

    def test_missing_goal_preserves_screen_baseline(self):
        k=np.zeros((17,2));k[15]=[-20,0];k[16]=[-20,0]
        ss=[dict(frame=f,keypoints=k,confidence=np.ones(17)) for f in [1,2,3]]
        r=approach_candidates(ss,[0,0],None,10)
        self.assertEqual(r['screen_x_candidate'],'right');self.assertEqual(r['goal_axis_candidate'],'unknown')

    def test_followthrough_right_extension(self):
        k=np.zeros((17,2));k[15]=[-5,5];k[16]=[8,2]
        ss=[dict(frame=f,keypoints=k,confidence=np.ones(17)) for f in [10,11,12]]
        self.assertEqual(followthrough_evidence(ss,[100,0],10)['candidate'],'right')

    def test_followthrough_missing_samples(self):
        self.assertEqual(followthrough_evidence([], [100,0], 10)['candidate'],'unknown')

    def test_followthrough_low_confidence_abstains(self):
        k=np.zeros((17,2));k[15]=[-5,5];k[16]=[8,2];c=np.ones(17);c[15]=.1
        ss=[dict(frame=f,keypoints=k,confidence=c) for f in [10,11,12]]
        self.assertEqual(followthrough_evidence(ss,[100,0],10)['candidate'],'unknown')

    def test_camera_prior_does_not_apply_to_other_views(self):
        r=camera_scoped_approach(dict(goal_axis_candidate='right',screen_x_candidate='right'),'goal_left')
        self.assertEqual(r['candidate'],'unknown')

    def test_camera_prior_prefers_axis_over_screen(self):
        r=camera_scoped_approach(dict(goal_axis_candidate='left',screen_x_candidate='right'),'goal_right')
        self.assertEqual(r['candidate'],'left');self.assertTrue(r['requires_review'])


if __name__ == '__main__': unittest.main()
