import unittest
from copy import deepcopy
from src.goalkeeper_direction_evidence import early_direction_evidence
from src.goalkeeper_observation_guard import guard_postkick_observations


def record(xs=None,frames=None):
    frames=list(range(60,73)) if frames is None else frames
    xs=[100+5*i for i in range(len(frames))] if xs is None else xs
    return {'fps':30,'kick_frame':60,'kick_frame_confidence':'supported',
        'dive_direction':'image_right','dive_height':'mid','auto_accepted':True,'review_required':False,
        'dive_review_required':False,'qa_warnings':[],
        'goal_identity_evidence':{'ambiguous_identity':False},
        'selected_candidate':{'qualifies':True,'is_selected_kicker':False,'pre_coverage':1,'pose_quality':.9,
          'pre_trajectory':[[f/30,100,100,100] for f in range(57,60)],
          'post_trajectory':[[f/30,x,100,100] for f,x in zip(frames,xs)]}}


class DirectionEvidenceTests(unittest.TestCase):
    def test_short_clear_dive_preserves_direction_but_not_height(self):
        r=record();old=deepcopy(r);out=guard_postkick_observations(r)
        self.assertEqual(out['dive_direction'],'image_right');self.assertEqual(out['dive_height'],'unknown')
        self.assertTrue(out['dive_direction_supported']);self.assertFalse(out['auto_accepted'])
        self.assertTrue(out['dive_height_review_required']);self.assertEqual(r,old)
    def test_image_left(self):
        r=record([100-5*i for i in range(13)]);r['dive_direction']='image_left'
        self.assertEqual(guard_postkick_observations(r)['dive_direction'],'image_left')
    def test_partial_static_not_center(self):
        r=record([100]*13);r['dive_direction']='center'
        self.assertEqual(guard_postkick_observations(r)['dive_direction'],'unknown')
    def test_one_outlier_cannot_establish_direction(self):
        self.assertFalse(early_direction_evidence(record([100]*12+[200]))['supported'])
    def test_long_gap_cannot_join_late_person(self):
        r=record([100]*4+[150+i*5 for i in range(9)],list(range(60,64))+list(range(75,84)))
        self.assertFalse(early_direction_evidence(r)['supported'])
    def test_missing_onset_is_unknown(self):
        r=record(frames=list(range(65,78)))
        self.assertFalse(early_direction_evidence(r)['supported'])
    def test_coordinate_teleport_rejected(self):
        r=record([100]*6+[300+i*5 for i in range(7)])
        self.assertFalse(early_direction_evidence(r)['supported'])
    def test_few_samples_cannot_be_inflated_by_duplicates(self):
        r=record([100,150],[60,72]);r['selected_candidate']['post_trajectory']*=10
        self.assertFalse(early_direction_evidence(r)['supported'])
    def test_oscillating_track_is_unknown(self):
        self.assertFalse(early_direction_evidence(record([100,110,100,115,105,120,110,125,115,130,120,140,150]))['supported'])
    def test_ambiguous_person_not_rescued(self):
        r=record();r['goal_identity_evidence']['ambiguous_identity']=True
        self.assertFalse(early_direction_evidence(r)['supported'])
    def test_candidate_disagreement_remains_unknown(self):
        r=record();r['dive_direction']='image_left'
        out=guard_postkick_observations(r)
        self.assertEqual(out['dive_direction'],'unknown')
        self.assertIn('early_motion_disagrees_with_legacy_candidate',out['dive_direction_evidence']['reasons'])
    def test_full_window_still_preserves_height(self):
        r=record(frames=list(range(60,91)));out=guard_postkick_observations(r)
        self.assertEqual(out['dive_height'],'mid');self.assertTrue(out['auto_accepted'])
    def test_old_policy_can_be_reproduced(self):
        self.assertEqual(guard_postkick_observations(record(),policy='legacy_full_window_v1')['dive_direction'],'unknown')
    def test_bad_fps_rejected(self):
        r=record();r['fps']=float('nan')
        with self.assertRaises(ValueError):early_direction_evidence(r)


if __name__=='__main__':unittest.main()
