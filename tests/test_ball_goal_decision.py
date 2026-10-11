import unittest
from src.ball_goal_decision import classify_goal_evidence


def sample(f,inside=True,outside=False):
    return {'frame':f,'independent_image_key':str(f),'identity_support':True,
            'geometry_supported':True,'inside_mouth':inside,
            'clearly_outside_mouth':outside,'arrival_progress_supported':True}


class GoalDecisionTests(unittest.TestCase):
    def test_two_dimensional_inside_sequence_is_only_likely(self):
        d=classify_goal_evidence([sample(f) for f in [10,11,12]],10,{})
        self.assertEqual(d['rule_outcome'],'goal_likely')
        self.assertFalse(d['whole_ball_crossing_established'])

    def test_net_turn_after_supported_entry_does_not_become_miss(self):
        d=classify_goal_evidence([sample(f) for f in [10,11,12]],10,
           {'turn_confirmed':True,'contact_prediction':{'frame_float':14},'support_frames':[15,16,17]})
        self.assertEqual(d['rule_outcome'],'goal_likely')
        self.assertTrue(d['later_turn_not_used_as_miss'])

    def test_keeper_turn_before_entry_gives_no_goal_hint(self):
        d=classify_goal_evidence([sample(12),sample(13)],10,
           {'turn_confirmed':True,'contact_prediction':{'frame_float':11.5},'support_frames':[12,13,14]})
        self.assertEqual(d['rule_outcome'],'no_goal_likely')

    def test_single_jump_or_inside_point_cannot_decide(self):
        self.assertEqual(classify_goal_evidence([sample(10)],10,{'turn_confirmed':False})['rule_outcome'],'unknown')

    def test_repeated_images_do_not_count_towards_entry(self):
        points=[{**sample(f),'independent_image_key':'same'} for f in [10,11,12]]
        self.assertEqual(classify_goal_evidence(points,10,{})['rule_outcome'],'unknown')

    def test_finite_ball_near_edge_without_clear_outside_does_not_fail(self):
        points=[sample(10,False,False),sample(11,False,False)]
        self.assertEqual(classify_goal_evidence(points,10,{})['rule_outcome'],'unknown')

    def test_two_distinct_fully_outside_observations_give_no_goal_hint(self):
        points=[sample(10,False,True),sample(11,False,True)]
        self.assertEqual(classify_goal_evidence(points,10,{})['rule_outcome'],'no_goal_likely')

    def test_late_tail_and_bad_identity_are_ignored(self):
        points=[sample(30,False,True),sample(31,False,True)]
        points+=[{**sample(f),'identity_support':False} for f in [10,11,12]]
        self.assertEqual(classify_goal_evidence(points,10,{})['rule_outcome'],'unknown')


if __name__=='__main__':unittest.main()
