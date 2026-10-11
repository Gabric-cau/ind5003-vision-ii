from copy import deepcopy
import unittest

from src.fresh_goal_guard import guard_keeper_occluded_goal


class FreshGoalGuardTest(unittest.TestCase):
    def fixture(self):
        result = {'rule_decision': {'rule_outcome': 'goal_likely', 'projected_entry_support_frame': 13,
                                   'evidence_frames': [10, 11, 13]},
                  'keeper': {'role': {'seed_track_id': 3}, 'selected_candidate': {'source_track_ids': [3]}},
                  'independent_contact_search': {'selected_candidate': {'contact_associated_track_id': 6}},
                  'trajectory': [{'frame': f, 'detection_id': str(f), 'x': x, 'y': 50, 'width': 6, 'height': 6}
                                 for f, x in ((10, 20), (11, 30), (13, 50))], 'rebound': {}}
        pose = {'tracks': [{'track_id': 3, 'observations': [{'frame': f, 'bbox': [45, 40, 70, 60]} for f in (10, 11, 13)]}]}
        return result, pose

    def test_occlusion_before_third_entry_point_abstains_without_inventing_save(self):
        result, pose = self.fixture()
        before = deepcopy(result)
        decision, evidence = guard_keeper_occluded_goal(result, pose)
        self.assertEqual(decision['rule_outcome'], 'unknown')
        self.assertEqual(evidence['first_keeper_overlap_frame'], 13)
        self.assertFalse(decision['keeper_occlusion_used_as_no_goal_evidence'])
        self.assertEqual(result, before)

    def test_later_keeper_overlap_cannot_erase_already_supported_projected_entry(self):
        result, pose = self.fixture()
        result['rule_decision']['projected_entry_support_frame'] = 11
        self.assertEqual(guard_keeper_occluded_goal(result, pose)[0]['rule_outcome'], 'goal_likely')

    def test_pose_keeper_track_is_used_when_selected_candidate_has_no_source_ids(self):
        result, pose = self.fixture()
        result['keeper']['selected_candidate'] = {}
        decision, evidence = guard_keeper_occluded_goal(result, pose)
        self.assertEqual(decision['rule_outcome'], 'unknown')
        self.assertEqual(evidence['first_keeper_overlap_frame'], 13)

    def test_rebound_support_on_keeper_body_abstains_and_unknown_is_never_promoted(self):
        result, pose = self.fixture()
        result['rule_decision']['rule_outcome'] = 'no_goal_likely'
        result['rebound']['support_ids'] = ['13']
        self.assertEqual(guard_keeper_occluded_goal(result, pose)[0]['rule_outcome'], 'unknown')
        result['rule_decision']['rule_outcome'] = 'unknown'
        self.assertEqual(guard_keeper_occluded_goal(result, pose)[0]['rule_outcome'], 'unknown')

    def test_missing_pose_box_is_not_an_imagined_occlusion_and_kicker_is_excluded(self):
        result, pose = self.fixture()
        self.assertEqual(guard_keeper_occluded_goal(result, {'tracks': []})[0]['rule_outcome'], 'goal_likely')
        result['independent_contact_search']['selected_candidate']['contact_associated_track_id'] = 3
        decision, evidence = guard_keeper_occluded_goal(result, pose)
        self.assertEqual(decision['rule_outcome'], 'goal_likely')
        self.assertEqual(evidence['keeper_source_track_ids'], [])


if __name__ == '__main__':
    unittest.main()
