from copy import deepcopy
import unittest

from src.fresh_camera_reference import live_reference_action, source_role_observations
from src.goalkeeper_pipeline import _centre_and_scale


def observation(frame, x=20):
    points = [[x+10, 20] for _ in range(17)]
    for index, point in {5: [x+5, 25], 6: [x+15, 25], 11: [x+5, 50],
                         12: [x+15, 50], 13: [x+2, 70], 14: [x+18, 70],
                         15: [x, 90], 16: [x+20, 90]}.items():
        points[index] = point
    return {'frame': frame, 'bbox': [x, 0, x+30, 100], 'keypoints': points,
            'keypoint_confidences': [.9]*17, 'box_confidence': .9}


class FreshCameraReferenceTest(unittest.TestCase):
    def setUp(self):
        self.pose = {'fps': 25, 'shot_start_frame': 0, 'shot_end_frame': 80,
            '_pose_file_sha256': 'pose-sha', 'inference_identity': {'source_sha256': 'source'},
            'goal_line_role_exclusions': {'excluded_track_ids': [2]},
            'tracks': [
                {'track_id': tid, 'observations': [observation(frame, x) for frame in range(80)]}
                for tid, x in ((1, 20), (2, 100), (3, 400))]}
        candidate = {'contact_associated_track_id': 1, 'contact_frame': 40,
                     'contact_bracket_frames': [39, 41],
                     'contact_timing_status': 'bounded_stationary_departure_candidate'}
        trajectory = [[frame/25, *_centre_and_scale(observation(frame, 100))]
                      for frame in range(80)]
        self.ball = {'independent_contact_search': {'selected_candidate': candidate},
            'keeper': {'role': {'status': 'candidate_supported', 'seed_track_id': 2},
                'selected_candidate': {'qualifies': True, 'source_track_ids': [2, 3],
                    'pre_trajectory': trajectory[:41], 'post_trajectory': trajectory[40:]}}}

    def test_native_pose_observations_are_reused_under_exact_track_and_frame(self):
        result = live_reference_action(self.pose, self.ball, self.ball, 5, 65)
        self.assertFalse(result['new_neural_inference_this_action'])
        self.assertEqual(result['measurement_source'], 'exact_source_pose_producer_observations')
        self.assertEqual(result['source_pose_sha256'], 'pose-sha')
        self.assertEqual([row['frame'] for row in result['roles']['kicker']], list(range(5, 65, 3)))
        self.assertEqual({row['source_track_id'] for row in result['roles']['keeper']}, {2})
        self.assertGreaterEqual(result['contiguous_support']['keeper']['longest_block_samples'], 10)

    def test_missing_exact_frame_is_not_repaired_with_another_pose_or_person(self):
        for track in self.pose['tracks']:
            if track['track_id'] in (1, 2):
                track['observations'] = [row for row in track['observations'] if row['frame'] != 5]
        result = live_reference_action(self.pose, self.ball, self.ball, 5, 15)
        self.assertNotIn(5, [row['frame'] for row in result['roles']['kicker']])
        self.assertNotIn(5, [row['frame'] for row in result['roles']['keeper']])
        self.assertIn(5, result['association_diagnostics']['keeper_unmatched_frames'])

    def test_wrong_pixel_keeper_trajectory_cannot_lock_nearby_person(self):
        self.ball['keeper']['selected_candidate']['pre_trajectory'][5][1] += 1
        roles, diagnostics = source_role_observations(self.pose, self.ball, self.ball)
        self.assertNotIn(5, roles['keeper'])
        self.assertIn(5, diagnostics['keeper_unmatched_frames'])

    def test_rejected_contact_or_goal_occupant_cannot_seed_reference(self):
        for changes in ({'contact_bracket_frames': [39, 80]},
                        {'contact_timing_status': 'unknown'},
                        {'contact_associated_track_id': 2}):
            ball = deepcopy(self.ball)
            ball['independent_contact_search']['selected_candidate'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                source_role_observations(self.pose, ball, ball)

    def test_reference_window_cannot_cross_source_camera_cut(self):
        with self.assertRaisesRegex(ValueError, 'cannot cross'):
            live_reference_action(self.pose, self.ball, self.ball, -1, 65)
        with self.assertRaisesRegex(ValueError, 'cannot cross'):
            live_reference_action(self.pose, self.ball, self.ball, 10, 81)

    def test_unsupported_keeper_does_not_create_reference_from_kit_colour(self):
        self.ball['keeper']['role']['status'] = 'abstain'
        result = live_reference_action(self.pose, self.ball, self.ball, 5, 65)
        self.assertEqual(result['roles']['keeper'], [])
        self.assertGreater(len(result['roles']['kicker']), 0)


if __name__ == '__main__':
    unittest.main()
