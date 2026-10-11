from copy import deepcopy
import unittest

import numpy as np

from src.fresh_ball_endpoint import classify_endpoint


class FreshBallEndpointTest(unittest.TestCase):
    def row(self, frame, x, y=80):
        return {'frame': frame, 'detection_id': str(frame), 'x': x, 'y': y,
                'width': 10, 'height': 10, 'stabilized_xy': [x, y],
                'confidence': .8, 'sources': ['fresh_neural_detection']}

    def classify(self, path, **kwargs):
        options = {'boundary_reason': 'scene_cut', 'identity_supported': True,
                   'end_frame_exclusive': max(row['frame'] for row in path)+1}
        options.update(kwargs)
        pool = options.pop('pool', {str(row['frame']): [row] for row in path})
        transforms = options.pop('transforms', {str(f): np.eye(3).tolist() for f in range(options['end_frame_exclusive'])})
        return classify_endpoint(path, pool, transforms, 0, 30., width=300, height=160, **options)

    def settled_path(self):
        return [self.row(0, 20), self.row(1, 45), self.row(2, 70)]+[self.row(f, 100+.1*(f % 2)) for f in range(3, 26)]

    def test_sustained_same_identity_settling_is_candidate_and_actual_source_point_not_final_truth(self):
        path = self.settled_path()
        result = self.classify(path)
        self.assertEqual(result['status'], 'stationary_end_candidate')
        candidate = result['stationary_end_candidate']
        measured = next(row for row in path if row['detection_id'] == candidate['detection_id'])
        self.assertEqual(candidate['source_xy'], [measured['x'], measured['y']])
        self.assertIsNone(result['final_position'])
        self.assertFalse(result['physical_ball_rest_established'])

    def test_settling_before_long_detection_loss_is_not_endpoint(self):
        result = self.classify(self.settled_path(), end_frame_exclusive=100)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['reason'], 'observed_settling_followed_by_unobserved_interval')

    def test_cut_and_replay_cut_censor_same_identity_flight_without_final_position(self):
        path = [self.row(f, 20+10*f) for f in range(5)]
        for boundary, expected in (('scene_cut', 'camera_cut_censored'), ('replay_cut_candidate', 'replay_cut_candidate')):
            result = self.classify(path, boundary_reason=boundary)
            self.assertEqual(result['status'], expected)
            self.assertIsNone(result['final_position'])

    def test_missing_camera_link_cannot_be_skipped_or_repaired_by_later_identity_matrix(self):
        path = self.settled_path()
        transforms = {str(f): np.eye(3).tolist() for f in range(26)}
        transforms['10'] = None
        result = self.classify(path, transforms=transforms)
        self.assertEqual(result['status'], 'camera_reference_lost')
        self.assertEqual(result['camera_prefix_stop_frame'], 10)

    def test_edge_departure_and_person_overlap_have_distinct_candidate_states(self):
        path = [self.row(0, 270), self.row(1, 283), self.row(2, 297)]
        self.assertEqual(self.classify(path, end_frame_exclusive=30)['status'], 'out_of_view_candidate')
        interior = [self.row(0, 40), self.row(1, 50), self.row(2, 60)]
        result = self.classify(interior, end_frame_exclusive=30, people={'2': [{'bbox': [55, 50, 75, 110]}]})
        self.assertEqual(result['status'], 'occluded_candidate')
        self.assertFalse(result['occlusion_verified'])

    def test_competing_identity_and_encoded_repeats_do_not_establish_stationary_end(self):
        path = self.settled_path()
        pool = {str(row['frame']): [row] for row in path}
        competing = {**self.row(20, 102), 'detection_id': 'other'}
        pool['20'].append(competing)
        self.assertEqual(self.classify(path, pool=pool)['reason'], 'competing_ball_identity_at_stationary_end')
        repeated = self.classify(path, duplicate_frames=list(range(5, 26)))
        self.assertNotEqual(repeated['status'], 'stationary_end_candidate')

    def test_unmeasured_or_changed_source_point_is_integrity_error(self):
        path = self.settled_path()
        pool = {str(row['frame']): [deepcopy(row)] for row in path}
        path[-1]['x'] += 1
        with self.assertRaisesRegex(ValueError, 'changed measured source'):
            self.classify(path, pool=pool)


if __name__ == '__main__':
    unittest.main()
