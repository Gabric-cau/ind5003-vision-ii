import copy
import unittest

import numpy as np

from scripts.replay_fresh_foot_v2 import cached_samples
from src.kicking_foot_evidence_v2 import kicking_foot_evidence_v2


def samples():
    output = []
    for frame in range(6, 27):
        points = np.tile([150., 130.], (17, 1))
        points[[11, 12]] = [[145, 150], [155, 150]]
        points[[13, 14]] = [[140, 190], [170+.3*max(0, frame-15), 190]]
        points[[15, 16]] = [[140+.2*max(0, frame-15), 230],
                            [180+7*max(0, frame-15), 230]]
        output.append(dict(frame=frame, bbox=[100, 100, 300, 240+(frame%3)*20],
                           keypoints=points.tolist(), confidence=[.95]*17))
    return output


def evaluate(normal=None, mirror=None, stationary=None, departure=None):
    return kicking_foot_evidence_v2(normal or samples(), mirror or samples(), [15, 17], 30.,
        stationary_ball_observations=stationary or [dict(frame=15, detection_id='15:1', xy=[140, 230])],
        departure_ball_observations=departure or [dict(frame=17, detection_id='17:2', xy=[500, 230]),
                                                 dict(frame=18, detection_id='18:3', xy=[650, 230])])


class PairedMotionTests(unittest.TestCase):
    def test_one_stationary_sample_can_support_motion_but_never_contact_truth(self):
        result = evaluate()
        self.assertEqual(result['candidate'], 'right')
        self.assertFalse(result['shoe_ball_contact_verified'])
        distance = result['stationary_ball_distance']['paired']
        self.assertEqual(distance['matched_stationary_frames'], 1)
        self.assertEqual(distance['observations'][0]['left_distance_body'], 0)
        self.assertFalse(result['distance_used_to_choose_anatomical_label'])

    def test_departure_distance_cannot_change_motion_label(self):
        near = evaluate(departure=[dict(frame=17, detection_id='17:2', xy=[145, 225]),
                                   dict(frame=18, detection_id='18:3', xy=[150, 225])])
        far = evaluate()
        self.assertEqual(near['candidate'], far['candidate'])
        self.assertEqual(near['stationary_ball_distance'], far['stationary_ball_distance'])
        self.assertNotEqual(near['causal_departure_observations'], far['causal_departure_observations'])

    def test_reflection_joint_disagreement_is_not_repaired_into_a_label(self):
        mirrored = samples()
        for row in mirrored:
            row['keypoints'][15], row['keypoints'][16] = row['keypoints'][16], row['keypoints'][15]
            row['keypoints'][13], row['keypoints'][14] = row['keypoints'][14], row['keypoints'][13]
        result = evaluate(mirror=mirrored)
        self.assertEqual(result['candidate'], 'unknown')
        self.assertFalse(result['anatomical_channel_permutation_applied'])

    def test_no_observed_departure_cannot_produce_candidate(self):
        result = evaluate(departure=[dict(frame=17, detection_id='17:2', xy=[500, 230])])
        self.assertEqual(result['candidate'], 'unknown')
        self.assertEqual(result['status'], 'actual_stationary_departure_bracket_not_supported')

    def test_missing_knees_do_not_become_confident_leg_chain(self):
        normal = samples()
        for row in normal:
            row['confidence'][13] = .1
        self.assertEqual(evaluate(normal=normal)['candidate'], 'unknown')

    def test_duplicate_cached_neural_view_is_rejected(self):
        sample = samples()[0]
        selected = dict(frame=sample['frame'], bbox=sample['bbox'], keypoints=sample['keypoints'],
                        keypoint_confidences=sample['confidence'])
        row = dict(frame=sample['frame'], reflected=False, selected=selected, selection_reason='accepted')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            cached_samples([row, copy.deepcopy(row)])


if __name__ == '__main__':
    unittest.main()
