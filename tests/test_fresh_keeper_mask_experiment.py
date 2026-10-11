from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.run_fresh_keeper_mask_experiment import plan_keeper_mask, _lossless_roi_video
from tests.test_fresh_keeper_height import inputs


class KeeperMaskPlanningTests(unittest.TestCase):
    def setup_inputs(self):
        pose, cache, camera, role = inputs()
        association = dict(observations=pose['tracks'][0]['observations'])
        return pose, cache, camera, role, association

    def test_automatic_role_seed_and_original_frame_mapping(self):
        values = self.setup_inputs()
        plan = plan_keeper_mask(*values)
        self.assertEqual(plan['status'], 'planned_source_role_mask_inference')
        self.assertEqual(plan['seed_frame'], 0)
        self.assertEqual(plan['end_frame_exclusive'], 36)
        self.assertEqual(plan['keeper_track_id'], 3)
        self.assertFalse(plan['new_neural_inference'])
        self.assertEqual(plan['expected_frame_count'], 36)

    def test_missing_pre_contact_torso_abstains_without_manual_prompt(self):
        pose, cache, camera, role, association = self.setup_inputs()
        for row in association['observations'][:15]:
            row['keypoint_confidences'][5] = .1
        result = plan_keeper_mask(pose, cache, camera, role, association)
        self.assertEqual(result['reason'], 'no_confident_pre_contact_keeper_torso_seed')

    def test_striker_and_keeper_roles_cannot_coincide(self):
        pose, cache, camera, role, association = self.setup_inputs()
        cache['contact']['contact_associated_track_id'] = 3
        self.assertEqual(plan_keeper_mask(pose, cache, camera, role, association)['reason'],
                         'keeper_role_is_independent_striker')

    def test_camera_break_limits_window_without_reconnecting(self):
        pose, cache, camera, role, association = self.setup_inputs()
        camera['transforms']['30'] = None
        result = plan_keeper_mask(pose, cache, camera, role, association)
        self.assertEqual(result['end_frame_exclusive'], 30)
        self.assertEqual(result['expected_frame_count'], 30)

    def test_camera_break_before_contact_abstains(self):
        pose, cache, camera, role, association = self.setup_inputs()
        camera['transforms']['10'] = None
        self.assertEqual(plan_keeper_mask(pose, cache, camera, role, association)['reason'],
                         'camera_reference_lost_before_contact')

    def test_lossless_crop_keeps_exact_decoded_source_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'FFV1'), 30., (80, 60))
            self.assertTrue(writer.isOpened())
            for frame in range(5):
                image = np.zeros((60, 80, 3), np.uint8)
                image[:, :, 0] = frame*40
                image[10:30, 12:42, 1] = 200
                writer.write(image)
            writer.release()
            records = _lossless_roi_video(source, directory/'crop.avi',
                dict(inference_roi_xyxy=[10, 8, 50, 40], seed_frame=1, end_frame_exclusive=4), 30.)
            self.assertEqual([record['frame'] for record in records], [1, 2, 3])
            self.assertEqual(len({record['crop_bgr_sha256'] for record in records}), 3)


if __name__ == '__main__':
    unittest.main()
