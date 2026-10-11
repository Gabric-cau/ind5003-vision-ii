import unittest
from unittest.mock import patch

import numpy as np

from src.goalkeeper_pipeline import _stabilize_trajectories, analyze_goalkeeper_result


class GoalkeeperStabilizationTests(unittest.TestCase):
    def run_stabilizer(self, matrices, *, missing_frame=None):
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        decoded = [(False, None) if frame == missing_frame else (True, image)
                   for frame in range(5)]
        corners = np.arange(16, dtype=np.float32).reshape(8, 1, 2)
        pre = [(frame / 10, 20 + frame * 10, 12, 10) for frame in range(3)]
        post = [(frame / 10, 20 + frame * 10, 12, 10) for frame in range(2, 5)]
        with patch("cv2.VideoCapture") as capture, \
                patch("cv2.goodFeaturesToTrack", return_value=corners), \
                patch("cv2.calcOpticalFlowPyrLK", return_value=(corners, np.ones((8, 1)), None)), \
                patch("cv2.estimateAffinePartial2D", side_effect=[(matrix, None) for matrix in matrices]) as estimate:
            capture.return_value.read.side_effect = decoded
            result = _stabilize_trajectories("synthetic-camera.mp4", pre, post, 10)
            self.assertTrue(capture.return_value.release.called)
            return result, estimate.call_count

    def test_complete_camera_translation_is_removed(self):
        matrix = np.array([[1., 0., 5.], [0., 1., 0.]])
        (pre, post, success), calls = self.run_stabilizer([matrix] * 4)
        self.assertEqual(calls, 4)
        self.assertEqual(success, 1.0)
        self.assertEqual([row[1] for row in pre + post], [20.] * 6)

    def test_failed_pair_cannot_rejoin_later_successful_pairs(self):
        matrix = np.array([[1., 0., 5.], [0., 1., 0.]])
        (pre, post, success), calls = self.run_stabilizer([matrix, None, matrix, matrix])
        self.assertEqual(calls, 2)
        self.assertEqual([row[0] for row in pre], [0., .1])
        self.assertEqual(post, [])
        self.assertEqual(success, .25)

    def test_decode_gap_invalidates_every_later_reference(self):
        matrix = np.array([[1., 0., 5.], [0., 1., 0.]])
        (pre, post, success), calls = self.run_stabilizer([matrix] * 4, missing_frame=2)
        self.assertEqual(calls, 1)
        self.assertEqual([row[0] for row in pre], [0., .1])
        self.assertEqual(post, [])
        self.assertEqual(success, .25)

    def test_missing_first_frame_never_returns_raw_coordinates_as_stable(self):
        (pre, post, success), calls = self.run_stabilizer([], missing_frame=0)
        self.assertEqual((pre, post, success), ([], [], 0.))
        self.assertEqual(calls, 0)

    def test_singular_pair_is_rejected(self):
        (pre, post, success), _ = self.run_stabilizer([np.zeros((2, 3))])
        self.assertEqual(len(pre), 1)
        self.assertEqual(post, [])
        self.assertEqual(success, 0.)

    def analyze_with_stable_windows(self, stable_pre, stable_post, success):
        pre = [(frame / 10, 50, 40, 100) for frame in range(6)]
        post = [(frame / 10, 50 + 100 * (frame / 10 - .5), 40, 100) for frame in range(5, 11)]
        selected = {"track_id": 1, "pre_trajectory": pre, "post_trajectory": post,
                    "pre_coverage": 1., "post_coverage": 1., "score": 1.}
        with patch("src.goalkeeper_pipeline.select_goalkeeper_track", return_value=(selected, [])), \
                patch("src.goalkeeper_pipeline._stabilize_trajectories", return_value=(stable_pre, stable_post, success)):
            return analyze_goalkeeper_result({"fps": 10, "tracks": []}, 5,
                                             pre_window_sec=.5, post_window_sec=.5)

    def test_late_failure_rejects_dive_even_with_ninety_percent_success(self):
        pre = [(frame / 10, 50, 40, 100) for frame in range(6)]
        post = [(frame / 10, 50 + 100 * (frame / 10 - .5), 40, 100) for frame in range(5, 10)]
        result = self.analyze_with_stable_windows(pre, post, .9)
        self.assertTrue(result["gk_prekick_stabilization_complete"])
        self.assertEqual(result["gk_prekick_movement_level"], "static")
        self.assertFalse(result["dive_stabilization_complete"])
        self.assertEqual(result["dive_direction"], "unknown")
        self.assertIsNone(result["dive_displacement_body"])
        self.assertFalse(result["auto_accepted"])
        self.assertIn("incomplete_camera_reference_chain", result["qa_warnings"])

    def test_complete_windows_keep_computed_features(self):
        pre = [(frame / 10, 50, 40, 100) for frame in range(6)]
        post = [(frame / 10, 50 + 100 * (frame / 10 - .5), 40, 100) for frame in range(5, 11)]
        result = self.analyze_with_stable_windows(pre, post, 1.)
        self.assertTrue(result["gk_prekick_stabilization_complete"])
        self.assertTrue(result["dive_stabilization_complete"])
        self.assertEqual(result["dive_direction"], "image_right")
        self.assertGreater(result["dive_displacement_body"], 0.)
        self.assertTrue(result["auto_accepted"])

    def test_early_failure_rejects_movement_instead_of_fabricating_zero(self):
        result = self.analyze_with_stable_windows([(0., 50, 40, 100), (.1, 50, 40, 100)], [], .1)
        self.assertEqual(result["gk_prekick_movement_level"], "unknown")
        self.assertIsNone(result["gk_prekick_mean_speed_px_s"])
        self.assertIsNone(result["gk_prekick_speed_body_s"])
        self.assertEqual(result["dive_direction"], "unknown")
        self.assertFalse(result["auto_accepted"])


if __name__ == "__main__":
    unittest.main()
