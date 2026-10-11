import unittest

from src.ball_outcome import BallOutcomeExtractor


class BallOutcomeTests(unittest.TestCase):
    def test_ball_trajectory_speed_and_normalised_final_position(self):
        result = BallOutcomeExtractor().extract_trajectory_and_speed(
            [(0.0, 100, 200), (0.1, 130, 240), (0.2, 160, 280)],
            frame_width=1280,
            frame_height=720,
            expected_frames=5,
            confidences=[0.4, 0.6, 0.8],
        )
        self.assertEqual(result["ball_speed_px_s"], 500.0)
        self.assertEqual(result["final_position"], (160.0, 280.0))
        self.assertEqual(result["final_position_normalized"][0], 0.125)
        self.assertAlmostEqual(result["final_position_normalized"][1], 0.38889)
        self.assertEqual(result["ball_track_coverage"], 0.6)
        self.assertFalse(result["ball_review_required"])

    def test_empty_track_is_reviewable(self):
        result = BallOutcomeExtractor().extract_trajectory_and_speed([])
        self.assertEqual(result["ball_track_points"], 0)
        self.assertTrue(result["ball_review_required"])

    def test_observed_outcome_overrides_geometry_fallback(self):
        extractor = BallOutcomeExtractor(goal_bbox=(0, 0, 100, 100))
        result = extractor.determine_outcome((50, 50), observed_outcome="no_goal")
        self.assertEqual(result["outcome"], "no_goal")
        self.assertEqual(result["outcome_source"], "observed_annotation")
        self.assertTrue(result["final_inside_goal_bbox"])

    def test_invalid_observed_outcome_is_rejected(self):
        with self.assertRaises(ValueError):
            BallOutcomeExtractor().determine_outcome((0, 0), observed_outcome="maybe")


if __name__ == "__main__":
    unittest.main()
