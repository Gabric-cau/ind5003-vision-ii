import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.ball_outcome import BallOutcomeExtractor
from src.ball_pipeline import process_ball_outcomes


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
        self.assertEqual(result["last_observed_time_sec"], .2)
        self.assertEqual(result["last_observed_position"], result["final_position"])
        self.assertEqual(result["final_position_semantics"], "last_observed_position")
        self.assertFalse(result["final_position_valid_as_event_endpoint"])
        self.assertIsNone(result["event_final_position"])

    def test_empty_track_is_reviewable(self):
        result = BallOutcomeExtractor().extract_trajectory_and_speed([])
        self.assertEqual(result["ball_track_points"], 0)
        self.assertTrue(result["ball_review_required"])
        self.assertIsNone(result["final_position"])
        self.assertIsNone(result["final_position_normalized"])
        self.assertIsNone(result["ball_speed_px_s"])
        self.assertEqual(result["event_final_position_status"], "no_observations")

    def test_normalized_position_requires_frame_dimensions(self):
        result = BallOutcomeExtractor().extract_trajectory_and_speed([(0., 12, 34)])
        self.assertEqual(result["final_position"], (12., 34.))
        self.assertIsNone(result["final_position_normalized"])

    def test_verified_event_endpoint_can_precede_later_observations(self):
        result = BallOutcomeExtractor().extract_trajectory_and_speed(
            [(0., 10, 20), (.1, 30, 40), (.2, 50, 60)],
            event_end_time_sec=.1, event_end_verified=True,
        )
        self.assertEqual(result["event_final_position"], (30., 40.))
        self.assertEqual(result["final_position"], (50., 60.))
        self.assertEqual(result["event_final_position_status"], "verified_observation")
        self.assertFalse(result["final_position_valid_as_event_endpoint"])

    def test_verified_last_frame_is_explicitly_validated(self):
        result = BallOutcomeExtractor().extract_trajectory_and_speed(
            [(0., 10, 20), (.1, 30, 40)], event_end_time_sec=.1, event_end_verified=True,
        )
        self.assertTrue(result["final_position_valid_as_event_endpoint"])

    def test_endpoint_occlusion_does_not_invent_position(self):
        result = BallOutcomeExtractor().extract_trajectory_and_speed(
            [(0., 10, 20), (.1, 30, 40)], event_end_time_sec=.5, event_end_verified=True,
        )
        self.assertIsNone(result["event_final_position"])
        self.assertEqual(result["event_final_position_status"], "verified_time_unobserved")

    def test_verification_requires_explicit_event_time(self):
        with self.assertRaises(ValueError):
            BallOutcomeExtractor().extract_trajectory_and_speed([(0., 10, 20)], event_end_verified=True)

    def test_missing_position_preserves_independent_outcome_without_geometry(self):
        result = BallOutcomeExtractor().determine_outcome(None, observed_outcome="goal")
        self.assertEqual(result["outcome"], "goal")
        self.assertEqual(result["outcome_source"], "observed_annotation")
        self.assertIsNone(result["final_inside_goal_bbox"])
        self.assertEqual(result["target_zone"], "unknown")
        self.assertEqual(BallOutcomeExtractor().determine_outcome(None)["outcome"], "unknown")

    def test_empty_track_pipeline_exports_null_positions_and_separate_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step = root / "steps.json"
            kicks = root / "kicks.csv"
            outcomes = root / "outcomes.csv"
            output = root / "features.csv"
            diagnostics = root / "diagnostics.json"
            step.write_text(json.dumps([{"clip_id": "sample", "video_path": "not-decoded.mp4"}]))
            kicks.write_text("clip_id,kick_frame,source\nsample,10,unverified_fixture\n")
            outcomes.write_text("clip_id,outcome,outcome_detail,confidence,source\nsample,goal,scored,high,external_fixture\n")
            model = Mock()
            with patch("src.ball_pipeline._read_window", return_value=([], [], 25., 1280, 720)):
                rows = process_ball_outcomes(str(step), str(kicks), str(outcomes), str(output),
                                             str(diagnostics), model, allow_unverified_annotations=True)
            model.predict.assert_not_called()
            self.assertIsNone(rows[0]["final_position_x_px"])
            self.assertIsNone(rows[0]["event_final_position_x_px"])
            self.assertEqual(rows[0]["outcome_source"], "external_fixture")
            self.assertEqual(rows[0]["outcome"], "goal")
            self.assertEqual(rows[0]["ball_track_status"], "review_required")
            with output.open(newline="") as handle:
                exported = next(csv.DictReader(handle))
            self.assertEqual(exported["final_position_x_px"], "")
            self.assertIsNone(json.loads(diagnostics.read_text())[0]["final_position_x_px"])

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
