"""Synthetic encoded-video checks; these do not establish real-clip accuracy.

The AVI is generated from reproducible texture and decoded through OpenCV, not
from mocked optical-flow points.  One test deliberately rejects an otherwise
valid frame pair to expose a broken cumulative-transform chain.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src import motion_stabilization as motion


class MotionStabilizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-camera-motion-",
                                                 dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.width, self.height = 320, 240
        rng = np.random.default_rng(20261009)
        noise = rng.integers(20, 235, (self.height, self.width), dtype=np.uint8)
        self.texture = cv2.GaussianBlur(noise, (3, 3), 0.6)
        for x in range(25, self.width - 20, 35):
            for y in range(30, self.height - 20, 35):
                cv2.rectangle(self.texture, (x, y), (x + 8, y + 8), 250, -1)

    def video(self, name="translation.avi", frames=24, dx=2.0, dy=1.0, texture=True):
        path = self.directory / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 30.0, (self.width, self.height))
        self.assertTrue(writer.isOpened(), "Synthetic-video writer must be usable for these integration tests")
        base = self.texture if texture else np.full_like(self.texture, 80)
        try:
            for frame_id in range(frames):
                affine = np.array([[1, 0, dx * frame_id], [0, 1, dy * frame_id]], dtype=float)
                image = cv2.warpAffine(base, affine, (self.width, self.height),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()
        return path

    def test_actual_avi_translation_compensated(self):
        path = self.video()
        result = motion.estimate_background_motion(str(path), 0, 24)
        self.assertTrue(result["decoded_complete"])
        self.assertEqual(result["status"], "supported")
        self.assertTrue(all(item["accepted"] for item in result["pairs"]))
        frames = np.arange(24)
        points = np.column_stack((100.0 + 2 * frames, 130.0 + frames))
        stabilized = np.asarray(motion.apply_transforms(points, frames, result["transforms"]))
        np.testing.assert_allclose(stabilized, np.tile([100., 130.], (24, 1)), atol=1.0)
        self.assertFalse(result["metric_calibrated"])
        self.assertFalse(result["human_verified"])
        json.dumps(result, allow_nan=False)

    def test_downscaled_estimation_returns_full_resolution_coordinates(self):
        path = self.video()
        result = motion.estimate_background_motion(str(path), 0, 24, target_width=160)
        self.assertEqual(result["status"], "supported")
        frames = np.arange(24)
        points = np.column_stack((100.0 + 2 * frames, 130.0 + frames))
        stabilized = np.asarray(motion.apply_transforms(points, frames, result["transforms"]))
        np.testing.assert_allclose(stabilized, np.tile([100., 130.], (24, 1)), atol=1.5)

    def test_nonzero_start_anchor_and_absolute_frame_ids(self):
        path = self.video(frames=30)
        result = motion.estimate_background_motion(str(path), 7, 24)
        self.assertEqual(set(result["transforms"]), {str(x) for x in range(7, 24)})
        self.assertEqual(result["pairs"][0]["frame"], 8)
        np.testing.assert_allclose(result["transforms"]["7"], np.eye(3), atol=1e-8)
        frames = np.arange(7, 24)
        points = np.column_stack((100.0 + 2 * frames, 130.0 + frames))
        stabilized = np.asarray(motion.apply_transforms(points, frames, result["transforms"]))
        # Coordinate origin is selected shot frame 7, not video frame 0.
        np.testing.assert_allclose(stabilized, np.tile([114., 137.], (17, 1)), atol=1.0)

    def test_degenerate_no_feature_video_cannot_claim_support(self):
        path = self.video(name="constant.avi", texture=False)
        result = motion.estimate_background_motion(str(path), 0, 24)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["success_fraction"], 0.0)
        self.assertTrue(result["decoded_complete"])
        self.assertTrue(all(not x["accepted"] for x in result["pairs"]))
        self.assertEqual(result["max_failed_pairs_in_a_row"], 23)

    def test_all_features_excluded_cannot_claim_support(self):
        path = self.video()
        boxes = {frame: [[0, 0, self.width, self.height]] for frame in range(24)}
        result = motion.estimate_background_motion(str(path), 0, 24, boxes)
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(all(x["tracked_features"] == 0 for x in result["pairs"]))

    def test_isolated_rejected_pair_breaks_cumulative_first_frame_support(self):
        path = self.video(frames=24)
        original = motion.cv2.goodFeaturesToTrack
        calls = 0

        def fail_one_pair(*args, **kwargs):
            nonlocal calls
            calls += 1
            return None if calls == 5 else original(*args, **kwargs)

        with patch.object(motion.cv2, "goodFeaturesToTrack", side_effect=fail_one_pair):
            result = motion.estimate_background_motion(str(path), 0, 24)
        rejected = [item for item in result["pairs"] if not item["accepted"]]
        self.assertEqual([item["frame"] for item in rejected], [5])
        self.assertGreater(result["success_fraction"], 0.9)
        self.assertNotEqual(result["status"], "supported",
                            "An unrecovered failed pair makes subsequent first-frame transforms incomplete")
        self.assertTrue(all(result["transforms"][str(frame)] is not None for frame in range(5)))
        self.assertTrue(all(result["transforms"][str(frame)] is None for frame in range(5, 24)))
        self.assertTrue(all(x["reference_chain_valid"] == (x["frame"] < 5) for x in result["pairs"]))
        points = np.column_stack((100.0 + 2 * np.arange(24), 130.0 + np.arange(24)))
        stabilized = np.asarray(motion.apply_transforms(points, np.arange(24), result["transforms"]), float)
        self.assertTrue(np.isfinite(stabilized[:5]).all())
        self.assertTrue(np.isnan(stabilized[5:]).all())

    def test_incomplete_decode_cannot_claim_complete_or_supported(self):
        path = self.video(frames=9)
        result = motion.estimate_background_motion(str(path), 0, 13)
        self.assertFalse(result["decoded_complete"])
        self.assertNotEqual(result["status"], "supported")
        self.assertTrue(all(int(key) < 9 for key in result["transforms"]))
        self.assertEqual(len(result["pairs"]), 8)

    def test_missing_transform_is_not_silently_identity(self):
        transforms = {"7": np.eye(3).tolist(), "9": None}
        result = np.asarray(motion.apply_transforms([[20, 30], [40, 50], [60, 70]], [7, 8, 9], transforms), float)
        np.testing.assert_allclose(result[0], [20, 30])
        self.assertTrue(np.all(np.isnan(result[1])),
                        "Missing transform must abstain, not masquerade as no camera motion")
        self.assertTrue(np.all(np.isnan(result[2])), "Explicitly invalid transform must abstain")

    def test_start_after_video_end_raises(self):
        path = self.video(frames=5)
        with self.assertRaises(ValueError):
            motion.estimate_background_motion(str(path), 20, 25)

    def test_invalid_interval_rejected(self):
        for start, end in [(-1, 5), (5, 5), (9, 3)]:
            with self.assertRaises(ValueError):
                motion.estimate_background_motion("not-needed.avi", start, end)


if __name__ == "__main__":
    unittest.main()
