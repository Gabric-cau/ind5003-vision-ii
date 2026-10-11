import unittest

import numpy as np

from src.kicker_analysis import KickerFeatureExtractor


EXPECTED_KEYS = {
    "step_count",
    "step_frames",
    "step_feet",
    "runup_start_frame",
    "kick_frame",
    "coverage",
    "confidence",
    "review_required",
    "warnings",
    "signal_frames",
    "phase_signal",
}


def make_pose_sequence(
    frame_count=61,
    period=12,
    amplitude=1.0,
    movement_start=0,
    scale=1.0,
    translation=None,
):
    frames = np.arange(frame_count, dtype=int)
    moving_time = np.maximum(frames - movement_start, 0)
    hip_x = moving_time * 0.08
    phase = np.zeros(frame_count, dtype=float)
    moving_mask = frames >= movement_start
    phase[moving_mask] = amplitude * np.sin(
        2.0
        * np.pi
        * (frames[moving_mask] - movement_start)
        / period
    )

    hip_centres = np.column_stack([
        hip_x,
        np.zeros(frame_count),
    ])
    left_hips = hip_centres + np.array([0.0, -0.10])
    right_hips = hip_centres + np.array([0.0, 0.10])
    left_ankles = hip_centres + np.column_stack([
        phase / 2.0,
        np.ones(frame_count),
    ])
    right_ankles = hip_centres + np.column_stack([
        -phase / 2.0,
        np.ones(frame_count),
    ])

    arrays = [
        left_ankles,
        right_ankles,
        left_hips,
        right_hips,
    ]

    if translation is None:
        translation = np.zeros((frame_count, 2), dtype=float)
    else:
        translation = np.asarray(translation, dtype=float)

        if translation.shape == (2,):
            translation = np.repeat(
                translation[None, :],
                frame_count,
                axis=0,
            )

    arrays = [
        values * scale + translation
        for values in arrays
    ]

    return (frames, *arrays)


def call_counter(sequence, **overrides):
    (
        frames,
        left_ankles,
        right_ankles,
        left_hips,
        right_hips,
    ) = sequence

    arguments = {
        "frames": frames.tolist(),
        "left_ankles": left_ankles.tolist(),
        "right_ankles": right_ankles.tolist(),
        "left_hips": left_hips.tolist(),
        "right_hips": right_hips.tolist(),
        "fps": 30.0,
        "kick_frame": int(frames[-1]),
        "runup_start_frame": 0,
    }
    arguments.update(overrides)

    return KickerFeatureExtractor().count_steps_from_pose(
        **arguments
    )


class TestKickerStepCount(unittest.TestCase):
    def test_flat_phase_signal_returns_zero_steps(self):
        sequence = make_pose_sequence(amplitude=0.0)
        result = call_counter(sequence)

        self.assertEqual(set(result), EXPECTED_KEYS)
        self.assertEqual(result["step_count"], 0)
        self.assertEqual(result["step_frames"], [])
        self.assertEqual(result["step_feet"], [])
        self.assertEqual(result["coverage"], 1.0)

    def test_synthetic_alternating_phase_counts_each_extremum(self):
        sequence = make_pose_sequence()
        result = call_counter(sequence)

        self.assertEqual(result["step_count"], 10)
        self.assertEqual(result["step_feet"], [
            "left",
            "right",
            "left",
            "right",
            "left",
            "right",
            "left",
            "right",
            "left",
            "right",
        ])
        self.assertTrue(np.allclose(
            result["step_frames"],
            [3, 9, 15, 21, 27, 33, 39, 45, 51, 57],
            atol=1,
        ))

    def test_count_is_invariant_to_translation_camera_drift_and_scale(
        self
    ):
        baseline = call_counter(make_pose_sequence())
        frames = np.arange(61, dtype=float)
        camera_translation = np.column_stack([
            40.0 + 0.02 * frames,
            -25.0 + 0.01 * frames,
        ])
        transformed = call_counter(make_pose_sequence(
            scale=3.5,
            translation=camera_translation,
        ))

        self.assertEqual(
            transformed["step_count"],
            baseline["step_count"],
        )
        self.assertEqual(
            transformed["step_frames"],
            baseline["step_frames"],
        )
        self.assertEqual(
            transformed["step_feet"],
            baseline["step_feet"],
        )

    def test_automatic_start_ignores_pre_run_stationary_jitter(self):
        sequence = list(make_pose_sequence(
            frame_count=73,
            movement_start=18,
        ))
        frames = sequence[0]
        jitter = 0.025 * np.sin(2.0 * np.pi * frames / 6.0)
        pre_run = frames < 18
        sequence[1][pre_run, 0] += jitter[pre_run] / 2.0
        sequence[2][pre_run, 0] -= jitter[pre_run] / 2.0

        result = call_counter(
            tuple(sequence),
            runup_start_frame=None,
        )

        self.assertGreaterEqual(result["runup_start_frame"], 16)
        self.assertLessEqual(result["runup_start_frame"], 20)
        self.assertTrue(all(
            frame >= result["runup_start_frame"]
            for frame in result["step_frames"]
        ))
        self.assertEqual(result["step_count"], 9)

    def test_final_support_foot_endpoint_is_retained(self):
        sequence = list(make_pose_sequence())
        frames = sequence[0]
        phase = np.sin(2.0 * np.pi * frames / 12.0)
        phase[48:] = np.linspace(0.0, 1.0, len(frames[48:]))
        hip_centres = (sequence[3] + sequence[4]) / 2.0
        sequence[1][:, 0] = hip_centres[:, 0] + phase / 2.0
        sequence[2][:, 0] = hip_centres[:, 0] - phase / 2.0

        result = call_counter(
            tuple(sequence),
            kicking_foot="right",
        )

        self.assertEqual(result["step_feet"][-1], "left")
        self.assertGreaterEqual(result["step_frames"][-1], 58)

    def test_mismatched_pose_or_confidence_lengths_raise_value_error(
        self
    ):
        sequence = make_pose_sequence(frame_count=12)
        (
            frames,
            left_ankles,
            right_ankles,
            left_hips,
            right_hips,
        ) = sequence
        extractor = KickerFeatureExtractor()

        with self.assertRaisesRegex(ValueError, "same length"):
            extractor.count_steps_from_pose(
                frames=frames.tolist(),
                left_ankles=left_ankles[:-1].tolist(),
                right_ankles=right_ankles.tolist(),
                left_hips=left_hips.tolist(),
                right_hips=right_hips.tolist(),
                fps=30.0,
                kick_frame=11,
            )

        with self.assertRaisesRegex(ValueError, "left_confidences"):
            extractor.count_steps_from_pose(
                frames=frames.tolist(),
                left_ankles=left_ankles.tolist(),
                right_ankles=right_ankles.tolist(),
                left_hips=left_hips.tolist(),
                right_hips=right_hips.tolist(),
                fps=30.0,
                kick_frame=11,
                left_confidences=[0.9] * 11,
            )

    def test_insufficient_pose_data_returns_reviewable_zero_result(self):
        sequence = make_pose_sequence(frame_count=1)
        result = call_counter(sequence, kick_frame=0)

        self.assertEqual(set(result), EXPECTED_KEYS)
        self.assertEqual(result["step_count"], 0)
        self.assertTrue(result["review_required"])
        self.assertIn("insufficient_pose_data", result["warnings"])
        self.assertEqual(result["signal_frames"], [])
        self.assertEqual(result["phase_signal"], [])


if __name__ == "__main__":
    unittest.main()
