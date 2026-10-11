"""Synthetic mechanics tests, NOT accuracy evaluation on football clips."""

import copy
import json
import unittest

import numpy as np

from src.penalty_phases import PhaseConfig, classify_penalty_phases


def synthetic_track(fps=30, offset=100, duration=6.0):
    """Lateral preparation, a short pause, forward approach, then recovery."""
    t = np.arange(int(duration * fps)) / fps
    hips = np.column_stack((100 + 90 * np.maximum(t - 1.5, 0),
                            100 + 30 * np.minimum(t, 1)))
    relative_left = np.column_stack((12 * np.sin(2 * np.pi * 2 * t), np.full(len(t), 50)))
    relative_right = np.column_stack((-12 * np.sin(2 * np.pi * 2 * t), np.full(len(t), 50)))
    return dict(frame_ids=np.arange(len(t)) + offset, fps=fps, hips=hips,
                left_ankles=hips + relative_left, right_ankles=hips + relative_right,
                box_heights=np.full(len(t), 100.0), confidences=np.ones((len(t), 3)),
                stabilized_hips=hips.copy(), kick_frame=offset + int(4.5 * fps),
                kick_confidence=0.9, camera_motion_status="supported",
                step_events=[{"frame": offset + int(s * fps), "foot": foot}
                             for s, foot in [(0.5, "left"), (1.7, "right"),
                                             (2.5, "left"), (4.5, "right"), (5.2, "left")]])


class PenaltyPhaseTests(unittest.TestCase):
    def test_net_progress_gate_rejects_early_body_sway(self):
        data = synthetic_track(offset=0)
        t = np.arange(len(data["hips"])) / data["fps"]
        replacement = np.c_[100 + 2 * np.sin(2 * np.pi * 2 * t) + 90 * np.maximum(t - 1.5, 0),
                            np.full(len(t), 100.)]
        correction = replacement - data["hips"]
        for name in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[name] += correction
        result = classify_penalty_phases(**data)
        self.assertGreaterEqual(result["approach_start_frame"], 30)
        self.assertLessEqual(result["approach_start_frame"], 49)
        self.assertIn("progress_confirmation", result["metrics"])

    def test_unknown_foot_still_has_candidate_progress_not_step_length(self):
        data = synthetic_track(offset=0)
        data["step_events"] = [{"frame": f, "foot": "unknown"} for f in range(55, 125, 6)]
        result = classify_penalty_phases(**data)
        self.assertTrue(result["metrics"]["candidate_interval_progress"])
        self.assertTrue(all(e["candidate_step_style"] == "uncertain" for e in result["events"]))
        self.assertIn("dense_low_progress_movement_image_plane_candidate", [x["tag"] for x in result["style_tags"]])

    def test_preparation_initiation_approach_strike_recovery(self):
        result = classify_penalty_phases(**synthetic_track())
        self.assertEqual(result["status"], "provisional")
        self.assertEqual([x["phase"] for x in result["events"]],
                         ["preparation_adjustment", "initiation", "approach", "strike", "recovery"])
        self.assertAlmostEqual(result["approach_start_frame"], 145, delta=4)
        self.assertIn("lateral_adjustment_relative_to_approach_image_plane",
                      [x["tag"] for x in result["style_tags"]])

    def test_no_claim_of_human_verification_or_physical_speed(self):
        data = synthetic_track()
        data["step_events"][0]["human_verified"] = True
        data["step_events"][0]["contact_status"] = "confirmed"
        result = classify_penalty_phases(**data)
        self.assertFalse(result["human_verified"])
        self.assertTrue(result["review_required"])
        self.assertFalse(result["evidence_score_is_probability"])
        for event in result["events"]:
            self.assertFalse(event["human_verified"])
            self.assertEqual(event["contact_status"], "unverified_candidate")
        self.assertNotIn("sprint", [x["tag"] for x in result["style_tags"]])
        self.assertNotIn("speed_mps", result["metrics"])

    def test_translation_and_uniform_scale_invariance(self):
        data = synthetic_track()
        before = classify_penalty_phases(**data)
        for key in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[key] = data[key] * 2.8 + [731, -451]
        data["box_heights"] *= 2.8
        after = classify_penalty_phases(**data)
        self.assertEqual(before["frame_labels"], after["frame_labels"])
        self.assertEqual([x["tag"] for x in before["style_tags"]], [x["tag"] for x in after["style_tags"]])
        self.assertEqual([x["candidate_step_style"] for x in before["events"]],
                         [x["candidate_step_style"] for x in after["events"]])

    def test_frame_offset_invariance(self):
        a, b = synthetic_track(offset=0), synthetic_track(offset=753)
        ra, rb = classify_penalty_phases(**a), classify_penalty_phases(**b)
        self.assertEqual(rb["approach_start_frame"] - ra["approach_start_frame"], 753)
        self.assertEqual([x["phase"] for x in ra["frame_labels"]], [x["phase"] for x in rb["frame_labels"]])
        self.assertEqual([x["start_frame"] + 753 for x in ra["segments"]], [x["start_frame"] for x in rb["segments"]])

    def test_rotation_and_reflection_use_relative_approach_axis(self):
        data = synthetic_track()
        before = classify_penalty_phases(**data)
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        for key in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[key] = data[key] @ rotation
        after = classify_penalty_phases(**data)
        self.assertEqual(before["frame_labels"], after["frame_labels"])
        self.assertEqual([x["tag"] for x in before["style_tags"]], [x["tag"] for x in after["style_tags"]])
        for key in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[key][:, 0] *= -1
        mirror = classify_penalty_phases(**data)
        self.assertEqual(after["frame_labels"], mirror["frame_labels"])
        self.assertEqual([x["candidate_step_style"] for x in after["events"]],
                         [x["candidate_step_style"] for x in mirror["events"]])

    def test_stabilized_hips_but_raw_hip_relative_ankles(self):
        data = synthetic_track()
        before = classify_penalty_phases(**data)
        camera = np.column_stack((np.arange(len(data["hips"])) * 5.0,
                                  np.arange(len(data["hips"])) * -2.0))
        for key in ("hips", "left_ankles", "right_ankles"):
            data[key] += camera
        after = classify_penalty_phases(**data)
        self.assertEqual(before["frame_labels"], after["frame_labels"])
        self.assertEqual([x["candidate_step_style"] for x in before["events"]],
                         [x["candidate_step_style"] for x in after["events"]])

    def test_absent_kick_anchor_abstains_from_phase_labels(self):
        data = synthetic_track()
        data["kick_frame"] = None
        result = classify_penalty_phases(**data)
        self.assertEqual(result["status"], "abstained")
        self.assertTrue(all(x["phase"] == "unknown" for x in result["frame_labels"]))
        self.assertTrue(result["metrics"]["movement_bouts"])
        self.assertEqual(result["style_tags"], [])

    def test_low_confidence_or_outside_anchor_abstains(self):
        for changes in ({"kick_confidence": 0.49}, {"kick_frame": 9999}, {"kick_frame": 120.5}):
            data = synthetic_track()
            data.update(changes)
            result = classify_penalty_phases(**data)
            self.assertFalse(result["kick_anchor"]["usable"])
            self.assertEqual(result["status"], "abstained")

    def test_long_missing_gap_not_interpolated_or_classified(self):
        data = synthetic_track()
        data["step_events"].append({"frame": 180, "foot": "left"})
        keep = (data["frame_ids"] < 170) | (data["frame_ids"] > 192)
        for key in ("frame_ids", "hips", "left_ankles", "right_ankles", "box_heights", "confidences", "stabilized_hips"):
            data[key] = data[key][keep]
        result = classify_penalty_phases(**data)
        gap = [x for x in result["frame_labels"] if 170 <= x["frame"] <= 192]
        self.assertTrue(all(x["phase"] == "unknown" and not x["pose_supported"] for x in gap))
        self.assertEqual(result["events"][-1]["phase"], "unknown")
        self.assertIn("event_in_pose_gap", result["events"][-1]["warnings"])
        self.assertIn("long_pose_gaps_not_interpolated", result["warnings"])

    def test_short_internal_gap_can_be_interpolated(self):
        data = synthetic_track()
        keep = data["frame_ids"] != 175
        for key in ("frame_ids", "hips", "left_ankles", "right_ankles", "box_heights", "confidences", "stabilized_hips"):
            data[key] = data[key][keep]
        result = classify_penalty_phases(**data)
        item = next(x for x in result["frame_labels"] if x["frame"] == 175)
        self.assertTrue(item["pose_supported"])
        self.assertFalse(item["pose_observed"])
        self.assertTrue(item["pose_interpolated"])
        self.assertEqual(result["metrics"]["interpolated_pose_frames"], 1)
        self.assertEqual(item["phase"], "approach")

    def test_anchor_inside_pose_gap_abstains(self):
        data = synthetic_track()
        data["confidences"][130:140] = 0.0
        result = classify_penalty_phases(**data)
        self.assertFalse(result["kick_anchor"]["usable"])
        self.assertIn("kick_anchor_in_pose_gap", result["warnings"])

    def test_low_confidence_ankles_do_not_become_small_steps(self):
        data = synthetic_track()
        data["confidences"][:, 1:] = 0.1
        result = classify_penalty_phases(**data)
        self.assertEqual(result["status"], "abstained")
        self.assertTrue(all(x["candidate_step_style"] == "uncertain" for x in result["events"]))

    def test_small_projected_ankle_excursion_is_not_deleted(self):
        data = synthetic_track()
        for key in ("left_ankles", "right_ankles"):
            data[key] = data["hips"] + (data[key] - data["hips"]) * 0.2
        result = classify_penalty_phases(**data)
        self.assertEqual(len(result["events"]), len(data["step_events"]))
        self.assertEqual(result["events"][2]["candidate_step_style"], "small_image_plane_ankle_excursion")
        self.assertEqual(result["events"][2]["phase"], "approach")

    def test_recovery_events_not_reported_as_approach(self):
        result = classify_penalty_phases(**synthetic_track())
        self.assertFalse(result["events"][-1]["within_inferred_approach"])
        self.assertFalse(result["events"][-1]["before_kick_anchor"])
        self.assertEqual(result["phase_event_counts"]["recovery"], 1)

    def test_stationary_body_does_not_create_runup_from_ankle_fidgets(self):
        data = synthetic_track()
        for key in ("left_ankles", "right_ankles"):
            data[key] = data[key] - data["hips"] + [100, 100]
        data["hips"][:] = [100, 100]
        data["stabilized_hips"][:] = [100, 100]
        result = classify_penalty_phases(**data)
        self.assertIsNone(result["approach_start_frame"])
        self.assertEqual(result["status"], "partially_segmented")
        self.assertIn("no_supported_terminal_approach_bout", result["warnings"])

    def test_observed_pace_increase_is_not_called_sprint(self):
        data = synthetic_track()
        t = np.arange(len(data["hips"])) / data["fps"]
        delta = np.maximum(t - 1.5, 0)
        hip_new = data["hips"].copy()
        hip_new[:, 0] = 100 + 100 * (0.03 * delta + delta ** 3)
        correction = hip_new - data["hips"]
        for key in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[key] += correction
        result = classify_penalty_phases(**data)
        names = [x["tag"] for x in result["style_tags"]]
        self.assertIn("pace_increase_image_plane", names)
        self.assertFalse(any("sprint" in x for x in names))

    def test_stutter_pause_remains_inside_approach(self):
        data = synthetic_track()
        t = np.arange(len(data["hips"])) / data["fps"]
        correction = np.column_stack((-90 * np.clip(t - 2.8, 0, 0.2), np.zeros(len(t))))
        for key in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[key] += correction
        result = classify_penalty_phases(**data)
        names = [x["tag"] for x in result["style_tags"]]
        self.assertIn("pause_or_stutter_image_plane_candidate", names)
        self.assertLess(result["approach_start_frame"], 160)

    def test_curved_path_remains_image_plane_description(self):
        data = synthetic_track()
        t = np.arange(len(data["hips"])) / data["fps"]
        curve = np.column_stack((np.zeros(len(t)), 80 * np.sin(np.clip((t - 1.5) / 3, 0, 1) * np.pi)))
        for key in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[key] += curve
        result = classify_penalty_phases(**data)
        tags = [x["tag"] for x in result["style_tags"]]
        self.assertTrue(any("curved" in x and "image_plane" in x for x in tags))
        self.assertFalse(any(x in {"circular_runup", "sprint", "physical_curve"} for x in tags))

    def test_projected_preparation_loop_is_only_a_candidate(self):
        data = synthetic_track(duration=8)
        t = np.arange(len(data["hips"])) / data["fps"]
        theta = np.minimum(t / 2.0, 1) * 2 * np.pi
        new_hips = np.column_stack((100 + 30 * np.cos(theta) + 90 * np.maximum(t - 3, 0),
                                    100 + 30 * np.sin(theta)))
        correction = new_hips - data["hips"]
        for key in ("hips", "left_ankles", "right_ankles", "stabilized_hips"):
            data[key] += correction
        data["kick_frame"] = 100 + int(6.5 * data["fps"])
        result = classify_penalty_phases(**data)
        loops = [x for x in result["style_tags"] if x["tag"] == "loop_like_preparation_image_plane_candidate"]
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0]["status"], "candidate")
        self.assertFalse(loops[0]["human_verified"])

    def test_clip_start_during_approach_is_left_censored(self):
        data = synthetic_track()
        keep = data["frame_ids"] >= 162
        for key in ("frame_ids", "hips", "left_ankles", "right_ankles", "box_heights", "confidences", "stabilized_hips"):
            data[key] = data[key][keep]
        result = classify_penalty_phases(**data)
        self.assertIn("approach_may_be_left_censored", result["warnings"])
        self.assertFalse(any(x["phase"] == "preparation_adjustment" for x in result["segments"]))

    def test_camera_unknown_keeps_explicit_uncertainty(self):
        data = synthetic_track()
        data["camera_motion_status"] = "unknown"
        data["stabilized_hips"] = None
        result = classify_penalty_phases(**data)
        self.assertIn("camera_motion_not_fully_resolved", result["warnings"])
        self.assertTrue(all(x["evidence_score"] <= 0.4 for x in result["style_tags"]))
        self.assertTrue(result["review_required"])

    def test_empty_track_preserves_unverified_event(self):
        result = classify_penalty_phases([], 30, [], [], [], [], step_events=[34])
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["events"][0]["frame"], 34)
        self.assertEqual(result["events"][0]["phase"], "unknown")

    def test_output_strict_json_and_no_input_mutation(self):
        data = synthetic_track()
        before = copy.deepcopy(data)
        data["step_events"].append({"frame": 5000, "source_score": float("nan")})
        result = classify_penalty_phases(**data)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["events"][-1]["source_score"], None)
        for key in ("hips", "left_ankles", "right_ankles", "frame_ids"):
            np.testing.assert_array_equal(data[key], before[key])
        self.assertNotIn("phase", data["step_events"][0])

    def test_segments_exclusive_end_partition_entire_timeline(self):
        result = classify_penalty_phases(**synthetic_track())
        segments = result["segments"]
        self.assertEqual(segments[0]["start_frame"], 100)
        self.assertEqual(segments[-1]["end_frame"], 280)
        for a, b in zip(segments[:-1], segments[1:]):
            self.assertEqual(a["end_frame"], b["start_frame"])
        self.assertEqual(sum(x["end_frame"] - x["start_frame"] for x in segments), 180)

    def test_fps_resampling_preserves_onset_time(self):
        a = classify_penalty_phases(**synthetic_track(fps=30, offset=0))
        b = classify_penalty_phases(**synthetic_track(fps=60, offset=0))
        self.assertAlmostEqual(a["approach_start_frame"] / 30, b["approach_start_frame"] / 60, delta=0.08)

    def test_invalid_frame_order_shape_and_fps_rejected(self):
        for key, value in (("fps", 0), ("frame_ids", [1, 1]), ("hips", [[1, 2]]),
                           ("camera_motion_status", "perfect")):
            data = synthetic_track()
            data[key] = value
            with self.assertRaises(ValueError):
                classify_penalty_phases(**data)

    def test_all_thresholds_are_disclosed_and_no_old_runup_input(self):
        result = classify_penalty_phases(**synthetic_track())
        self.assertEqual(result["thresholds"]["min_kick_confidence"], PhaseConfig().min_kick_confidence)
        self.assertEqual(result["annotation_source"], "heuristic_2d_pose_phase_classifier")
        with self.assertRaises(TypeError):
            classify_penalty_phases(**synthetic_track(), runup_start_frame=1)


if __name__ == "__main__":
    unittest.main()
