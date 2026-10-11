"""Synthetic mechanics/invariance tests; none establish real-video accuracy."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest

import numpy as np

from src.terminal_footwork import TerminalConfig, analyze_terminal_footwork


def fixture(events=(0, 12, 24, 36, 42, 48, 54), kick=60, fps=30):
    frames = np.arange(kick + 8)
    return dict(frame_ids=frames, fps=fps,
        stabilized_hips=np.column_stack((frames * 2., np.zeros(len(frames)))),
        box_heights=np.full(len(frames), 100.), camera_status="supported",
        phase_result={"approach_start_frame": 0, "kick_anchor": {"usable": True},
            "frame_labels": [{"frame": int(f), "pose_supported": True} for f in frames]},
        contact={"contact_frame": kick, "interval_frames": [kick-2, kick+3], "status": "supported"},
        events=[{"frame": f, "foot": "unknown"} for f in events])


class TerminalFootworkTests(unittest.TestCase):
    def test_terminal_compression_is_relative_not_last_n(self):
        r = analyze_terminal_footwork(**fixture())
        self.assertEqual(r["terminal"]["start_frame"], 36)
        self.assertIn("rhythm_compression", r["terminal"]["tags"])
        self.assertEqual(r["counts"]["terminal_adjustment_candidates"], 4)
        self.assertEqual(r["counts"]["terminal_count_excluding_boundary_event"], 3)

    def test_uniform_fast_steps_do_not_force_terminal_change(self):
        r = analyze_terminal_footwork(**fixture(tuple(range(0, 55, 6))))
        self.assertEqual(r["terminal"]["status"], "no_distinct_change_detected")
        self.assertIsNone(r["counts"]["terminal_adjustment_candidates"])

    def test_uniform_curved_route_is_not_terminal_adjustment(self):
        a = fixture(tuple(range(0, 55, 6)))
        theta = np.asarray(a["frame_ids"]) * .035
        a["stabilized_hips"] = 150 * np.column_stack((np.sin(theta), np.cos(theta)))
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["status"], "no_distinct_change_detected")
        self.assertIn("curved_image_plane", r["route"]["tags"])

    def test_curve_and_terminal_change_can_coexist(self):
        a = fixture()
        theta = np.asarray(a["frame_ids"]) * .04
        a["stabilized_hips"] = 150 * np.column_stack((np.sin(theta), np.cos(theta)))
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["status"], "candidate_change")
        self.assertIn("curved_image_plane", r["route"]["tags"])

    def test_one_short_final_interval_is_not_sustained(self):
        r = analyze_terminal_footwork(**fixture((0, 12, 24, 36, 48, 54)))
        self.assertEqual(r["terminal"]["status"], "no_distinct_change_detected")

    def test_early_compression_followed_by_normal_rhythm_is_not_terminal(self):
        r = analyze_terminal_footwork(**fixture((0, 3, 6, 18, 30, 42, 54)))
        self.assertNotIn("rhythm_compression", r["terminal"]["tags"])

    def test_terminal_lengthening_not_called_small_steps(self):
        r = analyze_terminal_footwork(**fixture((0, 6, 12, 18, 30, 42, 54)))
        self.assertIn("rhythm_lengthening", r["terminal"]["tags"])
        self.assertNotIn("lower_projected_progress", r["terminal"]["tags"])

    def test_contact_guard_cannot_create_a_change(self):
        r = analyze_terminal_footwork(**fixture((0, 10, 20, 30, 40, 50, 56, 58, 59, 60, 62)))
        self.assertEqual(r["terminal"]["status"], "no_distinct_change_detected")
        self.assertEqual(r["counts"]["contact_guard_candidates"], 3)
        self.assertEqual(r["counts"]["nominal_approach_candidates"], 9)

    def test_no_camera_allows_timing_but_not_route_or_progress(self):
        a = fixture()
        a.update(camera_status="partial", stabilized_hips=None)
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["tags"], ["rhythm_compression"])
        self.assertEqual(r["route"]["status"], "unavailable")

    def test_scale_zoom_disables_displacement(self):
        a = fixture()
        a["box_heights"] = np.linspace(50, 200, len(a["frame_ids"]))
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["tags"], ["rhythm_compression"])

    def test_long_gap_is_not_bridged_for_change(self):
        a = fixture()
        for row in a["phase_result"]["frame_labels"]:
            if 40 <= row["frame"] <= 45:
                row["pose_supported"] = False
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["status"], "insufficient_evidence")
        self.assertEqual(r["counts"]["pose_gap_candidates"], 1)

    def test_tail_gap_prevents_no_change_claim(self):
        a = fixture()
        for row in a["phase_result"]["frame_labels"]:
            if row["frame"] == 55:
                row["pose_supported"] = False
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["status"], "insufficient_evidence")

    def test_insufficient_is_not_zero(self):
        r = analyze_terminal_footwork(**fixture((36, 42, 48, 54)))
        self.assertIsNone(r["counts"]["terminal_adjustment_candidates"])
        self.assertEqual(r["counts"]["unresolved_prestrike_candidates"], 4)

    def test_unreliable_axis_abstains(self):
        r = analyze_terminal_footwork(**fixture(), candidate_axis_reliable=False)
        self.assertEqual(r["terminal"]["status"], "insufficient_evidence")

    def test_missing_contact_gives_null_counts(self):
        a = fixture()
        a["contact"].update(status="abstain", contact_frame=None)
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["status"], "abstained")
        self.assertIsNone(r["counts"]["nominal_approach_candidates"])

    def test_tentative_contact_is_explicit(self):
        a = fixture()
        a["contact"]["status"] = "tentative"
        self.assertIn("terminal_split_uses_tentative_contact", analyze_terminal_footwork(**a)["warnings"])

    def test_counts_partition_without_overlapping_guard(self):
        r = analyze_terminal_footwork(**fixture((0, 12, 24, 36, 42, 48, 54, 57, 59, 61)))
        self.assertEqual(r["partition_sum"], 9)
        self.assertEqual(sum(r["counts"][k] or 0 for k in ("before_terminal_candidates", "terminal_adjustment_candidates",
            "contact_guard_candidates", "pose_gap_candidates", "unresolved_prestrike_candidates", "no_distinct_split_candidates")), 9)

    def test_translation_scale_rotation_reflection_invariance(self):
        a = fixture()
        r = analyze_terminal_footwork(**a)
        a["stabilized_hips"] = a["stabilized_hips"] @ np.array([[0, -1], [-1, 0]]) * 2.7 + [600, 250]
        a["box_heights"] *= 2.7
        s = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["start_frame"], s["terminal"]["start_frame"])
        self.assertEqual(r["terminal"]["tags"], s["terminal"]["tags"])

    def test_absolute_frame_offset(self):
        a = fixture()
        a["frame_ids"] += 753
        for row in a["phase_result"]["frame_labels"]:
            row["frame"] += 753
        a["phase_result"]["approach_start_frame"] += 753
        a["contact"]["contact_frame"] += 753
        a["contact"]["interval_frames"] = [f+753 for f in a["contact"]["interval_frames"]]
        a["events"] = [{"frame": e["frame"]+753} for e in a["events"]]
        r = analyze_terminal_footwork(**a)
        self.assertEqual(r["terminal"]["start_frame"], 789)

    def test_fps_resampling_keeps_time_boundary(self):
        a = fixture(tuple(f*2 for f in (0, 12, 24, 36, 42, 48, 54)), kick=120, fps=60)
        a["stabilized_hips"] /= 2
        self.assertEqual(analyze_terminal_footwork(**a)["terminal"]["start_frame"], 72)

    def test_no_ankle_control_claim_and_json_finite(self):
        r = analyze_terminal_footwork(**fixture())
        self.assertEqual(r["ankle_control"]["status"], "not_measurable")
        self.assertFalse(r["human_verified"])
        json.dumps(r, allow_nan=False)

    def test_no_input_mutation(self):
        a = fixture()
        before = deepcopy(a["phase_result"])
        analyze_terminal_footwork(**a)
        self.assertEqual(a["phase_result"], before)

    def test_duplicate_events_rejected(self):
        with self.assertRaises(ValueError):
            analyze_terminal_footwork(**fixture((0, 0, 12)))

    def test_invalid_frames_rejected(self):
        a = fixture()
        a["frame_ids"] = [0, 2, 2]
        with self.assertRaises(ValueError):
            analyze_terminal_footwork(**a)

    def test_guarded_contact_window_validated(self):
        a = fixture()
        a["contact"]["interval_frames"] = [65, 70]
        with self.assertRaises(ValueError):
            analyze_terminal_footwork(**a)

    def test_config_needs_a_real_baseline(self):
        with self.assertRaises(ValueError):
            analyze_terminal_footwork(**fixture(), config=replace(TerminalConfig(), min_intervals_each_side=1))


if __name__ == "__main__":
    unittest.main()
