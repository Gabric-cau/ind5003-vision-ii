"""Style-aware count semantics tested on synthetic event/pose evidence."""
from copy import deepcopy
import json
import unittest

from src.terminal_footwork import analyze_terminal_footwork
from src.style_aware_steps import analyze_style_aware_steps


def fixture(events=(0, 10, 20, 30, 40, 50), kick=60):
    frames = list(range(kick+8))
    phase = {"approach_start_frame": 0, "kick_anchor": {"usable": True}, "style_tags": [], "warnings": [],
             "frame_labels": [{"frame": f, "pose_supported": True, "pose_observed": True} for f in frames]}
    contact = {"contact_frame": kick, "interval_frames": [kick-2, kick+3], "status": "supported"}
    event_rows = [{"frame": f} for f in events]
    f = analyze_terminal_footwork(frames, 30., [[2*i, 0] for i in frames], [100.]*len(frames),
        phase_result=phase, contact=contact, events=event_rows, camera_status="supported")
    brief = {"status": f["terminal"]["status"], "tags": f["terminal"]["tags"]}
    f["sensitivity"] = {"event_prominence": {k:deepcopy(brief) for k in ("sensitive", "central", "strict")},
        "change_rules": {k:deepcopy(brief) for k in ("permissive", "central", "conservative")},
        "contact_anchor": {k:deepcopy(brief) for k in ("early", "nominal", "late")}, "status_and_tags_agree": True}
    return {"phase_result": phase, "contact": contact, "footwork_result": f,
        "event_proposal": {"events": event_rows, "variants": {k:deepcopy(event_rows) for k in ("sensitive", "central", "strict")}}}


class StyleAwareStepTests(unittest.TestCase):
    def test_regular_rhythm_gets_continuous_candidate(self):
        s = analyze_style_aware_steps(fixture())
        self.assertEqual(s["attempt_style"], "continuous_approach_candidate")
        self.assertEqual(s["candidate_step_count"], 6)
        self.assertEqual(s["count_partition"]["continuous_approach"], 6)
        self.assertIsNone(s["count_partition"]["terminal_adjustment"])

    def test_short_regular_approach_need_not_force_change_point(self):
        r = fixture((30, 42, 54))
        self.assertEqual(r["footwork_result"]["terminal"]["status"], "insufficient_evidence")
        s = analyze_style_aware_steps(r)
        self.assertEqual(s["attempt_style"], "continuous_approach_candidate")
        self.assertIn("short_approach_regular_rhythm_not_a_proven_negative_change_test", s["cautions"])

    def test_two_events_insufficient_not_zero(self):
        s = analyze_style_aware_steps(fixture((42, 54)))
        self.assertEqual(s["attempt_style"], "undetermined")
        self.assertEqual(s["candidate_step_count"], 2)

    def test_variable_intervals_not_automatically_continuous(self):
        s = analyze_style_aware_steps(fixture((0, 12, 18, 34, 40, 54)))
        self.assertEqual(s["attempt_style"], "undetermined")

    def test_pause_blocks_continuous_class(self):
        r = fixture()
        r["phase_result"]["style_tags"] = [{"tag": "pause_or_stutter_image_plane_candidate"}]
        s = analyze_style_aware_steps(r)
        self.assertEqual(s["attempt_style"], "undetermined")
        self.assertIn("pause_evidence_prevents_continuous_label", s["reasons"])

    def test_supported_and_observed_coverage_are_distinct(self):
        r = fixture()
        for x in r["phase_result"]["frame_labels"][:30]:
            x["pose_observed"] = False
        self.assertEqual(analyze_style_aware_steps(r)["attempt_style"], "undetermined")

    def test_missing_pose_observed_metadata_is_not_perfect_coverage(self):
        r = fixture()
        for x in r["phase_result"]["frame_labels"]:
            x.pop("pose_observed")
        self.assertEqual(analyze_style_aware_steps(r)["attempt_style"], "undetermined")

    def test_terminal_change_splits_not_recounts(self):
        r = fixture((0, 12, 24, 36, 42, 48, 54))
        s = analyze_style_aware_steps(r)
        self.assertEqual(s["attempt_style"], "terminal_adjustment_candidate")
        self.assertEqual(s["count_policy"], "split_at_terminal_boundary")
        self.assertEqual(s["count_partition"]["before_terminal"], 3)
        self.assertEqual(s["count_partition"]["terminal_adjustment"], 4)
        self.assertEqual(s["candidate_step_count"], 7)

    def test_dense_uniform_rhythm_not_forced_into_terminal_class(self):
        s = analyze_style_aware_steps(fixture(tuple(range(0, 55, 3))))
        self.assertEqual(s["attempt_style"], "continuous_approach_candidate")
        self.assertEqual(s["candidate_step_count"], 19)

    def test_curve_is_independent_of_step_style(self):
        r = fixture()
        r["footwork_result"]["route"]["tags"] = ["curved_image_plane"]
        self.assertEqual(analyze_style_aware_steps(r)["attempt_style"], "continuous_approach_candidate")

    def test_disagreeing_change_variant_prevents_continuous_label(self):
        r = fixture()
        r["footwork_result"]["sensitivity"]["event_prominence"]["sensitive"]["status"] = "candidate_change"
        s = analyze_style_aware_steps(r)
        self.assertEqual(s["attempt_style"], "undetermined")
        self.assertEqual(s["candidate_step_count"], 6)

    def test_missing_sensitivity_does_not_pass(self):
        r = fixture()
        del r["footwork_result"]["sensitivity"]["change_rules"]["conservative"]
        self.assertEqual(analyze_style_aware_steps(r)["attempt_style"], "undetermined")

    def test_tail_or_axis_failure_prevents_continuous_label(self):
        for warning in ("terminal_tail_not_observed", "candidate_axis_unreliable_for_terminal_split"):
            r = fixture()
            r["footwork_result"]["warnings"].append(warning)
            self.assertEqual(analyze_style_aware_steps(r)["attempt_style"], "undetermined")

    def test_long_interval_gap_blocks_continuous_even_low_mad(self):
        r = fixture((0, 6, 12, 42, 48, 54))
        r["footwork_result"]["terminal"].update(status="no_distinct_change_detected", start_frame=None)
        s = analyze_style_aware_steps(r)
        self.assertEqual(s["attempt_style"], "undetermined")

    def test_contact_guard_in_total_but_not_continuous_core(self):
        s = analyze_style_aware_steps(fixture((0, 10, 20, 30, 40, 50, 59)))
        self.assertEqual(s["candidate_step_count"], 7)
        self.assertEqual(s["count_partition"]["continuous_approach"], 6)
        self.assertEqual(s["count_partition"]["contact_guard"], 1)

    def test_all_types_conserve_count(self):
        for ev in ((0,10,20,30,40,50), (0,12,24,36,42,48,54), (42,54)):
            s = analyze_style_aware_steps(fixture(ev))
            self.assertEqual(sum(v or 0 for v in s["count_partition"].values()), s["candidate_step_count"])

    def test_unresolved_input_retains_null_count(self):
        r = fixture()
        r["footwork_result"]["counts"].update(nominal_approach_candidates=None, contact_guard_candidates=None, pose_gap_candidates=None)
        r["footwork_result"]["analysis_window"] = None
        s = analyze_style_aware_steps(r)
        self.assertEqual(s["attempt_style"], "undetermined")
        self.assertIsNone(s["candidate_step_count"])
        self.assertTrue(all(x is None for x in s["count_partition"].values()))

    def test_contact_boundary_is_half_open_for_every_variant(self):
        s = analyze_style_aware_steps(fixture((0,10,20,30,40,50,58,59,60,63)))
        self.assertEqual(s["count_sensitivity"]["grid"]["central"], {"early": 6, "nominal": 8, "late": 9})
        self.assertFalse(s["count_sensitivity"]["is_confidence_interval"])

    def test_sensitivity_never_selects_more_favourable_count(self):
        r = fixture()
        r["event_proposal"]["variants"]["strict"] = [{"frame": 10}, {"frame": 30}]
        s = analyze_style_aware_steps(r)
        self.assertEqual(s["candidate_step_count"], 6)
        self.assertEqual(s["count_sensitivity"]["threshold_only_range"], [2,6])

    def test_user_reference_or_name_is_not_a_training_label(self):
        r = fixture()
        expected = analyze_style_aware_steps(r)
        r.update(clip_id="arbitrary_player", user_reference={"approximate_steps":999}, outcome="goal")
        self.assertEqual(analyze_style_aware_steps(r), expected)

    def test_no_claim_of_verified_steps_or_ankle_control(self):
        s = analyze_style_aware_steps(fixture())
        self.assertIsNone(s["verified_step_count"])
        self.assertFalse(s["human_verified"])
        self.assertFalse(s["accuracy_validated"])
        json.dumps(s, allow_nan=False)

    def test_no_mutation(self):
        r = fixture()
        before = deepcopy(r)
        analyze_style_aware_steps(r)
        self.assertEqual(before, r)


if __name__ == "__main__":
    unittest.main()
