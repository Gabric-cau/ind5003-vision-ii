"""Regression tests for close-up intros and clip-absolute frame coordinates."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src import step_pipeline as sp
from src.kicker_analysis import KickerFeatureExtractor


class ShotSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-shot-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.video = Path(self.temp.name) / "clip.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 90))
        self.assertTrue(writer.isOpened())
        for value in [20] * 30 + [200] * 90:
            writer.write(np.full((90, 160, 3), value, np.uint8))
        writer.release()

    def test_scan_includes_later_shots_without_masks(self):
        shots = sp._scan_camera_shots(str(self.video), 8, 30)
        self.assertEqual([(s["shot_start_frame"], s["shot_end_frame"]) for s in shots], [(0, 30), (30, 120)])

    def test_cap_is_relative_to_each_shot_not_clip_start(self):
        shots = sp._scan_camera_shots(str(self.video), 1.5, 30)
        self.assertEqual(shots[1]["shot_end_frame"], 75)
        self.assertEqual(shots[1]["camera_shot_end_frame"], 120)
        self.assertEqual(shots[1]["shot_end_reason"], "duration_cap")

    def test_same_colour_spatial_cut(self):
        # Same green hue/background, but a sudden brightness/layout change.
        before = np.full((90, 160, 3), (10, 70, 10), np.uint8)
        after = np.full((90, 160, 3), (20, 160, 20), np.uint8)
        with patch.object(sp, "_histogram_for_cut", side_effect=[np.array([[1.]], np.float32)] * 120), \
             patch.object(sp.cv2, "compareHist", return_value=4.0):
            writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 90))
            for frame in [before] * 30 + [after] * 90:
                writer.write(frame)
            writer.release()
            shots = sp._scan_camera_shots(str(self.video), 8, 30)
        self.assertEqual(shots[1]["shot_start_frame"], 30)

    def test_inference_excludes_intro_and_preserves_frame_ids(self):
        seen = []
        class Model:
            def predict(self, images, **kwargs):
                seen.extend(float(image.mean()) for image in images)
                return [SimpleNamespace(keypoints=None) for _ in images]
        frames, warnings = sp._run_pose_in_batches(
            str(self.video), 50, Model(), shot_start_frame=30,
            batch_size=8, pose_conf=.25, predict_kwargs={}, roi_inference_mode="full")
        self.assertEqual([f["frame"] for f in frames], list(range(30, 50)))
        self.assertEqual(len(seen), 20)
        self.assertTrue(all(value > 190 for value in seen))
        self.assertEqual(warnings, [])

    @staticmethod
    def fake_result(shot, success):
        return {"status": "ok" if success else "failed", "error": None if success else "NoKickerTrack",
                "selected_track_id": 0 if success else None, "selected_track_qualified": success,
                "track_candidates": [{"track_id": 0, "valid_pose_count": 80, "candidate_score": 7}] if success else [],
                "shot_start_frame": shot["shot_start_frame"], "shot_end_frame": shot["shot_end_frame"],
                "qa_warnings": [], "review_required": False, "auto_accepted": success}

    def test_failed_intro_does_not_stop_processing_and_rescue_requires_review(self):
        calls = []
        def process(*args, shot, **kwargs):
            calls.append((shot["shot_start_frame"], kwargs["use_field_mask"]))
            return self.fake_result(shot, shot["shot_start_frame"] > 0)
        with patch.object(sp, "_process_shot_steps", side_effect=process):
            result = sp.process_video_steps(str(self.video), object())
        self.assertEqual(calls, [(0, False), (30, False)])
        self.assertEqual(result["shot_start_frame"], 30)
        self.assertTrue(result["review_required"])
        self.assertFalse(result["auto_accepted"])

    def test_first_qualified_shot_wins_not_later_replay(self):
        with patch.object(sp, "_process_shot_steps", side_effect=lambda *a, shot, **kw: self.fake_result(shot, True)) as process:
            result = sp.process_video_steps(str(self.video), object())
        self.assertEqual(process.call_count, 1)
        self.assertEqual(result["shot_start_frame"], 0)

    def test_legacy_policy_still_stops_at_intro(self):
        with patch.object(sp, "_process_shot_steps", side_effect=lambda *a, shot, **kw: self.fake_result(shot, False)) as process:
            result = sp.process_video_steps(str(self.video), object(), shot_selection="first")
        self.assertEqual(process.call_count, 1)
        self.assertEqual(result["status"], "failed")

    def test_no_viable_shot_remains_failed(self):
        with patch.object(sp, "_process_shot_steps", side_effect=lambda *a, shot, **kw: self.fake_result(shot, False)):
            result = sp.process_video_steps(str(self.video), object())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["shot_selection"]["found_qualified_shot"])
        self.assertTrue(result["review_required"])

    def test_kick_and_candidate_score_are_invariant_to_intro_offset(self):
        track = {"track_id": 0, "observations": []}
        for frame in range(100):
            hip = np.array([60 + frame * .25, 50.])
            xy = np.ones((17, 2))
            xy[11], xy[12] = hip + [-2, 0], hip + [2, 0]
            swing = 18. if frame == 55 else np.sin(frame * .25)
            xy[15], xy[16] = hip + [swing, 35], hip + [-2, 35]
            track["observations"].append({"frame": frame, "bbox": [20 + frame * .25, 5, 100 + frame * .25, 89],
                                          "keypoints": xy.tolist(), "keypoint_confidences": [1.] * 17})
        shifted = copy.deepcopy(track)
        for item in shifted["observations"]:
            item["frame"] += 200
        kwargs = dict(shot_frame_count=100, frame_height=90, keypoint_conf=.15, extractor=KickerFeatureExtractor())
        original = sp._analyse_tracks({0: track}, **kwargs)
        offset = sp._analyse_tracks({0: shifted}, shot_start_frame=200, **kwargs)
        self.assertEqual(original[0], offset[0])
        self.assertEqual(original[2], offset[2])
        self.assertAlmostEqual(original[1][0]["candidate_score"], offset[1][0]["candidate_score"])
        self.assertEqual(original[1][0]["kick_detection"]["kick_frame"] + 200,
                         offset[1][0]["kick_detection"]["kick_frame"])

    def test_reject_invalid_window(self):
        with self.assertRaises(ValueError):
            sp._run_pose_in_batches(str(self.video), 10, object(), shot_start_frame=10,
                                    batch_size=8, pose_conf=.25, predict_kwargs={}, roi_inference_mode="full")

    def test_conflicting_leg_action_triggers_crop_refinement(self):
        # The clearer stationary official must not suppress a weaker runner's retry.
        summary = {
            0: {"box_height_ratio": .17, "pose_coverage": 1., "valid_pose_count": 100,
                "kick_peak_norm": .08, "lower_body_motion_q90_norm": .03},
            1: {"box_height_ratio": .14, "pose_coverage": .4, "valid_pose_count": 40,
                "kick_peak_norm": .23, "lower_body_motion_q90_norm": .06},
        }
        shot = sp._scan_camera_shots(str(self.video), 8, 30)[1]
        with patch.object(sp, "_run_pose_in_batches", return_value=([], [])) as inference, \
             patch.object(sp, "_analyse_tracks", side_effect=[(0, summary, True), (None, {}, False)]):
            result = sp._process_shot_steps(str(self.video), object(), shot=shot)
        self.assertEqual(inference.call_count, 2)
        self.assertEqual(inference.call_args.kwargs["shot_start_frame"], 30)
        self.assertEqual(inference.call_args.kwargs["roi_inference_mode"], "lower_tiles")
        self.assertIn("competing_track_has_stronger_leg_action", result["roi_fallback_reasons"])

    def test_public_rejects_unknown_selection_policy(self):
        with self.assertRaises(ValueError):
            sp.process_video_steps(str(self.video), object(), shot_selection="largest_person")

    def test_full_body_goalkeeper_closeup_is_not_accepted_as_runup_shot(self):
        def process(*args, shot, **kwargs):
            result = self.fake_result(shot, True)
            result["track_candidates"][0]["box_height_ratio"] = .75 if shot["shot_start_frame"] == 0 else .15
            return result
        with patch.object(sp, "_process_shot_steps", side_effect=process):
            result = sp.process_video_steps(str(self.video), object())
        self.assertEqual(result["shot_start_frame"], 30)
        self.assertEqual(result["shot_selection"]["attempts"][0]["scene_rejections"],
                         ["large_foreground_pose_likely_closeup"])

    def test_subsecond_fragment_cannot_win_even_with_qualified_pose(self):
        shots = sp._scan_camera_shots(str(self.video), 8, 30)
        shots[0]["shot_end_frame"] = 20
        with patch.object(sp, "_scan_camera_shots", return_value=shots), \
             patch.object(sp, "_process_shot_steps", side_effect=lambda *a, shot, **kw: self.fake_result(shot, True)):
            result = sp.process_video_steps(str(self.video), object())
        self.assertEqual(result["shot_start_frame"], 30)

    def test_goal_guard_skips_non_action_scene_without_detector_inference(self):
        def context(video, shot):
            return {"supported": shot["shot_start_frame"] > 0, "samples": []}
        with patch.object(sp, "_shot_goal_context", side_effect=context), \
             patch.object(sp, "_process_shot_steps", side_effect=lambda *a, shot, **kw: self.fake_result(shot, True)) as process:
            result = sp.process_video_steps(str(self.video), object(), require_goal_context=True)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(result["shot_start_frame"], 30)
        self.assertEqual(result["shot_selection"]["attempts"][0]["status"], "skipped")

    def test_all_context_rejections_stay_missing_instead_of_background_kick(self):
        with patch.object(sp, "_shot_goal_context", return_value={"supported": False, "samples": []}), \
             patch.object(sp, "_process_shot_steps") as process:
            result = sp.process_video_steps(str(self.video), object(), require_goal_context=True)
        process.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["kick_frame"])
        self.assertIsNone(result["step_count"])
        self.assertFalse(result["shot_selection"]["found_qualified_shot"])

    def test_plain_nonfield_pixels_do_not_establish_goal_context(self):
        shot = sp._scan_camera_shots(str(self.video), 8, 30)[0]
        result = sp._shot_goal_context(str(self.video), shot)
        self.assertFalse(result["supported"])
        self.assertEqual(len(result["samples"]), 3)

    def test_manual_event_slice_excludes_preceding_shot_but_retains_earlier_context(self):
        with patch.object(sp, "_process_shot_steps", side_effect=lambda *a, shot, **kw: self.fake_result(shot, True)) as process:
            result = sp.process_video_steps(str(self.video), object(), event_slice_interval=(50, 90))
        self.assertEqual(process.call_count, 1)
        self.assertEqual(result["shot_start_frame"], 30)
        self.assertEqual(result["shot_selection"]["attempts"][0]["reason"], "camera_shot_outside_manual_event_slice")

    def test_invalid_manual_event_slice_rejected(self):
        for interval in ((-1, 90), (50, 50), (50, 121), (False, 90)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                sp.process_video_steps(str(self.video), object(), event_slice_interval=interval)

    def test_goal_line_people_removed_from_kicker_ranking_without_dropping_keeper_evidence(self):
        tracks = {1: {"track_id": 1}, 2: {"track_id": 2}}
        shot = sp._scan_camera_shots(str(self.video), 8, 30)[1]
        with patch("src.goal_geometry.select_goal_occupant", return_value={"candidates": [{"track_id": 2}]}):
            candidates, evidence = sp._goal_line_actor_tracks(str(self.video), tracks, shot)
        self.assertEqual(list(candidates), [1])
        self.assertEqual(evidence["excluded_track_ids"], [2])
        self.assertEqual(list(tracks), [1, 2])

    def test_adaptive_scene_scan_keeps_full_timeline_and_relative_caps(self):
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 90))
        for color in [(10, 90, 10)] * 30 + [(10, 10, 240)] * 90:
            writer.write(np.full((90, 160, 3), color, np.uint8))
        writer.release()
        shots = sp._scan_adaptive_shots(str(self.video), 1.5)
        self.assertEqual(shots[0]["shot_start_frame"], 0)
        self.assertEqual(shots[-1]["camera_shot_end_frame"], 120)
        self.assertTrue(all(row["shot_end_frame"] - row["shot_start_frame"] <= 45 for row in shots))
        self.assertTrue(any(abs(row["shot_start_frame"] - 30) <= 1 for row in shots))
        self.assertTrue(all("adaptive" in row["boundary_method"] for row in shots))

    def test_late_keeper_fragment_reranks_with_existing_pose_instead_of_rejecting_whole_shot(self):
        shot = sp._scan_camera_shots(str(self.video), 8, 30)[1]
        tracks = {1: {"track_id": 1}, 9: {"track_id": 9}}
        summaries = {9: {"kick_detection": {"kick_frame": 80}}}
        remaining = {1: {"kick_detection": {"kick_frame": 85}}}
        context = {"samples": [{"frame": 70, "goal": {"status": "candidate_supported"}}]}
        roles = [{"status": "candidate_supported", "seed_track_id": 9},
                 {"status": "candidate_supported", "seed_track_id": 9}]
        with patch("src.goal_geometry.select_goal_occupant", side_effect=roles), \
             patch.object(sp, "_analyse_tracks", return_value=(1, remaining, True)) as rank:
            selected, excluded, evidence = sp._rerank_goal_occupant_selection(
                str(self.video), tracks, tracks, shot, context, (9, summaries, True),
                keypoint_conf=.15, extractor=object(), use_field_mask=True)
        self.assertEqual(selected[0], 1)
        self.assertEqual(excluded, [9])
        self.assertEqual(list(rank.call_args.args[0]), [1])
        self.assertEqual(len(evidence), 2)
        self.assertEqual(list(tracks), [1, 9])


if __name__ == "__main__":
    unittest.main()
