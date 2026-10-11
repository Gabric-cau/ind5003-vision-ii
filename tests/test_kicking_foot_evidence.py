"""Failure-mode tests for temporal hypotheses, not video accuracy tests."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.kicking_foot_evidence import associate_leg_trajectories, kicking_foot_evidence
from src.kicking_foot_redetection import COCO_MIRROR
from scripts.run_fresh_foot_experiment import (pose_scene_rejections, prepare_tasks,
                                             restore_crop_detections, selected_ball_observations,
                                             summarize_plan, timing_and_actor, validate_pose_model)


def observations(side="right", frames=range(15)):
    output = []
    for frame in frames:
        points = np.zeros((17, 2))
        points[11], points[12] = [40, 40], [60, 40]
        points[13], points[14] = [43, 65], [57, 65]
        points[15], points[16] = [45, 90], [55, 90]
        points[15 if side == "left" else 16, 0] += 2 * frame
        output.append({"frame": frame, "keypoints": points.tolist(),
                       "confidence": [0.9] * 17, "bbox": [0, 0, 100, 100]})
    return output


def test_sustained_motion_survives_one_large_opposite_ankle_outlier():
    normal = observations()
    normal[6]["keypoints"][15][0] += 70
    result = kicking_foot_evidence(normal, observations(), [4, 7], 30)
    assert result["candidate"] == "right"
    assert result["shoe_ball_contact_verified"] is False


def test_camera_translation_does_not_choose_a_foot():
    rows = observations()
    for row in rows:
        shift = np.array([10 * row["frame"], 0])
        row["keypoints"] = (np.asarray(row["keypoints"]) + shift).tolist()
    result = kicking_foot_evidence(rows, rows, [4, 7], 30)
    assert result["candidate"] == "right"


def test_consistent_left_is_not_a_right_foot_prior():
    rows = observations("left")
    assert kicking_foot_evidence(rows, rows, [4, 7], 30)["candidate"] == "left"


def test_mirror_anatomical_disagreement_abstains():
    result = kicking_foot_evidence(observations("left"), observations("right"), [4, 7], 30)
    assert result["candidate"] == "unknown"
    assert result["status"] == "conflicting_temporal_windows_or_views"


def test_sparse_samples_do_not_silently_bridge_a_gap():
    rows = observations(frames=[0, 4, 8, 12, 16])
    assert kicking_foot_evidence(rows, rows, [4, 7], 30)["candidate"] == "unknown"


def test_low_confidence_motion_is_missing_evidence():
    rows = observations()
    for row in rows:
        row["confidence"][16] = 0.1
    assert kicking_foot_evidence(rows, rows, [4, 7], 30)["candidate"] == "unknown"


def test_equal_motion_abstains():
    rows = observations()
    for row in rows:
        row["keypoints"][15][0] += 2 * row["frame"]
    assert kicking_foot_evidence(rows, rows, [4, 7], 30)["candidate"] == "unknown"


def test_both_views_repeating_a_temporal_channel_swap_do_not_validate_anatomy():
    rows = observations()
    for row in rows[7:]:
        row["keypoints"][15], row["keypoints"][16] = row["keypoints"][16], row["keypoints"][15]
    result = kicking_foot_evidence(rows, rows, [4, 7], 30)
    assert result["candidate"] == "unknown"
    assert result["status"] == "temporal_anatomical_channel_ambiguity"
    assert result["temporal_channel_checks"]["normal"]["channel_permutation_applied"] is False


def test_trajectory_matching_can_reconnect_a_channel_swap_without_claiming_anatomy():
    rows = observations(frames=range(12))
    for row in rows:
        row["keypoints"][15] = [30, 90]
        row["keypoints"][16] = [70, 90]
        row["keypoints"][13] = [35, 65]
        row["keypoints"][14] = [65, 65]
    for row in rows[6:]:
        for left, right in ((13, 14), (15, 16)):
            row["keypoints"][left], row["keypoints"][right] = row["keypoints"][right], row["keypoints"][left]
    result = associate_leg_trajectories(rows, 30)
    assert result["source_channel_switch_frames"] == [6]
    assert result["anatomical_side_A"] == "unknown"
    assert np.allclose(result["observations"][0]["trajectory_A_relative_knee_ankle"],
                       result["observations"][-1]["trajectory_A_relative_knee_ankle"])


def test_trajectory_labels_reset_after_a_long_gap():
    result = associate_leg_trajectories(observations(frames=[0, 1, 2, 10, 11]), 30)
    assert result["segments"] == 2
    assert result["observations"][3]["segment_start"] is True


def test_paired_local_ball_distance_can_support_an_approaching_foot():
    rows = observations(frames=range(15))
    for row in rows:
        row["keypoints"][15] = [20, 90]
        row["keypoints"][16] = [80 - 3 * row["frame"], 90]
    balls = [{"frame": frame, "xy": [50, 90]} for frame in range(10)]
    result = kicking_foot_evidence(rows, rows, [7, 9], 30, ball_observations=balls)
    assert result["candidate"] == "right"
    assert result["status"] == "distance_supported_contact_window_candidate"
    assert result["shoe_ball_contact_verified"] is False


def test_already_planted_near_ankle_is_not_distance_supported_striking_foot():
    rows = observations(frames=range(15))
    for row in rows:
        row["keypoints"][15] = [51, 90]
        row["keypoints"][16] = [95 - row["frame"], 90]
    balls = [{"frame": frame, "xy": [50, 90]} for frame in range(10)]
    result = kicking_foot_evidence(rows, rows, [7, 9], 30, ball_observations=balls)
    assert result["ball_proximity"]["normal"]["candidate"] == "unknown"
    assert result["ball_proximity"]["normal"]["approach_distance_reduction_body_heights"] == 0


def test_earlier_channel_flip_is_recorded_without_invalidating_stable_contact_window():
    rows = observations(frames=range(16))
    for row in rows:
        row["keypoints"][15] = [20, 90]
        row["keypoints"][16] = [80 - 3 * row["frame"], 90]
    rows[3]["keypoints"][15], rows[3]["keypoints"][16] = rows[3]["keypoints"][16], rows[3]["keypoints"][15]
    balls = [{"frame": frame, "xy": [50, 90]} for frame in range(10)]
    result = kicking_foot_evidence(rows, rows, [7, 9], 30, ball_observations=balls)
    assert result["candidate"] == "right"
    assert result["status"] == "distance_supported_contact_window_candidate"
    assert result["temporal_channel_checks"]["normal"]["ambiguous_pairs"]
    assert result["contact_window_channel_checks"]["normal"]["ambiguous_pairs"] == []


def test_local_anatomical_ambiguity_retains_ball_distance_diagnostics():
    rows = observations()
    for row in rows[7:]:
        row["keypoints"][15], row["keypoints"][16] = row["keypoints"][16], row["keypoints"][15]
    result = kicking_foot_evidence(rows, rows, [4, 7], 30,
                                  ball_observations=[{"frame": f, "xy": [50, 90]} for f in range(8)])
    assert result["candidate"] == "unknown"
    assert result["status"] == "temporal_anatomical_channel_ambiguity"
    assert "ball_proximity" in result


def test_ambiguous_ball_identity_is_not_collapsed_to_first_detection():
    with pytest.raises(ValueError, match="one identity"):
        kicking_foot_evidence(observations(), observations(), [4, 7], 30,
            ball_observations=[{"frame": 5, "xy": [50, 90]}, {"frame": 5, "xy": [70, 90]}])


def test_duplicate_pose_frame_is_rejected():
    rows = observations()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="one observation"):
        kicking_foot_evidence(rows, None, [4, 7], 30)


@pytest.mark.parametrize("fps,bracket", [(0, [4, 7]), (30, [7, 4]), (30, [4.2, 7])])
def test_invalid_time_basis_is_rejected(fps, bracket):
    with pytest.raises(ValueError):
        kicking_foot_evidence(observations(), None, bracket, fps)


def test_pose_timing_proposal_does_not_become_ball_departure_evidence():
    pose = {"shot_start_frame": 100, "shot_end_frame": 200, "kick_frame": 150, "selected_track_id": 3}
    result = timing_and_actor(pose, None, True)
    assert result["contact_bracket_frames"] == [148, 152]
    assert result["ball_departure_evidence_available"] is False
    assert result["timing_source"] == "pose_motion_proposal"


def test_contact_bracket_from_another_shot_is_rejected():
    pose = {"shot_start_frame": 100, "shot_end_frame": 200, "selected_track_id": 3}
    with pytest.raises(ValueError, match="inside the selected"):
        timing_and_actor(pose, {"contact_bracket_frames": [40, 45]}, False)


def test_explicitly_unresolved_bracket_is_not_reconstructed_from_other_timing_fields():
    pose = {"shot_start_frame": 100, "shot_end_frame": 200, "selected_track_id": 3}
    timing = timing_and_actor(pose, {"contact": {"contact_bracket_frames": None,
        "contact_frame": 150, "ball_departure_frames": [152, 154]}}, False)
    assert timing["contact_bracket_frames"] is None
    assert timing["ball_departure_evidence_available"] is False


def test_reflected_roi_coordinates_and_channels_are_undone_once(monkeypatch):
    import scripts.run_fresh_foot_experiment as runner
    original = np.column_stack((np.arange(17) + 10, np.arange(17) + 20)).astype(float)
    mirrored = original[COCO_MIRROR].copy()
    mirrored[:, 0] = 99 - mirrored[:, 0]
    monkeypatch.setattr(runner, "_detections", lambda prediction: [{
        "keypoints": mirrored.tolist(), "keypoint_confidences": [.9] * 17,
        "bbox": [69, 15, 94, 80], "box_confidence": .9}])
    restored = restore_crop_detections(None, [200, 300, 300, 400], True, 123)[0]
    assert np.allclose(restored["keypoints"], original + [200, 300])
    assert restored["bbox"] == [205, 315, 230, 380]


def test_fallback_qualified_track_cannot_override_rejected_camera_scene(tmp_path):
    pose = {"selected_track_qualified": True, "shot_selection": {
        "found_qualified_shot": False, "selected_shot_index": 2,
        "attempts": [{"shot_index": 2, "qualifies": False,
                      "scene_rejections": ["selected_kicker_is_goal_line_occupant"]}]}}
    images, tasks, rejected = prepare_tasks(tmp_path / "absent.mp4", pose, {})
    assert images == tasks == []
    assert "upstream_did_not_find_qualified_shot" in [row["reason"] for row in rejected]


def test_qualified_scene_requires_explicit_empty_scene_rejections():
    pose = {"selected_track_qualified": True, "shot_selection": {
        "found_qualified_shot": True, "selected_shot_index": 2,
        "attempts": [{"shot_index": 2, "qualifies": True}]}}
    assert pose_scene_rejections(pose) == ["selected_shot_has_scene_rejections_or_missing_scene_check"]
    pose["shot_selection"]["attempts"][0]["scene_rejections"] = []
    assert pose_scene_rejections(pose) == []


def test_different_contact_actor_does_not_inherit_selected_kicker_qualification(tmp_path):
    pose = {"selected_track_id": 2, "selected_track_qualified": True, "shot_selection": {
        "found_qualified_shot": True, "selected_shot_index": 0,
        "attempts": [{"shot_index": 0, "qualifies": True, "scene_rejections": []}]}}
    images, tasks, rejected = prepare_tasks(tmp_path / "absent.mp4", pose, {"actor_track_id_candidate": 8})
    assert images == tasks == []
    assert rejected == [{"reason": "reselected_contact_actor_has_insufficient_field_role_evidence"}]


def test_independent_contact_actor_keeps_reselection_and_role_evidence():
    pose = {"shot_start_frame": 100, "shot_end_frame": 200, "selected_track_id": 3}
    timing = timing_and_actor(pose, {"contact": {
        "contact_bracket_frames": [140, 144], "contact_associated_track_id": 8,
        "actor_reselected_by_ball_evidence": True, "actor_field_support": .8,
        "actor_goal_line_excluded": False, "seed_ambiguous_frames": [141]}}, False)
    assert timing["actor_track_id_candidate"] == 8
    assert timing["original_pose_selected_track_id"] == 3
    assert timing["actor_reselected_by_ball_evidence"] is True
    assert timing["seed_ambiguous_frames"] == [141]


def test_stabilized_stationary_center_is_not_fabricated_into_source_pixel_observations():
    contact = {"contact": {"stationary_ball_reference_xy": [123, 456],
                           "contact_frame": 10, "ball_departure_frames": [12, 13]}}
    assert selected_ball_observations(contact) is None


def test_selected_contact_observations_preserve_actual_detection_coordinates():
    samples = [{"frame": 9, "xy": [123, 456], "detection_id": "9:2"}]
    assert selected_ball_observations({"contact": {"stationary_ball_observations": samples}}) == samples


def test_departure_points_are_from_the_same_upstream_contact_payload():
    stationary = [{"frame": 9, "xy": [123, 456], "detection_id": "9:2"}]
    departure = [{"frame": 12, "xy": [145, 456], "detection_id": "12:1"}]
    assert selected_ball_observations({"contact": {
        "stationary_ball_observations": stationary,
        "departure_ball_observations": departure}}) == stationary + departure


@pytest.mark.parametrize("task", ["detect", "segment", "classify", None])
def test_detection_models_fail_explicitly_instead_of_producing_empty_pose_evidence(task):
    with pytest.raises(ValueError, match="requires task=pose"):
        validate_pose_model(SimpleNamespace(task=task))


def test_non_coco_pose_model_is_not_misread_as_left_right_ankles():
    with pytest.raises(ValueError, match="COCO-17"):
        validate_pose_model(SimpleNamespace(task="pose", model=SimpleNamespace(yaml={"kpt_shape": [21, 3]})))
    result = validate_pose_model(SimpleNamespace(task="pose", model=SimpleNamespace(yaml={"kpt_shape": [17, 3]})))
    assert result["model_loaded_and_checked"] is True


def test_plan_exposes_source_pixel_and_ball_contact_evidence_shortfalls():
    pose = {"fps": 30, "frame_width": 1280, "frame_height": 720}
    timing = {"contact_bracket_frames": [150, 154], "timing_source": "independent_ball_contact_candidate",
              "actor_track_id_candidate": 8, "seed_ambiguous_frames": [152]}
    tasks = [{"frame": f, "reflected": mirrored} for f in (149, 150, 151) for mirrored in (False, True)]
    plan = summarize_plan(pose, timing, tasks, [], [{"frame": 150, "xy": [300, 400]}])
    assert plan["planned_normal_pose_crops"] == plan["planned_mirrored_pose_crops"] == 3
    assert plan["planned_contact_ball_frame_matches"] == [150]
    assert plan["source_pixel_dimensions"] == [1280, 720]
    assert "fewer_than_three_planned_contact_frame_matched_ball_samples" in plan["warnings"]
    assert "local_ball_identity_ambiguity_will_keep_candidate_unknown" in plan["warnings"]
    assert plan["new_neural_inference"] is False
