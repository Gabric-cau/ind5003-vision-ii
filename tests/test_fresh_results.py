import pytest

from src.fresh_results import bind_stage, feature_row, summarize, combine_outcomes, compact_stage


def source():
    return {"clip_id": "event", "clip_sha256": "source", "event_id": 1,
            "match_id": "match", "kicker": "Kicker", "team": "team", "fps": 30.}


def test_rejects_stale_pose_binding_even_when_video_bytes_match():
    data = {"clip_id": "event", "human_verified": False,
            "inference_identity": {"source_sha256": "source", "pose_sha256": "old"}}
    with pytest.raises(ValueError, match="pose producer"):
        bind_stage("ball", data, source(), "new", "manifest")


def test_rejects_reference_dependent_pose_prediction():
    data = {"clip_id": "event", "human_verified": False, "new_neural_inference": True,
            "reference_labels_used_for_prediction": True, "inference_identity": {"source_sha256": "source"}}
    with pytest.raises(ValueError, match="source-only"):
        bind_stage("pose", data, source(), None, "manifest")


def test_missing_features_remain_missing_and_do_not_become_zero_measurements():
    row = feature_row(source(), {"pose": {"selected_track_id": 2, "step_count": 0}},
                      {"reference_outcome": "no_goal", "reference_supported": True})
    assert row["steps_pose_baseline"] == 0
    assert row["steps_phase_candidate"] is None
    assert row["ball_trajectory_observations"] is None
    assert row["contact_time_clip_sec_candidate"] is None
    assert row["contact_time_source_sec_candidate"] is None
    assert row["contact_producer_time_consistent"] is None
    assert row["event_final_ball_position_verified"] is False
    summary = summarize([row])
    assert summary["ball_evaluation"]["decided"] == 0
    assert summary["ball_evaluation"]["covered_reference_agreement"] is None


@pytest.mark.parametrize("fps", [50., 30000 / 1001])
def test_final_frame_clock_overrides_stale_producer_seconds_without_rewriting_artifact(fps):
    mapping = {**source(), "fps": fps, "source_start_frame": 4700}
    contact = {"contact_frame": 215, "contact_time_sec": 205 / fps,
               "contact_associated_track_id": 5}
    ball = {"source_mapping": mapping, "independent_contact_search": {"selected_candidate": contact}}
    row = feature_row(mapping, {"ball": ball}, {"reference_outcome": "goal", "reference_supported": True})
    assert row["contact_time_clip_sec_candidate"] == 215 / fps
    assert row["contact_time_source_sec_candidate"] == 4915 / fps
    assert row["contact_producer_time_consistent"] is False
    compact = compact_stage("ball", ball)
    assert compact["contact_timing_audit"]["producer_stored_time_sec"] == 205 / fps
    assert compact["contact_timing_audit"]["candidate_clip_time_sec"] == 215 / fps
    assert compact["contact_timing_audit"]["producer_artifact_modified"] is False
    assert compact["independent_contact_search"]["selected_candidate"] == contact
    assert ball["independent_contact_search"]["selected_candidate"]["contact_time_sec"] == 205 / fps


def test_abstention_is_not_a_correct_negative():
    base = feature_row(source(), {}, {"reference_outcome": "no_goal", "reference_supported": True})
    decided = {**base, "clip_id": "other", "ball_rule_outcome": "no_goal_likely"}
    summary = summarize([base, decided])
    assert summary["ball_evaluation"]["decided"] == 1
    assert summary["ball_evaluation"]["abstained"] == 1
    assert summary["ball_evaluation"]["decision_coverage"] == .5
    assert summary["ball_evaluation"]["reference_agreements"] == 1


def test_unrecorded_human_review_cannot_be_promoted_by_bundle_builder():
    data = {"clip_id": "event", "human_verified": True,
            "inference_identity": {"source_sha256": "source", "pose_sha256": "pose"}}
    with pytest.raises(ValueError, match="separate review record"):
        bind_stage("foot", data, source(), "pose", "manifest")


def test_endpoint_last_observation_cannot_be_promoted_to_verified_final_position():
    data = {"clip_id": "event", "human_verified": False, "source_sha256": "source",
            "pose_sha256": "pose", "final_position": [1, 2], "final_position_verified": False}
    with pytest.raises(ValueError, match="event-final"):
        bind_stage("endpoint", data, source(), "pose", "manifest")


def test_image_height_cannot_be_promoted_to_physical_height():
    data = {"clip_id": "event", "human_verified": False, "physical_height_metres": .7,
            "inference_identity": {"source_clip_sha256": "source", "pose_sha256": "pose"}}
    with pytest.raises(ValueError, match="physical height"):
        bind_stage("height", data, source(), "pose", "manifest")


@pytest.mark.parametrize("signal_candidate,signal_supported,exported_candidate", [
    ("low", False, "unknown"),
    ("unknown", True, "unknown"),
    ("unknown", False, "low"),
])
def test_invalid_height_reference_cannot_enter_bundle_as_supported_category(
        signal_candidate, signal_supported, exported_candidate):
    data = {"clip_id": "event", "human_verified": False,
            "inference_identity": {"source_clip_sha256": "source", "pose_sha256": "pose"},
            "observed_image_plane_height_candidate": exported_candidate,
            "observed_image_plane_signal": {"reference_depth_assumption_failed": True,
                "candidate": signal_candidate, "supported": signal_supported}}
    with pytest.raises(ValueError, match="Invalid height reference"):
        bind_stage("height", data, source(), "pose", "manifest")


def test_invalid_height_reference_diagnostics_remain_available_after_abstention():
    data = {"clip_id": "event", "human_verified": False,
            "inference_identity": {"source_clip_sha256": "source", "pose_sha256": "pose"},
            "observed_image_plane_height_candidate": "unknown",
            "observed_image_plane_signal": {"reference_depth_assumption_failed": True,
                "candidate": "unknown", "supported": False,
                "observed_maximum_torso_clearance_body": .2, "supporting_frames": [1, 2, 3]}}
    bound = bind_stage("height", data, source(), "pose", "manifest")
    assert bound["observed_image_plane_signal"]["observed_maximum_torso_clearance_body"] == .2
    assert bound["observed_image_plane_signal"]["supporting_frames"] == [1, 2, 3]


def test_valid_height_reference_can_retain_conditional_observed_category():
    data = {"clip_id": "event", "human_verified": False,
            "inference_identity": {"source_clip_sha256": "source", "pose_sha256": "pose"},
            "observed_image_plane_height_candidate": "low",
            "observed_image_plane_signal": {"reference_depth_assumption_failed": False,
                "candidate": "low", "supported": True}}
    assert bind_stage("height", data, source(), "pose", "manifest") == data


def test_conflicting_source_outcomes_abstain_without_using_reference():
    assert combine_outcomes("goal_likely", "no_goal_likely") == ("unknown", "source_evidence_conflict")
    assert combine_outcomes("unknown", "goal_likely") == ("goal_likely", "scoreboard_only")


def test_scoreboard_original_recording_must_match_the_context_source():
    data = {"clip_id": "event", "human_verified": False, "source_video_sha256": "other",
            "frame_index_basis": "original_full_video", "outcome_labels_read": False,
            "inference_identity": {"source_sha256": "source"}}
    with pytest.raises(ValueError, match="original source"):
        bind_stage("scoreboard", data, {**source(), "source_video_sha256": "original"}, "pose", "manifest")
