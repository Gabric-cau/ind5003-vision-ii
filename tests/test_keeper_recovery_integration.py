from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from scripts.predict_video_keeper import predict_analysis, supported_rule
from src.keeper_contact_recovery import bounded_contact, guard_recovered_prediction, recover_keeper
from tests.test_keeper_supervised import analysis_fixture, keeper
from scripts.recover_keeper_roi_pose import add_new_tracks, fullshot_contact_inputs, plan_goal_roi
from src.goalkeeper_pose_direction import pose_direction_evidence
from tests.test_goalkeeper_pose_direction import fixture as direction_fixture
from src.goalkeeper_pipeline import select_goalkeeper_track


def clock_inputs():
    source = {"clip_sha256": "a" * 64, "fps": 30, "frame_count": 300}
    pose = {"fps": 30, "shot_start_frame": 0, "shot_end_frame": 250}
    contact = {"video_sha256": "a" * 64, "manual_annotations_required": False,
               "automatic_window": {"automatic": True, "start_frame": 0, "end_frame_exclusive": 250},
               "prediction": {"frame": 60, "time_seconds": 2.0}}
    return pose, contact, source


def test_contact_candidate_clock_is_bounded_and_not_truth():
    pose, contact, source = clock_inputs()
    assert bounded_contact(pose, contact, source)[0] == 60
    changed = deepcopy(pose)
    changed["shot_start_frame"] = 80
    assert bounded_contact(changed, contact, source) == (None, "automatic_contact_outside_pose_shot")
    changed["shot_start_frame"] = None
    assert bounded_contact(changed, contact, source) == (None, "no_supported_pose_shot")


@pytest.mark.parametrize("corruption", ("hash", "clock", "manual", "window", "bool_frame"))
def test_bad_contact_provenance_rejected(corruption):
    pose, contact, source = clock_inputs()
    if corruption == "hash":
        contact["video_sha256"] = "b" * 64
    elif corruption == "clock":
        contact["prediction"]["time_seconds"] = 3
    elif corruption == "manual":
        contact["manual_annotations_required"] = True
    elif corruption == "window":
        contact["automatic_window"]["automatic"] = False
    else:
        contact["prediction"]["frame"] = True
    with pytest.raises(ValueError):
        bounded_contact(pose, contact, source)


def test_no_contact_does_not_fabricate_keeper_geometry():
    pose, contact, source = clock_inputs()
    contact["prediction"] = {"frame": None, "time_seconds": None}
    result = recover_keeper(pose, contact, source, "unused.mp4", {})
    assert result["dive_direction"] == result["dive_height"] == "unknown"
    assert result["reason"] == "automatic_contact_unavailable"


def test_recovery_keeps_role_identity_gate():
    pose, contact, source = clock_inputs()
    pose.update(tracks=[], clip_id="x")
    result = recover_keeper(pose, contact, source, "unused.mp4", {})
    assert result["status"] == "abstain"
    assert result["reason"] == "no_supported_goal_line_identity_at_automatic_contact"


def test_unqualified_tracks_cannot_be_rule_outputs():
    assert supported_rule({"status": "ok", "selected_candidate": None}) == "unknown"
    assert supported_rule({"status": "ok", "selected_candidate": {"qualifies": False},
                           "dive_direction": "image_right"}) == "unknown"


def test_existing_good_evidence_does_not_read_contact(tmp_path):
    directory, model = analysis_fixture(tmp_path)
    result = predict_analysis(directory, model, recover_from_contact=True)
    assert result["prediction"]["direction"] == "left"
    assert result["keeper_contact_recovery"] == {"requested": True, "applied": False}


def test_recovery_cannot_override_existing_supported_direction(tmp_path, monkeypatch):
    directory, model = analysis_fixture(tmp_path)
    pose_path = directory / "pose/unseen_penalty.json"
    pose = json.loads(pose_path.read_text())
    pose.update(fps=30, shot_start_frame=0, shot_end_frame=250)
    pose_path.write_text(json.dumps(pose))
    from src.runtime import file_sha256
    ball_path = directory / "ball/results/unseen_penalty.json"
    ball = json.loads(ball_path.read_text())
    ball["inference_identity"]["pose_sha256"] = file_sha256(pose_path)
    ball["keeper"]["selected_candidate"]["pre_coverage"] = 0.1
    ball_path.write_text(json.dumps(ball))
    manifest_path = directory / "source/source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["clips"][0].update(fps=30, frame_count=300)
    manifest_path.write_text(json.dumps(manifest))
    for path in (pose_path, ball_path):
        payload = json.loads(path.read_text())
        payload["inference_identity"]["source_manifest_sha256"] = file_sha256(manifest_path)
        path.write_text(json.dumps(payload))
    ball = json.loads(ball_path.read_text())
    ball["inference_identity"]["pose_sha256"] = file_sha256(pose_path)
    ball_path.write_text(json.dumps(ball))
    _, contact, _ = clock_inputs()
    contact["video_sha256"] = manifest["clips"][0]["clip_sha256"]
    path = directory / "trained_contact/prediction.json"
    path.parent.mkdir()
    path.write_text(json.dumps(contact))
    monkeypatch.setattr("src.keeper_contact_recovery.automatic_goal_geometry", lambda *args: {})
    recovered = keeper()
    recovered["dive_direction"] = "image_right"
    recovered["pose_direction_evidence"]["supported"] = True
    monkeypatch.setattr("src.keeper_contact_recovery.recover_keeper", lambda *args: recovered)
    result = predict_analysis(directory, model, recover_from_contact=True)
    assert result["prediction"]["direction"] == "left"
    assert result["existing_rule_direction"] == "left"
    assert result["keeper_contact_recovery"]["applied"]


def test_goal_roi_does_not_need_preexisting_person_identity():
    pose = {"fps": 30, "shot_start_frame": 0, "shot_end_frame": 200,
            "frame_width": 1280, "frame_height": 720, "tracks": []}
    geometry = {index: {"selected": {"bbox": [400, 200, 700, 300]}} for index in range(45, 60)}
    plan = plan_goal_roi(pose, geometry, 60)
    assert plan["seed_track_id"] is None
    assert plan["roi_xyxy"] == [340, 150, 760, 420]
    assert plan["start_frame"] == 15
    assert plan["end_frame_exclusive"] == 97
    assert plan["rotations"] == [0, 90, 270]


def test_unstable_goal_geometry_cannot_supply_roi():
    pose = {"fps": 30, "shot_start_frame": 0, "shot_end_frame": 200,
            "frame_width": 1280, "frame_height": 720}
    assert plan_goal_roi(pose, {}, 60) is None
    geometry = {index: {"selected": {"bbox": [index*100, 200, index*100+30, 220]}}
                for index in range(45, 60)}
    assert plan_goal_roi(pose, geometry, 60) is None


def test_roi_new_track_ids_do_not_collide_with_source_ids():
    pose = {"tracks": [{"track_id": 9, "observations": []}], "selected_track_id": 9}
    detection = {"frame": 10, "bbox": [20, 20, 40, 70], "box_confidence": .9,
                 "keypoints": [[30, 40]]*17, "keypoint_confidences": [.9]*17}
    original = deepcopy(pose)
    merged, additions = add_new_tracks(pose, {10: [detection]})
    assert pose == original
    assert merged["selected_track_id"] == 9
    assert additions[0]["track_id"] >= 10
    assert additions[0]["inference_track_scope"] == "automatic_goal_roi_pose_recovery"


def test_standing_window_cannot_yield_a_learned_dive_height():
    prediction = {"status": "candidate", "direction": "center", "height": "middle", "reasons": []}
    before = deepcopy(prediction)
    result = guard_recovered_prediction({"pose_direction_evidence": {
        "supported": False, "reasons": ["no_sustained_lateral_body_transition"]}}, prediction)
    assert prediction == before
    assert result["status"] == "abstain"
    assert result["direction"] == result["height"] == "unknown"
    assert "no_observed_sustained_dive_phase" in result["reasons"]


def test_observed_dive_phase_preserves_head_candidate():
    prediction = {"status": "candidate", "direction": "left", "height": "low", "reasons": []}
    result = guard_recovered_prediction({"pose_direction_evidence": {"supported": True},
        "dive_observability": {"window_supported": True},
        "goal_identity_evidence": {"seed_track_id": 7},
        "selected_candidate": {"source_track_ids": [7]}}, prediction)
    assert result["status"] == "candidate"
    assert result["height"] == "low"


@pytest.mark.parametrize("failure", ["censored_window", "source_switch"])
def test_early_body_direction_cannot_bypass_height_window_or_identity_guard(failure):
    prediction = {"status": "candidate", "direction": "left", "height": "middle", "reasons": []}
    keeper = {"pose_direction_evidence": {"supported": True},
        "dive_observability": {"window_supported": failure != "censored_window"},
        "goal_identity_evidence": {"seed_track_id": 7},
        "selected_candidate": {"source_track_ids": [7, 8] if failure == "source_switch" else [7]}}
    result = guard_recovered_prediction(keeper, prediction)
    assert result["status"] == "candidate" and result["direction"] == "left"
    assert result["height"] == result["dive_height"] == "unknown"
    assert prediction["height"] == "middle"
    reason = ("incomplete_recovered_height_window" if failure == "censored_window" else
              "unverified_identity_switch_in_height_features")
    assert reason in result["reasons"]


def test_lateral_preparation_steps_do_not_establish_a_dive():
    prediction = {"status": "candidate", "direction": "left", "height": "middle", "reasons": []}
    result = guard_recovered_prediction({"dive_direction_evidence": {"supported": True},
        "pose_direction_evidence": {"supported": False}}, prediction)
    assert result["lateral_motion_supported"] is True
    assert result["actual_dive_evidence_supported"] is False
    assert result["height"] == "unknown"


def fullshot_fixture(folder):
    pose, contact, source = clock_inputs()
    pose["shot_end_frame"] = 100
    pose["tracks"] = [{"track_id": 4, "observations": [{"frame": 70}]}]
    contact["prediction"] = {"frame": 150, "time_seconds": 5.0}
    contact["model_bindings"] = {"mode": "matched_training_protocol"}
    protocol = {"selection": "first_automatic_camera_shot", "analysis_end_frame_exclusive": 250,
                "contact_time_or_labels_used_to_select_window": False}
    artifacts = {"prediction": contact, "features": {
        "video_sha256": source["clip_sha256"], "fps": 30,
        "window": deepcopy(contact["automatic_window"]), "diagnostics": {"fullshot_protocol": protocol}},
        "evidence": {"video_sha256": source["clip_sha256"], "fullshot_protocol": protocol}}
    for name, item in artifacts.items():
        (folder/(name+".json")).write_text(json.dumps(item))
    return pose, source, artifacts


def test_fullshot_override_extends_analysis_bounds_without_fabricating_old_observations(tmp_path):
    pose, source, _ = fullshot_fixture(tmp_path)
    before = deepcopy(pose)
    contact, updated, adjustment, bindings = fullshot_contact_inputs(tmp_path, pose, source)
    assert pose == before
    assert updated["tracks"] == before["tracks"]
    assert updated["shot_end_frame"] == 250
    assert bounded_contact(updated, contact, source)[0] == 150
    assert adjustment["original_pose_shot_end_frame"] == 100
    assert adjustment["original_pose_hash_reinterpreted_as_new_inference"] is False
    assert set(bindings) == {"fullshot_prediction_sha256", "fullshot_features_sha256", "fullshot_evidence_sha256"}


@pytest.mark.parametrize('corruption', ['source', 'bounds', 'manual_window', 'transfer', 'clock', 'window_mismatch'])
def test_fullshot_override_rejects_unbound_or_incompatible_artifacts(tmp_path, corruption):
    pose, source, artifacts = fullshot_fixture(tmp_path)
    if corruption == 'source':
        artifacts['evidence']['video_sha256'] = 'other_video'
    elif corruption == 'bounds':
        artifacts['prediction']['automatic_window']['end_frame_exclusive'] = source['frame_count']+1
        artifacts['features']['window'] = artifacts['prediction']['automatic_window']
    elif corruption == 'manual_window':
        artifacts['features']['diagnostics']['fullshot_protocol']['contact_time_or_labels_used_to_select_window'] = True
    elif corruption == 'transfer':
        artifacts['prediction']['model_bindings']['mode'] = 'experimental_frozen_head_window_transfer'
    elif corruption == 'clock':
        artifacts['features']['fps'] = 50
    else:
        artifacts['features']['window']['end_frame_exclusive'] -= 1
    for name, item in artifacts.items():
        (tmp_path/(name+'.json')).write_text(json.dumps(item))
    with pytest.raises(ValueError):
        fullshot_contact_inputs(tmp_path, pose, source)


def test_exact_source_frame_binding_disambiguates_duplicate_detector_views():
    record, pose = direction_fixture()
    record['selected_candidate']['source_track_ids'] = [7, 8]
    pose['tracks'].append({'track_id': 8, 'observations': deepcopy(pose['tracks'][0]['observations'])})
    assert pose_direction_evidence(record, pose)['reasons'] == ['overlapping_selected_source_tracks']
    record['selected_candidate']['source_track_id_by_frame'] = {str(f): 7 for f in range(55, 73)}
    result = pose_direction_evidence(record, pose)
    assert result['supported']
    assert result['source_track_id'] == 7
    assert result['supporting_frames'] == [69, 70, 71, 72]


def test_exact_binding_does_not_clear_actual_identity_ambiguity_or_source_switch():
    record, pose = direction_fixture()
    record['selected_candidate']['source_track_ids'] = [7, 8]
    pose['tracks'].append({'track_id': 8, 'observations': deepcopy(pose['tracks'][0]['observations'])})
    record['selected_candidate']['source_track_id_by_frame'] = {str(f): 7 for f in range(55, 73)}
    record['goal_identity_evidence']['ambiguous_identity'] = True
    assert not pose_direction_evidence(record, pose)['supported']
    record['goal_identity_evidence']['ambiguous_identity'] = False
    record['selected_candidate']['source_track_id_by_frame'].update({str(f): 8 for f in range(69, 73)})
    result = pose_direction_evidence(record, pose)
    assert not result['supported']
    assert result['prefix_stop_reason'] == 'source_track_switch'


def test_selected_source_frame_binding_cannot_reference_unselected_person():
    record, pose = direction_fixture()
    record['selected_candidate']['source_track_id_by_frame'] = {'60': 99}
    assert pose_direction_evidence(record, pose)['reasons'] == ['invalid_selected_source_frame_binding']


def test_stitcher_keeps_the_actual_source_id_at_each_selected_frame():
    def observation(frame):
        return {'frame': frame, 'bbox': [90, 80, 110, 120], 'box_confidence': .9}
    pose = {'fps': 30, 'frame_width': 640, 'frame_height': 360, 'selected_track_id': None,
            'video_path': 'unused', 'tracks': [
                {'track_id': 7, 'observations': [observation(f) for f in range(0, 34)]},
                {'track_id': 8, 'observations': [observation(f) for f in range(34, 61)]}]}
    with patch('cv2.VideoCapture') as capture:
        capture.return_value.read.return_value = (False, None)
        selected, _ = select_goalkeeper_track(pose, 30, seed_track_id=7, tracking_policy='bounded')
    lineage = selected['source_track_id_by_frame']
    assert lineage['30'] == 7 and lineage['34'] == 8
    frames = {str(round(row[0]*30)) for row in selected['pre_trajectory']+selected['post_trajectory']}
    assert set(lineage) == frames
