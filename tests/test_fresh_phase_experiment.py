import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import scripts.run_fresh_phase_experiment as runner
from scripts.run_fresh_phase_experiment import (_analyse_pose, _bound_contact, _prefix_motion,
                                               _unknown_contact)


def _pose():
    observations = []
    for frame in range(120):
        x = 100. + max(0, frame - 20) * 1.2
        ankle_phase = 9 * np.sin(frame * np.pi / 6)
        points = np.zeros((17, 2))
        points[11:13] = [[x - 3, 100], [x + 3, 100]]
        points[15] = [x + ankle_phase, 150]
        points[16] = [x - ankle_phase, 150]
        observations.append({"frame": frame, "bbox": [x - 20, 60, x + 20, 160],
                             "keypoints": points.tolist(), "keypoint_confidences": [1.] * 17})
    return {"clip_id": "test_clip", "fps": 30., "selected_track_id": 2,
            "shot_start_frame": 0, "shot_end_frame": 120, "kick_frame": 45,
            "step_count": 999, "tracks": [{"track_id": 2, "observations": observations}]}


def _motion(*, missing=None):
    transforms = {str(frame): np.eye(3).tolist() for frame in range(120)}
    if missing is not None:
        transforms[str(missing)] = None
    return _prefix_motion({"status": "supported", "reference_chain_complete": missing is None,
                           "transforms": transforms}, 0, 120)


def _contact():
    return {"status": "tentative", "contact_frame": 80, "contact_time_sec": 80 / 30,
            "interval_frames": [78, 84], "confidence": .8, "fps": 30.,
            "contact_associated_track_id": 2, "warnings": [], "human_verified": False}


def test_pose_motion_peak_never_replaces_unknown_independent_contact():
    pose = _pose()
    result, _ = _analyse_pose(pose, _unknown_contact(30., "no_ball_contact"), _motion())
    assert pose["kick_frame"] == 45
    assert pose["step_count"] == 999
    assert result["phase_result"]["kick_anchor"]["frame"] is None
    assert result["candidate_step_count"] is None
    assert result["verified_step_count"] is None
    assert result["step_analysis"]["attempt_style"] == "undetermined"


def test_independent_contact_reanchors_count_without_using_pose_baseline():
    result, _ = _analyse_pose(_pose(), _contact(), _motion())
    assert result["phase_result"]["kick_anchor"]["frame"] == 80
    assert result["phase_result"]["kick_anchor"]["usable"]
    assert result["candidate_step_count"] is not None
    assert 0 < result["candidate_step_count"] < 30
    assert result["verified_step_count"] is None
    assert result["event_proposal"]["counts_are_contacts"] is False
    assert result["step_analysis"]["counts_are_verified_contacts"] is False
    assert set(result["count_sensitivity"]) == {"sensitive", "central", "strict"}
    json.dumps(result, allow_nan=False)


def test_motion_gap_nulls_all_later_reference_matrices_even_if_they_exist():
    motion = _motion(missing=30)
    assert motion["first_invalid_reference_frame"] == 30
    assert motion["valid_reference_frame_count"] == 30
    assert motion["transforms"]["29"] is not None
    assert all(motion["transforms"][str(frame)] is None for frame in range(30, 120))
    result, _ = _analyse_pose(_pose(), _contact(), motion)
    assert result["candidate_step_count"] is None
    assert result["phase_result"]["camera_motion_status"] == "partial"
    assert result["phase_result"]["kick_anchor"]["usable"] is False
    assert all(row["phase"] == "unknown" and not row["pose_supported"]
               for row in result["phase_result"]["frame_labels"] if row["frame"] >= 30)
    json.dumps(result, allow_nan=False)


def test_later_gap_preserves_valid_precontact_evidence_without_bridging_recovery():
    result, _ = _analyse_pose(_pose(), _contact(), _motion(missing=100))
    assert result["phase_result"]["kick_anchor"]["usable"]
    assert result["candidate_step_count"] is not None
    assert all(row["phase"] == "unknown" for row in result["phase_result"]["frame_labels"]
               if row["frame"] >= 100)
    assert result["footwork_result"]["route"]["status"] == "unavailable"


@pytest.mark.parametrize("raw", [None, [[1, 2], [3, 4]], np.full((3, 3), np.nan).tolist(), np.zeros((3, 3)).tolist()])
def test_invalid_reference_transform_never_recovers_downstream(raw):
    motion = {"transforms": {"0": np.eye(3).tolist(), "1": raw, "2": np.eye(3).tolist()},
              "reference_chain_complete": True}
    result = _prefix_motion(motion, 0, 3)
    assert result["reference_chain_complete"] is False
    assert result["transforms"]["1"] is None
    assert result["transforms"]["2"] is None


def _ball_fixture(tmp_path, *, actor=2, field_support=.8, ambiguous=False, excluded=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pose = _pose()
    if actor != 2:
        pose["tracks"].append({"track_id": actor, "observations": pose["tracks"][0]["observations"]})
    pose["track_candidates"] = [{"track_id": actor, "field_support": field_support}]
    pose["goal_line_role_exclusions"] = {"excluded_track_ids": [actor] if excluded else []}
    pose_path = tmp_path / "pose.json"
    pose_path.write_text(json.dumps(pose))
    source = {"clip_id": pose["clip_id"], "clip_sha256": "1" * 64}
    manifest_hash = "2" * 64
    identity = {"source_sha256": source["clip_sha256"],
                "pose_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
                "source_manifest_sha256": manifest_hash, "model_sha256": "3" * 64}
    stationary = [{"frame": frame, "xy": [10., 20.], "detection_id": frame} for frame in (77, 78, 79)]
    moving = [{"frame": frame, "xy": [float(frame - 60), 20.], "detection_id": frame} for frame in (81, 82)]
    candidate = {"contact_frame": 80, "score": .8, "interval_frames": [79, 81], "contact_bracket_frames": [79, 81],
                 "ball_departure_frames": [81, 82], "contact_associated_track_id": actor,
                 "camera_chain_measured": True, "warnings": [], "ankle_approach_supported": True,
                 "actor_goal_line_excluded": False, "contact_timing_status": "bounded_stationary_departure_candidate",
                 "goalward_departure_support": {"status": "supported_goalward_departure_candidate"},
                 "stationary_ball_observations": stationary, "departure_ball_observations": moving}
    search = {"status": "candidate_supported", "selected_candidate": candidate,
              "old_contact_or_outcome_labels_used": False, "full_selected_shot_search": [0, 120],
              "eligible_actor_ids": [actor], "goal_line_excluded_actor_ids": [actor] if excluded else [],
              "actor_identity_ambiguous": ambiguous}
    event = {"inference_identity": identity, "pose_sha256": identity["pose_sha256"],
             "kick_frame": 80, "contact": candidate}
    raw_cache = {"clip_id": "test_clip", "source_sha256": source["clip_sha256"],
                 "fresh_model_inference": True, "inference_identity": identity,
                 "detections": {str(row["frame"]): [{"x": row["xy"][0], "y": row["xy"][1],
                                 "detection_id": row["detection_id"]}] for row in stationary + moving}}
    ball_dir = tmp_path / "ball"
    for directory, value in (("contact_search", search), ("event_windows", event), ("detections", raw_cache)):
        folder = ball_dir / directory
        folder.mkdir(parents=True)
        (folder / "test_clip.json").write_text(json.dumps(value))
    result = {"clip_id": "test_clip", "inference_identity": identity, "new_neural_inference": True,
              "status": "candidate_event", "reference_labels_used_for_prediction": False}
    for directory, field in (("contact_search", "contact_search_sha256"), ("event_windows", "event_cache_sha256"),
                             ("detections", "source_cache_sha256")):
        result[field] = hashlib.sha256((ball_dir / directory / "test_clip.json").read_bytes()).hexdigest()
    (ball_dir / "results").mkdir()
    (ball_dir / "results/test_clip.json").write_text(json.dumps(result))
    return ball_dir, source, pose_path, pose, manifest_hash


def test_fresh_ball_contact_is_hash_bound_and_kept_tentative(tmp_path):
    args = _ball_fixture(tmp_path)
    contact, provenance = _bound_contact(*args)
    assert contact["status"] == "tentative"
    assert contact["contact_frame"] == 80
    assert contact["contact_frame"] != args[3]["kick_frame"]
    assert contact["human_verified"] is False
    assert contact["interval_frames"] == [79, 81]
    assert contact["contact_bracket_frames"] == [79, 81]
    assert contact["exact_contact_frame_established"] is False
    assert provenance["status"] == "source_bound_independent_contact_candidate"


def test_missing_ball_result_does_not_supply_a_pose_contact(tmp_path):
    pose = _pose()
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(pose))
    contact, _ = _bound_contact(tmp_path / "missing", {"clip_id": "test_clip"}, path, pose, "manifest")
    assert contact["status"] == "unknown"
    assert contact["contact_frame"] is None


def test_modified_pose_or_ball_cache_fails_source_binding(tmp_path):
    args = _ball_fixture(tmp_path)
    args[2].write_text(args[2].read_text() + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        _bound_contact(*args)
    args = _ball_fixture(tmp_path / "another")
    (args[0] / "event_windows/test_clip.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _bound_contact(*args)


def test_actor_mismatch_cannot_enter_phase_analysis(tmp_path):
    args = _ball_fixture(tmp_path)
    search_path = args[0] / "contact_search/test_clip.json"
    search = json.loads(search_path.read_text())
    search["selected_candidate"]["contact_associated_track_id"] = 99
    search_path.write_text(json.dumps(search))
    result_path = args[0] / "results/test_clip.json"
    result = json.loads(result_path.read_text())
    result["contact_search_sha256"] = hashlib.sha256(search_path.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result))
    contact, _ = _bound_contact(*args)
    assert contact["status"] == "unknown"
    assert contact["contact_frame"] is None


def test_independently_reselected_actor_is_used_with_pose_actor_preserved(tmp_path):
    args = _ball_fixture(tmp_path, actor=9)
    contact, provenance = _bound_contact(*args)
    assert contact["status"] == "tentative"
    assert contact["contact_associated_track_id"] == 9
    assert provenance["actor_reselected_by_ball_evidence"]
    result, track = _analyse_pose(args[3], contact, _motion())
    assert track["track_id"] == 9
    assert result["analysis_actor_track_id"] == 9
    assert result["original_pose_selected_track_id"] == 2
    assert result["actor_reselected_by_independent_ball_evidence"]
    assert args[3]["selected_track_id"] == 2


@pytest.mark.parametrize("guard", [{"field_support": .2}, {"ambiguous": True}, {"excluded": True}])
def test_reselected_actor_requires_field_and_unambiguous_nonkeeper_evidence(tmp_path, guard):
    contact, _ = _bound_contact(*_ball_fixture(tmp_path, actor=9, **guard))
    assert contact["status"] == "unknown"
    assert contact["contact_associated_track_id"] is None


def _update_candidate(args, change):
    folder = args[0]
    search_path = folder / "contact_search/test_clip.json"
    event_path = folder / "event_windows/test_clip.json"
    result_path = folder / "results/test_clip.json"
    search, event, result = (json.loads(path.read_text()) for path in (search_path, event_path, result_path))
    change(search["selected_candidate"])
    event["contact"] = search["selected_candidate"]
    search_path.write_text(json.dumps(search))
    event_path.write_text(json.dumps(event))
    result["contact_search_sha256"] = hashlib.sha256(search_path.read_bytes()).hexdigest()
    result["event_cache_sha256"] = hashlib.sha256(event_path.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result))


def test_ankle_proximity_without_actual_departure_bracket_is_unknown(tmp_path):
    args = _ball_fixture(tmp_path)
    _update_candidate(args, lambda candidate: candidate.update(contact_timing_status="unknown"))
    contact, _ = _bound_contact(*args)
    assert contact["status"] == "unknown"
    assert contact["contact_frame"] is None


def test_bracket_must_match_interval_and_actual_raw_source_points(tmp_path):
    args = _ball_fixture(tmp_path)
    _update_candidate(args, lambda candidate: candidate.update(interval_frames=[78, 84]))
    with pytest.raises(ValueError, match="contact candidate interval"):
        _bound_contact(*args)
    args = _ball_fixture(tmp_path / "wrong_pixel")
    def change_pixel(candidate):
        candidate["stationary_ball_observations"][0]["xy"][0] += 1.
    _update_candidate(args, change_pixel)
    with pytest.raises(ValueError, match="raw source detection"):
        _bound_contact(*args)


def test_cpu_replay_requires_byte_bound_original_neural_producer(tmp_path):
    args = _ball_fixture(tmp_path)
    folder = args[0]
    result_path = folder / "results/test_clip.json"
    original = json.loads(result_path.read_text())
    producer_folder = folder / "producer_results"
    producer_folder.mkdir()
    receipt = producer_folder / "test_clip.json"
    receipt.write_bytes(result_path.read_bytes())
    result = {**original, "new_neural_inference": False, "new_neural_inference_this_run": False,
              "fresh_neural_cache_used": True,
              "original_neural_inference_producer": {"inference_identity": original["inference_identity"],
                  "new_neural_inference": True, "fresh_model_inference": True,
                  "receipt_file": "producer_results/test_clip.json",
                  "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                  "detections_file": "detections/test_clip.json",
                  "detections_sha256": original["source_cache_sha256"]}}
    result_path.write_text(json.dumps(result))
    contact, provenance = _bound_contact(*args)
    assert contact["status"] == "tentative"
    assert provenance["neural_inference_this_run"] is False
    assert provenance["fresh_neural_cache_used"] is True
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="producer receipt hash mismatch"):
        _bound_contact(*args)


def test_fresh_cache_flag_without_original_producer_does_not_establish_contact(tmp_path):
    args = _ball_fixture(tmp_path)
    path = args[0] / "results/test_clip.json"
    result = json.loads(path.read_text())
    result.update(new_neural_inference=False, new_neural_inference_this_run=False, fresh_neural_cache_used=True)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="producer identity mismatch"):
        _bound_contact(*args)


def test_phase_cli_help_directly_from_unrelated_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/run_fresh_phase_experiment.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--ball-dir" in result.stdout
    assert "--pose-dir" in result.stdout


def _runner_fixture(tmp_path, monkeypatch):
    clips, poses, output = (tmp_path / name for name in ("clips", "poses", "phase"))
    clips.mkdir()
    poses.mkdir()
    video = clips / "test_clip.mp4"
    video.write_bytes(b"test source hash; decoder is replaced in this fixture")
    source = {"clip_id": "test_clip", "clip_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "clip_filename": video.name, "event_id": 1, "fps": 30., "frame_count": 120}
    (clips / "source_manifest.json").write_text(json.dumps({"clips": [source]}))
    pose = _pose()
    pose.update(selected_track_qualified=True, new_neural_inference=True,
                reference_labels_used_for_prediction=False,
                shot_selection={"found_qualified_shot": True, "selected_shot_index": 1,
                                "attempts": [{"shot_index": 1, "qualifies": True,
                                              "selected_track_id": 2, "scene_rejections": []}]},
                inference_identity={"source_sha256": source["clip_sha256"],
                    "settings": {"use_field_mask": True, "require_goal_context": True,
                        "event_slice_policy": "selected_shot_must_overlap_manual_course_slice_contact_remains_automatic"}})
    (poses / "test_clip.json").write_text(json.dumps(pose))
    monkeypatch.setattr(runner, "estimate_background_motion", lambda *args, **kwargs: _motion())
    monkeypatch.setattr(runner, "render_sheet", lambda *args, **kwargs: None)
    arguments = ["--clips-dir", str(clips), "--pose-dir", str(poses), "--ball-dir", str(tmp_path / "missing_ball"),
                 "--output-dir", str(output)]
    return arguments, poses, output


def test_resume_preserves_unknown_summary_and_rejects_modified_motion(tmp_path, monkeypatch):
    arguments, _, output = _runner_fixture(tmp_path, monkeypatch)
    assert runner.main(arguments) == 0
    initial = json.loads((output / "phase_experiment_summary.json").read_text())
    assert initial["rows"][0]["contact_status"] == "unknown"
    assert initial["rows"][0]["candidate_step_count"] is None
    assert runner.main(arguments + ["--resume"]) == 0
    repeated = json.loads((output / "phase_experiment_summary.json").read_text())
    assert repeated["rows"] == initial["rows"]
    camera_path = output / "camera_motion/test_clip.json"
    camera_path.write_text(camera_path.read_text() + "\n")
    with pytest.raises(ValueError, match="camera motion artifact mismatch"):
        runner.main(arguments + ["--resume"])


def test_pose_claiming_reference_prediction_is_rejected(tmp_path, monkeypatch):
    arguments, poses, _ = _runner_fixture(tmp_path, monkeypatch)
    path = poses / "test_clip.json"
    pose = json.loads(path.read_text())
    pose["reference_labels_used_for_prediction"] = True
    path.write_text(json.dumps(pose))
    with pytest.raises(ValueError, match="pose result is not source-only"):
        runner.main(arguments)
