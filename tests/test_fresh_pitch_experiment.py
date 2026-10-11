import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

import scripts.run_fresh_pitch_experiment as runner
from scripts.run_fresh_pitch_experiment import _pose_input, _sample_frames


def _pose_fixture(tmp_path):
    source = {"clip_id": "test_clip", "clip_sha256": "a" * 64}
    pose = {"clip_id": source["clip_id"], "selected_track_id": 2, "selected_track_qualified": True,
            "new_neural_inference": True,
            "shot_selection": {"found_qualified_shot": True, "selected_shot_index": 1,
                               "attempts": [{"shot_index": 1, "qualifies": True,
                                             "selected_track_id": 2, "scene_rejections": []}]},
            "inference_identity": {"source_sha256": source["clip_sha256"],
                                   "settings": {"use_field_mask": True, "require_goal_context": True,
                          "event_slice_policy": "selected_shot_must_overlap_manual_course_slice_contact_remains_automatic"}}}
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(pose))
    return source, pose, path


def test_pose_source_binding_and_guard_requirements(tmp_path):
    source, pose, path = _pose_fixture(tmp_path)
    selected, provenance = _pose_input(path, source)
    assert selected == pose
    assert provenance["status"] == "source_bound_automatic_pose_candidate"
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    source["clip_sha256"] = "b" * 64
    selected, provenance = _pose_input(path, source)
    assert selected is None
    assert provenance["status"] == "source_identity_mismatch"


@pytest.mark.parametrize("field,value,reason", [
    ("use_field_mask", "false", "pose_selection_scene_guards_missing"),
    ("require_goal_context", False, "pose_selection_scene_guards_missing"),
    ("event_slice_policy", None, "pose_manual_slice_overlap_policy_missing"),
    ("selected_track_qualified", False, "selected_person_not_qualified"),
    ("new_neural_inference", False, "fresh_neural_inference_not_recorded"),
])
def test_unsafe_or_prior_pose_is_not_projected(tmp_path, field, value, reason):
    source, pose, path = _pose_fixture(tmp_path)
    if field in pose["inference_identity"]["settings"]:
        pose["inference_identity"]["settings"][field] = value
    else:
        pose[field] = value
    path.write_text(json.dumps(pose))
    selected, provenance = _pose_input(path, source)
    assert selected is None
    assert provenance["status"] == reason


@pytest.mark.parametrize("rejection", ["found_false", "scene_rejection", "goal_occupant"])
def test_rejected_selected_actor_cannot_enter_pitch_trajectory(tmp_path, rejection):
    source, pose, path = _pose_fixture(tmp_path)
    if rejection == "found_false":
        pose["shot_selection"]["found_qualified_shot"] = False
    elif rejection == "scene_rejection":
        pose["shot_selection"]["attempts"][0]["scene_rejections"] = ["selected_kicker_is_goal_line_occupant"]
    else:
        pose["shot_selection"]["attempts"][0]["role_context"] = {"status": "candidate_supported", "seed_track_id": 2}
    path.write_text(json.dumps(pose))
    selected, provenance = _pose_input(path, source)
    assert selected is None
    assert provenance["status"] == ("selected_person_is_goal_line_occupant" if rejection == "goal_occupant"
                                     else "selected_pose_shot_scene_rejected")


def test_shot_sampling_never_crosses_end_exclusive_boundary():
    frames = _sample_frames([{"shot_start_frame": 10, "shot_end_frame": 15},
                             {"shot_start_frame": 30, "shot_end_frame": 34}], 3)
    assert frames == [10, 12, 13, 30, 31, 32, 33]
    assert 15 not in frames and 34 not in frames


def test_cli_unknown_geometry_source_hash_and_checked_cache(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    video = clips / "test_clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 64))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((64, 64, 3), 25, np.uint8))
    writer.release()
    clip_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    source = {"clip_id": "test_clip", "clip_sha256": clip_hash, "clip_filename": video.name,
              "event_id": 1, "source_start_frame": 100}
    (clips / "source_manifest.json").write_text(json.dumps({"clips": [source]}))
    output = tmp_path / "pitch"
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/run_fresh_pitch_experiment.py"), "--clips-dir", str(clips),
               "--output-dir", str(output), "--frame-stride", "2"]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "pitch_experiment_summary.json").read_text())
    assert report["cohort_size"] == 1
    assert report["clips_with_supported_geometry"] == 0
    assert report["clips_with_ground_contact_candidates"] == 0
    assert report["metric_accuracy_verified"] is False
    result = json.loads((output / "test_clip.json").read_text())
    assert result["pose_input"]["status"] == "unavailable"
    assert result["calibration_identity"]["source_clip_sha256"] == clip_hash
    assert result["frame_calibrations"][0]["source_frame_id"] == 100
    assert result["sparse_ground_contact_candidates"] == []
    assert (output / "test_clip_ground.png").exists()
    repeated = subprocess.run(command + ["--reuse-calibrations"], cwd=tmp_path, capture_output=True, text=True)
    assert repeated.returncode == 0, repeated.stderr
    result["calibration_identity"]["source_clip_sha256"] = "bad"
    (output / "test_clip.json").write_text(json.dumps(result))
    rejected = subprocess.run(command + ["--reuse-calibrations"], cwd=tmp_path, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "Calibration cache identity mismatch" in rejected.stderr


def test_cli_rejects_wrong_manifest_clip_hash(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "wrong.mp4").write_bytes(b"source bytes")
    (clips / "source_manifest.json").write_text(json.dumps({"clips": [{"clip_id": "test_clip",
                          "clip_sha256": "a" * 64, "clip_filename": "wrong.mp4"}]}))
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "out"
    completed = subprocess.run([sys.executable, str(root / "scripts/run_fresh_pitch_experiment.py"),
                                "--clips-dir", str(clips), "--output-dir", str(output)],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Source clip hash mismatch" in completed.stderr
    assert not (output / "test_clip.json").exists()


def _projection_fixture(tmp_path, monkeypatch):
    video = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 64))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.full((64, 64, 3), 100, np.uint8))
    writer.release()
    source = {"clip_id": "test_clip", "event_id": 1, "source_start_frame": 100,
              "frame_count": 10, "original_slice_start_frame_in_clip": 1,
              "original_slice_end_frame_in_clip": 9}
    identity = {key: "current" for key in ("source_manifest_sha256", "source_clip_sha256", "algorithm_sha256",
                    "goal_geometry_sha256", "field_masks_sha256", "opencv_version", "camera_scanner_sha256", "script_sha256")}
    identity["settings"] = {"frame_stride": 3}
    pose = {"selected_track_id": 2, "shot_start_frame": 1, "shot_end_frame": 9, "kick_frame": 999,
            "tracks": [{"track_id": 2, "observations": [{"frame": 4, "actor": 2}]},
                       {"track_id": 9, "observations": [{"frame": frame, "actor": 9} for frame in (1, 3, 4, 5, 6, 8)]}]}
    contact = {"status": "tentative", "contact_frame": 5, "interval_frames": [4, 6], "contact_associated_track_id": 9}
    monkeypatch.setattr(runner, "_scan_camera_shots", lambda *args, **kwargs: [{"shot_start_frame": 0, "shot_end_frame": 10}])
    monkeypatch.setattr(runner, "calibrate_pitch_frame", lambda *args, **kwargs:
                        {"status": "candidate_supported", "reason": "test_geometry"})
    def sole(frame, observation):
        assert observation["actor"] == 9
        return {"status": "contact_candidate", "sole_pixel": [32, 32], "foot": "left"}
    monkeypatch.setattr(runner, "foot_contact_candidate", sole)
    monkeypatch.setattr(runner, "project_ground_contacts", lambda *args, **kwargs:
                        {"coordinates_m_hypothesis": [[1., 2.]]})
    return source, video, pose, identity, contact


def test_pitch_uses_independent_actor_and_contact_interval_with_exact_frame_geometry(tmp_path, monkeypatch):
    source, video, pose, identity, contact = _projection_fixture(tmp_path, monkeypatch)
    result, _ = runner._run_clip(source, video, pose, identity, independent_contact=contact)
    points = result["sparse_ground_contact_candidates"]
    assert {point["frame_id"] for point in points} == {1, 3, 4, 5, 6, 8}
    assert all(point["actor_track_id"] == 9 and point["independent_contact_frame"] == 5 for point in points)
    assert all(point["source_frame_id"] == point["frame_id"] + 100 for point in points)
    assert {point["frame_id"] for point in points if point["phase"] == "pre_kick_candidate"} == {1, 3}
    assert {point["frame_id"] for point in points if point["phase"] == "contact_interval_candidate"} == {4, 5, 6}
    assert result["complete_runup_verified"] is False


def test_pose_peak_without_independent_contact_cannot_produce_runup_projection(tmp_path, monkeypatch):
    source, video, pose, identity, _ = _projection_fixture(tmp_path, monkeypatch)
    result, _ = runner._run_clip(source, video, pose, identity)
    assert result["supported_frame_count"] > 0
    assert result["sparse_ground_contact_candidates"] == []


def test_cached_geometry_preserves_old_sampling_origin_when_scanner_changes(tmp_path, monkeypatch):
    source, video, pose, identity, contact = _projection_fixture(tmp_path, monkeypatch)
    previous, _ = runner._run_clip(source, video, None, identity)
    previous["calibration_identity"] = {**identity, "camera_scanner_sha256": "old", "script_sha256": "old"}
    previous.pop("camera_shot_sampling_identity")
    def do_not_rescan(*args, **kwargs):
        raise AssertionError("Cached camera shots must keep their original sampling provenance")
    monkeypatch.setattr(runner, "_scan_camera_shots", do_not_rescan)
    result, _ = runner._run_clip(source, video, pose, identity, previous, contact)
    assert result["camera_shot_sampling_identity"]["camera_scanner_sha256"] == "old"
    assert result["new_camera_scanner_applied_to_cached_shots"] is False
    assert result["reused_calibration_identity"]["script_sha256"] == "old"
    assert result["calibration_identity"]["script_sha256"] == "current"
    assert result["ground_contact_candidate_count"] > 0
    identity["algorithm_sha256"] = "different_geometry"
    with pytest.raises(ValueError, match="Calibration cache identity mismatch"):
        runner._run_clip(source, video, pose, identity, previous, contact)
