"""Synthetic joint-fit checks are not measured course-video accuracy."""
import cv2
import hashlib
import json
import numpy as np
from pathlib import Path
import subprocess
import sys
import pytest

import src.joint_pitch_calibration as joint
from scripts.run_joint_pitch_calibration_pilot import _runup_segments
from src.joint_pitch_calibration import _ambiguous_assignments, calibrate_joint_pitch_frame
from src.pitch_calibration import PITCH_LINES, _project, project_ground_contacts
from tests.test_pitch_calibration import _scene


def test_joint_ground_fit_preserves_independent_side_evidence_and_exact_frame():
    frame, goal, matrix = _scene()
    result = calibrate_joint_pitch_frame(frame, frame_id=7, goal=goal)
    assert result["status"] == "candidate_supported", result["reason"]
    assert set(result["fit_line_names"]).isdisjoint(result["held_out_line_names"])
    assert "penalty_area_front" in result["fit_line_names"]
    assert set(result["held_out_line_names"]) == {"penalty_area_left", "penalty_area_right"}
    assert result["held_out_used_for_hypothesis_selection"] is False
    assert result["camera_intrinsics_known"] is False
    assert result["metric_accuracy_verified"] is False
    pixels, _ = _project(matrix, [[0., 11.]])
    recovered = project_ground_contacts(result, pixels, frame_id=7, point_kind="foot_ground_contact", on_ground=[True])
    np.testing.assert_allclose(recovered["coordinates_m_hypothesis"], [[0., 11.]], atol=.35)
    rejected = project_ground_contacts(result, pixels, frame_id=8, point_kind="foot_ground_contact", on_ground=[True])
    assert rejected["coordinates_m_hypothesis"] == [None]


def test_joint_fit_paint_alone_cannot_replace_independent_held_out_sides():
    frame, goal, matrix = _scene(held_out=False)
    points, _ = _project(matrix, PITCH_LINES["penalty_area_front"])
    cv2.line(frame, tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)), (245, 245, 245), 3)
    result = calibrate_joint_pitch_frame(frame, frame_id=7, goal=goal)
    assert result["candidate"] is not None
    assert result["status"] == "abstain"
    assert result["selected"] is None
    assert result["reason"] in {"independent_held_out_penalty_side_not_supported", "contradictory_held_out_penalty_side"}


def test_near_cost_different_geometric_assignments_abstain():
    first = {"selection_cost_px": 2., "pitch_to_image_hypothesis": np.eye(3).tolist()}
    matrix = np.eye(3)
    matrix[0, 2] = 20.
    second = {"selection_cost_px": 3., "pitch_to_image_hypothesis": matrix.tolist()}
    assert _ambiguous_assignments([first, second], 720)
    second["selection_cost_px"] = 20.
    assert not _ambiguous_assignments([first, second], 720)
    second["selection_cost_px"] = 3.
    matrix[0, 2] = 2.
    second["pitch_to_image_hypothesis"] = matrix.tolist()
    assert not _ambiguous_assignments([first, second], 720)


def test_ambiguous_goal_or_dark_field_cannot_generate_joint_geometry():
    frame, _, _ = _scene()
    result = calibrate_joint_pitch_frame(frame, goal={"status": "abstain", "selected": None})
    assert result["candidate"] is None
    assert result["selected"] is None
    dark = calibrate_joint_pitch_frame(np.zeros_like(frame))
    assert dark["reason"] == "insufficient_field_prior"


def test_unexplored_base_assignments_prevent_claim_of_unique_joint_hypothesis(monkeypatch):
    frame, goal, _ = _scene()
    original = joint.calibrate_pitch_frame
    def truncated(*args, **kwargs):
        result = original(*args, **kwargs)
        result["distinct_fit_count"] = len(result["alternative_fit_summaries"]) + 2
        return result
    monkeypatch.setattr(joint, "calibrate_pitch_frame", truncated)
    result = joint.calibrate_joint_pitch_frame(frame, goal=goal)
    assert result["reason"] == "unexplored_goal_area_assignments"
    assert result["selected"] is None


def test_joint_pilot_direct_cli_keeps_source_hash_and_unknown_geometry(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    video = clips / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 64))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.full((64, 64, 3), 25, np.uint8))
    writer.release()
    clip_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    source = {"clip_id": "tiny", "clip_sha256": clip_hash, "clip_filename": video.name,
              "frame_count": 2, "fps": 10., "event_id": 1, "source_start_frame": 100,
              "original_slice_start_frame_in_clip": 0, "original_slice_end_frame_in_clip": 2}
    (clips / "source_manifest.json").write_text(json.dumps({"clips": [source]}))
    output = tmp_path / "joint"
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts/run_joint_pitch_calibration_pilot.py"),
                                "--clips-dir", str(clips), "--frames", "0", "1", "--output-dir", str(output)],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "tiny.json").read_text())
    assert result["inference_identity"]["source_sha256"] == clip_hash
    assert result["sampled_frame_ids"] == [0, 1]
    assert all(row["source_frame_id"] == row["frame_id"] + 100 for row in result["frames"])
    assert all(row["joint"]["selected"] is None for row in result["frames"])
    assert all(row["coordinates_m_hypothesis"] == {"baseline": None, "joint": None} for row in result["frames"])
    assert result["metric_accuracy_verified"] is False
    assert result["human_verified"] is False
    assert result["sparse_ground_contact_candidates"] == []
    assert result["pre_kick_contact_candidate_count"] == 0
    assert result["pre_contact_path_segments_frame_ids"] == []
    assert (output / "tiny_ground.png").is_file()


def test_missing_scheduled_ground_point_breaks_sparse_runup_display_segments():
    points = [{"frame_id": frame} for frame in (8, 2)]
    assert [[point["frame_id"] for point in segment] for segment in _runup_segments(points, [2, 5, 8])] == [[2], [8]]
    points.append({"frame_id": 5})
    assert [[point["frame_id"] for point in segment] for segment in _runup_segments(points, [2, 5, 8])] == [[2, 5, 8]]
    with pytest.raises(ValueError, match="distinct sampled source frame"):
        _runup_segments([{"frame_id": 4}], [2, 5, 8])
