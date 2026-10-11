"""Synthetic geometry regressions; these do not measure real-video accuracy."""
import cv2
import hashlib
import json
import numpy as np
from pathlib import Path
import pytest
import subprocess
import sys

from src.pitch_calibration import (PITCH_LINES, _project, calibrate_pitch_frame,
                                   foot_contact_candidate, project_ground_contacts)


def _scene(*, held_out=True, wrong_front=False):
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), np.uint8)
    frame[35:-35, 60:-60] = (55, 145, 65)
    camera = np.array([-65., -45., 10.])
    target = np.array([0., -9., 0.])
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ camera
    intrinsic = np.array([[3200., 0, w / 2], [0, 3200., h / 2], [0, 0, 1]])
    matrix = intrinsic @ np.c_[rotation[:, 0], -rotation[:, 1], translation]
    for name, ends in PITCH_LINES.items():
        if name.startswith("penalty_") and not held_out:
            continue
        ends = np.asarray(ends, float)
        if name == "penalty_area_front" and wrong_front:
            ends[:, 1] += 2.5
        points, _ = _project(matrix, ends)
        cv2.line(frame, tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)),
                 (245, 245, 245), 3)
    world = np.array([[-3.66, 0, 2.44], [3.66, 0, 2.44], [3.66, 0, 0], [-3.66, 0, 0]], float)
    homogeneous = (world @ rotation.T + translation) @ intrinsic.T
    corners = homogeneous[:, :2] / homogeneous[:, 2, None]
    for a, b in ((0, 1), (0, 3), (1, 2)):
        cv2.line(frame, tuple(np.rint(corners[a]).astype(int)), tuple(np.rint(corners[b]).astype(int)),
                 (255, 255, 255), 3)
    goal = {"status": "candidate_supported", "selected": {"corners": corners.tolist()}}
    return frame, goal, matrix


def test_synthetic_pitch_recovers_ground_not_vertical_goal_plane():
    frame, goal, true_matrix = _scene()
    result = calibrate_pitch_frame(frame, frame_id=24, goal=goal)
    assert result["status"] == "candidate_supported", result["reason"]
    assert not result["metric_accuracy_verified"]
    assert not result["camera_intrinsics_known"]
    assert not result["valid_for_other_frames"]
    assert set(result["fit_line_names"]).isdisjoint(result["held_out_line_names"])
    ground = np.array([[0., 11.], [-1.5, 10.], [2., 13.]])
    pixels, _ = _project(true_matrix, ground)
    recovered = project_ground_contacts(result, pixels, frame_id=24,
                                         point_kind="foot_ground_contact", on_ground=[True] * 3)
    assert recovered["status"] == "candidate_supported"
    assert not recovered["metric_accuracy_verified"]
    np.testing.assert_allclose(recovered["coordinates_m_hypothesis"], ground, atol=.35)


@pytest.mark.parametrize("options", [{"held_out": False}, {"wrong_front": True}])
def test_fit_lines_alone_do_not_pass_without_independent_penalty_front(options):
    frame, goal, _ = _scene(**options)
    result = calibrate_pitch_frame(frame, frame_id=2, goal=goal)
    assert result["candidate"] is not None
    assert result["status"] == "abstain"
    assert result["selected"] is None
    assert result["reason"] == "held_out_penalty_front_not_supported"


def test_dark_frame_and_ambiguous_goal_abstain():
    result = calibrate_pitch_frame(np.zeros((720, 1280, 3), np.uint8))
    assert result["status"] == "abstain"
    assert result["candidate"] is None
    frame, _, _ = _scene()
    result = calibrate_pitch_frame(frame, goal={"status": "abstain", "selected": None})
    assert result["reason"] == "insufficient_or_ambiguous_goal_geometry"


@pytest.mark.parametrize("kind", ["ball", "hip", "ankle", "foot_ground_contact_guess"])
def test_noncontact_points_are_never_projected(kind):
    result = project_ground_contacts({}, [[100, 200]], frame_id=0, point_kind=kind, on_ground=[True])
    assert result["coordinates_m_hypothesis"] == [None]
    assert result["reason"] == "not_a_ground_contact_foot_point"


def test_contact_projection_rejects_airborne_outside_frame_outside_area_and_nonfinite():
    frame, goal, true_matrix = _scene()
    calibration = calibrate_pitch_frame(frame, frame_id=2, goal=goal)
    pixels, _ = _project(true_matrix, [[0, 11], [0, 10], [0, 22]])
    pixels = np.vstack([pixels, [-1, 300], [np.nan, 400]])
    result = project_ground_contacts(calibration, pixels, frame_id=2,
                                     point_kind="foot_ground_contact", on_ground=[True, False, True, True, True])
    assert result["coordinates_m_hypothesis"][0] is not None
    assert result["coordinates_m_hypothesis"][1:] == [None] * 4


def test_contact_projection_cannot_reuse_a_camera_frame_or_abstained_hypothesis():
    frame, goal, _ = _scene()
    calibration = calibrate_pitch_frame(frame, frame_id=2, goal=goal)
    result = project_ground_contacts(calibration, [[400, 400]], frame_id=3,
                                     point_kind="foot_ground_contact", on_ground=[True])
    assert result["reason"] == "calibration_frame_mismatch"
    for frame_id in (True, 2.0, "2"):
        result = project_ground_contacts(calibration, [[400, 400]], frame_id=frame_id,
                                         point_kind="foot_ground_contact", on_ground=[True])
        assert result["reason"] == "calibration_frame_mismatch"
    calibration["status"] = "abstain"
    result = project_ground_contacts(calibration, [[400, 400]], frame_id=2,
                                     point_kind="foot_ground_contact", on_ground=[True])
    assert result["reason"] == "calibration_not_supported"
    calibration["frame_id"] = None
    result = project_ground_contacts(calibration, [[400, 400]], frame_id=None,
                                     point_kind="foot_ground_contact", on_ground=[True])
    assert result["reason"] == "calibration_frame_mismatch"


@pytest.mark.parametrize("flag", ["false", 0, 1, None])
def test_contact_flags_require_actual_booleans(flag):
    with pytest.raises(ValueError, match="boolean"):
        project_ground_contacts({}, [[1, 2]], frame_id=1,
                                point_kind="foot_ground_contact", on_ground=[flag])


@pytest.mark.parametrize("kwargs", [{"frame_id": -1}, {"frame_id": True}, {"held_out_radius_px": np.nan},
                                   {"held_out_radius_px": -1}, {"focal_ratios": [np.inf]}])
def test_invalid_camera_or_quality_parameters_reject(kwargs):
    frame, goal, _ = _scene()
    with pytest.raises(ValueError):
        calibrate_pitch_frame(frame, goal=goal, **kwargs)


def test_direct_pilot_preserves_source_and_decode_evidence(tmp_path):
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 64))
    assert writer.isOpened()
    for value in (20, 40):
        writer.write(np.full((64, 64, 3), value, np.uint8))
    writer.release()
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    completed = subprocess.run([sys.executable, str(root / "scripts/run_pitch_calibration_pilot.py"),
                                "--video", str(video), "--times", "0", "0.1", "1",
                                "--output-dir", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "pitch_calibration_pilot.json").read_text())
    assert result["sample_count"] == 3
    assert result["supported_frame_count"] == 0
    assert result["metric_accuracy_verified"] is False
    assert result["human_reference_used"] is False
    assert result["source_hashes"][str(video)] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert result["algorithm_sha256"] == hashlib.sha256((root / "src/pitch_calibration.py").read_bytes()).hexdigest()
    assert result["records"][0]["decoded_frame_id"] == 0
    assert result["records"][1]["decoded_frame_id"] == 1
    assert len(result["records"][0]["decoded_pixels_sha256"]) == 64
    assert result["records"][2]["reason"] == "frame_decode_failed"
    assert result["records"][2]["calibration"] is None
    assert len(list(output.glob("*.jpg"))) == 2


def _foot_scene():
    frame = np.full((160, 160, 3), (55, 145, 65), np.uint8)
    cv2.rectangle(frame, (74, 90), (90, 105), (225, 225, 225), -1)
    points = np.zeros((17, 2))
    points[15] = [80, 98]
    points[16] = [100, 90]
    observation = {"bbox": [60, 15, 110, 107], "keypoints": points.tolist(),
                   "keypoint_confidences": [0.] * 15 + [.8, .8]}
    return frame, observation


def test_contact_pixels_are_sole_boundary_hypotheses_not_ankle_or_box_substitutes():
    frame, observation = _foot_scene()
    contact = foot_contact_candidate(frame, observation)
    assert contact["status"] == "contact_candidate"
    assert contact["sole_pixel"][1] > contact["ankle_pixel"][1]
    assert contact["sole_pixel"][1] < observation["bbox"][3]
    assert not contact["ground_contact_verified"]
    assert not contact["human_verified"]
    assert contact["grass_below_fraction"] >= .55
    frame[:] = (55, 145, 65)
    missing = foot_contact_candidate(frame, observation)
    assert missing["sole_pixel"] is None
    assert missing["reason"] == "no_visible_shoe_boundary"


def test_contact_heuristic_rejects_low_confidence_and_missing_grass():
    frame, observation = _foot_scene()
    observation["keypoint_confidences"] = [0.] * 17
    assert foot_contact_candidate(frame, observation)["reason"] == "no_confident_ankle"
    frame, observation = _foot_scene()
    frame[108:] = (0, 0, 0)
    assert foot_contact_candidate(frame, observation)["reason"] == "insufficient_grass_below_sole"
