from copy import deepcopy

import numpy as np
import pytest

from src import contact_tracking_recovery as recovery


def observation(frame, x, *, y=100., height=100., shift=(0., 0.)):
    delta = np.asarray(shift)
    points = np.tile([x, y], (17, 1)).astype(float)
    points[15] = [x - 8, y + height / 2 - 4]
    points[16] = [x + 8, y + height / 2 - 3]
    points += delta
    return {"frame": frame,
            "bbox": (np.array([x - 20, y - height / 2, x + 20, y + height / 2])
                     + np.tile(delta, 2)).tolist(),
            "keypoints": points.tolist(), "keypoint_confidences": [1.] * 17,
            "box_confidence": .9}


def evidence(observations, *, frame_ids=None, shifts=None, failures=None):
    frames = frame_ids if frame_ids is not None else sorted({o["frame"] for o in observations})
    transforms = {}
    for frame in frames:
        transform = np.eye(3)
        transform[:2, 2] = -np.asarray((shifts or {}).get(frame, [0., 0.]))
        transforms[str(frame)] = transform.tolist()
    return {"tracks": [{"track_id": index, "observations": [o]}
                       for index, o in enumerate(observations)],
            "balls": [], "transforms": transforms,
            "camera": {"failed_pairs": failures or []}, "candidate_contacts": []}


def test_stabilized_association_recovers_camera_pan_without_fabricating_observations():
    shifts = {0: (0., 0.), 2: (180., -40.), 6: (600., 25.)}
    observations = [observation(frame, 100 + frame, shift=shifts[frame]) for frame in shifts]
    data = evidence(observations, frame_ids=[0, 2, 4, 6], shifts=shifts)
    original = deepcopy(data)
    tracks, diagnostics = recovery.rebuild_camera_compensated_tracks(data, fps=30)
    assert len(tracks) == 1
    assert [o["frame"] for o in tracks[0]["observations"]] == [0, 2, 6]
    assert diagnostics["missing_keypoints_interpolated"] is False
    assert data == original
    for actual, expected in zip(tracks[0]["observations"], observations):
        assert actual["keypoints"] == expected["keypoints"]


@pytest.mark.parametrize("kind", ["camera_failure", "long_gap", "scale_jump", "teleport"])
def test_track_is_not_joined_across_unreliable_geometry(kind):
    frame = 90 if kind == "long_gap" else 2
    second = observation(frame, 110 if kind != "teleport" else 900,
                         height=400 if kind == "scale_jump" else 100)
    data = evidence([observation(0, 100), second], failures=[[0, 2]] if kind == "camera_failure" else [])
    tracks, _ = recovery.rebuild_camera_compensated_tracks(data, fps=30)
    assert len(tracks) == 2


def test_ambiguous_identity_at_crossing_is_rejected_instead_of_guessed():
    data = evidence([observation(0, 80), observation(0, 120), observation(2, 100)])
    tracks, diagnostics = recovery.rebuild_camera_compensated_tracks(data, fps=30)
    assert len(tracks) == 3
    assert diagnostics["ambiguous_associations_rejected"] == 1


def test_assignment_is_one_to_one_and_does_not_merge_simultaneous_people():
    data = evidence([observation(0, 100), observation(0, 400),
                     observation(2, 105), observation(2, 395)])
    tracks, _ = recovery.rebuild_camera_compensated_tracks(data, fps=30)
    assert len(tracks) == 2
    assert all([o["frame"] for o in t["observations"]] == [0, 2] for t in tracks)


def test_invalid_missing_and_duplicate_observations_are_not_synthetic_poses():
    valid = observation(0, 100)
    invalid = observation(2, 110)
    invalid["keypoints"][15][0] = float("nan")
    data = evidence([valid, deepcopy(valid), invalid])
    tracks, diagnostics = recovery.rebuild_camera_compensated_tracks(data, fps=30)
    assert sum(len(t["observations"]) for t in tracks) == 1
    assert diagnostics["rejected_invalid_observations"] == 1
    empty = evidence([], frame_ids=[0, 2, 4])
    result = recovery.recover_contact_evidence(empty, fps=30, frame_height=720, frame_width=1280)
    assert result["feature_tracks"] == []
    assert result["pose_roi_requests"] == []
    assert result["actor_id"] is None


def test_old_candidate_timestamps_are_not_selection_inputs():
    data = evidence([observation(0, 100), observation(2, 102), observation(4, 104)])
    before = recovery.recover_contact_evidence(data, fps=30, frame_height=720)
    data["candidate_contacts"] = [{"contact_frame": 4, "contact_associated_track_id": 999,
                                    "score": 1., "stationary_ball_reference_xy": [1, 1]}]
    after = recovery.recover_contact_evidence(data, fps=30, frame_height=720)
    assert before == after
    assert after["recovery_diagnostics"]["existing_input_candidate_times_used"] is False


def test_provisional_actor_only_requests_new_pixels_and_never_fills_pose(monkeypatch):
    frames = list(range(0, 22, 2))
    observed = [observation(frame, 100 + frame * 5) for frame in [0, 2, 4, 16, 20]]
    data = evidence(observed, frame_ids=frames)
    candidate = {"contact_associated_track_id": 0, "score": .2, "contact_frame": 16,
                 "stationary_ball_reference_xy": [188, 146], "stationary_ball_samples": 8,
                 "ankle_ball_distance_body_heights": .08, "ankle_approach_supported": False}
    monkeypatch.setattr(recovery, "_score_all_tracks", lambda *args: [deepcopy(candidate)])
    result = recovery.recover_contact_evidence(data, fps=30, frame_height=240, frame_width=320)
    assert result["actor_id"] is None
    assert result["feature_tracks"] == []
    assert result["feature_candidates"] == []
    assert result["recovery_diagnostics"]["roi_actor_is_provisional"] is True
    requests = result["pose_roi_requests"]
    assert {r["frame"] for r in requests} == {6, 8, 10, 12, 14, 18}
    assert sum(len(t["observations"]) for t in result["tracks"]) == len(observed)
    for request in requests:
        assert request["requires_new_pose_inference"] is True
        assert request["is_measured_pose"] is False
        x0, y0, x1, y1 = request["bbox"]
        assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240
        assert request["measured_anchor_frames"][0] < request["frame"] < request["measured_anchor_frames"][1]


def test_candidate_scoring_cannot_cross_failed_camera_prefix(monkeypatch):
    data = evidence([observation(frame, 100 + frame) for frame in [0, 2, 4, 6]], failures=[[2, 4]])
    data["balls"] = [{"frame": frame, "xy": [100, 146], "confidence": .9, "diameter": 6}
                     for frame in [0, 2, 4, 6]]
    captured = {}
    def score(balls, samples, transforms, *args):
        captured["ball_frames"] = [b["frame"] for b in balls]
        captured["pose_frames"] = [s["frame"] for track in samples.values() for s in track]
        captured["transform_frames"] = list(transforms)
        return []
    monkeypatch.setattr(recovery, "_score_all_tracks", score)
    recovery.recover_contact_evidence(data, fps=30, frame_height=720)
    assert captured["ball_frames"] == [0, 2]
    assert captured["pose_frames"] == [0, 2]
    assert captured["transform_frames"] == [0, 2]


@pytest.mark.parametrize("key", ["labels", "targets", "reference_contact_seconds", "manual_contact_frame"])
def test_manual_targets_rejected_before_evidence_is_read(key):
    with pytest.raises(ValueError, match="annotation"):
        recovery.recover_contact_evidence({key: 3}, fps=30, frame_height=720)
