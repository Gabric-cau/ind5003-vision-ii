from copy import deepcopy

import numpy as np
import pytest

from src import contact_video_features as features


def scene():
    ids = [0, 2, 4, 6, 8]
    observations, balls = [], []
    for index, frame in enumerate(ids):
        points = np.zeros((17, 2), dtype=float)
        points[5:17] = [
            [40, 35], [65, 36], [30, 60], [75, 62], [20, 80], [85, 82],
            [43, 100], [62, 102], [38, 140], [68, 145],
            [30 + 4 * index, 178], [70 - 5 * index, 184],
        ]
        points[11:13, 0] += index
        confidence = np.ones(17)
        confidence[15], confidence[16] = .8, .9
        observations.append({
            "frame": frame, "bbox": [10, 20, 100, 220],
            "keypoints": points.tolist(),
            "keypoint_confidences": confidence.tolist(), "box_confidence": .95,
        })
        x = 50 + max(0, index - 2) * 10
        balls.append({"frame": frame, "xy": [x, 180], "bbox": [x - 3, 177, x + 3, 183],
                      "confidence": .8, "diameter": 6})
    candidate = {"contact_associated_track_id": 7, "stationary_ball_reference_xy": [50, 180],
                 "score": .9, "contact_frame": 4, "contact_time_sec": 4 / 30,
                 "interval_frames": [2, 7], "ball_departure_frames": [6, 8],
                 "ankle_approach_supported": True}
    return {"ids": ids, "fps": 30., "height": 720,
            "tracks": [{"track_id": 7, "observations": observations}],
            "balls": balls, "transforms": {frame: np.eye(3) for frame in ids},
            "candidates": [candidate], "camera": {"failed_pairs": []}}


def matrix(data):
    return np.asarray(features.geometric_features(**data)[0])


def column(name):
    return features.FEATURE_NAMES.index(name)


def test_features_do_not_encode_absolute_clock_or_event_identity():
    data = scene()
    shifted = deepcopy(data)
    offset = 900
    shifted["ids"] = [frame + offset for frame in data["ids"]]
    shifted["transforms"] = {frame + offset: transform for frame, transform in data["transforms"].items()}
    for track in shifted["tracks"]:
        for observation in track["observations"]:
            observation["frame"] += offset
    for ball in shifted["balls"]:
        ball["frame"] += offset
    for candidate in shifted["candidates"]:
        candidate["contact_frame"] += offset
        candidate["contact_time_sec"] += offset / shifted["fps"]
        for field in ("interval_frames", "ball_departure_frames"):
            candidate[field] = [frame + offset for frame in candidate[field]]
    np.testing.assert_allclose(matrix(data), matrix(shifted))
    forbidden = ("time", "frame", "progress", "core_id", "match", "reference", "label")
    assert not any(token in name for name in features.FEATURE_NAMES for token in forbidden)
    assert len(features.FEATURE_NAMES) == len(set(features.FEATURE_NAMES))


def test_camera_translation_is_removed_from_pose_and_ball_geometry():
    data = scene()
    moved = deepcopy(data)
    for index, frame in enumerate(moved["ids"]):
        translation = np.array([50 + 7 * index, -10 + 3 * index])
        observation = moved["tracks"][0]["observations"][index]
        observation["keypoints"] = (np.asarray(observation["keypoints"]) + translation).tolist()
        observation["bbox"] = (np.asarray(observation["bbox"]) + np.tile(translation, 2)).tolist()
        ball = moved["balls"][index]
        ball["xy"] = (np.asarray(ball["xy"]) + translation).tolist()
        ball["bbox"] = (np.asarray(ball["bbox"]) + np.tile(translation, 2)).tolist()
        moved["transforms"][frame][:2, 2] = -translation
    np.testing.assert_allclose(matrix(data), matrix(moved), atol=1e-12)


def test_mirror_and_anatomical_swap_preserve_corresponding_channels():
    data = scene()
    mirrored = deepcopy(data)
    width = 1280
    swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    for observation in mirrored["tracks"][0]["observations"]:
        points = np.asarray(observation["keypoints"])[swap]
        points[:, 0] = width - points[:, 0]
        observation["keypoints"] = points.tolist()
        observation["keypoint_confidences"] = np.asarray(observation["keypoint_confidences"])[swap].tolist()
        left, top, right, bottom = observation["bbox"]
        observation["bbox"] = [width - right, top, width - left, bottom]
    for ball in mirrored["balls"]:
        ball["xy"][0] = width - ball["xy"][0]
    mirrored["candidates"][0]["stationary_ball_reference_xy"][0] = width - 50
    expected, actual = matrix(data), matrix(mirrored)
    for name in features.FEATURE_NAMES:
        counterpart = ("right_" + name[5:] if name.startswith("left_") else
                       "left_" + name[6:] if name.startswith("right_") else name)
        np.testing.assert_allclose(actual[:, column(name)], expected[:, column(counterpart)], atol=1e-12)


def test_missing_pose_and_ball_do_not_create_motion_across_a_gap():
    data = scene()
    data["tracks"][0]["observations"] = [o for o in data["tracks"][0]["observations"] if o["frame"] != 4]
    data["balls"] = [b for b in data["balls"] if b["frame"] != 4]
    result = matrix(data)
    assert np.isfinite(result).all()
    assert result[2, column("pose_visible")] == 0
    assert result[2, column("ball_visible")] == 0
    for name in ("left_ankle_speed", "right_ankle_speed", "hip_speed", "ball_speed"):
        assert result[3, column(name)] == 0
    assert result[3, column("pose_visible")] == 1
    assert result[3, column("ball_visible")] == 1


def test_low_confidence_ankle_is_masked_and_empty_scene_abstains():
    data = scene()
    for observation in data["tracks"][0]["observations"]:
        observation["keypoint_confidences"][15] = .1
    result = matrix(data)
    for name in ("left_ankle_visible", "left_ankle_seed_distance", "left_ankle_speed", "left_knee_extension"):
        np.testing.assert_array_equal(result[:, column(name)], 0)
    np.testing.assert_array_equal(result[:, column("right_ankle_visible")], 1)
    data.update(tracks=[], balls=[], candidates=[])
    rows, actor, baseline = features.geometric_features(**data)
    assert np.isfinite(rows).all()
    assert actor["actor_id"] is None
    assert baseline["status"] == "abstain"
    assert baseline["time_seconds"] is None


@pytest.mark.parametrize("score,status", [(.75, "supported"), (.50, "tentative"), (.499, "abstain")])
def test_rule_baseline_preserves_inclusive_score_thresholds(score, status):
    candidate = scene()["candidates"][0]
    candidate["score"] = score
    result = features.rule_baseline([candidate], {"failed_pairs": []}, 20)
    assert result["status"] == status
    assert (result["time_seconds"] is None) == (status == "abstain")
    assert result["full_original_pipeline"] is False
    assert result["sparse_to_dense_refinement"] is False


@pytest.mark.parametrize("change", ["no_departure", "no_approach", "camera_failure", "competing_actor"])
def test_supported_rule_requires_departure_approach_camera_and_identity(change):
    candidates = scene()["candidates"]
    camera = {"failed_pairs": []}
    if change == "no_departure":
        candidates[0]["ball_departure_frames"] = None
    elif change == "no_approach":
        candidates[0]["ankle_approach_supported"] = False
    elif change == "camera_failure":
        camera["failed_pairs"] = [[4, 6]]
    else:
        other = deepcopy(candidates[0])
        other.update(contact_associated_track_id=8, score=.84)
        candidates.append(other)
    assert features.rule_baseline(candidates, camera, 20)["status"] == "tentative"


def test_camera_failure_after_contact_support_does_not_invalidate_it():
    result = features.rule_baseline(scene()["candidates"], {"failed_pairs": [[10, 12]]}, 20)
    assert result["status"] == "supported"
    assert result["camera_reliable_through_support"] is True
    empty = features.rule_baseline([], {"failed_pairs": []}, 20)
    assert empty["status"] == "abstain"
    assert empty["time_seconds"] is None


@pytest.mark.parametrize("field", ["reference_contact_seconds", "reference_dataset", "contact_seconds"])
def test_reference_inputs_are_rejected_before_models_or_media_are_accessed(field):
    with pytest.raises(ValueError, match="Annotation data"):
        features.extract_contact_record({field: 1.8}, None, None, [])


def test_changed_video_hash_is_rejected_before_inference(tmp_path, monkeypatch):
    video = tmp_path / "core_01.mp4"
    video.write_bytes(b"changed media")
    monkeypatch.setattr(features, "_scan_camera_shots", lambda *args: pytest.fail("must reject before decoding"))
    sample = {"video_path": str(video), "video_sha256": "0" * 64}
    with pytest.raises(ValueError, match="checksum"):
        features.extract_contact_record(sample, None, None, [])
