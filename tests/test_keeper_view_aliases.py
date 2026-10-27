from copy import deepcopy

import numpy as np
import pytest

from src.goalkeeper_pose_direction import pose_direction_evidence
from src.keeper_view_aliases import alias_guard_inputs
from tests.test_goalkeeper_pose_direction import fixture


def alias_fixture():
    record, pose = fixture()
    record["goal_identity_evidence"]["seed_track_id"] = 7
    record["selected_candidate"]["source_track_ids"] = [7, 8]
    record["selected_candidate"]["source_track_id_by_frame"] = {
        str(frame): 7 if frame < 69 else 8 for frame in range(55, 73)
    }
    for observation in pose["tracks"][0]["observations"]:
        observation["bbox"] = [60, 60, 140, 130]
    duplicate = deepcopy(pose["tracks"][0])
    duplicate["track_id"] = 8
    for observation in duplicate["observations"]:
        observation["source_roi"] = "automatic_goal_roi_rotation_90"
    pose["tracks"].append(duplicate)
    pose["selected_track_id"] = 99
    image = np.full((500, 500, 3), [20, 150, 40], dtype=np.uint8)
    return record, pose, lambda frame: image


def test_same_person_cross_view_alias_preserves_exact_native_observation_and_body_guard():
    record, pose, reader = alias_fixture()
    before = deepcopy((record, pose))
    assert pose_direction_evidence(record, pose)["prefix_stop_reason"] == "source_track_switch"
    derived, view, audit = alias_guard_inputs(record, pose, "unused", frame_reader=reader)
    assert (record, pose) == before
    assert audit["applied"] and audit["aliases"] == {"8": 7}
    assert audit["identity_verified"] is False
    assert audit["reference_labels_used"] is False
    assert len(audit["pairs"][0]["support"]) >= 3
    assert all(len(item["source_pixels_sha256"]) == 64 for item in audit["pairs"][0]["support"])
    assert derived["selected_candidate"]["native_source_track_id_by_frame"]["69"] == 8
    assert derived["selected_candidate"]["source_track_id_by_frame"]["69"] == 7
    assert len(view["tracks"]) == 1
    assert len(view["tracks"][0]["observations"]) == 18
    assert view["tracks"][0]["observations"][-1]["native_source_track_id"] == 8
    evidence = pose_direction_evidence(derived, view)
    assert evidence["supported"] and evidence["direction"] == "image_left"


def test_late_overlap_cannot_erase_earlier_separate_people_evidence():
    record, pose, reader = alias_fixture()
    earlier = pose["tracks"][1]["observations"][0]
    earlier["bbox"] = [260, 60, 340, 130]
    earlier["keypoints"] = [[x+200, y] for x, y in earlier["keypoints"]]
    derived, view, audit = alias_guard_inputs(record, pose, "unused", frame_reader=reader)
    assert not audit["applied"]
    assert audit["pairs"][0]["separated_counterexample_frames"] == [55]
    assert audit["pairs"][0]["reason"] == "simultaneously_separated_people_counterexample"
    assert not pose_direction_evidence(derived, view)["supported"]


@pytest.mark.parametrize("case", ["two_frames", "same_view", "low_torso_confidence", "wrong_torso"])
def test_weak_duplicate_evidence_does_not_clear_source_switch(case):
    record, pose, reader = alias_fixture()
    observations = pose["tracks"][1]["observations"]
    if case == "two_frames":
        pose["tracks"][1]["observations"] = [o for o in observations if o["frame"] in (68, 69)]
    for observation in observations:
        if case == "same_view":
            observation.pop("source_roi")
        elif case == "low_torso_confidence":
            observation["keypoint_confidences"][5] = .2
        elif case == "wrong_torso":
            for joint in (5, 6, 11, 12):
                observation["keypoints"][joint][0] += 25
    _, _, audit = alias_guard_inputs(record, pose, "unused", frame_reader=reader)
    assert not audit["applied"]
    assert audit["pairs"][0]["reason"] == "insufficient_local_cross_view_evidence"


def test_known_kicker_is_never_aliased_to_goalkeeper():
    record, pose, reader = alias_fixture()
    pose["selected_track_id"] = 8
    _, _, audit = alias_guard_inputs(record, pose, "unused", frame_reader=reader)
    assert not audit["applied"]
    assert audit["pairs"][0]["reason"] == "selected_kicker_cannot_alias_keeper"


def test_alias_does_not_bridge_an_unsupported_third_source():
    record, pose, reader = alias_fixture()
    record["selected_candidate"]["source_track_ids"].append(9)
    record["selected_candidate"]["source_track_id_by_frame"]["72"] = 9
    third = deepcopy(pose["tracks"][1])
    third["track_id"] = 9
    pose["tracks"].append(third)
    derived, view, audit = alias_guard_inputs(record, pose, "unused", frame_reader=reader)
    assert audit["aliases"] == {"8": 7}
    assert derived["selected_candidate"]["source_track_id_by_frame"]["72"] == 9
    assert pose_direction_evidence(derived, view)["prefix_stop_reason"] == "source_track_switch"
