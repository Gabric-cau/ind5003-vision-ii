from copy import deepcopy

from src.shoe_landing_events import propose_shoe_landings
from src.shoe_pivot_support import propose_pivot_landings
from tests.test_shoe_landing_events import fixture


def test_fixed_heel_support_survives_toe_motion_without_counting_twice():
    value = fixture()
    for row in value["records"]:
        if row["frame"] >= 5:
            row["landmarks_source_xy"][31][0] += 3 * (row["frame"] - 4)
    assert propose_shoe_landings(**value)["candidate_count"] == 0
    result = propose_pivot_landings(**value)
    assert result["candidate_count"] == 1
    assert result["landings"][0]["support_landmark"] == "heel"
    assert result["landings"][0]["stance_frames"] == [4, 5, 6, 7]


def test_rigid_stance_is_not_counted_as_heel_plus_toe_plus_baseline():
    baseline = propose_shoe_landings(**fixture())
    result = propose_pivot_landings(**fixture())
    assert result["landings"] == baseline["landings"]
    assert result["pivot_added_count"] == 0


def test_stationary_pose_and_missing_mirror_do_not_create_pivot_landings():
    assert propose_pivot_landings(**fixture([80] * 8))["candidate_count"] == 0
    value = deepcopy(fixture())
    value["records"] = [row for row in value["records"] if not row["reflected"]]
    assert propose_pivot_landings(**value)["candidate_count"] is None


def test_normal_heel_and_mirrored_toe_cannot_substitute_for_same_point_support():
    value = fixture()
    for row in value["records"]:
        if row["frame"] >= 5:
            index = 29 if row["reflected"] else 31
            row["landmarks_source_xy"][index][0] += 3 * (row["frame"] - 4)
    assert propose_pivot_landings(**value)["candidate_count"] == 0
