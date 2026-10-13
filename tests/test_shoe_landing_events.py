from copy import deepcopy

import numpy as np
import pytest

from src.shoe_landing_events import LandingConfig, propose_shoe_landings


def fixture(left_x=None, *, pan=0., fps=25.):
    left_x = left_x if left_x is not None else [80, 80, 92, 104, 116, 116, 116, 116]
    records, matrices, hashes = [], {}, {}
    for frame, x in enumerate(left_x):
        offset = pan * frame
        points = np.full((33, 2), [130. + offset, 80.])
        points[23] = [110 + 2 * frame + offset, 115]
        points[24] = [140 + 2 * frame + offset, 115]
        points[29], points[31] = [x + offset, 195], [x + 8 + offset, 195]
        points[30], points[32] = [180 + offset, 195], [188 + offset, 195]
        for reflected in (False, True):
            records.append(dict(frame=frame, reflected=reflected,
                landmarks_source_xy=points.tolist(), landmark_confidence=[1.] * 33,
                anchor_bbox=[offset, 0, 260 + offset, 200], roi_identity_status="accepted"))
        matrices[str(frame)] = [[1., 0., -offset], [0., 1., 0.], [0., 0., 1.]]
        hashes[str(frame)] = f"decoded-image-{frame}"
    return dict(records=records, fps=fps, interval_frames=[0, len(left_x) - 1],
                transforms=matrices, decoded_frame_hashes=hashes)


def test_visible_swing_then_stance_is_a_proposal_not_a_contact_or_total_count():
    result = propose_shoe_landings(**fixture())
    assert result["candidate_count"] == 1
    event = result["landings"][0]
    assert (event["frame"], event["foot"]) == (4, "left")
    assert event["stance_frames"] == [4, 5, 6]
    assert event["distinct_decoded_stance_observations"] == 3
    assert result["striking_foot"] == "unknown"
    assert result["full_approach_visible"] is False
    assert result["count_complete"] is False
    assert result["total_run_up_steps"] is None
    assert result["accuracy_validated"] is False


def test_background_camera_translation_is_removed_before_speed_detection():
    result = propose_shoe_landings(**fixture(pan=17.))
    assert [(row["frame"], row["foot"]) for row in result["landings"]] == [(4, "left")]
    static = propose_shoe_landings(**fixture([80] * 8, pan=17.))
    assert static["candidate_count"] == 0


def test_initial_stationary_support_is_not_an_extra_landing():
    result = propose_shoe_landings(**fixture([80] * 8))
    assert result["candidate_count"] == 0
    assert result["landings"] == []


def test_entire_repeated_frame_sequence_cannot_be_counted_as_observations():
    value = fixture()
    value["decoded_frame_hashes"] = {str(frame): "one-actual-image" for frame in range(8)}
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] is None
    assert result["landings"] is None
    assert len(result["paired_unique_pose_frames"]) == 1
    assert sum(row["reason"] == "repeated_decoded_pixels" for row in result["diagnostics"]) == 7


def test_repeated_stance_images_do_not_inflate_three_distinct_observations():
    value = fixture()
    for frame in range(5, 8):
        value["decoded_frame_hashes"][str(frame)] = value["decoded_frame_hashes"]["4"]
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] == 0


@pytest.mark.parametrize("missing", ["2", "4", "5"])
def test_camera_chain_gap_cannot_be_bridged_into_a_transition(missing):
    value = fixture()
    value["transforms"][missing] = None
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] in (None, 0)
    assert any(row["reason"] == "missing_measurable_camera_transform" for row in result["diagnostics"])


def test_payload_pair_rejection_overrides_a_supplied_matrix():
    value = fixture()
    value["transforms"] = dict(transforms=value["transforms"], pairs=[dict(frame=5, accepted=False, reference_chain_valid=False)])
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] == 0


def test_missing_mirror_and_low_shoe_confidence_preserve_missingness():
    value = fixture()
    value["records"] = [row for row in value["records"] if not row["reflected"]]
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] is None
    assert result["landings"] is None
    value = fixture()
    for row in value["records"]:
        row["landmark_confidence"][29] = .1
        row["landmark_confidence"][30] = .1
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] is None


def test_landmarks_from_different_decoded_pixels_cannot_enter_a_proposal():
    value = fixture()
    for row in value["records"]:
        row["frame_pixel_sha256"] = "some-other-image"
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] is None
    assert all(row["reason"] == "pose_decoded_pixel_hash_disagreement" for row in result["diagnostics"])


@pytest.mark.parametrize("mutation", ["missing", "outside"])
def test_hip_evidence_must_be_confident_and_belong_to_observed_actor_box(mutation):
    value = fixture()
    for row in value["records"]:
        if mutation == "missing":
            row["landmark_confidence"][23] = .1
        else:
            row["landmarks_source_xy"][23] = [400, 115]
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] is None


def test_too_short_high_rate_window_is_missing_evidence_not_zero_steps():
    assert propose_shoe_landings(**fixture(fps=1000.))["candidate_count"] is None


def test_normal_mirror_shoe_disagreement_cannot_create_a_paired_event():
    value = fixture()
    for row in value["records"]:
        if row["reflected"]:
            for index in [29, 31]:
                row["landmarks_source_xy"][index][0] += 40
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] == 0
    assert any(row["reason"] == "normal_mirror_shoe_disagreement" for row in result["diagnostics"])


def test_paired_coordinates_do_not_replace_independent_motion_agreement():
    value = fixture()
    for row in value["records"]:
        if row["reflected"] and row["frame"] >= 5:
            for index in [29, 31]:
                row["landmarks_source_xy"][index][0] += 3 * (row["frame"] - 4)
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] == 0
    assert result["view_proposals"]["normal"]["left"]
    assert not result["view_proposals"]["mirrored_inverse_mapped"]["left"]


def test_temporal_anatomical_channel_swap_abstains():
    value = fixture([80] * 8)
    for row in value["records"]:
        if row["frame"] >= 3:
            for left, right in [(29, 30), (31, 32)]:
                row["landmarks_source_xy"][left], row["landmarks_source_xy"][right] = (
                    row["landmarks_source_xy"][right], row["landmarks_source_xy"][left])
    result = propose_shoe_landings(**value)
    assert result["status"] == "temporal_anatomical_channel_ambiguity"
    assert result["candidate_count"] is None
    assert result["temporal_channel_ambiguous_frames"]["normal"] == [3]


def test_pose_jump_cannot_arm_a_stance_candidate():
    value = fixture([80, 80, 230, 230, 230, 230, 230, 230])
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] == 0
    assert any(row["reason"] == "pose_jump_or_scale_discontinuity" for row in result["diagnostics"])


def test_long_pose_gap_resets_prior_swing_evidence():
    value = fixture()
    value["records"] = [row for row in value["records"] if row["frame"] not in (4, 5, 6)]
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] == 0


def test_last_stationary_cutoff_excludes_later_stance_support():
    value = fixture()
    value["interval_frames"] = [0, 5]
    result = propose_shoe_landings(**value)
    assert result["candidate_count"] == 0
    assert max(result["paired_unique_pose_frames"]) == 5


def test_shoe_stability_above_actor_box_bottom_is_not_stance_evidence():
    value = fixture()
    for row in value["records"]:
        for index in [29, 31]:
            row["landmarks_source_xy"][index][1] = 100
    assert propose_shoe_landings(**value)["candidate_count"] == 0


def test_stationary_hips_do_not_support_a_run_up_landing():
    value = fixture()
    for row in value["records"]:
        row["landmarks_source_xy"][23] = [110, 115]
        row["landmarks_source_xy"][24] = [140, 115]
    assert propose_shoe_landings(**value)["candidate_count"] == 0


def test_sparse_source_frame_matrix_gap_is_not_assumed_static_camera():
    value = fixture()
    value["records"] = [row for row in value["records"] if row["frame"] != 5]
    del value["transforms"]["5"]
    assert propose_shoe_landings(**value)["candidate_count"] == 0


@pytest.mark.parametrize("mutation", ["fps", "frame", "reflected", "shape", "confidence", "bbox", "duplicate", "interval", "hashes", "transforms"])
def test_malformed_input_raises_instead_of_silently_becoming_zero(mutation):
    value = fixture()
    if mutation == "fps":
        value["fps"] = float("nan")
    elif mutation == "frame":
        value["records"][0]["frame"] = True
    elif mutation == "reflected":
        value["records"][0]["reflected"] = 0
    elif mutation == "shape":
        value["records"][0]["landmarks_source_xy"] = [[1, 2]]
    elif mutation == "confidence":
        value["records"][0]["landmark_confidence"][29] = 1.5
    elif mutation == "bbox":
        value["records"][0]["anchor_bbox"] = [0, 0, 0, 0]
    elif mutation == "duplicate":
        value["records"].append(deepcopy(value["records"][0]))
    elif mutation == "interval":
        value["interval_frames"] = [8, 0]
    elif mutation == "hashes":
        value["decoded_frame_hashes"] = []
    else:
        value["transforms"] = []
    with pytest.raises(ValueError):
        propose_shoe_landings(**value)


@pytest.mark.parametrize("setting", [dict(minimum_stance_observations=2),
    dict(minimum_swing_observations=3.5), dict(stance_speed_max_body_per_second=.6),
    dict(confidence_min=True), dict(maximum_scale_ratio=.5)])
def test_fixed_configuration_rejects_incoherent_settings(setting):
    with pytest.raises(ValueError):
        LandingConfig(**setting)
