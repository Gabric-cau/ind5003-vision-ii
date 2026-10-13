from copy import deepcopy

import numpy as np
import pytest

from src.fresh_results import bind_stage, compact_stage, write_json
from src.kicking_foot_strike_window import kicking_foot_strike_window
from src.mediapipe_shoe_evidence import MIRROR_PAIRS, coco_samples, restore_landmarks, shoe_ball_distances
from src.shoe_pose33_fusion import SCHEMA, artifact, recompute_shoe_region, select_candidate, validate_fused_ball_contract


@pytest.mark.parametrize("original,shoe,expected,reason", [
    ("left", "unknown", "left", "v2_retained_shoe_unavailable_or_abstained"),
    ("right", "unknown", "right", "v2_retained_shoe_unavailable_or_abstained"),
    ("unknown", "left", "left", "shoe_region_fallback_for_v2_unknown"),
    ("unknown", "right", "right", "shoe_region_fallback_for_v2_unknown"),
    ("left", "left", "left", "v2_and_shoe_region_agree"),
    ("right", "right", "right", "v2_and_shoe_region_agree"),
    ("left", "right", "unknown", "v2_and_shoe_region_conflict"),
    ("right", "left", "unknown", "v2_and_shoe_region_conflict"),
    ("unknown", "unknown", "unknown", "both_methods_unavailable_or_abstained"),
])
def test_frozen_selection_preserves_old_candidates_and_abstains_on_conflicts(original, shoe, expected, reason):
    assert select_candidate(original, shoe) == (expected, reason)


def fixture(tmp_path):
    source = dict(clip_id="event", clip_sha256="source", fps=30., frame_count=60)
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, dict(clips=[source]))
    manifest_hash = artifact(manifest_path)["sha256"]
    pose = dict(tracks=[dict(track_id=1, observations=[])])
    pose_path = tmp_path / "pose.json"
    write_json(pose_path, pose)
    pose_hash = artifact(pose_path)["sha256"]
    identity = dict(source_sha256="source", pose_sha256=pose_hash, source_manifest_sha256=manifest_hash)
    common = dict(clip_id="event", source_mapping=source, inference_identity=identity,
                  human_verified=False, reference_labels_used_for_prediction=False)
    stationary = [dict(frame=frame, xy=[100., 220.], detection_id=f"{frame}:ball") for frame in (18, 19, 20)]
    departing = [dict(frame=frame, xy=[60., 200.], detection_id=f"{frame}:ball") for frame in (22, 23)]
    payload = dict(observed_stationary_departure_bracket_frames=[20, 22], contact_associated_track_id=1,
                   stationary_ball_observations=stationary, departure_ball_observations=departing, seed_ambiguous_frames=[])
    contact = dict(**deepcopy(common), contact=payload)
    contact_path = tmp_path / "contact.json"
    write_json(contact_path, contact)
    original = dict(**deepcopy(common), new_neural_inference=True, upstream_scene_rejections=[],
        model_task_validation=dict(actual_task="pose"),
        plan=dict(source_pixel_dimensions=[200, 240]),
        timing=dict(contact_bracket_frames=[20, 22], actor_track_id_candidate=1))
    original["inference_identity"]["contact_sha256"] = artifact(contact_path)["sha256"]
    original_path = tmp_path / "original.json"
    write_json(original_path, original)
    old = dict(**deepcopy(common), kicking_foot_candidate="right",
               original_neural_producer=artifact(original_path))
    old_path = tmp_path / "v2.json"
    write_json(old_path, old)
    points = np.full((33, 2), [100., 40.])
    points[[0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]] = [
        [100, 20], [80, 60], [120, 60], [85, 130], [115, 130], [85, 170], [115, 170],
        [90, 220], [110, 220], [88, 220], [108, 220], [95, 220], [115, 220]]
    records = []
    for frame in range(16, 25):
        for reflected in (False, True):
            raw = points.copy()
            if reflected:
                for left, right in MIRROR_PAIRS:
                    raw[[left, right]] = raw[[right, left]]
                raw[:, 0] = 199 - raw[:, 0]
            raw /= [200, 240]
            restored, confidence = restore_landmarks(raw, np.ones(33), [0, 0, 200, 240], reflected)
            records.append(dict(frame=frame, reflected=reflected, crop_xyxy=[0, 0, 200, 240],
                anchor_bbox=[80, 20, 120, 220], rival_source_boxes=[],
                timestamp_ms=round(frame / 30 * 1000), decoded_timestamp_ms=frame / 30 * 1000,
                landmarks_source_xy=restored.tolist(), landmark_confidence=confidence.tolist(),
                original_normalized_xy=raw.tolist(), original_visibility=[1.] * 33,
                original_presence=[1.] * 33, roi_identity_status="accepted"))
    normal, mirror = ([r for r in records if r["reflected"] is flag] for flag in (False, True))
    motion = kicking_foot_strike_window(coco_samples(normal, shoe_centroids=True),
        coco_samples(mirror, shoe_centroids=True), [20, 22], 30.,
        stationary_ball_observations=stationary, departure_ball_observations=departing)
    assert motion["candidate"] == "unknown"  # Both static legs must abstain despite ball proximity.
    distances = {name: shoe_ball_distances(rows, [100., 220.], 200.)
                 for name, rows in (("normal", normal), ("mirrored_inverse_mapped", mirror))}
    result = dict(clip_id="event", source_mapping=source, source_fps=30., contact_bracket_frames=[20, 22],
        source_coordinate_shoe_swing_candidate="unknown", source_coordinate_shoe_contact_region_candidate="unknown",
        evidence=dict(shoe_centroid_motion=motion), stationary_ball_reference=dict(frame=20, xy=[100., 220.]),
        observed_preparation_scale_source_px=200., shoe_ball_distances=distances,
        shoe_region_near_bracket_frames={name: {side: [19, 20, 21, 22] for side in ("left", "right")} for name in distances},
        joint_near_shoe_and_paired_pose_frames=[], local_ball_identity_ambiguous_frames=[],
        human_verified=False, reference_labels_used_for_prediction=False)
    cache_path, plan_path, records_path = [tmp_path / name for name in ("cache.json", "plan.json", "records.json")]
    roi_cache = [dict(frame=row["frame"], reflected=False, selection_reason="accepted",
                      anchor_bbox=row["anchor_bbox"], crop_xyxy=[0, 0, 200, 240]) for row in normal]
    roi_plan = [dict(row, planned_original_crop_xyxy=row["crop_xyxy"]) for row in roi_cache]
    for path, value in ((cache_path, roi_cache), (plan_path, roi_plan), (records_path, records)):
        write_json(path, value)
    protocol = dict(clip_id="event", source_sha256="source", source_manifest_sha256=manifest_hash,
        rival_pose_sha256=pose_hash, contact_sha256=artifact(contact_path)["sha256"],
        original_neural_producer_sha256=artifact(original_path)["sha256"], roi_cache_sha256=artifact(cache_path)["sha256"],
        model_sha256="model", actor_track_id_candidate=1, code_sha256={},
        settings=dict(running_mode="VIDEO", num_poses=1, roi_reference_changes_between_frames=False,
            landmark_visibility_presence_min=.5, shoe_contact_region_maximum_body_distance=.08,
            detection_confidence=.5, presence_confidence=.5, fixed_union_roi_xyxy=[0, 0, 200, 240]),
        human_verified=False, reference_labels_read=False)
    result_path, protocol_path = tmp_path / "shoe.json", tmp_path / "protocol.json"
    write_json(result_path, result)
    write_json(protocol_path, protocol)
    cohort_path, summary_path = tmp_path / "cohort.json", tmp_path / "summary.json"
    write_json(cohort_path, dict(events=39, source_manifest_sha256=manifest_hash, model_sha256="model",
                                running_mode="VIDEO", reference_labels_read=False))
    write_json(summary_path, dict(rows=[dict(clip_id="event", candidate="unknown", shoe_region_candidate="unknown",
                                            result_sha256=artifact(result_path)["sha256"])]))
    bindings = dict(v2_result=artifact(old_path), original_neural_result=artifact(original_path),
        pose=artifact(pose_path), source_manifest=artifact(manifest_path),
        shoe_cohort_protocol=artifact(cohort_path), shoe_cohort_summary=artifact(summary_path),
        shoe_result=artifact(result_path), shoe_protocol=artifact(protocol_path), shoe_poses33=artifact(records_path),
        shoe_roi_plan=artifact(plan_path), roi_cache=artifact(cache_path), contact=artifact(contact_path), shoe_code_snapshot={})
    reason = "v2_retained_shoe_unavailable_or_abstained"
    data = dict(**deepcopy(common), schema_version=SCHEMA, kicking_foot_candidate="right", reason=reason,
        temporal_evidence=dict(v2_candidate="right", shoe_region_candidate="unknown",
                               selection_reason=reason, producer_bindings=bindings))
    return dict(data=data, source=source, pose_hash=pose_hash, manifest_hash=manifest_hash,
                result=result, protocol=protocol, contact=contact, original=original, pose=pose, records=records,
                roi_plan=roi_plan, roi_cache=roi_cache)


def bind(value):
    return bind_stage("foot", value["data"], value["source"], value["pose_hash"], value["manifest_hash"])


def test_live_dual_contract_retains_v2_when_static_shoes_near_ball_abstain(tmp_path):
    value = fixture(tmp_path)
    assert bind(value)["kicking_foot_candidate"] == "right"
    assert compact_stage("foot", value["data"])["temporal_evidence"]["producer_bindings"]
    validate_fused_ball_contract(value["data"], dict(independent_contact_search=dict(selected_candidate=value["contact"]["contact"])))


def test_live_unavailable_new_producer_does_not_retract_original_candidate(tmp_path):
    value = fixture(tmp_path)
    bindings = value["data"]["temporal_evidence"]["producer_bindings"]
    value["original"]["new_neural_inference"] = False
    write_json(bindings["original_neural_result"]["path"], value["original"])
    bindings["original_neural_result"] = artifact(bindings["original_neural_result"]["path"])
    old_path = bindings["v2_result"]["path"]
    import json
    from pathlib import Path
    old = json.loads(Path(old_path).read_text())
    old["original_neural_producer"] = bindings["original_neural_result"]
    write_json(old_path, old)
    bindings["v2_result"] = artifact(old_path)
    unavailable_path = tmp_path / "unavailable.json"
    write_json(unavailable_path, dict(source_mapping=value["source"],
        upstream_neural_producer_sha256=bindings["original_neural_result"]["sha256"]))
    for name in ("shoe_result", "shoe_protocol", "shoe_poses33", "shoe_roi_plan", "roi_cache", "contact", "shoe_code_snapshot"):
        bindings.pop(name)
    bindings["shoe_unavailable"] = artifact(unavailable_path)
    assert bind(value)["kicking_foot_candidate"] == "right"


@pytest.mark.parametrize("name", ["shoe_result", "shoe_protocol", "shoe_poses33", "contact", "v2_result"])
def test_changed_actual_producer_bytes_are_rejected_by_stage_binding(tmp_path, name):
    value = fixture(tmp_path)
    path = value["data"]["temporal_evidence"]["producer_bindings"][name]["path"]
    write_json(path, dict(changed=True))
    with pytest.raises(ValueError, match="stale or missing"):
        bind(value)


@pytest.mark.parametrize("field", ["source_sha256", "rival_pose_sha256", "contact_sha256", "model_sha256"])
def test_rehashed_protocol_cannot_bind_a_different_source_pose_contact_or_model(tmp_path, field):
    value = fixture(tmp_path)
    bindings = value["data"]["temporal_evidence"]["producer_bindings"]
    value["protocol"][field] = "different"
    write_json(bindings["shoe_protocol"]["path"], value["protocol"])
    bindings["shoe_protocol"] = artifact(bindings["shoe_protocol"]["path"])
    with pytest.raises(ValueError, match="source/pose/contact/producer mismatch"):
        bind(value)


def test_candidate_field_cannot_promote_proximity_or_swing_without_region_evidence(tmp_path):
    value = fixture(tmp_path)
    value["data"]["temporal_evidence"]["shoe_region_candidate"] = "right"
    value["data"]["temporal_evidence"]["selection_reason"] = "v2_and_shoe_region_agree"
    value["data"]["reason"] = "v2_and_shoe_region_agree"
    with pytest.raises(ValueError, match="two bound producers"):
        bind(value)


def test_same_source_but_different_selected_ball_contact_cannot_enter_bundle(tmp_path):
    value = fixture(tmp_path)
    payload = deepcopy(value["contact"]["contact"])
    payload["observed_stationary_departure_bracket_frames"] = [30, 32]
    with pytest.raises(ValueError, match="bundle ball contact"):
        validate_fused_ball_contract(value["data"], dict(independent_contact_search=dict(selected_candidate=payload)))


@pytest.mark.parametrize("mutation,error", [("moving_roi", "fixed ROI"), ("duplicate_clock", "monotonic"),
    ("rival", "rival identity"), ("raw_mirror", "landmark mapping"), ("anchor", "original source actor ROI"),
    ("plan", "original source crop tasks")])
def test_raw_33_point_geometry_and_temporal_contract_is_rechecked(tmp_path, mutation, error):
    value = fixture(tmp_path)
    records = value["records"]
    if mutation == "moving_roi":
        records[2]["crop_xyxy"] = [1, 0, 200, 240]
    elif mutation == "duplicate_clock":
        records[2]["timestamp_ms"] = records[0]["timestamp_ms"]
    elif mutation == "rival":
        value["pose"]["tracks"].append(dict(track_id=2, observations=[dict(frame=16, bbox=[80, 20, 120, 220])]))
    elif mutation == "raw_mirror":
        records[1]["original_normalized_xy"][31][0] += .2
    elif mutation == "anchor":
        records[0]["anchor_bbox"] = [90, 20, 130, 220]
    else:
        value["roi_plan"][0]["anchor_bbox"] = [90, 20, 130, 220]
    with pytest.raises(ValueError, match=error):
        recompute_shoe_region(records, value["result"], value["protocol"], value["contact"], value["original"],
                             value["pose"], value["roi_plan"], value["roi_cache"])
