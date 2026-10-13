"""Conservative foot selection with live, dual-producer source contracts."""
import json
from pathlib import Path

import numpy as np

from src.final_results import sha256
from src.kicking_foot_strike_window import kicking_foot_strike_window
from src.mediapipe_shoe_evidence import (
    SHOE_PAIRS, coco_samples, restore_landmarks, shoe_ball_distances, source_body_box,
)
from src.pose_refinement import select_anchored_detection


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fresh-foot-pose33-fusion-v1"


def select_candidate(original, shoe_region):
    if original not in {"left", "right", "unknown"} or shoe_region not in {"left", "right", "unknown"}:
        raise ValueError("Foot producers must explicitly decide or abstain")
    if original != "unknown" and shoe_region != "unknown":
        return ((original, "v2_and_shoe_region_agree") if original == shoe_region
                else ("unknown", "v2_and_shoe_region_conflict"))
    if original != "unknown":
        return original, "v2_retained_shoe_unavailable_or_abstained"
    if shoe_region != "unknown":
        return shoe_region, "shoe_region_fallback_for_v2_unknown"
    return "unknown", "both_methods_unavailable_or_abstained"


def artifact(path):
    path = Path(path).resolve()
    try:
        recorded = str(path.relative_to(ROOT))
    except ValueError:
        recorded = str(path)
    return {"path": recorded, "sha256": sha256(path)}


def load_artifact(binding, name):
    path = Path(binding["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or sha256(path) != binding.get("sha256"):
        raise ValueError(f"Foot fusion stale or missing {name} artifact")
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(data, source, pose_hash, manifest_hash, label):
    identity = data.get("inference_identity", {})
    if (data.get("clip_id") != source["clip_id"]
            or data.get("source_mapping") != source
            or identity.get("source_sha256") != source["clip_sha256"]
            or identity.get("pose_sha256") != pose_hash
            or identity.get("source_manifest_sha256") != manifest_hash):
        raise ValueError(f"Foot fusion {label} source/pose/manifest mismatch")
    if data.get("human_verified") is not False or data.get("reference_labels_used_for_prediction") is not False:
        raise ValueError(f"Foot fusion {label} must remain source-only and unverified")


def recompute_shoe_region(records, result, protocol, contact, producer, pose, roi_plan, roi_cache):
    """Reconstruct the frozen rule and recheck the accepted actor/crop geometry."""
    settings = protocol["settings"]
    if (settings.get("running_mode") != "VIDEO" or settings.get("num_poses") != 1
            or settings.get("roi_reference_changes_between_frames") is not False
            or settings.get("landmark_visibility_presence_min") != .5
            or settings.get("shoe_contact_region_maximum_body_distance") != .08
            or settings.get("detection_confidence") != .5 or settings.get("presence_confidence") != .5):
        raise ValueError("Foot fusion shoe protocol differs from frozen guarded rule")
    payload = contact.get("contact", contact)
    bracket = producer["timing"]["contact_bracket_frames"]
    if (result["contact_bracket_frames"] != bracket
            or payload.get("observed_stationary_departure_bracket_frames") != bracket
            or protocol["actor_track_id_candidate"] != payload["contact_associated_track_id"]
            or protocol["actor_track_id_candidate"] != producer["timing"]["actor_track_id_candidate"]):
        raise ValueError("Foot fusion shoe contact/actor mismatch")
    if (producer.get("upstream_scene_rejections") != []
            or producer.get("model_task_validation", {}).get("actual_task") != "pose"):
        raise ValueError("Foot fusion shoe producer lacks scene and pose-model guards")
    source, fps = result["source_mapping"], result["source_fps"]
    if fps != source["fps"]:
        raise ValueError("Foot fusion shoe decoder/source clock mismatch")
    normal = [row for row in records if row["reflected"] is False]
    mirror = [row for row in records if row["reflected"] is True]
    frames = [row["frame"] for row in normal]
    if (not frames or frames != sorted(set(frames))
            or [row["frame"] for row in mirror] != frames
            or len(normal) + len(mirror) != len(records)
            or any(not isinstance(frame, int) or isinstance(frame, bool)
                   or not 0 <= frame < source["frame_count"] for frame in frames)):
        raise ValueError("Foot fusion shoe frames must be unique paired source frames")
    fixed_roi = settings["fixed_union_roi_xyxy"]
    original_tasks = [row for row in roi_cache if row["reflected"] is False
                      and row.get("selection_reason") == "accepted"]
    if [row["frame"] for row in original_tasks] != frames:
        raise ValueError("Foot fusion shoe frames differ from the original accepted ROI tasks")
    bounds = np.asarray([row["crop_xyxy"] for row in original_tasks])
    expected_roi = [int(bounds[:, 0].min()), int(bounds[:, 1].min()),
                    int(bounds[:, 2].max()), int(bounds[:, 3].max())]
    width, height = producer["plan"]["source_pixel_dimensions"]
    expected_plan = [dict(row, planned_original_crop_xyxy=row["crop_xyxy"], crop_xyxy=expected_roi)
                     for row in original_tasks]
    if (fixed_roi != expected_roi or roi_plan != expected_plan
            or not (0 <= fixed_roi[0] < fixed_roi[2] <= width and 0 <= fixed_roi[1] < fixed_roi[3] <= height)):
        raise ValueError("Foot fusion shoe fixed ROI plan differs from original source crop tasks")
    anchors = {row["frame"]: row["anchor_bbox"] for row in original_tasks}
    expected_rivals = {}
    for track in pose["tracks"]:
        if track["track_id"] == protocol["actor_track_id_candidate"]:
            continue
        for observation in track["observations"]:
            if observation["frame"] in frames:
                expected_rivals.setdefault(observation["frame"], []).append(observation["bbox"])
    for view in (normal, mirror):
        timestamps = [row["timestamp_ms"] for row in view]
        if timestamps != sorted(set(timestamps)):
            raise ValueError("Foot fusion shoe VIDEO timestamps must be monotonic")
        for row in view:
            if (row["crop_xyxy"] != fixed_roi or row["timestamp_ms"] != round(row["frame"] / fps * 1000)
                    or abs(row["decoded_timestamp_ms"] - row["frame"] / fps * 1000) > max(1., 100 / fps)
                    or row.get("rival_source_boxes") != expected_rivals.get(row["frame"], [])):
                raise ValueError("Foot fusion shoe fixed ROI, source clock or rival identity changed")
            if row["anchor_bbox"] != anchors[row["frame"]]:
                raise ValueError("Foot fusion shoe anchor differs from the original source actor ROI")
            if row.get("landmarks_source_xy") is None:
                continue
            points, confidence = restore_landmarks(row["original_normalized_xy"],
                np.minimum(row["original_visibility"], row["original_presence"]), fixed_roi, row["reflected"])
            if (not np.allclose(points, row["landmarks_source_xy"], atol=1e-9)
                    or not np.allclose(confidence, row["landmark_confidence"], atol=1e-9)):
                raise ValueError("Foot fusion shoe source landmark mapping changed")
            body_box = source_body_box(points, confidence)
            selected, reason = (select_anchored_detection([dict(bbox=body_box)], row["anchor_bbox"],
                rivals=expected_rivals.get(row["frame"], [])) if body_box else (None, "missing_torso"))
            if selected is None or reason != "accepted" or row.get("roi_identity_status") != "accepted":
                raise ValueError("Foot fusion shoe pose does not match the original actor")
    first, departure = bracket
    stationary, departing = payload["stationary_ball_observations"], payload["departure_ball_observations"]
    evidence = kicking_foot_strike_window(coco_samples(normal, shoe_centroids=True),
        coco_samples(mirror, shoe_centroids=True), bracket, fps,
        stationary_ball_observations=stationary, departure_ball_observations=departing)
    if evidence != result["evidence"]["shoe_centroid_motion"]:
        raise ValueError("Foot fusion shoe motion does not reproduce from poses33")
    target = next(row["xy"] for row in stationary if row["frame"] == first)
    scale = float(np.median([row["anchor_bbox"][3] - row["anchor_bbox"][1]
                             for row in normal if row["frame"] < first]))
    distances = {name: shoe_ball_distances(rows, target, scale)
                 for name, rows in (("normal", normal), ("mirrored_inverse_mapped", mirror))}
    if (result["stationary_ball_reference"] != dict(frame=first, xy=target)
            or result["observed_preparation_scale_source_px"] != scale
            or result["shoe_ball_distances"] != distances):
        raise ValueError("Foot fusion shoe distances do not reproduce from poses/contact")
    near = {name: {side: [row["frame"] for row in rows if first - 1 <= row["frame"] <= departure
                            and row[side]["confident"] and row[side]["stationary_point_distance_body"] <= .08]
                   for side in SHOE_PAIRS} for name, rows in distances.items()}
    swing = evidence["candidate"]
    accepted = {row["frame"] for row in evidence.get("accepted_paired_frames", [])}
    joint = (sorted(set(near["normal"][swing]) & set(near["mirrored_inverse_mapped"][swing]) & accepted)
             if swing != "unknown" else [])
    region = swing if joint else "unknown"
    ambiguous = sorted(set(producer.get("local_seed_identity_ambiguous_frames", [])
        + [frame for frame in payload.get("seed_ambiguous_frames", []) if first - 5 <= frame <= departure]))
    if ambiguous:
        swing, region = "unknown", "unknown"
    if (result["source_coordinate_shoe_swing_candidate"] != swing
            or result["source_coordinate_shoe_contact_region_candidate"] != region
            or result["shoe_region_near_bracket_frames"] != near
            or result["joint_near_shoe_and_paired_pose_frames"] != joint
            or result["local_ball_identity_ambiguous_frames"] != ambiguous):
        raise ValueError("Foot fusion shoe candidate or contact identity gate does not reproduce")
    return region


def validate_fused_foot(data, source, pose_hash, manifest_hash):
    evidence = data["temporal_evidence"]
    bindings = evidence["producer_bindings"]
    old = load_artifact(bindings["v2_result"], "V2 result")
    original = load_artifact(bindings["original_neural_result"], "original neural result")
    pose = load_artifact(bindings["pose"], "pose")
    manifest = load_artifact(bindings["source_manifest"], "source manifest")
    if (bindings["pose"]["sha256"] != pose_hash or bindings["source_manifest"]["sha256"] != manifest_hash
            or [row for row in manifest["clips"] if row["clip_id"] == source["clip_id"]] != [source]):
        raise ValueError("Foot fusion actual pose/manifest mismatch")
    _identity(old, source, pose_hash, manifest_hash, "V2")
    _identity(original, source, pose_hash, manifest_hash, "original neural")
    if old["original_neural_producer"]["sha256"] != bindings["original_neural_result"]["sha256"]:
        raise ValueError("Foot fusion V2 neural producer mismatch")
    cohort = load_artifact(bindings["shoe_cohort_protocol"], "shoe cohort protocol")
    summary = load_artifact(bindings["shoe_cohort_summary"], "shoe cohort summary")
    if (cohort.get("source_manifest_sha256") != manifest_hash or cohort.get("events") != 39
            or cohort.get("running_mode") != "VIDEO" or cohort.get("reference_labels_read") is not False):
        raise ValueError("Foot fusion shoe cohort source/protocol mismatch")
    row = [row for row in summary["rows"] if row["clip_id"] == source["clip_id"]]
    if len(row) != 1:
        raise ValueError("Foot fusion shoe cohort membership mismatch")
    new = "unknown"
    if "shoe_result" in bindings:
        result = load_artifact(bindings["shoe_result"], "shoe result")
        protocol = load_artifact(bindings["shoe_protocol"], "shoe protocol")
        records = load_artifact(bindings["shoe_poses33"], "shoe poses33")
        contact = load_artifact(bindings["contact"], "contact")
        roi_plan = load_artifact(bindings["shoe_roi_plan"], "shoe ROI plan")
        roi_cache = load_artifact(bindings["roi_cache"], "original ROI cache")
        expected = dict(source_sha256=source["clip_sha256"], source_manifest_sha256=manifest_hash,
            rival_pose_sha256=pose_hash, contact_sha256=bindings["contact"]["sha256"],
            original_neural_producer_sha256=bindings["original_neural_result"]["sha256"],
            roi_cache_sha256=bindings["roi_cache"]["sha256"], model_sha256=cohort["model_sha256"])
        if any(protocol.get(key) != value for key, value in expected.items()):
            raise ValueError("Foot fusion shoe source/pose/contact/producer mismatch")
        if (result.get("clip_id") != source["clip_id"] or result.get("source_mapping") != source
                or protocol.get("clip_id") != source["clip_id"]
                or result.get("reference_labels_used_for_prediction") is not False
                or protocol.get("reference_labels_read") is not False
                or result.get("human_verified") is not False or protocol.get("human_verified") is not False
                or row[0].get("result_sha256") != bindings["shoe_result"]["sha256"]
                or original["inference_identity"].get("contact_sha256") != bindings["contact"]["sha256"]):
            raise ValueError("Foot fusion shoe result is not its bound source-only cohort producer")
        _identity(contact, source, pose_hash, manifest_hash, "contact")
        for name, bound in bindings["shoe_code_snapshot"].items():
            path = Path(bound["path"])
            path = path if path.is_absolute() else ROOT / path
            if sha256(path) != bound["sha256"] or protocol["code_sha256"].get(name) != bound["sha256"]:
                raise ValueError("Foot fusion shoe code snapshot mismatch")
        new = recompute_shoe_region(records, result, protocol, contact, original, pose, roi_plan, roi_cache)
        if (row[0]["shoe_region_candidate"] != new
                or row[0]["candidate"] != result["source_coordinate_shoe_swing_candidate"]):
            raise ValueError("Foot fusion cohort summary differs from actual shoe producer")
    else:
        unavailable = load_artifact(bindings["shoe_unavailable"], "shoe unavailable")
        if (original.get("new_neural_inference") is not False
                or unavailable.get("source_mapping") != source
                or unavailable.get("upstream_neural_producer_sha256") != bindings["original_neural_result"]["sha256"]
                or row[0]["shoe_region_candidate"] != "unknown"):
            raise ValueError("Foot fusion unavailable shoe producer mismatch")
    selected, reason = select_candidate(old["kicking_foot_candidate"], new)
    if (evidence["v2_candidate"] != old["kicking_foot_candidate"]
            or evidence["shoe_region_candidate"] != new or evidence["selection_reason"] != reason
            or data["kicking_foot_candidate"] != selected or data.get("reason") != reason):
        raise ValueError("Foot fusion selection does not match its two bound producers")
    return contact if "shoe_result" in bindings else None


def validate_fused_ball_contract(data, ball):
    bindings = data["temporal_evidence"]["producer_bindings"]
    if "contact" in bindings:
        contact = load_artifact(bindings["contact"], "contact")
        if (ball.get("independent_contact_search") or {}).get("selected_candidate") != contact.get("contact", contact):
            raise ValueError("Foot fusion contact differs from the bundle ball contact")
