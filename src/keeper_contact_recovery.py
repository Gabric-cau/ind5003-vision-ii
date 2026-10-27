"""Recover keeper evidence from an automatic contact without accepting ball rules."""
from __future__ import annotations

from copy import deepcopy
import math

from src.goalkeeper_observation_guard import guard_postkick_observations
from src.goalkeeper_pipeline import analyze_goalkeeper_result
from src.goalkeeper_role_association import select_goalkeeper_role


def bounded_contact(pose, contact, source):
    """Validate clocks and provenance; a trained contact stays a timing candidate."""
    if contact.get("video_sha256") != source.get("clip_sha256"):
        raise ValueError("Keeper recovery contact/source hash mismatch")
    if contact.get("manual_annotations_required") is not False:
        raise ValueError("Keeper recovery requires an automatic contact prediction")
    window = contact.get("automatic_window") or {}
    if window.get("automatic") is not True:
        raise ValueError("Keeper recovery requires an automatic contact window")
    prediction = contact.get("prediction") or {}
    frame, seconds = prediction.get("frame"), prediction.get("time_seconds")
    if frame is None or seconds is None:
        return None, "automatic_contact_unavailable"
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise ValueError("Contact frame must be an integer")
    fps, source_fps = float(pose["fps"]), float(source["fps"])
    if not all(math.isfinite(value) and value > 0 for value in (fps, source_fps)):
        raise ValueError("Invalid keeper source clock")
    if not math.isclose(fps, source_fps, rel_tol=1e-5):
        raise ValueError("Keeper source/pose clocks differ")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        raise ValueError("Contact seconds must be finite")
    if abs(frame / fps - seconds) > 0.51 / fps:
        raise ValueError("Contact frame/time disagree")
    if any(pose.get(key) is None for key in ("shot_start_frame", "shot_end_frame")):
        return None, "no_supported_pose_shot"
    start = max(0, int(pose["shot_start_frame"]), int(window["start_frame"]))
    end = min(int(source["frame_count"]), int(pose["shot_end_frame"]), int(window["end_frame_exclusive"]))
    if not start <= frame < end:
        return None, "automatic_contact_outside_pose_shot"
    # A full one-second observation window is required by the retained guard.
    if frame + round(fps) >= min(int(source["frame_count"]), int(pose["shot_end_frame"])):
        return None, "automatic_contact_has_no_full_post_window"
    return frame, "source_bound_trained_contact_candidate"


def recover_keeper(pose, contact, source, video_path, geometries, *, allow_cross_view_aliases=False):
    """Retain role, camera and observation guards while removing the ball gate."""
    frame, reason = bounded_contact(pose, contact, source)
    output = {
        "status": "abstain", "dive_direction": "unknown", "dive_height": "unknown",
        "reason": reason, "human_verified": False, "accuracy_validated": False,
        "reference_labels_used_for_prediction": False,
        "recovery_policy": "automatic_contact_independent_keeper_v1",
        "new_keeper_neural_inference": False,
        "contact_timing_is_human_truth": False,
    }
    if frame is None:
        return output
    recovered_pose = deepcopy(pose)
    recovered_pose["video_path"] = str(video_path)
    recovered_pose["kick_frame"] = frame
    role = select_goalkeeper_role(recovered_pose, geometries, frame)
    output.update(role=role, kick_frame=frame)
    if role.get("seed_track_id") is None:
        output["reason"] = "no_supported_goal_line_identity_at_automatic_contact"
        return output
    keeper = analyze_goalkeeper_result(
        recovered_pose, frame, seed_track_id=role["seed_track_id"],
        pre_window_sec=1.5, post_window_sec=1.0, tracking_policy="appearance_bounded",
        use_field_mask=True, kick_frame_confidence="uncalibrated_model_candidate",
        kick_frame_source="source_bound_trained_contact_candidate",
    )
    keeper["goal_identity_evidence"] = role
    alias_audit = None
    if allow_cross_view_aliases:
        from src.keeper_view_aliases import alias_guard_inputs
        keeper, guard_pose, alias_audit = alias_guard_inputs(keeper, recovered_pose, video_path)
    else:
        guard_pose = recovered_pose
    guarded = guard_postkick_observations(keeper, policy="pose_direction_v3", pose=guard_pose)
    if alias_audit is not None:
        guarded["cross_view_alias_evidence"] = alias_audit
    guarded.update({key: value for key, value in output.items()
                    if key not in ("status", "dive_direction", "dive_height", "reason")})
    guarded.update(
        direction_coordinate_system="source_image_x_right",
        goalkeeper_relative_direction="unknown", identity_verified=False,
        reason="keeper_replayed_with_retained_identity_camera_and_observation_guards",
    )
    return guarded


def guard_recovered_prediction(keeper, prediction):
    """Separate an observed early dive from complete height-head evidence."""
    result = deepcopy(prediction)
    body = keeper.get("pose_direction_evidence") or {}
    motion = keeper.get("dive_direction_evidence") or {}
    actual_dive = body.get("supported") is True
    window_supported = (keeper.get("dive_observability") or {}).get("window_supported") is True
    selected = keeper.get("selected_candidate") or {}
    source_ids = selected.get("source_track_ids") or []
    seed = (keeper.get("goal_identity_evidence") or {}).get("seed_track_id")
    identity_continuous = seed is not None and set(source_ids) == {seed}
    result["lateral_motion_supported"] = motion.get("supported") is True
    result["actual_dive_evidence_supported"] = actual_dive
    result["height_observation_window_supported"] = window_supported
    result["height_identity_continuity_supported"] = identity_continuous
    if not actual_dive and result.get("status") != "abstain":
        result.update(status="abstain", direction="unknown", height="unknown",
                      dive_direction="unknown", dive_height="unknown")
        result["reasons"] = list(dict.fromkeys(list(result.get("reasons", [])) +
                                             ["no_observed_sustained_dive_phase"]))
    elif actual_dive and not (window_supported and identity_continuous):
        result.update(height="unknown", dive_height="unknown")
        reasons = []
        if not window_supported:
            reasons.append("incomplete_recovered_height_window")
        if not identity_continuous:
            reasons.append("unverified_identity_switch_in_height_features")
        result["reasons"] = list(dict.fromkeys(list(result.get("reasons", []))+reasons))
    return result


def automatic_goal_geometry(video_path, pose, frame):
    """Decode only the pre-contact role window if saved goal geometry is absent."""
    import cv2
    from src.goal_geometry import detect_goal_frame

    start = max(int(pose["shot_start_frame"]), frame - round(0.5 * float(pose["fps"])))
    cap, geometry = cv2.VideoCapture(str(video_path)), {}
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode keeper recovery source")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for current in range(start, frame):
            ok, image = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != current + 1:
                raise ValueError("Keeper source frame clock mismatch")
            geometry[current] = detect_goal_frame(image)
    finally:
        cap.release()
    return geometry
