"""Separate early direction evidence from whole-window dive observability.

Coverage describes observation completeness, never correct identity. A short
prefix cannot establish height or a whole-window ``center`` conclusion.
"""
from copy import deepcopy
import math
from src.goalkeeper_direction_evidence import early_direction_evidence
from src.goalkeeper_pose_direction import pose_direction_evidence


def guard_postkick_observations(record, *, duration_sec=1.0, min_fraction=.80,
                               boundary_tolerance_sec=.10, max_gap_sec=.20,
                               policy='separate_direction_v2', pose=None):
    if policy not in ('separate_direction_v2','legacy_full_window_v1','pose_direction_v3'):
        raise ValueError('Unknown observation policy')
    if duration_sec <= 0 or not 0 <= min_fraction <= 1:
        raise ValueError("Invalid observation policy")
    out = deepcopy(record)
    fps = float(record.get("fps") or 30.0)
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be positive and finite")
    kick = int(record["kick_frame"])
    end = kick + round(duration_sec * fps)
    selected = record.get("selected_candidate") or {}
    frames = sorted({round(float(r[0]) * fps)
                     for r in selected.get("post_trajectory", [])
                     if r and math.isfinite(float(r[0]))
                     and kick <= round(float(r[0]) * fps) <= end})
    expected = end - kick + 1
    fraction = len(frames) / expected
    gaps = [b-a for a,b in zip(frames, frames[1:])]
    reasons = []
    tolerance = round(boundary_tolerance_sec * fps)
    if not frames:
        reasons.append("no_postkick_observations")
    else:
        if frames[0] - kick > tolerance:
            reasons.append("missing_dive_onset")
        if end - frames[-1] > tolerance:
            reasons.append("right_censored_dive_window")
        if max(gaps, default=0) > round(max_gap_sec * fps):
            reasons.append("postkick_observation_gap")
    if fraction < min_fraction:
        reasons.append("insufficient_observed_frame_fraction")
    out["candidate_dive_direction"] = record.get("dive_direction", "unknown")
    out["candidate_dive_height"] = record.get("dive_height", "unknown")
    out["pre_guard_auto_accepted"] = bool(record.get("auto_accepted"))
    out["dive_observability"] = {
        "policy": "fixed_1s_postkick_v1", "duration_sec": duration_sec,
        "min_fraction": min_fraction, "boundary_tolerance_sec": boundary_tolerance_sec,
        "max_gap_sec": max_gap_sec, "expected_frames": expected,
        "observed_frames": len(frames), "observed_fraction": round(fraction, 4),
        "first_observed_frame": frames[0] if frames else None,
        "last_observed_frame": frames[-1] if frames else None,
        "max_gap_frames": max(gaps, default=0), "reasons": reasons,
        "window_supported": not reasons, "identity_verified": False,
    }
    if reasons:
        out["dive_direction"] = "unknown"
        out["dive_height"] = "unknown"
        out["dive_review_required"] = True
        out["review_required"] = True
        out["auto_accepted"] = False
        out["qa_warnings"] = list(dict.fromkeys(list(record.get("qa_warnings", [])) + reasons))
    if policy in ('separate_direction_v2','pose_direction_v3'):
        evidence=early_direction_evidence(record,max_gap_sec=max_gap_sec,
                                          onset_tolerance_sec=boundary_tolerance_sec)
        candidate=record.get('dive_direction','unknown')
        supported_early=evidence['supported'] and candidate in ('image_left','image_right') and evidence['direction']==candidate
        if evidence['supported'] and not supported_early:
            evidence['reasons'].append('early_motion_disagrees_with_legacy_candidate')
        if reasons and supported_early:
            out['dive_direction']=evidence['direction']
        out['dive_feature_policy']=policy
        out['dive_direction_evidence']=evidence
        out['dive_direction_supported']=bool(not reasons and candidate!='unknown' or reasons and supported_early)
        out['dive_direction_support_status']=('early_motion_candidate' if reasons and supported_early else
            'full_window_legacy_candidate' if not reasons and candidate!='unknown' else 'insufficient_direction_evidence')
        out['dive_direction_review_required']=bool(reasons or record.get('dive_review_required',True))
        out['dive_height_window_supported']=not reasons
        out['dive_height_review_required']=bool(reasons or record.get('dive_review_required',True))
        out['goalkeeper_relative_direction']='unknown'
        out['direction_coordinate_system']='source_image_x_right'
        # Whole-record QA remains rejected if height/full-window evidence is missing.
    if policy == 'pose_direction_v3':
        out['dive_feature_policy']=policy
        body=pose_direction_evidence(record,pose)
        out['pose_direction_evidence']=body
        out['pose_vs_motion_conflict']=bool(body['supported'] and out['dive_direction'] not in ('unknown',body['direction']))
        # Actual body transition takes priority over box-center travel, including
        # pre-steps in the opposite direction. Conflict and prior output stay explicit.
        out['pre_pose_dive_direction']=out['dive_direction']
        if body['supported']:
            out['dive_direction']=body['direction']
            out['dive_direction_supported']=True
            out['dive_direction_support_status']='lateral_body_pose_candidate'
            out['dive_direction_review_required']=True
        if out['pose_vs_motion_conflict']:
            out['dive_direction_review_required']=True
    out["human_verified"] = False
    out["accuracy_validated"] = False
    return out
