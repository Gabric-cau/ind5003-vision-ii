"""Robust temporal kicking-foot hypotheses from saved anatomical pose channels.

Motion, ankle-ball proximity and actual shoe-ball contact are distinct evidence.
This module never reads reviewer labels or assumes a player's dominant foot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class EvidenceParameters:
    confidence_min: float = 0.4
    precontact_seconds: float = 0.13
    postdeparture_seconds: float = 0.20
    maximum_pair_gap_seconds: float = 0.07
    minimum_pairs: int = 4
    motion_ratio_min: float = 1.25
    pair_support_fraction_min: float = 0.65
    minimum_motion_body_heights_per_second: float = 0.3
    mirror_relative_error_max: float = 0.15
    channel_jump_body_heights_min: float = 0.12
    crossed_assignment_cost_ratio_max: float = 0.55
    ball_distance_body_heights_max: float = 0.25
    ball_distance_margin_body_heights_min: float = 0.05
    ball_approach_body_heights_min: float = 0.03
    minimum_contact_samples: int = 3


def _validate_samples(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in samples:
        frame = item.get("frame")
        if not isinstance(frame, (int, np.integer)) or isinstance(frame, bool) or frame < 0:
            raise ValueError("Pose frame IDs must be nonnegative integers")
        points = np.asarray(item.get("keypoints"), float)
        confidence = np.asarray(item.get("confidence", item.get("keypoint_confidences")), float)
        box = np.asarray(item.get("bbox"), float)
        if points.shape != (17, 2) or confidence.shape != (17,) or box.shape != (4,):
            raise ValueError("Each sample needs COCO-17 xy/confidence and xyxy box")
        if not np.isfinite(box).all() or box[3] <= box[1] or box[2] <= box[0]:
            raise ValueError("Pose boxes must have positive finite dimensions")
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("Keypoint confidence must be finite and between zero and one")
        rows.append({"frame": int(frame), "points": points, "confidence": confidence,
                     "height": float(box[3] - box[1])})
    rows.sort(key=lambda row: row["frame"])
    if len({row["frame"] for row in rows}) != len(rows):
        raise ValueError("A view must contain one observation per frame")
    return rows


def _relative(row: Mapping[str, Any], parameters: EvidenceParameters) -> np.ndarray | None:
    indices = [11, 12, 15, 16]
    if (min(row["confidence"][indices]) < parameters.confidence_min
            or not np.isfinite(row["points"][indices]).all()):
        return None
    hips = row["points"][[11, 12]].mean(axis=0)
    return (row["points"][[15, 16]] - hips) / row["height"]


def _window_motion(rows: list[dict[str, Any]], start: int, end: int, fps: float,
                   parameters: EvidenceParameters) -> dict[str, Any]:
    selected = [row for row in rows if start <= row["frame"] <= end]
    pairs = []
    for first, second in zip(selected, selected[1:]):
        elapsed = (second["frame"] - first["frame"]) / fps
        if elapsed > parameters.maximum_pair_gap_seconds:
            continue
        left, right = _relative(first, parameters), _relative(second, parameters)
        if left is None or right is None:
            continue
        speeds = np.linalg.norm(right - left, axis=1) / elapsed
        pairs.append({"frames": [first["frame"], second["frame"]],
                      "left_speed": float(speeds[0]), "right_speed": float(speeds[1])})
    result = {"candidate": "unknown", "window_frames": [start, end],
              "observed_samples": len(selected), "valid_pairs": len(pairs), "pairs": pairs,
              "reason": "insufficient_continuous_anatomical_pose"}
    if len(pairs) < parameters.minimum_pairs:
        return result
    speeds = np.asarray([[pair["left_speed"], pair["right_speed"]] for pair in pairs])
    medians = np.median(speeds, axis=0)
    side = int(np.argmax(medians))
    ratio = float(medians[side] / max(medians[1 - side], 1e-9))
    support = float(np.mean(speeds[:, side] >= parameters.motion_ratio_min * speeds[:, 1 - side]))
    result.update(median_speeds={"left": float(medians[0]), "right": float(medians[1])},
                  motion_units="person_box_heights_per_source_second_relative_to_hips",
                  median_motion_ratio=ratio, pair_support_fraction=support,
                  reason="weak_or_inconsistent_temporal_motion")
    if (ratio >= parameters.motion_ratio_min and support >= parameters.pair_support_fraction_min
            and medians[side] >= parameters.minimum_motion_body_heights_per_second):
        result.update(candidate="left" if side == 0 else "right",
                      reason="sustained_relative_ankle_motion_hypothesis")
    return result


def _mirror_check(normal: list[dict[str, Any]], mirrored: list[dict[str, Any]],
                  start: int, end: int, parameters: EvidenceParameters) -> dict[str, Any]:
    other = {row["frame"]: row for row in mirrored}
    errors = []
    for row in normal:
        if not start <= row["frame"] <= end or row["frame"] not in other:
            continue
        left, right = _relative(row, parameters), _relative(other[row["frame"]], parameters)
        if left is not None and right is not None:
            errors.append(float(np.median(np.linalg.norm(left - right, axis=1))))
    median = float(np.median(errors)) if errors else None
    return {"matched_frames": len(errors), "median_relative_ankle_error_body_heights": median,
            "consistent": bool(len(errors) >= parameters.minimum_pairs + 1
                               and median <= parameters.mirror_relative_error_max),
            "scope": "original-coordinate anatomical channels must already be inverse-mapped"}


def _temporal_channel_check(rows: list[dict[str, Any]], start: int, end: int, fps: float,
                            parameters: EvidenceParameters) -> dict[str, Any]:
    selected = [row for row in rows if start <= row["frame"] <= end]
    ambiguities = []
    for first, second in zip(selected, selected[1:]):
        if (second["frame"] - first["frame"]) / fps > parameters.maximum_pair_gap_seconds:
            continue
        left, right = _relative(first, parameters), _relative(second, parameters)
        if left is None or right is None:
            continue
        direct = float(np.linalg.norm(right - left, axis=1).sum())
        crossed = float(np.linalg.norm(right[::-1] - left, axis=1).sum())
        if (direct >= parameters.channel_jump_body_heights_min
                and crossed < parameters.crossed_assignment_cost_ratio_max * direct):
            ambiguities.append({"frames": [first["frame"], second["frame"]],
                                "direct_assignment_cost": direct, "crossed_assignment_cost": crossed})
    return {"ambiguous_pairs": ambiguities, "channel_permutation_applied": False,
            "interpretation": "crossing limbs or channel swaps require observation; no automatic anatomical relabeling"}


def associate_leg_trajectories(samples: Sequence[Mapping[str, Any]], fps: float,
                              parameters: EvidenceParameters | None = None) -> dict[str, Any]:
    """Associate two visible legs without asserting anatomical left and right.

    Hungarian matching uses hip-relative knee and ankle positions. A/B labels
    reset after a gap and remain trajectory identities, even when both pose
    views agree. This diagnostic does not override a striking-foot abstention.
    """
    parameters = parameters or EvidenceParameters()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source FPS must be positive and finite")
    rows = _validate_samples(samples)
    output, previous, previous_frame, segment = [], None, None, -1
    for row in rows:
        indices = [11, 12, 13, 14, 15, 16]
        if (min(row["confidence"][indices]) < parameters.confidence_min
                or not np.isfinite(row["points"][indices]).all()):
            continue
        hips = row["points"][[11, 12]].mean(axis=0)
        legs = np.asarray([row["points"][[13, 15]], row["points"][[14, 16]]])
        legs = (legs - hips) / row["height"]
        reset = previous_frame is None or (row["frame"] - previous_frame) / fps > parameters.maximum_pair_gap_seconds
        if reset:
            assignment, margin = np.arange(2), None
            segment += 1
        else:
            distances = np.linalg.norm(previous[:, None, :, :] - legs[None, :, :, :], axis=3)
            costs = .4 * distances[:, :, 0] + .6 * distances[:, :, 1]
            _, assignment = linear_sum_assignment(costs)
            direct, crossed = float(costs[0, 0] + costs[1, 1]), float(costs[0, 1] + costs[1, 0])
            margin = abs(direct - crossed)
        connected = legs[assignment]
        output.append({"frame": row["frame"], "segment_id": segment,
                       "trajectory_A_source_channel": "left" if assignment[0] == 0 else "right",
                       "trajectory_B_source_channel": "left" if assignment[1] == 0 else "right",
                       "trajectory_A_relative_knee_ankle": connected[0].tolist(),
                       "trajectory_B_relative_knee_ankle": connected[1].tolist(),
                       "assignment_cost_margin": margin, "ambiguous_assignment": margin is not None and margin < .03,
                       "segment_start": bool(reset)})
        previous, previous_frame = connected, row["frame"]
    switches = [row["frame"] for first, row in zip(output, output[1:])
                if not row["segment_start"] and row["trajectory_A_source_channel"] != first["trajectory_A_source_channel"]]
    return {"method": "Hungarian_hip_relative_knee_ankle_association", "observations": output,
            "source_channel_switch_frames": switches, "segments": segment + 1,
            "anatomical_side_A": "unknown", "anatomical_side_B": "unknown", "human_verified": False,
            "scope": "continuous image-plane leg hypotheses, not anatomical labels or ground contacts"}


def _ball_proximity(rows: list[dict[str, Any]], balls: Sequence[Mapping[str, Any]],
                    start: int, departure: int, parameters: EvidenceParameters) -> dict[str, Any]:
    matched = []
    lookup = {}
    for observation in balls:
        frame, point = observation.get("frame"), np.asarray(observation.get("xy"), float)
        if (not isinstance(frame, (int, np.integer)) or isinstance(frame, bool)
                or frame < 0 or point.shape != (2,) or not np.isfinite(point).all()):
            raise ValueError("Ball observations need integer frames and finite source-pixel xy")
        if frame in lookup:
            raise ValueError("Ball observations must select one identity per frame")
        lookup[int(frame)] = {"xy": point, "detection_id": observation.get("detection_id")}
    for row in rows:
        if row["frame"] not in lookup or not start - 5 <= row["frame"] <= departure:
            continue
        if _relative(row, parameters) is None:
            continue
        ball = lookup[row["frame"]]
        distances = np.linalg.norm(row["points"][[15, 16]] - ball["xy"], axis=1) / row["height"]
        matched.append({"frame": row["frame"], "ball_source_xy": ball["xy"].tolist(),
                        "ball_detection_id": ball["detection_id"],
                        "left_distance": float(distances[0]), "right_distance": float(distances[1])})
    contact = [row for row in matched if start - 1 <= row["frame"] <= departure]
    result = {"candidate": "unknown", "reason": "insufficient_frame_matched_ball_pose",
              "contact_samples": len(contact), "observations": matched,
              "ball_identity_verified": False, "shoe_ball_contact_verified": False}
    if len(contact) < parameters.minimum_contact_samples:
        return result
    values = np.asarray([[row["left_distance"], row["right_distance"]] for row in contact])
    medians = np.median(values, axis=0)
    side = int(np.argmin(medians))
    margin = float(medians[1 - side] - medians[side])
    support = float(np.mean(values[:, side] < values[:, 1 - side]))
    early = [row for row in matched if row["frame"] < start - 1]
    approach = (float(np.median([row["left_distance" if side == 0 else "right_distance"] for row in early])
                      - medians[side]) if early else None)
    result.update(median_distance_body_heights={"left": float(medians[0]), "right": float(medians[1])},
                  distance_margin_body_heights=margin, nearest_side_support_fraction=support,
                  approach_distance_reduction_body_heights=approach,
                  reason="ankle_ball_distance_or_approach_not_separated")
    if (medians[side] <= parameters.ball_distance_body_heights_max
            and margin >= parameters.ball_distance_margin_body_heights_min and support >= .75
            and approach is not None and approach >= parameters.ball_approach_body_heights_min):
        result.update(candidate="left" if side == 0 else "right",
                      reason="approaching_ankle_with_separated_local_ball_distance")
    return result


def kicking_foot_evidence(normal_samples: Sequence[Mapping[str, Any]],
                          mirrored_samples: Sequence[Mapping[str, Any]] | None,
                          contact_bracket: Sequence[int], fps: float,
                          parameters: EvidenceParameters | None = None,
                          *, ball_observations: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return an auditable hypothesis, never proof of shoe-ball contact.

    Paired mirror runs are preferred. A separated, approaching local ankle-ball
    distance can provide a distinct candidate; motion disagreement is retained.
    Camera-side and player-name priors are never used to resolve anatomy.
    """
    parameters = parameters or EvidenceParameters()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Source FPS must be positive and finite")
    if (len(contact_bracket) != 2 or any(not isinstance(value, (int, np.integer))
                                       or isinstance(value, bool) for value in contact_bracket)
            or not 0 <= contact_bracket[0] <= contact_bracket[1]):
        raise ValueError("Contact bracket must be ordered nonnegative integer frame IDs")
    start, departure = map(int, contact_bracket)
    low = max(0, start - round(parameters.precontact_seconds * fps))
    high = departure + round(parameters.postdeparture_seconds * fps)
    normal = _validate_samples(normal_samples)
    mirrored = _validate_samples(mirrored_samples) if mirrored_samples is not None else None
    windows = {"bracket_context": [low, high], "departure_followthrough": [departure, high]}
    views = {"normal": {name: _window_motion(normal, *bounds, fps, parameters)
                         for name, bounds in windows.items()}}
    if mirrored is not None:
        views["mirrored_inverse_mapped"] = {
            name: _window_motion(mirrored, *bounds, fps, parameters) for name, bounds in windows.items()}
    labels = [window["candidate"] for view in views.values() for window in view.values()
              if window["candidate"] != "unknown"]
    result = {"schema_version": 1, "method": "continuous_contact_window_ankle_evidence_v2",
              "candidate": "unknown", "status": "insufficient_temporal_evidence",
              "contact_bracket_frames": [start, departure], "source_fps": float(fps),
              "parameters": asdict(parameters), "views": views, "component_labels": labels,
              "requires_review": True, "human_verified": False,
              "shoe_ball_contact_verified": False, "confidence_calibrated": False,
              "reviewer_labels_read": False, "ground_calibrated": False,
              "limitations": ["pose anatomical channels may be wrong", "source clock may contain slow motion",
                              "relative ankle motion is not observed shoe-ball contact",
                              "upstream actor and departure identities require verification"]}
    checks = {"normal": _temporal_channel_check(normal, low, high, fps, parameters)}
    if mirrored is not None:
        checks["mirrored_inverse_mapped"] = _temporal_channel_check(mirrored, low, high, fps, parameters)
    result["temporal_channel_checks"] = checks
    local_checks = {"normal": _temporal_channel_check(normal, max(0, start - 2), departure + 1, fps, parameters)}
    if mirrored is not None:
        local_checks["mirrored_inverse_mapped"] = _temporal_channel_check(
            mirrored, max(0, start - 2), departure + 1, fps, parameters)
    result["contact_window_channel_checks"] = local_checks
    result["leg_trajectory_association"] = {
        "normal": associate_leg_trajectories([row for row in normal_samples if low <= row["frame"] <= high], fps, parameters)}
    if mirrored_samples is not None:
        result["leg_trajectory_association"]["mirrored_inverse_mapped"] = associate_leg_trajectories(
            [row for row in mirrored_samples if low <= row["frame"] <= high], fps, parameters)
    local_ambiguity = any(check["ambiguous_pairs"] for check in local_checks.values())
    if ball_observations:
        proximity = {"normal": _ball_proximity(normal, ball_observations, start, departure, parameters)}
        if mirrored is not None:
            proximity["mirrored_inverse_mapped"] = _ball_proximity(mirrored, ball_observations, start, departure, parameters)
        result["ball_proximity"] = proximity
        proximity_labels = [value["candidate"] for value in proximity.values()]
        local_mirror = _mirror_check(normal, mirrored, max(0, start - 2), departure + 1, parameters) if mirrored is not None else None
        result["contact_window_mirror_consistency"] = local_mirror
        if (not local_ambiguity and mirrored is not None and local_mirror["consistent"] and "unknown" not in proximity_labels
                and len(set(proximity_labels)) == 1):
            result.update(candidate=proximity_labels[0], status="distance_supported_contact_window_candidate",
                          motion_disagreement_present=any(label != proximity_labels[0] for label in labels))
            return result
    if local_ambiguity:
        result["status"] = "temporal_anatomical_channel_ambiguity"
        return result
    if len(set(labels)) > 1:
        result["status"] = "conflicting_temporal_windows_or_views"
        return result
    if mirrored is not None:
        check = _mirror_check(normal, mirrored, low, high, parameters)
        result["mirror_consistency"] = check
        if not check["consistent"]:
            result["status"] = "anatomical_channels_unstable_under_reflection"
            return result
        supported_views = sum(any(window["candidate"] != "unknown" for window in view.values())
                              for view in views.values())
        if labels and supported_views == 2:
            result.update(candidate=labels[0], status="consistent_temporal_motion_candidate")
    elif len(labels) == len(windows):
        result.update(candidate=labels[0], status="single_view_temporal_motion_candidate")
    return result
