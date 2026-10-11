"""Conservative, camera-aware phase descriptions for a single penalty pose track.

This module is a new heuristic, not a trained action classifier or a measurement
of physical sprint speed.  It uses *clip-absolute* frame IDs.  It never consumes
the old step counter's ``runup_start_frame`` and never treats pose-derived event
candidates as confirmed foot contacts.  An independently estimated, sufficiently
supported kick anchor is needed to distinguish pre-strike and recovery phases.

Coordinates and box heights are in pixels.  A caller may supply background-
stabilized hip positions, but raw hips must still be supplied because ankle
activity is computed relative to the hip in the same original camera frame.
No homography is assumed: lateral/curved/loop tags describe *projected* paths,
not measured ground-plane paths, true circles, or a footballer's intent.

All thresholds are explicit, unvalidated engineering choices.  Returned
``evidence_score`` values are heuristic scores, not calibrated probabilities.
Synthetic tests validate mechanics/invariances, never real-video accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PhaseConfig:
    """Scale is median observed person-box height; times are in seconds."""

    min_keypoint_confidence: float = 0.35
    min_kick_confidence: float = 0.50
    min_track_seconds: float = 0.75
    min_track_samples: int = 8
    max_interpolation_gap_seconds: float = 0.15
    smoothing_seconds: float = 0.12
    min_hip_speed: float = 0.12
    min_relative_ankle_speed: float = 0.10
    relative_motion_threshold: float = 0.22
    min_motion_bout_seconds: float = 0.20
    bridge_pause_seconds: float = 0.25
    max_approach_to_kick_gap_seconds: float = 0.50
    initiation_seconds: float = 0.45
    strike_before_seconds: float = 0.10
    strike_after_seconds: float = 0.20
    boundary_uncertainty_seconds: float = 0.15
    small_excursion_threshold: float = 0.09
    regular_excursion_threshold: float = 0.15
    max_box_scale_ratio: float = 1.50
    progress_confirmation_seconds: float = 0.40
    min_confirmation_progress: float = 0.08
    min_progress_direction_consistency: float = 0.65
    compact_max_candidate_interval_seconds: float = 0.32
    compact_max_forward_progress: float = 0.20
    min_compact_candidate_intervals: int = 3


def _safe(value: Any) -> Any:
    """Make output serializable with ``json.dumps(..., allow_nan=False)``."""
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _dense_samples(frames, values, valid, grid, max_gap):
    """Interpolate only short bounded gaps; never extend a track's endpoints."""
    result = np.full((len(grid), values.shape[1]), np.nan)
    good = np.flatnonzero(valid)
    if not len(good):
        return result
    indices = frames[good] - grid[0]
    result[indices] = values[good]
    for left, right in zip(good[:-1], good[1:]):
        gap = int(frames[right] - frames[left])
        if 1 < gap <= max_gap:
            span = np.arange(frames[left], frames[right] + 1) - grid[0]
            alpha = np.linspace(0.0, 1.0, gap + 1)[:, None]
            result[span] = values[left] * (1.0 - alpha) + values[right] * alpha
    return result


def _smooth(values, radius):
    """Coordinate-wise local median, confined to uninterrupted valid runs."""
    result = values.copy()
    valid = np.all(np.isfinite(values), axis=1)
    for start, end in _runs(valid):
        for i in range(start, end):
            result[i] = np.median(values[max(start, i - radius):min(end, i + radius + 1)], axis=0)
    return result


def _velocity(values, fps):
    velocity = np.full_like(values, np.nan)
    valid = np.all(np.isfinite(values), axis=1)
    for start, end in _runs(valid):
        if end - start > 1:
            velocity[start:end] = np.gradient(values[start:end], axis=0) * fps
    return velocity


def _runs(mask):
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))


def _bridge(mask, valid, max_gap):
    result = mask.copy()
    for start, end in _runs(~mask):
        if start > 0 and end < len(mask) and end - start <= max_gap and np.all(valid[start:end]):
            result[start:end] = True
    return result


def _path_features(points, fps):
    """Low-frequency image-plane path features; missing spans are not joined."""
    if len(points) < 4 or not np.all(np.isfinite(points)):
        return None
    stride = max(1, int(round(fps * 0.10)))
    sampled = points[np.unique(np.r_[np.arange(0, len(points), stride), len(points) - 1])]
    delta = np.diff(sampled, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    length = float(np.sum(lengths))
    chord_vector = sampled[-1] - sampled[0]
    chord = float(np.linalg.norm(chord_vector))
    direction = chord_vector / max(chord, 1e-9)
    perpendicular = np.array([-direction[1], direction[0]])
    deviation = float(np.max(np.abs((sampled - sampled[0]) @ perpendicular))) if chord > 1e-5 else None
    meaningful = delta[lengths > 0.015]
    turn = 0.0
    if len(meaningful) > 2:
        a, b = meaningful[:-1], meaningful[1:]
        angles = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))
        turn = float(np.sum(angles))
    return {
        "path_length_box_heights": length,
        "net_displacement_box_heights": chord,
        "path_to_chord_ratio": length / chord if chord > 0.03 else None,
        "max_cross_axis_deviation_box_heights": deviation,
        "signed_turn_radians": turn,
    }


def classify_penalty_phases(
    frame_ids: Sequence[int],
    fps: float,
    hips: Sequence[Sequence[float]],
    left_ankles: Sequence[Sequence[float]],
    right_ankles: Sequence[Sequence[float]],
    box_heights: Sequence[float],
    *,
    confidences: Optional[Sequence[Sequence[float]]] = None,
    stabilized_hips: Optional[Sequence[Sequence[float]]] = None,
    kick_frame: Optional[int] = None,
    kick_confidence: float = 0.0,
    step_events: Sequence[Any] = (),
    camera_motion_status: str = "unknown",
    config: Optional[PhaseConfig] = None,
) -> dict:
    """Segment a pose track and label supplied, unverified step candidates.

    ``confidences`` is an N x 3 array ordered hip / left ankle / right ankle.
    An absent array means confidence is unavailable, not human-verified.
    ``step_events`` accepts integer frame IDs or mappings with ``frame`` (also
    ``frame_id`` / ``landing_frame``) and optional ``foot`` (left/right).
    ``camera_motion_status`` accepts supported / partial / unavailable / unknown;
    even supported stabilization does not establish metric ground geometry.

    Phase segments have inclusive ``start_frame`` and exclusive ``end_frame``.
    No events are invented, removed, or promoted to ground truth.  The function
    does not change the original step counter's total.  Low-quality or missing
    spans explicitly receive ``unknown``.  Run-up phases require an adequate
    kick anchor and a terminal, sustained body-motion bout.  The short
    ``initiation`` interval is an operational boundary, not an acceleration
    claim; pace increase is reported separately only when observed.
    """
    cfg = config or PhaseConfig()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    raw_frames = np.asarray(frame_ids)
    if raw_frames.ndim != 1 or (len(raw_frames) and (not np.all(np.isfinite(raw_frames)) or
            not np.all(raw_frames == np.floor(raw_frames)))):
        raise ValueError("frame_ids must be one-dimensional integer frame IDs")
    frames = raw_frames.astype(int)
    if len(frames) and (frames[0] < 0 or np.any(np.diff(frames) <= 0)):
        raise ValueError("frame_ids must be nonnegative, unique, and increasing")
    if len(frames) and frames[-1] - frames[0] > 1_000_000:
        raise ValueError("track span exceeds the single-clip safety limit")
    n = len(frames)
    arrays = {}
    for name, values in (("hips", hips), ("left_ankles", left_ankles), ("right_ankles", right_ankles)):
        array = np.asarray(values, dtype=float)
        if n == 0 and array.size == 0:
            array = np.empty((0, 2))
        if array.shape != (n, 2):
            raise ValueError(f"{name} must have shape (N, 2)")
        arrays[name] = array
    heights = np.asarray(box_heights, dtype=float)
    if heights.shape != (n,):
        raise ValueError("box_heights must have shape (N,)")
    conf = np.ones((n, 3)) if confidences is None else np.asarray(confidences, dtype=float)
    if conf.shape != (n, 3):
        raise ValueError("confidences must have shape (N, 3): hip, left ankle, right ankle")
    if camera_motion_status not in {"supported", "partial", "unavailable", "unknown"}:
        raise ValueError("unrecognized camera_motion_status")
    trajectory = arrays["hips"] if stabilized_hips is None else np.asarray(stabilized_hips, dtype=float)
    if trajectory.shape != (n, 2):
        raise ValueError("stabilized_hips must have shape (N, 2)")

    warnings = []
    if confidences is None:
        warnings.append("keypoint_confidences_unavailable")
    if camera_motion_status != "supported":
        warnings.append("camera_motion_not_fully_resolved")
    if stabilized_hips is None and camera_motion_status == "supported":
        warnings.append("camera_support_supplied_without_stabilized_coordinates")
    base = {
        "schema_version": "penalty-phases-v1",
        "status": "abstained", "human_verified": False,
        "annotation_source": "heuristic_2d_pose_phase_classifier",
        "review_required": True,
        "evidence_score_is_probability": False,
        "coordinate_space": "stabilized_image_plane" if stabilized_hips is not None else "image_plane",
        "camera_motion_status": camera_motion_status,
        "thresholds": asdict(cfg), "fps": float(fps),
        "limitations": [
            "No calibrated ground-plane distances, physical sprint speeds, or true-circle claims.",
            "Step events and foot contacts remain unverified candidates.",
            "Phase boundaries and evidence scores are heuristic, not validated accuracy.",
            "No player identity or old run-up-start label is used as ground truth.",
        ],
        "warnings": warnings, "segments": [], "frame_labels": [], "style_tags": [],
        "events": [], "phase_event_counts": {}, "metrics": {},
        "kick_anchor": {"frame": kick_frame, "confidence": kick_confidence, "usable": False},
        "approach_start_frame": None,
    }
    if n == 0:
        warnings.append("empty_pose_track")
        base["events"] = [_event_without_track(event, "empty_pose_track") for event in step_events]
        if step_events:
            base["phase_event_counts"] = {"unknown": len(step_events)}
        return _safe(base)

    grid = np.arange(frames[0], frames[-1] + 1)
    height_valid = np.isfinite(heights) & (heights > 0)
    scale = float(np.median(heights[height_valid])) if np.any(height_valid) else 1.0
    max_gap = max(1, int(round(cfg.max_interpolation_gap_seconds * fps)))
    radius = max(1, int(round(cfg.smoothing_seconds * fps / 2)))
    hip_valid = height_valid & np.all(np.isfinite(trajectory), axis=1) & np.isfinite(conf[:, 0]) & (conf[:, 0] >= cfg.min_keypoint_confidence)
    hip_dense = _smooth(_dense_samples(frames, trajectory / scale, hip_valid, grid, max_gap), radius)
    relatives = []
    all_observed_valid = hip_valid.copy()
    for k, side in enumerate(("left_ankles", "right_ankles"), 1):
        relative = (arrays[side] - arrays["hips"]) / scale
        valid = hip_valid & np.all(np.isfinite(relative), axis=1) & np.isfinite(conf[:, k]) & (conf[:, k] >= cfg.min_keypoint_confidence)
        all_observed_valid &= valid
        relatives.append(_smooth(_dense_samples(frames, relative, valid, grid, max_gap), radius))
    hip_velocity = _velocity(hip_dense, fps)
    speed = np.linalg.norm(hip_velocity, axis=1)
    foot_speed = np.stack([np.linalg.norm(_velocity(x, fps), axis=1) for x in relatives], axis=1)
    # Both ankles are needed for the categorical phase decision; one-foot
    # occlusion must not silently become evidence for an absent small step.
    valid = np.isfinite(speed) & np.all(np.isfinite(foot_speed), axis=1)
    observed = np.zeros(len(grid), dtype=bool)
    observed[frames[all_observed_valid] - grid[0]] = True
    interpolated = valid & ~observed
    if np.any(interpolated):
        warnings.append("short_pose_gaps_interpolated")
    activity = np.max(foot_speed, axis=1)
    coverage = float(np.mean(valid))
    labels = np.full(len(grid), "unknown", dtype=object)
    box_ratio = float(np.percentile(heights[height_valid], 90) / np.percentile(heights[height_valid], 10)) if np.any(height_valid) else None
    if box_ratio is not None and box_ratio > cfg.max_box_scale_ratio:
        warnings.append("substantial_box_scale_change_projection_or_zoom")
    if coverage < 0.80:
        warnings.append("incomplete_pose_coverage")
    if np.any(np.diff(frames) > max_gap):
        warnings.append("long_pose_gaps_not_interpolated")
    enough_track = (int(np.sum(hip_valid)) >= cfg.min_track_samples and
                    (frames[-1] - frames[0]) / fps >= cfg.min_track_seconds and
                    np.sum(valid) >= cfg.min_track_samples)
    if not enough_track:
        warnings.append("insufficient_pose_track")

    anchor_ok = (kick_frame is not None and np.isfinite(kick_frame) and float(kick_frame).is_integer() and
                 frames[0] <= kick_frame <= frames[-1] and np.isfinite(kick_confidence) and
                 kick_confidence >= cfg.min_kick_confidence)
    if not anchor_ok:
        warnings.append("kick_anchor_missing_low_confidence_or_outside_track")
    anchor_index = int(kick_frame - frames[0]) if anchor_ok else None
    if anchor_ok and not valid[anchor_index]:
        warnings.append("kick_anchor_in_pose_gap")
        anchor_ok = False
    base["kick_anchor"]["usable"] = bool(anchor_ok)

    pre = valid & ((grid < kick_frame) if anchor_ok else True)
    speed_threshold = max(cfg.min_hip_speed, float(np.percentile(speed[pre], 80)) * cfg.relative_motion_threshold) if np.any(pre) else cfg.min_hip_speed
    moving = pre & (speed >= speed_threshold) & (activity >= cfg.min_relative_ankle_speed)
    moving = _bridge(moving, valid & pre, int(round(cfg.bridge_pause_seconds * fps)))
    bouts = [(int(grid[a]), int(grid[b - 1] + 1)) for a, b in _runs(moving)
             if (b - a) / fps >= cfg.min_motion_bout_seconds]
    base["metrics"].update({
        "coverage": coverage, "observed_samples": n, "timeline_frames": len(grid),
        "observed_valid_pose_frames": int(np.sum(observed)),
        "interpolated_pose_frames": int(np.sum(interpolated)),
        "median_box_height_px": scale, "box_height_p90_p10_ratio": box_ratio,
        "hip_speed_threshold_box_heights_per_second": speed_threshold,
        "movement_bouts": [{"start_frame": a, "end_frame": b} for a, b in bouts],
    })
    onset = None
    direction = None
    strike_start = strike_end = None
    if enough_track and anchor_ok:
        strike_start = max(int(grid[0]), int(kick_frame - round(cfg.strike_before_seconds * fps)))
        strike_end = min(int(grid[-1] + 1), int(kick_frame + round(cfg.strike_after_seconds * fps) + 1))
        # Terminal direction is based on the last supported pre-strike path,
        # excluding the kick itself.  This is relative to approach, not screen X.
        end = max(0, strike_start - int(grid[0]))
        lookback = max(0, end - int(round(0.80 * fps)))
        ids = np.flatnonzero(valid[lookback:end]) + lookback
        if len(ids) >= max(4, int(fps * 0.15)) and np.max(np.diff(ids)) <= max_gap:
            vector = hip_dense[ids[-1]] - hip_dense[ids[0]]
            if np.linalg.norm(vector) >= 0.08:
                direction = vector / np.linalg.norm(vector)
        eligible = [(a, b) for a, b in bouts if 0 <= kick_frame - b <= cfg.max_approach_to_kick_gap_seconds * fps]
        if eligible:
            onset, _ = eligible[-1]
            if direction is not None:
                # A preceding lateral shuffle can be moving continuously.  Find
                # the final sustained progression along the eventual approach.
                projected = hip_velocity @ direction
                aligned = moving & (projected >= 0.45 * speed_threshold)
                aligned = _bridge(aligned, valid & pre, int(round(cfg.bridge_pause_seconds * fps)))
                directional_bouts = [(int(grid[a]), int(grid[b - 1] + 1)) for a, b in _runs(aligned)
                                     if (b - a) / fps >= cfg.min_motion_bout_seconds]
                directional = [(a, b) for a, b in directional_bouts
                               if a >= onset and 0 <= kick_frame - b <= cfg.max_approach_to_kick_gap_seconds * fps]
                if directional:
                    onset = directional[-1][0]
                base["metrics"]["speed_only_onset_frame"] = onset
                # Instantaneous hip speed also reacts to standing body sway.
                # Require net progress over a short window before accepting
                # this early onset; do not restart at later stutter pauses.
                window = max(3, int(round(cfg.progress_confirmation_seconds * fps)))
                first = onset - int(grid[0])
                last = strike_start - int(grid[0])
                confirmation_direction = hip_dense[max(first, last - 1)] - hip_dense[first]
                if np.linalg.norm(confirmation_direction) > .08:
                    confirmation_direction /= np.linalg.norm(confirmation_direction)
                else:
                    confirmation_direction = direction
                confirmed = []
                for i in range(first, max(first, last - window + 1)):
                    sample = hip_dense[i:i + window + 1]
                    if len(sample) != window + 1 or not np.all(valid[i:i + window + 1]):
                        continue
                    delta = sample[-1] - sample[0]
                    progress = float(delta @ confirmation_direction)
                    path_length = float(np.sum(np.linalg.norm(np.diff(sample, axis=0), axis=1)))
                    consistency = progress / max(path_length, 1e-9)
                    if (progress >= cfg.min_confirmation_progress and
                            consistency >= cfg.min_progress_direction_consistency):
                        confirmed.append((int(grid[i]), progress, consistency))
                if confirmed:
                    onset = confirmed[0][0]
                    base["metrics"]["progress_confirmation"] = {
                        "start_frame": onset, "window_frames": window,
                        "forward_progress_box_heights": confirmed[0][1],
                        "direction_consistency": confirmed[0][2]}
                    if onset != base["metrics"]["speed_only_onset_frame"]:
                        warnings.append("early_body_motion_separated_by_net_progress_gate")
                else:
                    warnings.append("net_progress_confirmation_unavailable")
                    # Keep the boundary provisional rather than claiming that
                    # a slow/projected approach cannot exist.
            if onset >= strike_start:
                onset = None
        if onset is None:
            warnings.append("no_supported_terminal_approach_bout")
        else:
            labels[(grid < onset) & valid] = "preparation_adjustment"
            initiation_end = min(strike_start, onset + max(1, int(round(cfg.initiation_seconds * fps))))
            labels[(grid >= onset) & (grid < initiation_end) & valid] = "initiation"
            labels[(grid >= initiation_end) & (grid < strike_start) & valid] = "approach"
            if onset - grid[0] <= radius + max_gap:
                warnings.append("approach_may_be_left_censored")
        labels[(grid >= strike_start) & (grid < strike_end) & valid] = "strike"
        labels[(grid >= strike_end) & valid] = "recovery"

    # Conservative, interpretable style tags.  They never claim a measured
    # football tactic, true field-plane circle, sprint, or exact step length.
    camera_score = {"supported": 0.85, "partial": 0.60, "unavailable": 0.40, "unknown": 0.40}[camera_motion_status]
    evidence_score = min(camera_score, float(kick_confidence) if anchor_ok else 0.35, coverage)

    def tag(name, metrics, indices):
        if not len(indices):
            return
        base["style_tags"].append({"tag": name, "start_frame": int(grid[indices[0]]),
            "end_frame": int(grid[indices[-1]] + 1), "evidence_score": evidence_score,
            "status": "candidate", "metrics": metrics, "human_verified": False})

    if onset is not None:
        base["approach_start_frame"] = int(onset)
        for phase, ids in (("preparation", np.flatnonzero(valid & (grid < onset))),
                           ("approach", np.flatnonzero(valid & (grid >= onset) & (grid < strike_start)))):
            path = _path_features(hip_dense[ids], fps) if len(ids) and np.all(np.diff(ids) == 1) else None
            base["metrics"][f"{phase}_path"] = path
            if path and path["path_length_box_heights"] >= 0.35:
                deviation = path["max_cross_axis_deviation_box_heights"]
                ratio = path["path_to_chord_ratio"]
                if deviation is not None and deviation >= 0.12 and ratio is not None and ratio >= 1.08:
                    tag(f"curved_{phase}_image_plane", path, ids)
                if (phase == "preparation" and abs(path["signed_turn_radians"]) >= 4.5 and
                        path["net_displacement_box_heights"] / path["path_length_box_heights"] < 0.35):
                    tag("loop_like_preparation_image_plane_candidate", path, ids)
            if phase == "preparation" and len(ids) >= 4 and direction is not None and np.all(np.diff(ids) == 1):
                offsets = hip_dense[ids] - hip_dense[ids[0]]
                perpendicular = np.array([-direction[1], direction[0]])
                cross = float(np.ptp(offsets @ perpendicular))
                forward = float(np.ptp(offsets @ direction))
                if cross >= 0.12 and cross > 1.3 * forward:
                    tag("lateral_adjustment_relative_to_approach_image_plane",
                        {"cross_axis_range_box_heights": cross, "approach_axis_range_box_heights": forward}, ids)
        first = valid & (grid >= onset) & (grid < onset + fps * 0.35)
        later = valid & (grid >= onset + fps * 0.35) & (grid < min(strike_start, onset + fps * 0.85))
        if np.sum(first) >= 3 and np.sum(later) >= 3:
            a, b = float(np.median(speed[first])), float(np.median(speed[later]))
            if b > 1.7 * max(a, 0.03) and b - a > 0.15:
                tag("pace_increase_image_plane", {"early_speed_box_heights_per_second": a,
                    "later_speed_box_heights_per_second": b}, np.flatnonzero(first | later))
        inside = valid & (grid >= onset) & (grid < strike_start)
        pause = inside & (speed < speed_threshold)
        pause_runs = [(a, b) for a, b in _runs(pause) if (b - a) / fps >= 0.12 and
                      a > 0 and b < len(grid) and inside[a - 1] and inside[b]]
        if pause_runs:
            tag("pause_or_stutter_image_plane_candidate", {"pause_intervals": [
                {"start_frame": int(grid[a]), "end_frame": int(grid[b - 1] + 1)} for a, b in pause_runs]}, np.flatnonzero(inside))

    boundary_indices = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    for a, b in zip(boundary_indices[:-1], boundary_indices[1:]):
        label = str(labels[a])
        base["segments"].append({"phase": label, "start_frame": int(grid[a]),
            "end_frame": int(grid[b - 1] + 1), "start_seconds": float(grid[a] / fps),
            "end_seconds": float((grid[b - 1] + 1) / fps),
            "evidence_score": evidence_score if label != "unknown" else 0.0,
            "boundary_uncertainty_frames": int(round(cfg.boundary_uncertainty_seconds * fps)),
            "human_verified": False})
    base["frame_labels"] = [{"frame": int(f), "phase": str(label), "pose_supported": bool(v),
                             "pose_observed": bool(o), "pose_interpolated": bool(ip)}
                            for f, label, v, o, ip in zip(grid, labels, valid, observed, interpolated)]
    base["signals"] = {"frames": grid.tolist(), "hip_speed_box_heights_per_second": speed.tolist(),
                       "relative_ankle_speed_box_heights_per_second": activity.tolist()}
    phase_boundaries = grid[boundary_indices[1:-1]]
    for event in step_events:
        item = _event_without_track(event, None)
        f = item["frame"]
        if f is None or not grid[0] <= f <= grid[-1]:
            item["warnings"].append("event_outside_pose_track")
        else:
            i = f - grid[0]
            item["phase"] = str(labels[i])
            item["phase_evidence_score"] = evidence_score if valid[i] and labels[i] != "unknown" else 0.0
            item["before_kick_anchor"] = bool(f < kick_frame) if anchor_ok else None
            item["within_inferred_approach"] = bool(onset <= f < kick_frame) if onset is not None else None
            item["near_phase_boundary"] = bool(len(phase_boundaries) and np.min(abs(phase_boundaries - f)) <= cfg.boundary_uncertainty_seconds * fps)
            if item["near_phase_boundary"]:
                item["warnings"].append("phase_assignment_near_uncertain_boundary")
            side = str(item.get("foot", "")).lower()
            which = 0 if side in {"left", "l"} else 1 if side in {"right", "r"} else None
            if which is None:
                item["warnings"].append("event_foot_unspecified")
            elif valid[i]:
                radius_event = max(2, int(round(0.15 * fps)))
                window = relatives[which][max(0, i - radius_event):min(len(grid), i + radius_event + 1)]
                if len(window) >= 5 and np.all(np.isfinite(window)):
                    # Euclidean diameter is rotation/reflection invariant. It
                    # measures a local ankle excursion, never footstep length.
                    excursion = float(np.max(np.linalg.norm(window[:, None] - window[None, :], axis=2)))
                    item["local_ankle_excursion_box_heights"] = excursion
                    if excursion < cfg.small_excursion_threshold:
                        item["candidate_step_style"] = "small_image_plane_ankle_excursion"
                    elif excursion > cfg.regular_excursion_threshold:
                        item["candidate_step_style"] = "larger_image_plane_ankle_excursion"
                    else:
                        item["candidate_step_style"] = "uncertain"
            if not valid[i]:
                item["warnings"].append("event_in_pose_gap")
        base["events"].append(item)
    for item in base["events"]:
        phase = item["phase"]
        base["phase_event_counts"][phase] = base["phase_event_counts"].get(phase, 0) + 1
    # A foot-independent descriptor answers the compact-movement question
    # without inventing anatomical landing feet or true ground step lengths.
    intervals = []
    pre_events = sorted([e for e in base["events"] if e["frame"] is not None and
                         e.get("within_inferred_approach")], key=lambda e: e["frame"])
    if direction is not None and camera_motion_status == "supported" and stabilized_hips is not None:
        for a, b in zip(pre_events[:-1], pre_events[1:]):
            ia, ib = a["frame"] - grid[0], b["frame"] - grid[0]
            if ib <= ia or not np.all(valid[ia:ib + 1]):
                continue
            forward = float((hip_dense[ib] - hip_dense[ia]) @ direction)
            dt = (b["frame"] - a["frame"]) / fps
            intervals.append({"start_frame": a["frame"], "end_frame": b["frame"],
                "candidate_interval_seconds": dt, "forward_hip_displacement_box_heights": forward,
                "compact_progress_candidate": bool(0 <= forward <= cfg.compact_max_forward_progress and dt <= cfg.compact_max_candidate_interval_seconds),
                "human_verified": False})
        for a, b in _runs(np.array([item["compact_progress_candidate"] for item in intervals], bool)):
            if b - a < cfg.min_compact_candidate_intervals:
                continue
            compact = intervals[a:b]
            base["style_tags"].append({"tag": "dense_low_progress_movement_image_plane_candidate",
                "start_frame": compact[0]["start_frame"], "end_frame": compact[-1]["end_frame"] + 1,
                "status": "candidate", "human_verified": False, "evidence_score": evidence_score,
                "metrics": {"candidate_intervals": len(intervals), "compact_intervals": len(compact)},
                "limitation": "Hip progress between unverified extrema is not physical step length or verified cadence."})
    base["metrics"]["candidate_interval_progress"] = intervals
    small = [x for x in base["events"] if x["candidate_step_style"] == "small_image_plane_ankle_excursion"
             and x.get("within_inferred_approach")]
    if len(small) >= 2:
        base["style_tags"].append({"tag": "multiple_small_ankle_excursion_candidates",
            "start_frame": min(x["frame"] for x in small), "end_frame": max(x["frame"] for x in small) + 1,
            "status": "candidate", "human_verified": False, "evidence_score": evidence_score,
            "metrics": {"candidate_event_count": len(small)},
            "limitation": "Small projected excursion does not establish a physically small step or stutter."})
    if enough_track and anchor_ok:
        base["status"] = "provisional" if onset is not None else "partially_segmented"
    return _safe(base)


def _event_without_track(event, warning):
    item = dict(event) if isinstance(event, Mapping) else {"frame": event}
    frame = item.get("frame", item.get("frame_id", item.get("landing_frame")))
    try:
        value = float(frame)
        frame = int(value) if np.isfinite(value) and value.is_integer() else None
    except (ValueError, TypeError):
        frame = None
    item.update({"frame": frame, "phase": "unknown", "candidate_step_style": "uncertain",
                 "contact_status": "unverified_candidate", "human_verified": False,
                 "phase_evidence_score": 0.0, "before_kick_anchor": None,
                 "within_inferred_approach": None, "near_phase_boundary": None,
                 "warnings": [warning] if warning else []})
    return item
