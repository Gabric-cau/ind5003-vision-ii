"""Reassociate measured poses in camera-compensated coordinates.

This optional recovery pass reads cached neural evidence only. It never fills
missing keypoints. Suggested crops interpolate an actor's location between
measured observations solely to request another pose-model observation.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.kick_contact import _all_track_samples, _score_all_tracks, _xy_transform


@dataclass(frozen=True)
class RecoverySettings:
    maximum_gap_seconds: float = 2.0
    maximum_scale_ratio: float = 2.5
    maximum_speed_body_heights_per_second: float = 8.0
    base_prediction_gate: float = .45
    gap_prediction_gate_per_second: float = .6
    ambiguity_margin: float = .08
    minimum_actor_observations: int = 3
    roi_padding_body_heights: float = .5


def _geometry(observation, transform):
    x0, y0, x1, y1 = observation["bbox"]
    center = _xy_transform(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), transform)[0]
    vertical = _xy_transform(np.array([[(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]), transform)
    horizontal = _xy_transform(np.array([[x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]]), transform)
    return center, max(1., float(np.linalg.norm(vertical[1] - vertical[0]))), max(
        1., float(np.linalg.norm(horizontal[1] - horizontal[0])))


def _observations(evidence, transforms):
    by_frame = defaultdict(list)
    seen = set()
    rejected = 0
    for track in evidence.get("tracks", []):
        for original in track.get("observations", []):
            observation = deepcopy(original)
            frame = observation.get("frame")
            box = np.asarray(observation.get("bbox", []), dtype=float)
            points = np.asarray(observation.get("keypoints", []), dtype=float)
            confidence = np.asarray(observation.get("keypoint_confidences", []), dtype=float)
            if (not isinstance(frame, int) or frame not in transforms or box.shape != (4,)
                    or points.shape != (17, 2) or confidence.shape != (17,)
                    or not np.isfinite(box).all() or not np.isfinite(points).all()
                    or not np.isfinite(confidence).all() or box[2] <= box[0] or box[3] <= box[1]):
                rejected += 1
                continue
            key = (frame, tuple(np.round(box, 5)), tuple(np.round(points.ravel(), 5)))
            if key in seen:
                continue
            seen.add(key)
            observation["source_track_id"] = track.get("track_id")
            by_frame[frame].append(observation)
    return by_frame, rejected


def _prediction(track, frame, fps, transforms, settings):
    observations = track["observations"]
    recent = observations[-6:]
    centers = [_geometry(o, transforms[o["frame"]])[0] for o in recent]
    last_center, last_height, _ = _geometry(recent[-1], transforms[recent[-1]["frame"]])
    slopes = []
    for first in range(len(recent)):
        for second in range(first + 1, len(recent)):
            dt = (recent[second]["frame"] - recent[first]["frame"]) / fps
            if dt > 0:
                slopes.append((centers[second] - centers[first]) / dt)
    velocity = np.median(slopes, axis=0) if slopes else np.zeros(2)
    speed = float(np.linalg.norm(velocity))
    limit = settings.maximum_speed_body_heights_per_second * last_height
    if speed > limit:
        velocity = velocity * limit / speed
    dt = (frame - observations[-1]["frame"]) / fps
    return last_center + velocity * dt, last_center, last_height, dt


def _association_cost(track, observation, fps, transforms, settings):
    predicted, last, old_height, dt = _prediction(track, observation["frame"], fps, transforms, settings)
    if not 0 < dt <= settings.maximum_gap_seconds:
        return math.inf
    center, height, _ = _geometry(observation, transforms[observation["frame"]])
    ratio = max(height, old_height) / min(height, old_height)
    scale = max(height, old_height)
    distance = float(np.linalg.norm(center - predicted) / scale)
    gate = settings.base_prediction_gate + settings.gap_prediction_gate_per_second * dt
    raw_speed = float(np.linalg.norm(center - last) / scale)
    if (ratio > settings.maximum_scale_ratio or distance > gate
            or raw_speed > settings.maximum_speed_body_heights_per_second * dt + .25):
        return math.inf
    return distance + .25 * abs(math.log(ratio))


def rebuild_camera_compensated_tracks(evidence, *, fps, settings=None):
    """One-to-one association of real observations, breaking at camera failures."""
    settings = settings or RecoverySettings()
    if not np.isfinite(fps) or fps <= 0 or settings.maximum_gap_seconds <= 0:
        raise ValueError("Positive FPS and maximum gap are required")
    transforms = {int(frame): np.asarray(matrix, dtype=float)
                  for frame, matrix in evidence.get("transforms", {}).items()}
    if any(matrix.shape != (3, 3) or not np.isfinite(matrix).all() for matrix in transforms.values()):
        raise ValueError("Camera transforms must be finite 3-by-3 matrices")
    by_frame, rejected = _observations(evidence, transforms)
    boundaries = sorted({int(pair[1]) for pair in evidence.get("camera", {}).get("failed_pairs", [])})
    tracks = []
    ambiguous_pairs = 0
    linked_gaps = []
    for frame, detections in sorted(by_frame.items()):
        detections.sort(key=lambda o: tuple(o["bbox"]))
        active = []
        for track in tracks:
            last_frame = track["observations"][-1]["frame"]
            if (0 < (frame - last_frame) / fps <= settings.maximum_gap_seconds
                    and not any(last_frame < boundary <= frame for boundary in boundaries)):
                active.append(track)
        costs = np.full((len(active), len(detections)), np.inf)
        for i, track in enumerate(active):
            for j, detection in enumerate(detections):
                costs[i, j] = _association_cost(track, detection, fps, transforms, settings)
        assigned = set()
        if costs.size:
            rows, columns = linear_sum_assignment(np.where(np.isfinite(costs), costs, 1e9))
            for i, j in zip(rows, columns):
                cost = costs[i, j]
                if not np.isfinite(cost):
                    continue
                alternatives = np.concatenate((np.delete(costs[i], j), np.delete(costs[:, j], i)))
                if alternatives.size and np.any(alternatives <= cost + settings.ambiguity_margin):
                    ambiguous_pairs += 1
                    continue
                track, detection = active[i], detections[j]
                previous = track["observations"][-1]["frame"]
                detection["track_id"] = track["track_id"]
                track["observations"].append(detection)
                assigned.add(j)
                linked_gaps.append({"track_id": track["track_id"], "start_frame": previous,
                                    "end_frame": frame, "cost": float(cost)})
        for j, detection in enumerate(detections):
            if j not in assigned:
                detection["track_id"] = len(tracks)
                tracks.append({"track_id": len(tracks), "observations": [detection]})
    return tracks, {"settings": asdict(settings), "input_track_count": len(evidence.get("tracks", [])),
                    "output_track_count": len(tracks), "measured_observation_count": sum(
                        len(t["observations"]) for t in tracks), "rejected_invalid_observations": rejected,
                    "ambiguous_associations_rejected": ambiguous_pairs,
                    "camera_failure_boundaries": boundaries, "associations": linked_gaps,
                    "missing_keypoints_interpolated": False, "human_labels_used": False}


def _fallback_actor(tracks, balls, transforms, fps, settings):
    """Require repeated foot/ball proximity and motion in compensated space."""
    by_frame = defaultdict(list)
    for ball in balls:
        by_frame[int(ball["frame"])].append(ball)
    ranked = []
    for track in tracks:
        observations = track["observations"]
        if len(observations) < settings.minimum_actor_observations:
            continue
        centers, scales, proximity = [], [], []
        for observation in observations:
            frame = observation["frame"]
            center, scale, _ = _geometry(observation, transforms[frame])
            centers.append(center)
            scales.append(scale)
            confidence = np.asarray(observation["keypoint_confidences"])
            valid = [j for j in (15, 16) if confidence[j] >= .25]
            if not valid or not by_frame[frame]:
                continue
            ankles = _xy_transform(np.asarray(observation["keypoints"])[valid], transforms[frame])
            ball_points = _xy_transform(np.array([b["xy"] for b in by_frame[frame]]), transforms[frame])
            proximity.append(float(np.min(np.linalg.norm(ankles[:, None] - ball_points[None, :], axis=2))) / scale)
        close = [distance for distance in proximity if distance <= .6]
        movement = float(np.linalg.norm(centers[-1] - centers[0]) / np.median(scales))
        if len(close) < settings.minimum_actor_observations or movement < .15:
            continue
        score = 1. - float(np.median(close)) + .1 * min(movement, 3.)
        ranked.append((score, track["track_id"]))
    ranked.sort(reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] <= settings.ambiguity_margin):
        return None
    return ranked[0][1]


def _gap_rois(track, frame_ids, transforms, fps, width, height, settings):
    if track is None or width is None:
        return []
    requests = []
    observations = track["observations"]
    for left, right in zip(observations, observations[1:]):
        start, end = left["frame"], right["frame"]
        missing = [frame for frame in frame_ids if start < frame < end]
        if not missing or (end - start) / fps > settings.maximum_gap_seconds:
            continue
        a, ah, aw = _geometry(left, transforms[start])
        b, bh, bw = _geometry(right, transforms[end])
        for frame in missing:
            if frame not in transforms:
                continue
            fraction = (frame - start) / (end - start)
            center = a + fraction * (b - a)
            body_height = ah + fraction * (bh - ah)
            body_width = aw + fraction * (bw - aw)
            padding = settings.roi_padding_body_heights * body_height
            half = np.array([body_width / 2 + padding, body_height / 2 + padding])
            corners = np.array([center - half, center + [-half[0], half[1]],
                                center + [half[0], -half[1]], center + half])
            try:
                source = _xy_transform(corners, np.linalg.inv(transforms[frame]))
            except np.linalg.LinAlgError:
                continue
            x0, y0 = np.floor(source.min(axis=0)).astype(int)
            x1, y1 = np.ceil(source.max(axis=0)).astype(int)
            box = [max(0, int(x0)), max(0, int(y0)), min(int(width), int(x1)), min(int(height), int(y1))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            requests.append({"frame": int(frame), "bbox": box, "track_id": track["track_id"],
                             "measured_anchor_frames": [start, end],
                             "reason": "gap_between_geometrically_associated_actor_observations",
                             "is_measured_pose": False, "requires_new_pose_inference": True})
    return requests


def _provisional_roi_actor(candidates, track_by_id, transforms, settings):
    """A long-gap approach can request new pixels without becoming a prediction."""
    choices = []
    for candidate in candidates:
        track = track_by_id[candidate["contact_associated_track_id"]]
        if (len(track["observations"]) < settings.minimum_actor_observations
                or candidate["stationary_ball_samples"] < 3
                or candidate["ankle_ball_distance_body_heights"] > .28):
            continue
        seed = np.asarray(candidate["stationary_ball_reference_xy"])
        before = []
        for observation in track["observations"]:
            if observation["frame"] >= candidate["contact_frame"]:
                continue
            confidence = np.asarray(observation["keypoint_confidences"])
            valid = [index for index in (15, 16) if confidence[index] >= .25]
            if not valid:
                continue
            frame = observation["frame"]
            points = _xy_transform(np.asarray(observation["keypoints"])[valid], transforms[frame])
            _, scale, _ = _geometry(observation, transforms[frame])
            before.append(float(np.min(np.linalg.norm(points - seed, axis=1))) / scale)
        distant = [distance for distance in before if distance > .6]
        if len(distant) >= 3:
            choices.append((candidate["score"], track["track_id"]))
    choices.sort(reverse=True)
    if not choices or (len(choices) > 1 and choices[0][0] - choices[1][0] <= settings.ambiguity_margin
                       and choices[0][1] != choices[1][1]):
        return None
    return choices[0][1]


def recover_contact_evidence(evidence, *, fps, frame_height, frame_width=None,
                             frame_ids=None, settings=None):
    """Return measured tracks, recomputed candidates and optional pose crop requests.

    Candidates use only the camera-reliable prefix of the selected shot. The
    returned ``feature_tracks`` contains the chosen actor, or is empty when
    neither the contact rule nor repeated ball proximity supports an identity.
    Existing candidate timestamps in the input evidence are not consulted.
    """
    if any(key in evidence for key in ("labels", "targets", "reference_contact_seconds", "manual_contact_frame")):
        raise ValueError("Recovery accepts neural evidence, not annotation inputs")
    settings = settings or RecoverySettings()
    tracks, diagnostics = rebuild_camera_compensated_tracks(evidence, fps=fps, settings=settings)
    transforms = {int(frame): np.asarray(matrix, dtype=float) for frame, matrix in evidence["transforms"].items()}
    frame_ids = sorted(transforms) if frame_ids is None else list(frame_ids)
    if any(frame not in transforms for frame in frame_ids):
        raise ValueError("Every sampled frame requires a camera transform")
    boundaries = diagnostics["camera_failure_boundaries"]
    prefix_end = min(boundaries) if boundaries else math.inf
    prefix_tracks = [{"track_id": track["track_id"], "observations": [
        o for o in track["observations"] if o["frame"] < prefix_end]} for track in tracks]
    prefix_tracks = [track for track in prefix_tracks if track["observations"]]
    balls = [ball for ball in evidence.get("balls", []) if ball["frame"] < prefix_end
             and ball["frame"] in transforms and np.asarray(ball["xy"]).shape == (2,)
             and np.isfinite(ball["xy"]).all()]
    prefix_transforms = {frame: matrix for frame, matrix in transforms.items() if frame < prefix_end}
    stride = int(np.median(np.diff(frame_ids))) if len(frame_ids) > 1 else 1
    candidates = _score_all_tracks(balls, _all_track_samples({"tracks": prefix_tracks}),
                                   prefix_transforms, fps, max(1, stride), "causal_first_departure")
    track_by_id = {track["track_id"]: track for track in prefix_tracks}
    actor_candidates = [candidate for candidate in candidates
        if candidate["score"] >= .5 and candidate["ankle_approach_supported"]
        and len(track_by_id[candidate["contact_associated_track_id"]]["observations"])
        >= settings.minimum_actor_observations]
    if actor_candidates:
        actor_id = actor_candidates[0]["contact_associated_track_id"]
        actor_method = "recovered_stationary_ball_approach_candidate"
        feature_candidates = actor_candidates
    else:
        actor_id = _fallback_actor(prefix_tracks, balls, prefix_transforms, fps, settings)
        actor_method = "repeated_ball_proximity_and_compensated_motion" if actor_id is not None else "abstain"
        feature_candidates = []
    actor = track_by_id.get(actor_id)
    roi_actor_id = actor_id if actor_id is not None else _provisional_roi_actor(
        candidates, track_by_id, prefix_transforms, settings)
    roi_actor = track_by_id.get(roi_actor_id)
    diagnostics.update(actor_id=actor_id, actor_selection_method=actor_method,
                       actor_measured_samples=len(actor["observations"]) if actor else 0,
                       roi_actor_id=roi_actor_id,
                       roi_actor_is_provisional=actor_id is None and roi_actor_id is not None,
                       reliable_prefix_end_frame=None if math.isinf(prefix_end) else prefix_end,
                       existing_input_candidate_times_used=False,
                       available_frame_count=len(frame_ids))
    return {"tracks": tracks, "feature_tracks": [actor] if actor else [],
            "candidates": candidates, "feature_candidates": feature_candidates,
            "actor_id": actor_id, "recovery_diagnostics": diagnostics,
            "pose_roi_requests": _gap_rois(roi_actor, frame_ids, transforms, fps,
                                             frame_width, frame_height, settings)}
