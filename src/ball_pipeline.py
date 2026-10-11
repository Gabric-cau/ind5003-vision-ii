"""Manifest-aligned ball tracking and observed outcome export."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.ball_outcome import BallOutcomeExtractor
from src.annotation_provenance import is_human_verified, require_verified


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _read_csv_index(path: str, key: str = "clip_id") -> Dict[str, Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _pose_anchors(
    result: Mapping[str, Any], kick_frame: int, goal_center: Optional[Tuple[float, float]] = None
) -> List[Tuple[float, float]]:
    """Return all credible ankle positions at contact, not just the chosen kicker.

    The earlier step-counting pass can occasionally choose the referee.  The
    ball seed is therefore matched against every on-field pose, while the
    goal-direction prior distinguishes the penalty taker from the goalkeeper.
    """

    moving_anchors: List[Tuple[float, float]] = []
    fallback_anchors: List[Tuple[float, float]] = []
    minimum_field_y = 0.48 * float(result.get("frame_height") or 720.0)
    for track in result.get("tracks", []) or []:
        observations = track.get("observations", []) or []
        nearby = [item for item in observations if abs(int(item.get("frame", -999)) - kick_frame) <= 2]
        if not nearby:
            continue
        item = min(nearby, key=lambda value: abs(int(value["frame"]) - kick_frame))
        pre_observations = [
            value for value in observations
            if kick_frame - 15 <= int(value.get("frame", -999)) <= kick_frame
        ]
        centres = []
        for value in pre_observations:
            box = value.get("bbox") or []
            if len(box) >= 4:
                centres.append(((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0))
        movement = float(np.hypot(centres[-1][0] - centres[0][0], centres[-1][1] - centres[0][1])) if len(centres) >= 2 else 0.0
        points = item.get("keypoints") or []
        confidences = item.get("keypoint_confidences") or []
        for index in (15, 16):
            if index >= len(points):
                continue
            confidence = float(confidences[index]) if index < len(confidences) else 1.0
            point = points[index]
            if (
                confidence >= 0.15 and len(point) >= 2
                and all(math.isfinite(float(v)) for v in point[:2])
                and float(point[1]) >= minimum_field_y
            ):
                candidate = (float(point[0]), float(point[1]))
                fallback_anchors.append(candidate)
                far_from_goal = goal_center is None or float(np.hypot(candidate[0] - goal_center[0], candidate[1] - goal_center[1])) >= 180.0
                if movement >= 12.0 and far_from_goal:
                    moving_anchors.append(candidate)

    sequence = result.get("selected_pose_sequence") or {}
    frames = sequence.get("frames") or []
    if frames:
        index = min(range(len(frames)), key=lambda i: abs(int(frames[i]) - kick_frame))
        for side in ("left", "right"):
            values = sequence.get(f"{side}_ankles") or []
            confidences = sequence.get(f"{side}_confidences") or []
            if index >= len(values):
                continue
            confidence = float(confidences[index]) if index < len(confidences) else 1.0
            point = values[index]
            if (
                confidence >= 0.15 and len(point) >= 2
                and all(math.isfinite(float(v)) for v in point[:2])
                and float(point[1]) >= minimum_field_y
            ):
                fallback_anchors.append((float(point[0]), float(point[1])))
    if moving_anchors:
        lowest_field_y = max(point[1] for point in moving_anchors)
        moving_anchors = [point for point in moving_anchors if point[1] >= lowest_field_y - 40.0]
    if moving_anchors and goal_center is not None:
        distances = [float(np.hypot(x - goal_center[0], y - goal_center[1])) for x, y in moving_anchors]
        farthest = max(distances)
        moving_anchors = [
            point for point, distance in zip(moving_anchors, distances)
            if distance >= farthest - 120.0
        ]
    if not moving_anchors and goal_center is not None:
        fallback_anchors = [
            point for point in fallback_anchors
            if float(np.hypot(point[0] - goal_center[0], point[1] - goal_center[1])) >= 180.0
        ]
    anchors = moving_anchors or fallback_anchors
    deduplicated: List[Tuple[float, float]] = []
    for candidate in anchors:
        if not any(np.hypot(candidate[0] - x, candidate[1] - y) < 3.0 for x, y in deduplicated):
            deduplicated.append(candidate)
    return deduplicated


def _read_window(video_path: str, start_frame: int, end_frame: int) -> Tuple[List[int], List[np.ndarray], float, int, int]:
    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_ids: List[int] = []
    frames: List[np.ndarray] = []
    for frame_id in range(max(0, start_frame), end_frame + 1):
        ok, frame = capture.read()
        if not ok:
            break
        frame_ids.append(frame_id)
        frames.append(frame)
    capture.release()
    return frame_ids, frames, fps, width, height


def _visual_ball_score(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    height, width = frame.shape[:2]
    box_width, box_height = max(2.0, x2 - x1), max(2.0, y2 - y1)
    padding = max(10.0, 1.8 * max(box_width, box_height))
    px1, py1 = max(0, int(x1 - padding)), max(0, int(y1 - padding))
    px2, py2 = min(width, int(x2 + padding) + 1), min(height, int(y2 + padding) + 1)
    patch = frame[py1:py2, px1:px2]
    inner = frame[max(0, int(y1)):min(height, int(y2) + 1), max(0, int(x1)):min(width, int(x2) + 1)]
    if patch.size == 0 or inner.size == 0:
        return 0.0, 0.0, 0.0
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    inner_hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    green = (patch_hsv[..., 0] >= 28) & (patch_hsv[..., 0] <= 95) & (patch_hsv[..., 1] >= 45) & (patch_hsv[..., 2] >= 40)
    white = (inner_hsv[..., 1] <= 95) & (inner_hsv[..., 2] >= 120)
    green_ratio = float(np.mean(green))
    white_ratio = float(np.mean(white))
    roundness = min(box_width, box_height) / max(box_width, box_height)
    score = 1.4 * green_ratio + 1.1 * white_ratio + 0.35 * roundness
    return score, green_ratio, white_ratio


def _extract_detections(
    frame_ids: Sequence[int], predictions: Sequence[Any], frames: Optional[Sequence[np.ndarray]] = None
) -> Dict[int, List[Dict[str, float]]]:
    output: Dict[int, List[Dict[str, float]]] = {int(frame): [] for frame in frame_ids}
    frame_lookup = {int(frame_id): frame for frame_id, frame in zip(frame_ids, frames or [])}
    for frame_id, prediction in zip(frame_ids, predictions):
        boxes = getattr(prediction, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            continue
        xyxy = _to_numpy(boxes.xyxy)
        confidences = _to_numpy(boxes.conf).reshape(-1)
        for box, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = map(float, box[:4])
            box_width, box_height = x2 - x1, y2 - y1
            if box_width < 2.0 or box_height < 2.0 or box_width > 85.0 or box_height > 85.0:
                continue
            aspect = box_width / max(box_height, 1e-6)
            if not 0.20 <= aspect <= 5.0:
                continue
            visual_score, green_ratio, white_ratio = _visual_ball_score(
                frame_lookup[int(frame_id)], x1, y1, x2, y2
            ) if int(frame_id) in frame_lookup else (0.0, 0.0, 0.0)
            output[int(frame_id)].append(
                {
                    "frame": int(frame_id),
                    "x": round((x1 + x2) / 2.0, 3),
                    "y": round((y1 + y2) / 2.0, 3),
                    "confidence": round(float(confidence), 6),
                    "width": round(box_width, 3),
                    "height": round(box_height, 3),
                    "visual_score": round(visual_score, 5),
                    "green_ratio": round(green_ratio, 5),
                    "white_ratio": round(white_ratio, 5),
                }
            )
    return output


def _distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return float(np.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])))


def _ballistic_envelope(path: Sequence[Mapping[str, float]]) -> List[Dict[str, float]]:
    """Trim post-impact clutter and retain outward progress from contact."""

    if len(path) <= 2:
        return [dict(item) for item in path]
    start = path[0]
    radial = [_distance(start, item) for item in path]
    peak_index = int(np.argmax(radial))
    candidate_path = path[: peak_index + 1]
    selected: List[Dict[str, float]] = [dict(candidate_path[0])]
    furthest = 0.0
    for item in candidate_path[1:]:
        progress = _distance(start, item)
        if progress >= furthest + 3.0:
            selected.append(dict(item))
            furthest = progress
    peak = dict(path[peak_index])
    if int(selected[-1]["frame"]) != int(peak["frame"]):
        selected.append(peak)
    return selected


def select_ball_track(
    detections: Mapping[int, Sequence[Mapping[str, float]]],
    kick_frame: int,
    anchors: Sequence[Tuple[float, float]],
    *,
    goal_center: Optional[Tuple[float, float]] = None,
    anchor_radius: float = 150.0,
    beam_width: int = 80,
    preserve_completed: bool = False,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    """Use the contact-foot anchor and a motion-constrained beam search."""

    starts: List[Tuple[float, Dict[str, float]]] = []
    for frame_id in range(kick_frame - 3, kick_frame + 3):
        for item in detections.get(frame_id, []):
            anchor_distance = (
                min(np.hypot(item["x"] - x, item["y"] - y) for x, y in anchors)
                if anchors else None
            )
            if anchor_distance is not None and anchor_distance > 125.0:
                continue
            start_score = 3.0 * float(item["confidence"]) + 1.8 * float(item.get("visual_score", 0.0))
            if anchor_distance is not None:
                start_score -= 1.2 * min(2.5, anchor_distance / 75.0)
            if goal_center is not None:
                distance_from_goal = float(np.hypot(item["x"] - goal_center[0], item["y"] - goal_center[1]))
                start_score += 0.45 * min(2.0, distance_from_goal / 300.0)
            start_score -= 0.10 * abs(frame_id - kick_frame)
            starts.append((start_score, dict(item)))
    starts.sort(key=lambda pair: pair[0], reverse=True)
    if not starts:
        if anchors:
            fallback_track, fallback_diagnostics = select_ball_track(
                detections,
                kick_frame,
                [],
                goal_center=goal_center,
                anchor_radius=anchor_radius,
                beam_width=beam_width,
                preserve_completed=preserve_completed,
            )
            fallback_diagnostics["contact_anchor_fallback"] = True
            fallback_diagnostics["original_anchor_count"] = len(anchors)
            return fallback_track, fallback_diagnostics
        return [], {"reason": "no_contact_anchor_detection", "anchor_count": len(anchors)}

    states: List[Dict[str, Any]] = [
        {"score": score, "path": [item], "velocity": None, "misses": 0}
        for score, item in starts[:40]
    ]
    final_frame = max(detections) if detections else kick_frame
    first_frame = min(int(state["path"][0]["frame"]) for state in states)
    completed_states = []
    for frame_id in range(first_frame + 1, final_frame + 1):
        expanded: List[Dict[str, Any]] = []
        for state in states:
            last = state["path"][-1]
            if int(last["frame"]) >= frame_id:
                expanded.append(state)
                continue
            # Keeping a state alive permits short detector gaps.
            if int(state["misses"]) < 7:
                expanded.append({**state, "score": state["score"] - 0.06, "misses": state["misses"] + 1})
            elif preserve_completed:
                completed_states.append(state)
            gap = frame_id - int(last["frame"])
            for candidate_raw in detections.get(frame_id, []):
                candidate = dict(candidate_raw)
                jump = _distance(last, candidate)
                speed = jump / max(1, gap)
                if speed > 285.0:
                    continue
                velocity = state["velocity"]
                residual = 0.0
                if velocity is not None:
                    predicted_x = float(last["x"]) + float(velocity[0]) * gap
                    predicted_y = float(last["y"]) + float(velocity[1]) * gap
                    residual = float(np.hypot(candidate["x"] - predicted_x, candidate["y"] - predicted_y))
                    tolerance = max(55.0, 0.80 * float(np.hypot(*velocity)) * gap + 20.0)
                    if residual > tolerance:
                        continue
                edge_score = 2.2 * float(candidate["confidence"]) + 0.45 * float(candidate.get("visual_score", 0.0)) + 0.18
                if len(state["path"]) == 1:
                    edge_score += min(1.8, speed / 45.0)
                else:
                    edge_score -= residual / 95.0
                if frame_id >= kick_frame + 4 and speed < 2.5:
                    edge_score -= 0.65
                instantaneous = (
                    (float(candidate["x"]) - float(last["x"])) / gap,
                    (float(candidate["y"]) - float(last["y"])) / gap,
                )
                if velocity is None:
                    new_velocity = instantaneous
                else:
                    new_velocity = (0.60 * velocity[0] + 0.40 * instantaneous[0], 0.60 * velocity[1] + 0.40 * instantaneous[1])
                expanded.append(
                    {
                        "score": state["score"] + edge_score,
                        "path": state["path"] + [candidate],
                        "velocity": new_velocity,
                        "misses": 0,
                    }
                )
        expanded.sort(
            key=lambda state: state["score"]
            + min(4.0, _distance(state["path"][0], state["path"][-1]) / 75.0)
            + 0.04 * len(state["path"]),
            reverse=True,
        )
        # De-duplicate paths that terminate at nearly the same candidate.
        states = []
        signatures = set()
        for state in expanded:
            last = state["path"][-1]
            signature = (int(last["frame"]), round(float(last["x"]) / 8), round(float(last["y"]) / 8), len(state["path"]))
            if signature in signatures:
                continue
            signatures.add(signature)
            states.append(state)
            if len(states) >= beam_width:
                break
        if not states:
            break

    def final_score(state: Mapping[str, Any]) -> float:
        path = state["path"]
        displacement = _distance(path[0], path[-1])
        score = float(state["score"]) + min(5.0, displacement / 70.0) + 0.08 * len(path)
        if goal_center is not None:
            start_vector = np.asarray([goal_center[0] - path[0]["x"], goal_center[1] - path[0]["y"]], dtype=float)
            movement = np.asarray([path[-1]["x"] - path[0]["x"], path[-1]["y"] - path[0]["y"]], dtype=float)
            start_distance = float(np.linalg.norm(start_vector))
            end_distance = float(np.hypot(path[-1]["x"] - goal_center[0], path[-1]["y"] - goal_center[1]))
            progress = start_distance - end_distance
            alignment = float(np.dot(start_vector, movement) / max(1e-6, np.linalg.norm(start_vector) * np.linalg.norm(movement)))
            score += max(-10.0, min(10.0, progress / 45.0))
            score += 2.5 * alignment
        return score

    if preserve_completed:
        states = states + completed_states
    if not states:
        return [], {"reason": "all_motion_paths_expired", "anchor_count": len(anchors),
                    "candidate_start_count": len(starts)}
    best = max(states, key=final_score)
    path = sorted(best["path"], key=lambda item: int(item["frame"]))
    # One candidate per frame, retaining the last (highest-scoring beam choice).
    unique = {int(item["frame"]): dict(item) for item in path}
    path = _ballistic_envelope([unique[key] for key in sorted(unique)])
    start = path[0]
    anchor_distance = (
        min(float(np.hypot(start["x"] - x, start["y"] - y)) for x, y in anchors)
        if anchors else None
    )
    diagnostics = {
        "reason": "selected",
        "anchor_count": len(anchors),
        "start_anchor_distance_px": round(anchor_distance, 3) if anchor_distance is not None else None,
        "beam_score": round(final_score(best), 4),
        "candidate_start_count": len(starts),
        "goal_center": list(goal_center) if goal_center is not None else None,
    }
    if goal_center is not None and len(path) >= 2:
        start_vector = np.asarray([goal_center[0] - path[0]["x"], goal_center[1] - path[0]["y"]], dtype=float)
        movement = np.asarray([path[-1]["x"] - path[0]["x"], path[-1]["y"] - path[0]["y"]], dtype=float)
        end_distance = float(np.hypot(path[-1]["x"] - goal_center[0], path[-1]["y"] - goal_center[1]))
        diagnostics["progress_to_goal_px"] = round(float(np.linalg.norm(start_vector)) - end_distance, 3)
        diagnostics["goal_alignment"] = round(
            float(np.dot(start_vector, movement) / max(1e-6, np.linalg.norm(start_vector) * np.linalg.norm(movement))), 4
        )
    return path, diagnostics


def process_ball_outcomes(
    step_diagnostics_json: str,
    kick_frames_csv: str,
    outcomes_csv: str,
    output_csv: str,
    diagnostics_json: str,
    model: Any,
    *,
    goalkeeper_diagnostics_json: Optional[str] = None,
    pre_frames: int = 3,
    post_frames: int = 32,
    image_size: int = 1280,
    detector_confidence: float = 0.01,
    only_clip_ids: Optional[Sequence[str]] = None,
    allow_unverified_annotations: bool = False,
) -> List[Dict[str, Any]]:
    step_data = json.loads(Path(step_diagnostics_json).read_text(encoding="utf-8"))
    if isinstance(step_data, Mapping):
        step_data = step_data.get("results", [])
    step_by_clip = {row["clip_id"]: row for row in step_data}
    kick_rows = _read_csv_index(kick_frames_csv)
    outcome_rows = _read_csv_index(outcomes_csv)
    goalkeeper_by_clip: Dict[str, Mapping[str, Any]] = {}
    if goalkeeper_diagnostics_json and Path(goalkeeper_diagnostics_json).exists():
        goalkeeper_data = json.loads(Path(goalkeeper_diagnostics_json).read_text(encoding="utf-8"))
        goalkeeper_by_clip = {row["clip_id"]: row for row in goalkeeper_data}
    clip_ids = list(kick_rows)
    if only_clip_ids:
        wanted = set(only_clip_ids)
        clip_ids = [clip_id for clip_id in clip_ids if clip_id in wanted]
    missing = [clip_id for clip_id in clip_ids if clip_id not in step_by_clip or clip_id not in outcome_rows]
    if missing:
        raise ValueError(f"Missing diagnostics/outcome annotations: {missing}")
    require_verified([kick_rows[key] for key in clip_ids], "Kick frames", allow_unverified_annotations)

    features: List[Dict[str, Any]] = []
    rich_results: List[Dict[str, Any]] = []
    extractor = BallOutcomeExtractor()
    for clip_id in clip_ids:
        source = step_by_clip[clip_id]
        kick_frame = int(kick_rows[clip_id]["kick_frame"])
        video_path = str(source["video_path"])
        frame_ids, frames, fps, width, height = _read_window(video_path, kick_frame - pre_frames, kick_frame + post_frames)
        goalkeeper = goalkeeper_by_clip.get(clip_id, {})
        goalkeeper_candidate = goalkeeper.get("selected_candidate") or {}
        if not goalkeeper_candidate:
            fallback_candidates = [
                item for item in (goalkeeper.get("track_candidates") or [])
                if item.get("baseline_x") is not None and item.get("baseline_y") is not None
                and (float(item.get("baseline_x") or 0.0) != 0.0 or float(item.get("baseline_y") or 0.0) != 0.0)
            ]
            qualified = [item for item in fallback_candidates if item.get("qualifies")]
            goalkeeper_candidate = max(qualified or fallback_candidates, key=lambda item: float(item.get("score") or 0.0), default={})
        goal_center = None
        if goalkeeper_candidate.get("baseline_x") is not None and goalkeeper_candidate.get("baseline_y") is not None:
            goal_center = (float(goalkeeper_candidate["baseline_x"]), float(goalkeeper_candidate["baseline_y"]))
        anchors = _pose_anchors(source, kick_frame, goal_center)
        predictions = model.predict(frames, imgsz=image_size, conf=detector_confidence, classes=[32], verbose=False) if frames else []
        detections = _extract_detections(frame_ids, predictions, frames)
        track, selection = select_ball_track(detections, kick_frame, anchors, goal_center=goal_center)
        trajectory = [(item["frame"] / fps, item["x"], item["y"]) for item in track]
        track_features = extractor.extract_trajectory_and_speed(
            trajectory,
            fps=fps,
            frame_width=width,
            frame_height=height,
            expected_frames=(int(track[-1]["frame"]) - int(track[0]["frame"]) + 1) if track else len(frame_ids),
            confidences=[item["confidence"] for item in track],
        )
        outcome_annotation = outcome_rows[clip_id]
        final_position = track_features["final_position"]
        normalized_position = track_features["final_position_normalized"]
        event_position = track_features["event_final_position"]
        event_normalized = track_features["event_final_position_normalized"]
        outcome = extractor.determine_outcome(final_position, observed_outcome=outcome_annotation["outcome"])
        selection_review = bool(
            selection.get("reason") != "selected"
            or selection.get("contact_anchor_fallback")
            or selection.get("goal_center") is None
            or float(selection.get("progress_to_goal_px") or -999.0) < 45.0
            or float(selection.get("goal_alignment") or -1.0) < 0.45
        )
        review_required = bool(track_features["ball_review_required"] or selection_review)
        row = {
            "clip_id": clip_id,
            "match_key": source.get("match_key"),
            "penalty_number": source.get("penalty_number"),
            "kicker": source.get("kicker"),
            "team": source.get("team"),
            "kick_frame": kick_frame,
            "kick_frame_source": kick_rows[clip_id].get("source", "unknown"),
            "input_annotation_status": "human_verified" if is_human_verified(kick_rows[clip_id]) else "ai_assisted_unverified",
            "fps": round(fps, 4),
            "frame_width": width,
            "frame_height": height,
            "ball_track_start_frame": int(track[0]["frame"]) if track else None,
            "ball_track_end_frame": int(track[-1]["frame"]) if track else None,
            **track_features,
            "final_position_x_px": final_position[0] if final_position is not None else None,
            "final_position_y_px": final_position[1] if final_position is not None else None,
            "final_position_x_norm": normalized_position[0] if normalized_position is not None else None,
            "final_position_y_norm": normalized_position[1] if normalized_position is not None else None,
            "last_observed_position_x_px": final_position[0] if final_position is not None else None,
            "last_observed_position_y_px": final_position[1] if final_position is not None else None,
            "event_final_position_x_px": event_position[0] if event_position is not None else None,
            "event_final_position_y_px": event_position[1] if event_position is not None else None,
            "event_final_position_x_norm": event_normalized[0] if event_normalized is not None else None,
            "event_final_position_y_norm": event_normalized[1] if event_normalized is not None else None,
            "outcome": outcome["outcome"],
            "outcome_detail": outcome_annotation["outcome_detail"],
            "outcome_confidence": outcome_annotation["confidence"],
            "outcome_source": outcome_annotation["source"],
            "ball_review_required": review_required,
            "ball_track_status": "review_required" if review_required else "auto_accepted",
        }
        # Tuple-valued fields are represented by explicit scalar CSV columns.
        row.pop("final_position")
        row.pop("final_position_normalized")
        row.pop("last_observed_position")
        row.pop("last_observed_position_normalized")
        row.pop("event_final_position")
        row.pop("event_final_position_normalized")
        features.append(row)
        rich_results.append(
            {
                **row,
                "video_path": video_path,
                "pose_anchors": anchors,
                "selection": selection,
                "trajectory": track,
                "detections": {str(key): value for key, value in detections.items()},
            }
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]) if features else ["clip_id"])
        writer.writeheader()
        writer.writerows(features)
    diagnostic_path = Path(diagnostics_json)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(rich_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return features
