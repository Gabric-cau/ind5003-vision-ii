"""Manifest-aligned goalkeeper analysis from retained pose evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from src.goalkeeper_analysis import GoalkeeperFeatureExtractor
from src.annotation_provenance import is_human_verified, require_verified


LEFT_HIP = 11
RIGHT_HIP = 12


def _centre_and_scale(observation: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    points = np.asarray(observation.get("keypoints", []), dtype=float)
    confidence = np.asarray(observation.get("keypoint_confidences", []), dtype=float)
    box = np.asarray(observation.get("bbox", [0, 0, 1, 1]), dtype=float)
    scale = max(1.0, float(box[3] - box[1]))
    if points.shape == (17, 2) and confidence.shape == (17,) and np.all(
        confidence[[LEFT_HIP, RIGHT_HIP]] >= 0.10
    ):
        centre = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
        quality = float(np.mean(confidence[[LEFT_HIP, RIGHT_HIP]]))
    else:
        centre = np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        quality = float(observation.get("box_confidence") or 0.0) * 0.75
    return float(centre[0]), float(centre[1]), scale, quality


def _trajectory(
    observations: Sequence[Mapping[str, Any]],
    start_frame: int,
    end_frame: int,
    fps: float,
) -> List[Tuple[float, float, float, float]]:
    rows: List[Tuple[float, float, float, float]] = []
    seen: set[int] = set()
    for observation in observations:
        frame = int(observation["frame"])
        if frame in seen or not start_frame <= frame <= end_frame:
            continue
        seen.add(frame)
        x, y, scale, _ = _centre_and_scale(observation)
        rows.append((frame / fps, x, y, scale))
    return sorted(rows)


def _stabilize_trajectories(
    video_path: str,
    pre: Sequence[Sequence[float]],
    post: Sequence[Sequence[float]],
    fps: float,
    *,
    person_boxes_by_frame: Optional[Mapping[int, Sequence[Sequence[float]]]] = None,
    use_field_mask: bool = False,
) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]], float]:
    """Return only the prefix with a complete camera chain into the first frame.

    A missing decoded frame or failed pair cannot be repaired by applying the
    next pair: its displacement relative to the first frame is then unknown.
    """

    if not pre and not post:
        return [], [], 0.0
    import cv2

    all_rows = list(pre) + list(post)
    frames = sorted({int(round(float(row[0]) * fps)) for row in all_rows})
    start, end = min(frames), max(frames)
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, previous = capture.read()
    if not ok:
        capture.release()
        return [], [], 0.0
    previous_bgr = previous
    previous = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
    transforms: Dict[int, np.ndarray] = {start: np.eye(3, dtype=float)}
    forward = np.eye(3, dtype=float)
    successes = 0
    try:
        for frame_id in range(start + 1, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            current = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
            feature_mask = None
            if use_field_mask:
                from src.field_masks import background_feature_mask
                feature_mask, _ = background_feature_mask(
                    previous_bgr, (person_boxes_by_frame or {}).get(frame_id - 1, []))
                feature_mask = cv2.resize(feature_mask, (previous.shape[1], previous.shape[0]), interpolation=cv2.INTER_NEAREST)
            corners = cv2.goodFeaturesToTrack(previous, maxCorners=500, qualityLevel=0.01, minDistance=8, mask=feature_mask)
            matrix = None
            if corners is not None and len(corners) >= 8:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
                if tracked is not None and status is not None:
                    valid = status.reshape(-1).astype(bool)
                    if np.sum(valid) >= 8:
                        matrix, _ = cv2.estimateAffinePartial2D(
                            corners[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=2.0
                        )
            if (matrix is None or not np.all(np.isfinite(matrix))
                    or abs(float(np.linalg.det(matrix[:, :2]))) < 1e-9):
                break
            full = np.eye(3, dtype=float)
            full[:2] = matrix
            forward = full @ forward
            successes += 1
            transforms[frame_id] = np.linalg.inv(forward)
            previous = current
            previous_bgr = frame
    finally:
        capture.release()

    def convert(rows):
        converted = []
        for row in rows:
            frame_id = int(round(float(row[0]) * fps))
            transform = transforms.get(frame_id)
            if transform is None:
                continue
            point = transform @ np.asarray([float(row[1]) * 0.5, float(row[2]) * 0.5, 1.0])
            linear_scale = math.sqrt(abs(float(np.linalg.det(transform[:2, :2]))))
            converted.append((float(row[0]), float(point[0] * 2.0), float(point[1] * 2.0), float(row[3]) * linear_scale))
        return converted

    success_rate = successes / max(1, end - start)
    return convert(pre), convert(post), float(success_rate)


def _candidate_summary(
    track: Mapping[str, Any],
    *,
    kick_frame: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    pre_window_sec: float,
    post_window_sec: float,
    kicker_centre: Optional[np.ndarray],
    is_selected_kicker: bool,
) -> Dict[str, Any]:
    observations = track.get("observations", [])
    pre_start = int(round(kick_frame - pre_window_sec * fps))
    post_end = int(round(kick_frame + post_window_sec * fps))
    pre = _trajectory(observations, pre_start, kick_frame, fps)
    post = _trajectory(observations, kick_frame, post_end, fps)
    pre_expected = max(1, kick_frame - pre_start + 1)
    post_expected = max(1, post_end - kick_frame + 1)
    pre_coverage = min(1.0, len(pre) / pre_expected)
    post_coverage = min(1.0, len(post) / post_expected)

    quality_values = [
        _centre_and_scale(item)[3]
        for item in observations
        if pre_start <= int(item["frame"]) <= post_end
    ]
    pose_quality = float(np.median(quality_values)) if quality_values else 0.0
    if pre:
        baseline = np.median(np.asarray([[row[1], row[2]] for row in pre[-5:]]), axis=0)
        scale = max(1.0, float(np.median([row[3] for row in pre])))
    else:
        baseline = np.zeros(2, dtype=float)
        scale = 1.0
    if post:
        post_centres = np.asarray([[row[1], row[2]] for row in post])
        displacement = np.linalg.norm(post_centres - baseline, axis=1) / scale
        max_displacement = float(np.max(displacement))
    else:
        max_displacement = 0.0
    if len(pre) >= 2:
        pre_values = np.asarray(pre, dtype=float)
        dt = np.diff(pre_values[:, 0])
        valid_dt = (dt > 0) & (dt <= 0.25)
        pre_speed = np.hypot(np.diff(pre_values[:, 1]), np.diff(pre_values[:, 2]))
        pre_speed = pre_speed / np.maximum(dt, 1e-9) / np.maximum(
            (pre_values[:-1, 3] + pre_values[1:, 3]) / 2.0, 1.0
        )
        pre_speed_body_s = float(np.median(pre_speed[valid_dt])) if np.any(valid_dt) else 0.0
        pre_path_range_body = float(
            np.linalg.norm(np.quantile(pre_values[:, 1:3], 0.90, axis=0) - np.quantile(pre_values[:, 1:3], 0.10, axis=0))
            / scale
        )
    else:
        pre_speed_body_s = 0.0
        pre_path_range_body = 0.0
    distance_from_kicker = (
        float(np.linalg.norm(baseline - kicker_centre) / max(1, frame_width))
        if kicker_centre is not None and pre
        else 0.0
    )
    upper_field_prior = float(np.clip(1.0 - baseline[1] / max(1, frame_height), 0.0, 1.0)) if pre else 0.0
    distant_player_prior = float(np.clip(1.0 - scale / max(1, frame_height * 0.35), 0.0, 1.0)) if pre else 0.0

    # Coverage and a goalkeeper-like post-contact displacement dominate. The
    # known kicker track is strongly penalised but remains auditable.
    score = (
        1.6 * pre_coverage
        + 1.2 * post_coverage
        + min(1.8, max_displacement) * 1.7
        + 0.5 * pose_quality
        + min(0.8, distance_from_kicker * 2.0)
        + 3.8 * upper_field_prior
        + 0.60 * distant_player_prior
        - min(2.2, pre_speed_body_s * 1.8)
        - min(1.2, pre_path_range_body * 1.2)
    )
    qualifies = bool(len(pre) >= 4 and len(post) >= 2 and pre_coverage >= 0.10)
    return {
        "track_id": int(track.get("track_id", -1)),
        "score": round(score, 5),
        "qualifies": qualifies,
        "is_selected_kicker": is_selected_kicker,
        "pre_count": len(pre),
        "post_count": len(post),
        "pre_coverage": round(pre_coverage, 4),
        "post_coverage": round(post_coverage, 4),
        "pose_quality": round(pose_quality, 4),
        "max_post_displacement_body": round(max_displacement, 4),
        "pre_speed_body_s": round(pre_speed_body_s, 4),
        "pre_path_range_body": round(pre_path_range_body, 4),
        "distance_from_kicker_frame_width": round(distance_from_kicker, 4),
        "upper_field_prior": round(upper_field_prior, 4),
        "distant_player_prior": round(distant_player_prior, 4),
        "baseline_x": round(float(baseline[0]), 3),
        "baseline_y": round(float(baseline[1]), 3),
        "body_scale_px": round(float(scale), 3),
        "pre_trajectory": pre,
        "post_trajectory": post,
    }


def select_goalkeeper_track(
    result: Mapping[str, Any],
    kick_frame: int,
    *,
    pre_window_sec: float = 1.5,
    post_window_sec: float = 1.0,
    seed_track_id: int | None = None,
    tracking_policy: str = "legacy",
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rank track candidates and return the best qualified goalkeeper."""

    if tracking_policy not in {"legacy", "bounded", "appearance_bounded"}:
        raise ValueError("Unknown tracking_policy")
    fps = float(result.get("fps") or 25.0)
    width = int(result.get("frame_width") or 1)
    height = int(result.get("frame_height") or 1)
    selected_kicker = result.get("selected_track_id")
    tracks = [track for track in result.get("tracks", []) if isinstance(track, Mapping)]
    kicker_centre: Optional[np.ndarray] = None
    for track in tracks:
        if track.get("track_id") != selected_kicker:
            continue
        sequence = _trajectory(
            track.get("observations", []),
            int(round(kick_frame - 0.25 * fps)),
            kick_frame,
            fps,
        )
        if sequence:
            kicker_centre = np.median(np.asarray([[row[1], row[2]] for row in sequence]), axis=0)

    summaries = [
        _candidate_summary(
            track,
            kick_frame=kick_frame,
            fps=fps,
            frame_width=width,
            frame_height=height,
            pre_window_sec=pre_window_sec,
            post_window_sec=post_window_sec,
            kicker_centre=kicker_centre,
            is_selected_kicker=track.get("track_id") == selected_kicker,
        )
        for track in tracks
    ]
    # A goalkeeper must stand on the playing surface. This inexpensive image
    # cue rejects spectators, stewards, and substitutes detected behind the
    # advertising boards or goal net.
    try:
        import cv2

        sample_frame = max(0, int(round(kick_frame - 0.5 * fps)))
        capture = cv2.VideoCapture(str(result.get("video_path", "")))
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
        ok, frame = capture.read()
        capture.release()
        if ok:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for item in summaries:
                scale = max(8.0, float(item["body_scale_px"]))
                cx = float(item["baseline_x"])
                cy = float(item["baseline_y"]) + 0.48 * scale
                x0 = max(0, int(cx - 0.42 * scale))
                x1 = min(width, int(cx + 0.42 * scale))
                y0 = max(0, int(cy - 0.08 * scale))
                y1 = min(height, int(cy + 0.28 * scale))
                patch = hsv[y0:y1, x0:x1]
                if patch.size:
                    green = (patch[..., 0] >= 28) & (patch[..., 0] <= 95) & (patch[..., 1] >= 45)
                    green_ratio = float(np.mean(green))
                else:
                    green_ratio = 0.0
                item["field_green_ratio"] = round(green_ratio, 4)
                item["score"] = round(float(item["score"]) + 2.4 * green_ratio, 5)
    except Exception:
        for item in summaries:
            item["field_green_ratio"] = None
    # Seed the goalkeeper close to the goal line, then stitch across tracker-ID
    # changes caused by the rapid dive. Referees and kickers are closer to the
    # camera (larger y); touchline officials are rejected near image edges.
    seed_pool = [
        item
        for item in summaries
        if item["pre_count"] >= 2
        and (item.get("field_green_ratio") is None or item["field_green_ratio"] >= 0.45)
        and 0.10 <= float(item["baseline_x"]) / max(1, width) <= 0.88
    ]
    seed = None
    if seed_track_id is not None:
        seed = next(
            (item for item in summaries if int(item["track_id"]) == int(seed_track_id)),
            None,
        )
    if seed_pool:
        seed = seed or max(
            seed_pool,
            key=lambda item: (
                5.0 * float(item["upper_field_prior"])
                + 0.8 * float(item["distant_player_prior"])
                + 0.5 * min(1.0, float(item["pre_coverage"]))
                + 0.5 * min(1.0, float(item["max_post_displacement_body"]))
                - 0.4 * min(2.0, float(item["pre_speed_body_s"])),
                -int(item["track_id"]),
            ),
        )

    stitched = None
    if seed is not None and seed["pre_trajectory"]:
        pool_by_frame: Dict[int, List[Tuple[float, float, float, float, int]]] = {}
        for item in summaries:
            for row in item["pre_trajectory"] + item["post_trajectory"]:
                frame = int(round(float(row[0]) * fps))
                pool_by_frame.setdefault(frame, []).append((*row, int(item["track_id"])))

        anchor_row = min(
            seed["pre_trajectory"],
            key=lambda row: abs(float(row[0]) * fps - kick_frame),
        )
        anchor = np.asarray(anchor_row[1:3], dtype=float)
        anchor_scale = float(anchor_row[3])
        appearance = {}
        appearance_diagnostics = {"status":"not_requested"}
        if tracking_policy == "appearance_bounded":
            from src.person_appearance import appearance_scores
            appearance,appearance_diagnostics=appearance_scores(result,int(seed["track_id"]),int(round(anchor_row[0]*fps)),
                max(0,int(round(kick_frame-pre_window_sec*fps))),int(round(kick_frame+post_window_sec*fps)))

        def follow(frame_ids: Sequence[int], start: np.ndarray, start_scale: float):
            chosen = []
            previous = start.copy()
            previous_scale = start_scale
            missing = 0
            for frame in frame_ids:
                options = pool_by_frame.get(frame, [])
                ranked = []
                for row in options:
                    color_similarity=appearance.get((frame,int(row[4])))
                    if tracking_policy == "appearance_bounded" and (color_similarity is None or color_similarity<.50):
                        continue
                    bounds=result.get("goal_geometry_bounds")
                    if tracking_policy == "appearance_bounded" and bounds:
                        gh=max(1,bounds[3]-bounds[1])
                        if not bounds[1]-.15*gh <= row[2] <= bounds[3]+1.30*gh:
                            continue
                    point = np.asarray(row[1:3], dtype=float)
                    scale = max(1.0, float(row[3]))
                    distance = float(np.linalg.norm(point - previous) / max(previous_scale, scale, 1.0))
                    scale_change = abs(math.log(scale / max(previous_scale, 1.0)))
                    appearance_cost=.10*(1-color_similarity) if tracking_policy == "appearance_bounded" else 0
                    ranked.append((distance + 0.25 * scale_change + appearance_cost, row))
                if ranked:
                    _, best = min(ranked, key=lambda pair: pair[0])
                    # Do not bridge a long detector gap by jumping from the
                    # goalkeeper to a nearby referee. A real per-frame dive
                    # moves much less than one body height between detections.
                    gate = 0.82
                    if min(ranked, key=lambda pair: pair[0])[0] <= gate:
                        chosen.append(best)
                        previous = np.asarray(best[1:3], dtype=float)
                        previous_scale = float(best[3])
                        missing = 0
                        continue
                missing += 1
                if tracking_policy != "legacy" and missing > max(2, int(round(.20 * fps))):
                    # Stop, rather than reconnecting an identity after an
                    # arbitrarily long absence. Missing samples remain missing.
                    break
            return chosen

        pre_start = int(round(kick_frame - pre_window_sec * fps))
        post_end = int(round(kick_frame + post_window_sec * fps))
        backward = follow(range(kick_frame, pre_start - 1, -1), anchor, anchor_scale)
        baseline_row = backward[0] if backward else (*anchor_row, int(seed["track_id"]))
        forward = follow(
            range(kick_frame + 1, post_end + 1),
            np.asarray(baseline_row[1:3], dtype=float),
            float(baseline_row[3]),
        )
        if tracking_policy != "legacy":
            # The seed location belongs to its actual observation time, not
            # necessarily the kick frame. Never teleport a stale seed forward.
            anchor_frame = int(round(anchor_row[0] * fps))
            seeded_row = (*anchor_row, int(seed["track_id"]))
            backward = [seeded_row] + follow(range(anchor_frame-1, pre_start-1, -1), anchor, anchor_scale)
            forward = follow(range(anchor_frame+1, post_end+1), anchor, anchor_scale)
            all_rows = sorted(backward + forward)
            backward = [r for r in all_rows if int(round(r[0]*fps)) <= kick_frame]
            forward = [r for r in all_rows if int(round(r[0]*fps)) > kick_frame]
            baseline_row = backward[-1] if backward else seeded_row
        pre_trajectory = sorted([tuple(row[:4]) for row in backward])
        post_trajectory = [tuple(baseline_row[:4])] + [tuple(row[:4]) for row in forward]
        used_ids = sorted({int(row[4]) for row in backward + forward})
        pre_coverage = len(pre_trajectory) / max(1, kick_frame - pre_start + 1)
        post_coverage = len(post_trajectory) / max(1, post_end - kick_frame + 1)
        stitched = {
            **seed,
            "track_id": "stitched:" + "+".join(str(item) for item in used_ids),
            "source_track_ids": used_ids,
            "pre_trajectory": pre_trajectory,
            "post_trajectory": post_trajectory,
            "pre_count": len(pre_trajectory),
            "post_count": len(post_trajectory),
            "pre_coverage": round(min(1.0, pre_coverage), 4),
            "post_coverage": round(min(1.0, post_coverage), 4),
            "qualifies": bool(len(pre_trajectory) >= 4 and len(post_trajectory) >= 2),
            "selection_method": "goal_line_seed_plus_temporal_stitching",
            "tracking_policy": tracking_policy,
            "max_gap_seconds": .20 if tracking_policy != "legacy" else None,
            "appearance_diagnostics": appearance_diagnostics,
        }

    summaries.sort(key=lambda item: (-float(item["score"]), int(item["track_id"])))
    if seed_track_id is not None:
        selected = stitched if stitched and stitched["qualifies"] else None
    elif tracking_policy != "legacy":
        selected = stitched if stitched and stitched["qualifies"] else None
    else:
        selected = stitched if stitched and stitched["qualifies"] else next(
            (item for item in summaries if item["qualifies"]), None
        )
    return selected, summaries


def _has_distinct_competitor(selected, candidates):
    """A selected fragment must not be counted as its own competing person."""
    own_ids = set(selected.get("source_track_ids") or [selected["track_id"]])
    return any(item["track_id"] not in own_ids and item.get("qualifies")
               and item["score"] >= selected["score"] * .90 for item in candidates)


def analyze_goalkeeper_result(
    result: Mapping[str, Any],
    kick_frame: int,
    *,
    kick_frame_confidence: str = "unknown",
    kick_frame_source: str = "unknown",
    pre_window_sec: float = 1.5,
    post_window_sec: float = 1.0,
    extractor: Optional[GoalkeeperFeatureExtractor] = None,
    seed_track_id: int | None = None,
    use_field_mask: bool = False,
    tracking_policy: str = "legacy",
) -> Dict[str, Any]:
    extractor = extractor or GoalkeeperFeatureExtractor()
    selected, candidates = select_goalkeeper_track(
        result,
        kick_frame,
        pre_window_sec=pre_window_sec,
        post_window_sec=post_window_sec,
        seed_track_id=seed_track_id,
        tracking_policy=tracking_policy,
    )
    warnings: List[str] = []
    pre_stabilization_complete = False
    dive_stabilization_complete = False
    if selected is None:
        warnings.append("no_qualified_goalkeeper_track")
        features = extractor.analyze([], [], pre_window_sec=pre_window_sec, post_window_sec=post_window_sec)
        selected_id = None
    else:
        boxes_by_frame: Dict[int, List[Sequence[float]]] = {}
        if use_field_mask:
            for track in result.get("tracks", []):
                for obs in track.get("observations", []):
                    boxes_by_frame.setdefault(int(obs["frame"]), []).append(obs["bbox"])
        stable_pre, stable_post, stabilization_success = _stabilize_trajectories(
            str(result.get("video_path", "")),
            selected["pre_trajectory"], selected["post_trajectory"],
            float(result.get("fps") or 25.0),
            person_boxes_by_frame=boxes_by_frame, use_field_mask=use_field_mask,
        )
        features = extractor.analyze(
            stable_pre,
            stable_post,
            pre_window_sec=pre_window_sec,
            post_window_sec=post_window_sec,
        )
        pre_stabilization_complete = bool(selected["pre_trajectory"]) and len(stable_pre) == len(selected["pre_trajectory"])
        post_stabilization_complete = bool(selected["post_trajectory"]) and len(stable_post) == len(selected["post_trajectory"])
        dive_stabilization_complete = pre_stabilization_complete and post_stabilization_complete
        if not dive_stabilization_complete:
            warnings.append("incomplete_camera_reference_chain")
            unknown_features = extractor.analyze([], [], pre_window_sec=pre_window_sec, post_window_sec=post_window_sec)
            for key, value in unknown_features.items():
                if key.startswith("dive_") or (not pre_stabilization_complete and key.startswith("gk_prekick_")):
                    features[key] = value
            features["dive_displacement_body"] = None
            features["review_required"] = True
        selected_id = str(selected["track_id"])
        selected["stabilized_pre_trajectory"] = stable_pre
        selected["stabilized_post_trajectory"] = stable_post
        selected["stabilization_success_rate"] = round(stabilization_success, 4)
        selected["camera_reference_chain_complete"] = dive_stabilization_complete
        if stabilization_success < 0.75:
            warnings.append("low_camera_stabilization_success")
        if selected["pre_coverage"] < 0.50:
            warnings.append("low_prekick_track_coverage")
        if selected["post_coverage"] < 0.35:
            warnings.append("low_postkick_track_coverage")
        ambiguous = (_has_distinct_competitor(selected, candidates) if tracking_policy != "legacy"
                     else len(candidates) > 1 and candidates[1]["score"] >= selected["score"] * .90)
        if seed_track_id is None and ambiguous:
            warnings.append("ambiguous_goalkeeper_track")

    review_required = bool(features["review_required"] or warnings)
    return {
        "clip_id": result.get("clip_id"),
        "video_path": result.get("video_path"),
        "status": "ok" if selected is not None else "review_required",
        "fps": result.get("fps"),
        "frame_width": result.get("frame_width"),
        "frame_height": result.get("frame_height"),
        "kick_frame": int(kick_frame),
        "kick_frame_confidence": kick_frame_confidence,
        "kick_frame_source": kick_frame_source,
        "tracking_policy": tracking_policy,
        "goalkeeper_track_id": selected_id,
        "gk_prekick_stabilization_complete": pre_stabilization_complete,
        "dive_stabilization_complete": dive_stabilization_complete,
        **features,
        "review_required": review_required,
        "auto_accepted": bool(not review_required),
        "qa_warnings": warnings,
        "selected_candidate": selected,
        "track_candidates": candidates,
    }


CSV_FIELDS = [
    "clip_id", "match_key", "match_title", "penalty_number", "kicker", "team", "video_path",
    "status", "fps", "kick_frame", "kick_frame_confidence", "kick_frame_source",
    "goalkeeper_track_id", "goalkeeper_identity_reviewed", "goalkeeper_identity_confidence",
    "input_annotation_status",
    "gk_prekick_stabilization_complete", "dive_stabilization_complete",
    "gk_prekick_movement_level", "gk_prekick_mean_speed_px_s",
    "gk_prekick_speed_body_s", "gk_prekick_lateral_range_body", "gk_prekick_coverage",
    "gk_prekick_confidence", "dive_direction", "dive_height", "dive_lateral_displacement_body",
    "dive_vertical_displacement_body", "dive_displacement_body", "dive_coverage", "dive_confidence",
    "review_required", "auto_accepted", "qa_warnings",
]


def _manifest_metadata(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    metadata: Dict[str, Dict[str, Any]] = {}
    for match_key, match in manifest.items():
        for penalty in match.get("penalties", []):
            clip_id = f"{match_key}_p{int(penalty['number']):02d}_{penalty['kicker']}"
            metadata[clip_id] = {
                "match_key": match_key,
                "match_title": match.get("match_title"),
                "penalty_number": int(penalty["number"]),
                "kicker": penalty["kicker"],
                "team": penalty.get("team"),
            }
    return metadata


def process_goalkeeper_diagnostics(
    step_diagnostics_json: str,
    kick_frames_csv: str,
    manifest_path: str,
    output_csv: str,
    diagnostics_json: str,
    *,
    fallback_pose_json: str | None = None,
    track_reviews_csv: str | None = None,
    pre_window_sec: float = 1.5,
    post_window_sec: float = 1.0,
    allow_unverified_annotations: bool = False,
    use_field_mask: bool = False,
) -> List[Dict[str, Any]]:
    """Produce one reproducible goalkeeper record per manifest penalty."""

    step_results = json.loads(Path(step_diagnostics_json).read_text(encoding="utf-8"))
    result_by_id = {str(item["clip_id"]): item for item in step_results}
    if fallback_pose_json and Path(fallback_pose_json).is_file():
        fallback_results = json.loads(Path(fallback_pose_json).read_text(encoding="utf-8"))
        for item in fallback_results:
            result_by_id[str(item["clip_id"])] = item
    with Path(kick_frames_csv).open("r", encoding="utf-8", newline="") as handle:
        kick_by_id = {row["clip_id"]: row for row in csv.DictReader(handle)}
    metadata = _manifest_metadata(Path(manifest_path))
    seed_reviews: Dict[str, Dict[str, str]] = {}
    if track_reviews_csv and Path(track_reviews_csv).is_file():
        with Path(track_reviews_csv).open("r", encoding="utf-8", newline="") as handle:
            seed_reviews = {row["clip_id"]: row for row in csv.DictReader(handle)}
    if set(kick_by_id) != set(metadata):
        raise ValueError("Kick-frame annotations must match the canonical manifest exactly")
    require_verified(kick_by_id.values(), "Kick frames", allow_unverified_annotations)
    require_verified(seed_reviews.values(), "Goalkeeper seeds", allow_unverified_annotations)

    extractor = GoalkeeperFeatureExtractor()
    outputs: List[Dict[str, Any]] = []
    for clip_id, meta in metadata.items():
        source = result_by_id.get(clip_id, {"clip_id": clip_id, "tracks": []})
        kick = kick_by_id[clip_id]
        seed_review = seed_reviews.get(clip_id)
        output = analyze_goalkeeper_result(
            source,
            int(kick["kick_frame"]),
            kick_frame_confidence=kick.get("confidence", "unknown"),
            kick_frame_source=kick.get("source", "unknown"),
            pre_window_sec=pre_window_sec,
            post_window_sec=post_window_sec,
            extractor=extractor,
            use_field_mask=use_field_mask,
            seed_track_id=(
                int(seed_review["goalkeeper_seed_track_id"])
                if seed_review and seed_review.get("goalkeeper_seed_track_id")
                else None
            ),
        )
        output["goalkeeper_identity_reviewed"] = is_human_verified(seed_review or {})
        output["input_annotation_status"] = (
            "human_verified" if is_human_verified(kick) and (not seed_review or is_human_verified(seed_review))
            else "ai_assisted_unverified"
        )
        output["goalkeeper_identity_confidence"] = (
            seed_review.get("confidence") if seed_review else "automatic"
        )
        output.update(meta)
        outputs.append(output)

    csv_path = Path(output_csv)
    json_path = Path(diagnostics_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for output in outputs:
            row = dict(output)
            row["qa_warnings"] = json.dumps(row.get("qa_warnings", []), separators=(",", ":"))
            writer.writerow(row)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return outputs


def infer_goalkeeper_pose_window(
    video_path: str,
    model: Any,
    kick_frame: int,
    *,
    clip_id: str,
    fps: float = 30.0,
    post_window_sec: float = 1.0,
    batch_size: int = 8,
    pose_conf: float = 0.08,
    imgsz: int = 960,
) -> Dict[str, Any]:
    """Run a targeted high-resolution pose pass through the dive window."""

    import cv2
    from src.step_pipeline import _build_tracks, _run_pose_in_batches

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    try:
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    end_frame = int(round(kick_frame + post_window_sec * actual_fps)) + 1
    if frame_count > 0:
        end_frame = min(end_frame, frame_count)
    pose_frames, warnings = _run_pose_in_batches(
        video_path,
        end_frame,
        model,
        batch_size=batch_size,
        pose_conf=pose_conf,
        predict_kwargs={"imgsz": imgsz, "conf": pose_conf},
        roi_inference_mode="goal_band",
    )
    tracks = _build_tracks(pose_frames)
    return {
        "clip_id": clip_id,
        "video_path": video_path,
        "fps": actual_fps,
        "frame_width": width,
        "frame_height": height,
        "selected_track_id": None,
        "pose_mode": "goal_band_high_resolution",
        "pose_warnings": warnings,
        "pose_frames": pose_frames,
        "tracks": [tracks[key] for key in sorted(tracks)],
    }


__all__ = [
    "analyze_goalkeeper_result",
    "infer_goalkeeper_pose_window",
    "process_goalkeeper_diagnostics",
    "select_goalkeeper_track",
]
