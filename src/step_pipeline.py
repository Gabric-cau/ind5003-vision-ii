"""Pose-based penalty run-up step counting.

This module deliberately does not import or construct an Ultralytics model.  A
caller-owned pose model is passed to :func:`process_video_steps` (or shared by
all clips in :func:`process_manifest_steps`), which avoids loading model
weights once per video.

The pipeline is intentionally independent of ByteTrack/LAP.  It associates
people with a deterministic greedy nearest-centre/box-scale tracker and keeps
the original per-frame boxes, COCO-17 keypoints, and keypoint confidences for
QA.  Camera shots are analysed independently in chronological order until a
qualified full-body kicker candidate is found. Frame IDs remain clip-absolute.
When full-frame pose inference has no credible kicker candidate, ``roi_mode``
``"auto"`` retries overlapping lower-left/lower-right crops and maps every
coordinate back to the original frame.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from scipy.signal import find_peaks

from src.camera_angles import ShotDetector
from src.kicker_analysis import KickerFeatureExtractor


# COCO pose indices used by Ultralytics pose models.
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_ANKLE = 15
RIGHT_ANKLE = 16
COCO_KEYPOINT_COUNT = 17


def _append_warning(warnings: List[str], warning: str) -> None:
    """Append a warning once while preserving deterministic order."""

    if warning and warning not in warnings:
        warnings.append(warning)


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert torch/Ultralytics-like objects to NumPy without importing torch."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy values and non-finite floats for strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _normalise_keypoints(
    xy: np.ndarray,
    confidences: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return exactly ``(N, 17, 2)`` coordinates and ``(N, 17)`` confidence."""

    if xy.ndim == 2:
        xy = xy[None, ...]
    if xy.ndim != 3 or xy.shape[-1] < 2:
        return (
            np.empty((0, COCO_KEYPOINT_COUNT, 2), dtype=float),
            np.empty((0, COCO_KEYPOINT_COUNT), dtype=float),
            confidences is None,
        )

    xy = np.asarray(xy[..., :2], dtype=float)
    n_people = int(xy.shape[0])
    fixed_xy = np.zeros((n_people, COCO_KEYPOINT_COUNT, 2), dtype=float)
    copied = min(COCO_KEYPOINT_COUNT, int(xy.shape[1]))
    fixed_xy[:, :copied] = xy[:, :copied]

    confidence_missing = confidences is None
    if confidences is None:
        finite = np.all(np.isfinite(fixed_xy), axis=2)
        nonzero = np.any(np.abs(fixed_xy) > 1e-9, axis=2)
        fixed_conf = (finite & nonzero).astype(float)
    else:
        confidences = np.asarray(confidences, dtype=float)
        if confidences.ndim == 1:
            confidences = confidences[None, ...]
        fixed_conf = np.zeros((n_people, COCO_KEYPOINT_COUNT), dtype=float)
        if confidences.ndim == 2:
            rows = min(n_people, int(confidences.shape[0]))
            cols = min(COCO_KEYPOINT_COUNT, int(confidences.shape[1]))
            fixed_conf[:rows, :cols] = confidences[:rows, :cols]

    invalid = ~np.all(np.isfinite(fixed_xy), axis=2)
    fixed_xy[~np.isfinite(fixed_xy)] = 0.0
    fixed_conf[~np.isfinite(fixed_conf)] = 0.0
    fixed_conf[invalid] = 0.0
    fixed_conf = np.clip(fixed_conf, 0.0, 1.0)
    return fixed_xy, fixed_conf, confidence_missing


def _box_from_keypoints(keypoints: np.ndarray, confidences: np.ndarray) -> List[float]:
    """Build a conservative box when a pose result unexpectedly lacks boxes."""

    usable = (confidences > 0.0) & np.all(np.isfinite(keypoints), axis=1)
    if not np.any(usable):
        return [0.0, 0.0, 1.0, 1.0]
    points = keypoints[usable]
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    pad_x = max(1.0, float(x2 - x1) * 0.08)
    pad_y = max(1.0, float(y2 - y1) * 0.08)
    return [float(x1 - pad_x), float(y1 - pad_y), float(x2 + pad_x), float(y2 + pad_y)]


def _extract_result_detections(result: Any, frame_id: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract JSON-friendly pose detections from one Ultralytics result."""

    warnings: List[str] = []
    keypoint_object = getattr(result, "keypoints", None)
    if keypoint_object is None:
        return [], warnings

    xy = _to_numpy(getattr(keypoint_object, "xy", None))
    confidence = _to_numpy(getattr(keypoint_object, "conf", None))
    if xy is None:
        data = _to_numpy(getattr(keypoint_object, "data", None))
        if data is None or data.ndim < 3 or data.shape[-1] < 2:
            return [], warnings
        xy = data[..., :2]
        if confidence is None and data.shape[-1] >= 3:
            confidence = data[..., 2]

    keypoints, keypoint_confidences, confidence_missing = _normalise_keypoints(xy, confidence)
    if confidence_missing and len(keypoints):
        _append_warning(warnings, "keypoint_confidences_unavailable")

    boxes_object = getattr(result, "boxes", None)
    boxes = _to_numpy(getattr(boxes_object, "xyxy", None)) if boxes_object is not None else None
    box_confidences = _to_numpy(getattr(boxes_object, "conf", None)) if boxes_object is not None else None
    if boxes is not None:
        boxes = np.asarray(boxes, dtype=float)
        if boxes.ndim == 1:
            boxes = boxes[None, ...]
    if box_confidences is not None:
        box_confidences = np.asarray(box_confidences, dtype=float).reshape(-1)

    detections: List[Dict[str, Any]] = []
    for person_index in range(len(keypoints)):
        if boxes is not None and person_index < len(boxes) and boxes.shape[1] >= 4:
            box = np.asarray(boxes[person_index, :4], dtype=float)
            if not np.all(np.isfinite(box)):
                box = np.asarray(
                    _box_from_keypoints(keypoints[person_index], keypoint_confidences[person_index]),
                    dtype=float,
                )
        else:
            box = np.asarray(
                _box_from_keypoints(keypoints[person_index], keypoint_confidences[person_index]),
                dtype=float,
            )

        # Normalise inverted/zero-size boxes before tracking.
        x1, y1, x2, y2 = (float(item) for item in box)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x2 = max(x2, x1 + 1.0)
        y2 = max(y2, y1 + 1.0)

        if box_confidences is not None and person_index < len(box_confidences):
            box_confidence: Optional[float] = float(box_confidences[person_index])
            if not math.isfinite(box_confidence):
                box_confidence = None
        else:
            box_confidence = None

        detections.append(
            {
                "frame": int(frame_id),
                "bbox": [x1, y1, x2, y2],
                "box_confidence": box_confidence,
                "keypoints": keypoints[person_index].astype(float).tolist(),
                "keypoint_confidences": keypoint_confidences[person_index].astype(float).tolist(),
            }
        )

    if boxes is not None and len(boxes) != len(keypoints):
        _append_warning(warnings, "pose_box_keypoint_count_mismatch")
    return detections, warnings


def _histogram_for_cut(frame: np.ndarray) -> np.ndarray:
    """Match the inexpensive HSV histogram used by ``ShotDetector``."""

    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    return histogram


def _detect_first_shot_end(
    video_path: str,
    max_shot_sec: float,
    scene_cut_threshold: float,
) -> Dict[str, Any]:
    """Find the first cut without scanning beyond the configured time cap.

    ``shot_end_frame`` is exclusive.  The cut frame itself therefore never
    enters pose inference.
    """

    if max_shot_sec <= 0:
        raise ValueError("max_shot_sec must be positive")

    # Reuse the repository's threshold semantics, while stopping early rather
    # than calling detect_cuts(), which scans the complete source clip.
    threshold = float(ShotDetector(threshold=scene_cut_threshold).threshold)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap_frames = max(1, int(math.ceil(max_shot_sec * fps)))
        minimum_cut_frame = max(1, int(round(0.50 * fps)))

        previous_histogram: Optional[np.ndarray] = None
        decoded_count = 0
        cut_frame: Optional[int] = None
        frame_index = 0

        while frame_index < cap_frames:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_count = frame_index + 1
            histogram = _histogram_for_cut(frame)
            if previous_histogram is not None and frame_index >= minimum_cut_frame:
                difference = float(
                    cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_CHISQR)
                )
                if math.isfinite(difference) and difference > threshold:
                    cut_frame = frame_index
                    break
            previous_histogram = histogram
            frame_index += 1

        if cut_frame is not None:
            shot_end_frame = int(cut_frame)
            reason = "scene_cut"
        else:
            shot_end_frame = int(decoded_count)
            reached_known_end = source_frame_count > 0 and decoded_count >= source_frame_count
            if decoded_count >= cap_frames and not reached_known_end:
                reason = "duration_cap"
            else:
                reason = "video_end"

        if shot_end_frame <= 0:
            raise ValueError(f"No readable frames in video: {video_path}")
        return {
            "fps": fps,
            "source_frame_count": source_frame_count,
            "frame_width": width,
            "frame_height": height,
            "shot_end_frame": shot_end_frame,
            "shot_end_sec": float(shot_end_frame / fps),
            "shot_end_reason": reason,
        }
    finally:
        capture.release()


def _scan_camera_shots(
    video_path: str, max_shot_sec: float, scene_cut_threshold: float,
) -> List[Dict[str, Any]]:
    """Scan the whole clip, without masks, then bound inference within each shot.

    Histogram changes alone miss a wide-to-close cut with the same grass colour.
    A large spatial luminance change provides a second, conservative cut cue.
    These are heuristic boundaries, not human-verified live/replay labels.
    """
    if max_shot_sec <= 0 or scene_cut_threshold <= 0:
        raise ValueError("Shot duration and scene-cut threshold must be positive")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boundaries = [0]
    previous_hist = previous_gray = None
    frame_index = 0
    min_gap = max(1, round(0.25 * fps))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hist = _histogram_for_cut(frame)
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if previous_hist is not None and frame_index - boundaries[-1] >= min_gap:
                hist_delta = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CHISQR))
                spatial = cv2.absdiff(gray, previous_gray)
                spatial_mean = float(spatial.mean())
                changed_fraction = float(np.mean(spatial > 30))
                spatial_cut = ((spatial_mean > 25.0 and changed_fraction > 0.35 and hist_delta > 2.0)
                               or (spatial_mean > 60.0 and changed_fraction > 0.75))
                if (math.isfinite(hist_delta) and hist_delta > scene_cut_threshold) or spatial_cut:
                    boundaries.append(frame_index)
            previous_hist, previous_gray = hist, gray
            frame_index += 1
    finally:
        cap.release()
    if frame_index == 0:
        raise ValueError(f"No readable frames in video: {video_path}")
    boundaries.append(frame_index)
    limit = max(1, int(math.ceil(max_shot_sec * fps)))
    shots = []
    for shot_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        analysis_end = min(end, start + limit)
        shots.append({
            "shot_index": shot_index, "shot_start_frame": start,
            "shot_start_sec": start / fps, "shot_end_frame": analysis_end,
            "shot_end_sec": analysis_end / fps, "camera_shot_end_frame": end,
            "shot_end_reason": ("duration_cap" if analysis_end < end else
                                "video_end" if end == frame_index else "scene_cut"),
            "fps": fps, "frame_width": width, "frame_height": height,
            "source_frame_count": frame_index,
        })
    return shots


def _normalise_prediction_results(prediction: Any) -> List[Any]:
    """Normalise model output while avoiding iteration over one Results object."""

    if prediction is None:
        return []
    if isinstance(prediction, (list, tuple)):
        return list(prediction)
    if hasattr(prediction, "boxes") or hasattr(prediction, "keypoints"):
        return [prediction]
    return list(prediction)


def _roi_specs(mode: str, width: int, height: int) -> List[Dict[str, Any]]:
    """Return full-frame, goalkeeper-band, or lower-field crop definitions."""

    if mode == "full":
        return [{"name": "full", "x0": 0, "y0": 0, "x1": width, "y1": height}]
    if mode == "goal_band":
        return [
            {
                "name": "goal_band",
                "x0": 0,
                "y0": 0,
                "x1": width,
                "y1": max(1, int(math.ceil(height * 0.85))),
            }
        ]
    if mode != "lower_tiles":
        raise ValueError("ROI inference mode must be 'full', 'goal_band', or 'lower_tiles'")

    # Each crop covers the lower two thirds vertically and two thirds
    # horizontally.  Their middle-third overlap prevents a player on the tile
    # boundary from being systematically truncated.
    y0 = max(0, height // 3)
    left_end = max(1, int(math.ceil(width * 2.0 / 3.0)))
    right_start = min(width - 1, width // 3) if width > 1 else 0
    return [
        {"name": "lower_left", "x0": 0, "y0": y0, "x1": left_end, "y1": height},
        {"name": "lower_right", "x0": right_start, "y0": y0, "x1": width, "y1": height},
    ]


def _prepare_roi(frame: np.ndarray, spec: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop and magnify an ROI to full-frame dimensions for small-player recall."""

    height, width = frame.shape[:2]
    x0 = int(spec["x0"])
    y0 = int(spec["y0"])
    x1 = int(spec["x1"])
    y1 = int(spec["y1"])
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Empty ROI generated for {spec['name']}")
    if x0 == 0 and y0 == 0 and x1 == width and y1 == height:
        image = frame
    else:
        image = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    transform = {
        "name": str(spec["name"]),
        "x0": float(x0),
        "y0": float(y0),
        "scale_x": float((x1 - x0) / max(width, 1)),
        "scale_y": float((y1 - y0) / max(height, 1)),
    }
    return image, transform


def _map_detection_to_source(detection: MutableMapping[str, Any], transform: Mapping[str, Any]) -> None:
    """Map a resized crop detection back into original video coordinates."""

    x0 = float(transform["x0"])
    y0 = float(transform["y0"])
    scale_x = float(transform["scale_x"])
    scale_y = float(transform["scale_y"])
    box = detection["bbox"]
    detection["bbox"] = [
        x0 + float(box[0]) * scale_x,
        y0 + float(box[1]) * scale_y,
        x0 + float(box[2]) * scale_x,
        y0 + float(box[3]) * scale_y,
    ]

    keypoints = np.asarray(detection["keypoints"], dtype=float)
    confidences = np.asarray(detection["keypoint_confidences"], dtype=float)
    valid = (confidences > 0.0) & np.all(np.isfinite(keypoints), axis=1)
    keypoints[valid, 0] = x0 + keypoints[valid, 0] * scale_x
    keypoints[valid, 1] = y0 + keypoints[valid, 1] * scale_y
    keypoints[~valid] = 0.0
    detection["keypoints"] = keypoints.tolist()
    detection["source_roi"] = str(transform["name"])


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _detection_quality(detection: Mapping[str, Any]) -> float:
    keypoint_confidences = np.asarray(detection["keypoint_confidences"], dtype=float)
    lower = keypoint_confidences[[LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]]
    lower_quality = float(np.mean(lower))
    box_quality = detection.get("box_confidence")
    return lower_quality + (0.20 * float(box_quality) if box_quality is not None else 0.0)


def _deduplicate_frame_detections(detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Suppress duplicate people produced in the overlap between ROI tiles."""

    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-_detection_quality(item[1]), item[0]),
    )
    kept: List[Dict[str, Any]] = []
    for _, detection in ranked:
        if any(_box_iou(detection["bbox"], existing["bbox"]) >= 0.55 for existing in kept):
            continue
        kept.append(detection)
    return sorted(
        kept,
        key=lambda detection: (
            (float(detection["bbox"][0]) + float(detection["bbox"][2])) / 2.0,
            (float(detection["bbox"][1]) + float(detection["bbox"][3])) / 2.0,
        ),
    )


def _run_pose_in_batches(
    video_path: str,
    shot_end_frame: int,
    model: Any,
    *,
    batch_size: int,
    pose_conf: Optional[float],
    predict_kwargs: Optional[Mapping[str, Any]],
    roi_inference_mode: str,
    shot_start_frame: int = 0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run caller-owned YOLO pose inference on bounded batches of frames/ROIs."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if model is None:
        raise ValueError("A caller-owned pose model must be supplied")
    if not 0 <= shot_start_frame < shot_end_frame:
        raise ValueError("Pose window must satisfy 0 <= start < end")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    warnings: List[str] = []
    detections_by_frame: Dict[int, List[Dict[str, Any]]] = {
        frame: [] for frame in range(shot_start_frame, shot_end_frame)
    }
    inputs: List[np.ndarray] = []
    contexts: List[Tuple[int, Dict[str, Any]]] = []

    kwargs = dict(predict_kwargs or {})
    kwargs.setdefault("verbose", False)
    if pose_conf is not None:
        kwargs.setdefault("conf", float(pose_conf))
    if roi_inference_mode in {"lower_tiles", "goal_band"}:
        # Distant kickers can be only ~60 px tall in the source frame.  The
        # cropped fallback needs the extra input resolution, while the normal
        # full-frame pass intentionally stays at the caller's (usually 640)
        # setting because that is more stable for nearby left/right ankles.
        requested_size = int(kwargs.get("imgsz", 640))
        kwargs["imgsz"] = max(960, requested_size)

    def flush_batch() -> None:
        if not inputs:
            return
        predictor = getattr(model, "predict", None)
        if predictor is None:
            raise TypeError("Pose model must expose a predict(...) method")
        prediction = predictor(list(inputs), **kwargs)
        results = _normalise_prediction_results(prediction)
        if len(results) != len(contexts):
            _append_warning(warnings, "pose_result_count_mismatch")

        for item_index, (frame_id, transform) in enumerate(contexts):
            if item_index >= len(results):
                continue
            frame_detections, frame_warnings = _extract_result_detections(
                results[item_index], frame_id
            )
            for warning in frame_warnings:
                _append_warning(warnings, warning)
            for detection in frame_detections:
                _map_detection_to_source(detection, transform)
                detections_by_frame[frame_id].append(detection)
        inputs.clear()
        contexts.clear()

    try:
        # Decode the prefix instead of depending on codec-specific seek precision.
        frame_index = 0
        while frame_index < shot_start_frame:
            if not capture.grab():
                raise ValueError("Video ended before the selected shot start")
            frame_index += 1
        while frame_index < shot_end_frame:
            ok, frame = capture.read()
            if not ok:
                _append_warning(warnings, "video_decode_ended_before_shot_end")
                break
            height, width = frame.shape[:2]
            for spec in _roi_specs(roi_inference_mode, width, height):
                image, transform = _prepare_roi(frame, spec)
                inputs.append(image)
                contexts.append((frame_index, transform))
                if len(inputs) >= batch_size:
                    flush_batch()
            frame_index += 1
        flush_batch()
    finally:
        capture.release()

    pose_frames = [
        {
            "frame": frame_id,
            "detections": _deduplicate_frame_detections(detections_by_frame[frame_id]),
        }
        for frame_id in range(shot_start_frame, shot_end_frame)
    ]
    return pose_frames, warnings


def _merge_pose_frames(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge full-frame and fallback ROI detections without duplicating people."""

    first_map = {int(item["frame"]): list(item.get("detections", [])) for item in first}
    second_map = {int(item["frame"]): list(item.get("detections", [])) for item in second}
    frame_ids = sorted(set(first_map) | set(second_map))
    return [
        {
            "frame": frame_id,
            "detections": _deduplicate_frame_detections(
                first_map.get(frame_id, []) + second_map.get(frame_id, [])
            ),
        }
        for frame_id in frame_ids
    ]


def _box_geometry(box: Sequence[float]) -> Tuple[np.ndarray, float, float]:
    width = max(1.0, float(box[2]) - float(box[0]))
    height = max(1.0, float(box[3]) - float(box[1]))
    centre = np.asarray(
        [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0],
        dtype=float,
    )
    return centre, math.sqrt(width * height), height


class _NearestCentreBoxTracker:
    """Small deterministic multi-person tracker with no external assignment API."""

    def __init__(
        self,
        *,
        max_frame_gap: int = 3,
        centre_gate: float = 1.35,
        max_scale_ratio: float = 3.0,
    ) -> None:
        self.max_frame_gap = int(max_frame_gap)
        self.centre_gate = float(centre_gate)
        self.max_scale_ratio = float(max_scale_ratio)
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 0

    @staticmethod
    def _predicted_centre(observations: Sequence[Mapping[str, Any]], frame: int) -> np.ndarray:
        last = observations[-1]
        last_centre, _, _ = _box_geometry(last["bbox"])
        if len(observations) < 2:
            return last_centre
        previous = observations[-2]
        previous_centre, _, _ = _box_geometry(previous["bbox"])
        dt = max(1, int(last["frame"]) - int(previous["frame"]))
        velocity = (last_centre - previous_centre) / float(dt)
        horizon = max(0, frame - int(last["frame"]))
        return last_centre + velocity * float(horizon)

    def _new_track(self, detection: Dict[str, Any]) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        detection["track_id"] = track_id
        self.tracks[track_id] = {"track_id": track_id, "observations": [dict(detection)]}
        return track_id

    def update(self, frame: int, detections: Sequence[Dict[str, Any]]) -> None:
        ordered = sorted(
            detections,
            key=lambda detection: (
                _box_geometry(detection["bbox"])[0][0],
                _box_geometry(detection["bbox"])[0][1],
            ),
        )
        active_track_ids = [
            track_id
            for track_id, track in sorted(self.tracks.items())
            if frame - int(track["observations"][-1]["frame"]) <= self.max_frame_gap
        ]

        candidate_pairs: List[Tuple[float, int, int]] = []
        for track_id in active_track_ids:
            track = self.tracks[track_id]
            previous = track["observations"][-1]
            predicted_centre = self._predicted_centre(track["observations"], frame)
            _, previous_scale, _ = _box_geometry(previous["bbox"])
            gap = max(1, frame - int(previous["frame"]))

            for detection_index, detection in enumerate(ordered):
                detection_centre, detection_scale, _ = _box_geometry(detection["bbox"])
                scale_ratio = max(previous_scale, detection_scale) / max(
                    1.0, min(previous_scale, detection_scale)
                )
                normaliser = max(1.0, previous_scale, detection_scale)
                centre_distance = float(np.linalg.norm(detection_centre - predicted_centre) / normaliser)
                expanded_gate = self.centre_gate * (1.0 + 0.20 * max(0, gap - 1))
                if centre_distance > expanded_gate or scale_ratio > self.max_scale_ratio:
                    continue
                cost = centre_distance + 0.30 * abs(math.log(max(scale_ratio, 1e-6)))
                candidate_pairs.append((cost, track_id, detection_index))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidate_pairs):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            detection = ordered[detection_index]
            detection["track_id"] = track_id
            self.tracks[track_id]["observations"].append(dict(detection))
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        for detection_index, detection in enumerate(ordered):
            if detection_index not in assigned_detections:
                self._new_track(detection)


def _track_pose_sequence(
    track: Mapping[str, Any],
    keypoint_conf: float,
) -> Dict[str, Any]:
    """Extract aligned, fully valid hip/ankle samples from a raw track."""

    frames: List[int] = []
    left_ankles: List[List[float]] = []
    right_ankles: List[List[float]] = []
    left_hips: List[List[float]] = []
    right_hips: List[List[float]] = []
    left_confidences: List[float] = []
    right_confidences: List[float] = []
    box_heights: List[float] = []

    for observation in sorted(track["observations"], key=lambda item: int(item["frame"])):
        keypoints = np.asarray(observation["keypoints"], dtype=float)
        confidences = np.asarray(observation["keypoint_confidences"], dtype=float)
        if keypoints.shape != (COCO_KEYPOINT_COUNT, 2) or confidences.shape != (
            COCO_KEYPOINT_COUNT,
        ):
            continue
        required = [LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
        if np.any(confidences[required] < keypoint_conf):
            continue
        if not np.all(np.isfinite(keypoints[required])):
            continue
        if any(np.all(np.abs(keypoints[index]) < 1e-9) for index in required):
            continue

        _, _, box_height = _box_geometry(observation["bbox"])
        frames.append(int(observation["frame"]))
        left_ankles.append(keypoints[LEFT_ANKLE].astype(float).tolist())
        right_ankles.append(keypoints[RIGHT_ANKLE].astype(float).tolist())
        left_hips.append(keypoints[LEFT_HIP].astype(float).tolist())
        right_hips.append(keypoints[RIGHT_HIP].astype(float).tolist())
        left_confidences.append(float(confidences[LEFT_ANKLE]))
        right_confidences.append(float(confidences[RIGHT_ANKLE]))
        box_heights.append(float(box_height))

    return {
        "frames": frames,
        "left_ankles": left_ankles,
        "right_ankles": right_ankles,
        "left_hips": left_hips,
        "right_hips": right_hips,
        "left_confidences": left_confidences,
        "right_confidences": right_confidences,
        "box_heights": box_heights,
    }


def _motion_metrics(sequence: Mapping[str, Any], kick_frame: Optional[int]) -> Dict[str, float]:
    """Compute box-scale-normalised pre-kick hip and lower-body motion."""

    frames = np.asarray(sequence["frames"], dtype=int)
    if len(frames) < 2:
        return {
            "hip_displacement_norm": 0.0,
            "hip_path_norm": 0.0,
            "lower_body_motion_q90_norm": 0.0,
            "kick_peak_norm": 0.0,
        }
    left_ankles = np.asarray(sequence["left_ankles"], dtype=float)
    right_ankles = np.asarray(sequence["right_ankles"], dtype=float)
    left_hips = np.asarray(sequence["left_hips"], dtype=float)
    right_hips = np.asarray(sequence["right_hips"], dtype=float)
    heights = np.maximum(1.0, np.asarray(sequence["box_heights"], dtype=float))
    hip_centres = (left_hips + right_hips) / 2.0
    left_relative = left_ankles - hip_centres
    right_relative = right_ankles - hip_centres

    if kick_frame is None:
        pre_mask = np.ones(len(frames), dtype=bool)
    else:
        pre_mask = frames <= int(kick_frame)
    pre_indices = np.flatnonzero(pre_mask)
    if len(pre_indices) < 2:
        pre_indices = np.arange(len(frames))

    first_count = min(3, len(pre_indices))
    last_count = min(3, len(pre_indices))
    first_hip = np.median(hip_centres[pre_indices[:first_count]], axis=0)
    last_hip = np.median(hip_centres[pre_indices[-last_count:]], axis=0)
    typical_height = float(np.median(heights[pre_indices]))
    hip_displacement = float(np.linalg.norm(last_hip - first_hip) / max(1.0, typical_height))

    hip_path: List[float] = []
    lower_motion: List[float] = []
    for first_index, second_index in zip(pre_indices[:-1], pre_indices[1:]):
        if frames[second_index] - frames[first_index] != 1:
            continue
        scale = max(1.0, float((heights[first_index] + heights[second_index]) / 2.0))
        hip_path.append(float(np.linalg.norm(hip_centres[second_index] - hip_centres[first_index]) / scale))
        left_motion = float(np.linalg.norm(left_relative[second_index] - left_relative[first_index]) / scale)
        right_motion = float(np.linalg.norm(right_relative[second_index] - right_relative[first_index]) / scale)
        lower_motion.append(max(left_motion, right_motion))

    lower_q90 = float(np.quantile(lower_motion, 0.90)) if lower_motion else 0.0
    kick_peak = float(max(lower_motion)) if lower_motion else 0.0
    return {
        "hip_displacement_norm": hip_displacement,
        "hip_path_norm": float(np.sum(hip_path)) if hip_path else 0.0,
        "lower_body_motion_q90_norm": lower_q90,
        "kick_peak_norm": kick_peak,
    }


def _detect_live_kick_frame(
    sequence: Mapping[str, Any],
    *,
    shot_frame_count: int,
    fps: float = 30.0,
    shot_start_frame: int = 0,
) -> Dict[str, Any]:
    """Find the strongest kicking-leg burst in the live portion of a shot.

    Taking the single largest ankle displacement over the whole camera shot
    often selects a post-kick recovery step.  A penalty strike is instead the
    strongest scale-normalised hip-relative ankle-motion peak before the final
    18 percent of the camera shot, which is normally ball flight and recovery.
    Robust outlier statistics are retained in the diagnostics for QA.
    """

    frames = np.asarray(sequence.get("frames", []), dtype=int)
    left_ankles = np.asarray(sequence.get("left_ankles", []), dtype=float)
    right_ankles = np.asarray(sequence.get("right_ankles", []), dtype=float)
    left_hips = np.asarray(sequence.get("left_hips", []), dtype=float)
    right_hips = np.asarray(sequence.get("right_hips", []), dtype=float)
    heights = np.maximum(
        1.0,
        np.asarray(sequence.get("box_heights", []), dtype=float),
    )

    if (
        len(frames) < 3
        or any(len(values) != len(frames) for values in (
            left_ankles,
            right_ankles,
            left_hips,
            right_hips,
            heights,
        ))
    ):
        return {
            "kick_frame": None,
            "peak_motion": 0.0,
            "peak_motion_norm": 0.0,
            "peak_foot": "unknown",
            "frame_pair": None,
            "method": "live_motion_outlier",
        }

    hip_centres = (left_hips + right_hips) / 2.0
    left_relative = left_ankles - hip_centres
    right_relative = right_ankles - hip_centres
    records: List[Dict[str, Any]] = []

    for index in range(len(frames) - 1):
        if int(frames[index + 1]) - int(frames[index]) != 1:
            continue
        left_motion = float(np.linalg.norm(
            left_relative[index + 1] - left_relative[index]
        ))
        right_motion = float(np.linalg.norm(
            right_relative[index + 1] - right_relative[index]
        ))
        raw_motion = max(left_motion, right_motion)
        scale = max(1.0, float((heights[index] + heights[index + 1]) / 2.0))
        records.append({
            "frame": int(frames[index + 1]),
            "pair": (int(frames[index]), int(frames[index + 1])),
            "raw_motion": raw_motion,
            "motion_norm": raw_motion / scale,
            "foot": "left" if left_motion >= right_motion else "right",
            "sequence_index": index + 1,
        })

    if len(records) < 2:
        return {
            "kick_frame": None,
            "peak_motion": 0.0,
            "peak_motion_norm": 0.0,
            "peak_foot": "unknown",
            "frame_pair": None,
            "method": "live_motion_outlier",
        }

    values = np.asarray([item["motion_norm"] for item in records], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    q90 = float(np.quantile(values, 0.90))
    outlier_threshold = max(
        0.045,
        q90 * 1.35,
        median + 4.5 * 1.4826 * mad,
    )
    peak_indices, _ = find_peaks(
        values,
        distance=max(2, int(round(float(fps) * 0.10))),
        prominence=max(0.006, 1.4826 * mad),
    )
    latest_live_frame = shot_start_frame + max(1, int(round(shot_frame_count * 0.82)))
    minimum_lead_frames = max(3, int(round(float(fps) * 0.15)))
    first_track_frame = int(frames[0])
    candidates: List[Dict[str, Any]] = []

    for peak_index in peak_indices:
        item = records[int(peak_index)]
        if item["motion_norm"] < outlier_threshold:
            continue
        if item["frame"] > latest_live_frame:
            continue
        if item["frame"] - first_track_frame < minimum_lead_frames:
            continue

        current_index = int(item["sequence_index"])
        lookback_start = max(0, current_index - max(5, int(round(float(fps) * 1.5))))
        earlier_hips = hip_centres[lookback_start:current_index]
        if len(earlier_hips):
            earlier_centre = np.median(earlier_hips[: min(3, len(earlier_hips))], axis=0)
            body_scale = float(np.median(heights[lookback_start:current_index + 1]))
            approach_displacement = float(
                np.linalg.norm(hip_centres[current_index] - earlier_centre)
                / max(body_scale, 1.0)
            )
        else:
            approach_displacement = 0.0

        if approach_displacement < 0.05:
            continue
        item = dict(item)
        item["approach_displacement_norm"] = approach_displacement
        candidates.append(item)

    # At the stable 640px full-frame setting (and 960px cropped fallback),
    # the strike is normally the largest scale-normalised ankle burst for the
    # kicker.  A maximum very late in a long shot is commonly a post-kick
    # recovery/celebration movement.  In that case, prefer the strongest local
    # peak in the central live window where contact occurs, while retaining the
    # late maximum in diagnostics.
    eligible = [
        item for item in records
        if item["frame"] <= latest_live_frame
    ] or records
    selected = max(eligible, key=lambda item: item["motion_norm"])
    method = "bounded_live_motion_max"
    late_recovery_rejected: Optional[Dict[str, Any]] = None
    if selected["frame"] > shot_start_frame + int(round(shot_frame_count * 0.75)):
        central_start = shot_start_frame + int(round(shot_frame_count * 0.55))
        central_end = shot_start_frame + int(round(shot_frame_count * 0.75))
        central_peaks = [
            records[int(index)]
            for index in peak_indices
            if central_start <= records[int(index)]["frame"] <= central_end
            and values[int(index)] >= max(0.04, median + 2.0 * 1.4826 * mad)
        ]
        if central_peaks:
            late_recovery_rejected = dict(selected)
            selected = max(central_peaks, key=lambda item: item["motion_norm"])
            method = "central_live_peak_after_late_recovery_rejection"

    return {
        "kick_frame": int(selected["frame"]),
        "peak_motion": round(float(selected["raw_motion"]), 2),
        "peak_motion_norm": round(float(selected["motion_norm"]), 6),
        "peak_foot": str(selected["foot"]),
        "frame_pair": tuple(selected["pair"]),
        "method": method,
        "motion_outlier_threshold_norm": round(outlier_threshold, 6),
        "rejected_late_recovery_frame": (
            int(late_recovery_rejected["frame"])
            if late_recovery_rejected is not None
            else None
        ),
    }


def _analyse_tracks(
    tracks: Mapping[int, Mapping[str, Any]],
    *,
    shot_frame_count: int,
    frame_height: int,
    keypoint_conf: float,
    extractor: KickerFeatureExtractor,
    use_field_mask: bool = False,
    shot_start_frame: int = 0,
    fps: float = 30.0,
) -> Tuple[Optional[int], Dict[int, Dict[str, Any]], bool]:
    """Rank kicker candidates by coverage, run-up translation, and leg motion."""

    summaries: Dict[int, Dict[str, Any]] = {}
    minimum_pose_frames = max(8, int(math.ceil(shot_frame_count * 0.15)))

    for track_id, track in sorted(tracks.items()):
        observations = track["observations"]
        sequence = _track_pose_sequence(track, keypoint_conf)
        valid_count = len(sequence["frames"])
        if observations:
            span = int(observations[-1]["frame"]) - int(observations[0]["frame"]) + 1
        else:
            span = 0
        detection_coverage = float(len(observations) / max(1, shot_frame_count))
        pose_coverage = float(valid_count / max(1, shot_frame_count))
        within_track_coverage = float(valid_count / max(1, span))

        kick_detection: Dict[str, Any]
        if valid_count >= 2:
            try:
                kick_detection = _detect_live_kick_frame(
                    sequence,
                    shot_frame_count=shot_frame_count,
                    shot_start_frame=shot_start_frame,
                    fps=fps,
                )
            except (TypeError, ValueError) as exc:
                kick_detection = {
                    "kick_frame": None,
                    "peak_motion": 0.0,
                    "peak_motion_norm": 0.0,
                    "peak_foot": "unknown",
                    "frame_pair": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            kick_detection = {
                "kick_frame": None,
                "peak_motion": 0.0,
                "peak_foot": "unknown",
                "frame_pair": None,
            }

        motion = _motion_metrics(sequence, kick_detection.get("kick_frame"))
        median_box_height = (
            float(np.median(sequence["box_heights"])) if sequence["box_heights"] else 0.0
        )
        box_height_ratio = median_box_height / max(1.0, float(frame_height))
        median_box_centre_y = (
            float(np.median([
                (float(observation["bbox"][1]) + float(observation["bbox"][3])) / 2.0
                for observation in observations
            ]))
            if observations
            else 0.0
        )
        vertical_position_ratio = median_box_centre_y / max(1.0, float(frame_height))
        kick_frame_value = kick_detection.get("kick_frame")
        kick_timing_ratio = (
            float(kick_frame_value - shot_start_frame) / max(1.0, float(shot_frame_count))
            if kick_frame_value is not None
            else 1.0
        )
        timing_quality = max(0.0, 1.0 - abs(kick_timing_ratio - 0.55) / 0.45)
        # Long run-ups can legitimately traverse several body lengths.  Keep
        # this diagnostic for QA, but do not punish the strongest pre-kick
        # mover merely for covering more ground than nearby officials.
        implausible_translation = max(0.0, motion["hip_displacement_norm"] - 2.5)
        adequate_coverage = valid_count >= minimum_pose_frames and pose_coverage >= 0.25
        substantial_motion = (
            motion["hip_displacement_norm"] >= 0.08
            or motion["lower_body_motion_q90_norm"] >= 0.018
            or motion["kick_peak_norm"] >= 0.04
        )
        qualified = bool(
            adequate_coverage
            and substantial_motion
            and kick_frame_value is not None
            and 0.08 <= kick_timing_ratio <= 0.82
        )

        score = (
            3.0 * pose_coverage
            + 1.2 * within_track_coverage
            + 0.8 * min(motion["hip_displacement_norm"], 1.5)
            + 18.0 * min(motion["lower_body_motion_q90_norm"], 0.40)
            + 4.0 * min(motion["kick_peak_norm"], 0.60)
            + 2.5 * min(box_height_ratio * 4.0, 1.0)
            + 1.5 * min(vertical_position_ratio, 1.0)
            + 1.5 * timing_quality
        )
        field_values = [float(obs["field_support"]) for obs in observations if "field_support" in obs]
        field_support = float(np.median(field_values)) if len(field_values) >= 2 else None
        # Bounded soft prior: masks cannot identify the kicker among on-field people.
        field_penalty = 3.0 * (1.0 - field_support) if use_field_mask and field_support is not None else 0.0
        score -= field_penalty
        summaries[track_id] = {
            "field_support": field_support,
            "field_mask_samples": len(field_values),
            "field_mask_penalty": field_penalty,
            "track_id": int(track_id),
            "observation_count": len(observations),
            "valid_pose_count": valid_count,
            "detection_coverage": detection_coverage,
            "pose_coverage": pose_coverage,
            "within_track_coverage": within_track_coverage,
            "median_box_height": median_box_height,
            "box_height_ratio": box_height_ratio,
            "vertical_position_ratio": vertical_position_ratio,
            "kick_timing_ratio": kick_timing_ratio,
            "timing_quality": timing_quality,
            "implausible_translation_penalty": implausible_translation,
            **motion,
            "adequate_coverage": adequate_coverage,
            "substantial_motion": substantial_motion,
            "qualified_kicker_candidate": qualified,
            "candidate_score": float(score),
            "kick_detection": kick_detection,
        }

    if not summaries:
        return None, summaries, False
    qualified_ids = [
        track_id for track_id, summary in summaries.items() if summary["qualified_kicker_candidate"]
    ]
    pool = qualified_ids or [
        track_id for track_id, summary in summaries.items() if summary["valid_pose_count"] >= 2
    ]
    if not pool:
        return None, summaries, False
    selected_id = sorted(
        pool,
        key=lambda track_id: (
            -float(summaries[track_id]["candidate_score"]),
            -int(summaries[track_id]["valid_pose_count"]),
            int(track_id),
        ),
    )[0]
    return int(selected_id), summaries, bool(qualified_ids)


def _build_tracks(pose_frames: Sequence[MutableMapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    tracker = _NearestCentreBoxTracker()
    for pose_frame in pose_frames:
        tracker.update(int(pose_frame["frame"]), pose_frame.get("detections", []))
    return tracker.tracks


def _estimate_runup_start(
    sequence: Mapping[str, Any],
    kick_frame: int,
    fps: float,
) -> Dict[str, Any]:
    """Find the first sustained stationary-to-motion transition before contact.

    A run-up may contain a hesitation or a stutter step.  Selecting the last
    transition truncates those earlier, valid steps, so the operational
    boundary is the first intentional sustained movement after the initial
    stationary period.
    """

    frames = np.asarray(sequence["frames"], dtype=int)
    left_hips = np.asarray(sequence["left_hips"], dtype=float)
    right_hips = np.asarray(sequence["right_hips"], dtype=float)
    heights = np.maximum(1.0, np.asarray(sequence["box_heights"], dtype=float))
    mask = frames <= int(kick_frame)
    frames = frames[mask]
    heights = heights[mask]
    if len(frames) < 3:
        return {
            "runup_start_frame": int(frames[0]) if len(frames) else None,
            "left_censored": True,
            "transition_found": False,
            "speed_threshold": None,
            "signal_frames": frames.astype(int).tolist(),
            "hip_speed_norm": [],
        }

    hip_centres = ((left_hips + right_hips) / 2.0)[mask]
    speeds = np.full(len(frames), np.nan, dtype=float)
    for index in range(1, len(frames)):
        frame_delta = int(frames[index] - frames[index - 1])
        if frame_delta <= 0 or frame_delta > 2:
            continue
        scale = max(1.0, float((heights[index] + heights[index - 1]) / 2.0))
        speeds[index] = float(
            np.linalg.norm(hip_centres[index] - hip_centres[index - 1])
            / (scale * frame_delta)
        )

    smoothed = speeds.copy()
    for index in range(len(speeds)):
        start = max(1, index - 1)
        end = min(len(speeds), index + 2)
        window = speeds[start:end]
        finite = window[np.isfinite(window)]
        if len(finite):
            smoothed[index] = float(np.median(finite))

    finite_speeds = smoothed[np.isfinite(smoothed)]
    if not len(finite_speeds):
        return {
            "runup_start_frame": int(frames[0]),
            "left_censored": True,
            "transition_found": False,
            "speed_threshold": None,
            "signal_frames": frames.astype(int).tolist(),
            "hip_speed_norm": _json_safe(smoothed.tolist()),
        }

    baseline = float(np.quantile(finite_speeds, 0.25))
    deviation = float(np.median(np.abs(finite_speeds - np.median(finite_speeds))))
    threshold = float(np.clip(baseline + 2.5 * deviation, 0.006, 0.025))
    moving = np.isfinite(smoothed) & (smoothed >= threshold)
    sustain = max(2, int(round(0.12 * fps)))
    stationary = max(2, int(round(0.12 * fps)))
    candidates: List[int] = []

    for index in range(stationary, max(stationary, len(frames) - sustain + 1)):
        before = moving[index - stationary : index]
        after = moving[index : index + sustain]
        if len(after) < sustain:
            continue
        contiguous_after = np.all(np.diff(frames[index - 1 : index + sustain]) <= 2)
        mostly_stationary = int(np.sum(before)) <= max(1, stationary // 3)
        sustained_motion = int(np.sum(after)) >= int(math.ceil(0.75 * sustain))
        if contiguous_after and mostly_stationary and sustained_motion:
            candidates.append(max(0, index - 1))

    transition_frame: Optional[int] = None
    onset_preroll = max(2, int(round(0.35 * fps)))

    if candidates:
        selected_index = candidates[0]
        transition_frame = int(frames[selected_index])
        # Hip-centre velocity usually rises only after the first swing has
        # begun.  Keep a short pre-roll so the first new landing is inside the
        # analysis interval while the initially planted foot remains outside.
        runup_start = int(max(frames[0], transition_frame - onset_preroll))
        left_censored = False
        transition_found = True
    else:
        early_end = min(len(moving), 1 + sustain)
        early_motion = moving[1:early_end]
        early_displacement = float(
            np.linalg.norm(hip_centres[min(len(hip_centres) - 1, sustain)] - hip_centres[0])
            / max(1.0, float(np.median(heights[: early_end + 1])))
        )
        left_censored = bool(
            (len(early_motion) and np.mean(early_motion) >= 0.60) or early_displacement >= 0.05
        )
        sustained_indices = [
            index
            for index in range(1, max(1, len(moving) - sustain + 1))
            if np.sum(moving[index : index + sustain]) >= int(math.ceil(0.75 * sustain))
        ]
        if sustained_indices:
            transition_frame = int(frames[max(0, sustained_indices[0] - 1)])
        runup_start = (
            int(frames[0])
            if left_censored or transition_frame is None
            else int(max(frames[0], transition_frame - onset_preroll))
        )
        transition_found = False

    return {
        "runup_start_frame": runup_start,
        "left_censored": left_censored,
        "transition_found": transition_found,
        "speed_threshold": threshold,
        "signal_frames": frames.astype(int).tolist(),
        "hip_speed_norm": _json_safe(smoothed.tolist()),
        "transition_frame": transition_frame,
        "onset_preroll_frames": onset_preroll,
        "transition_candidate_frames": [
            int(frames[index]) for index in candidates
        ],
    }


def _call_step_counter(
    extractor: KickerFeatureExtractor,
    sequence: Mapping[str, Any],
    *,
    fps: float,
    kick_frame: int,
    runup_start_frame: Optional[int],
    kicking_foot: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Call the evolving extractor API, filtering unsupported optional kwargs."""

    warnings: List[str] = []
    method = getattr(extractor, "count_steps_from_pose", None)
    if method is None:
        raise AttributeError("KickerFeatureExtractor.count_steps_from_pose is not available")

    arguments: Dict[str, Any] = {
        "frames": sequence["frames"],
        "left_ankles": sequence["left_ankles"],
        "right_ankles": sequence["right_ankles"],
        "left_hips": sequence["left_hips"],
        "right_hips": sequence["right_hips"],
        "fps": float(fps),
        "kick_frame": int(kick_frame),
        "runup_start_frame": runup_start_frame,
        "kicking_foot": kicking_foot,
        "left_confidences": sequence["left_confidences"],
        "right_confidences": sequence["right_confidences"],
    }

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        supported = set(signature.parameters)
        dropped = sorted(set(arguments) - supported)
        arguments = {key: value for key, value in arguments.items() if key in supported}
        if any(key in dropped for key in ("left_confidences", "right_confidences")):
            _append_warning(warnings, "step_counter_does_not_accept_keypoint_confidences")

    result = method(**arguments)
    if not isinstance(result, Mapping):
        raise TypeError("count_steps_from_pose must return a mapping")
    step_result = dict(result)
    step_result.setdefault("step_count", 0)
    step_result.setdefault("step_frames", [])
    step_result.setdefault("step_feet", [])
    step_result.setdefault("runup_start_frame", runup_start_frame)
    step_result.setdefault("kick_frame", int(kick_frame))
    step_result.setdefault("coverage", 0.0)
    step_result.setdefault("confidence", 0.0)
    step_result.setdefault("review_required", True)
    step_result.setdefault("warnings", [])
    step_result.setdefault("signal_frames", sequence["frames"])
    step_result.setdefault("phase_signal", [])
    return _json_safe(step_result), warnings


def _empty_step_result(
    *,
    kick_frame: Optional[int] = None,
    runup_start_frame: Optional[int] = None,
    warning: Optional[str] = None,
) -> Dict[str, Any]:
    warnings = [warning] if warning else []
    return {
        "step_count": None,
        "step_frames": [],
        "step_feet": [],
        "runup_start_frame": runup_start_frame,
        "kick_frame": kick_frame,
        "coverage": 0.0,
        "confidence": 0.0,
        "review_required": True,
        "warnings": warnings,
        "signal_frames": [],
        "phase_signal": [],
    }


def _goal_line_actor_tracks(video_path, tracks, shot):
    """Exclude standing goal-line occupants before ranking kicker action."""
    from src.goal_geometry import select_goal_occupant

    start, end = int(shot["shot_start_frame"]), int(shot["shot_end_frame"])
    frame_ids = sorted({start, min(end - 1, start + 5),
                        min(end - 1, start + round(.30 * shot["fps"]))})
    capture = cv2.VideoCapture(video_path)
    excluded, evidence = set(), []
    try:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Cannot decode pre-action actor-role frame")
            role = select_goal_occupant(frame, list(tracks.values()), frame_id)
            excluded.update(row["track_id"] for row in role.get("candidates", []))
            evidence.append({"frame": frame_id, "goal_line_evidence": role})
    finally:
        capture.release()
    return {key: value for key, value in tracks.items() if key not in excluded}, {
        "excluded_track_ids": sorted(excluded), "samples": evidence,
        "rule": "preaction_goal_mouth_and_goal_line_foot_proximity", "human_verified": False}


def _rerank_goal_occupant_selection(video_path, tracks, actor_tracks, shot, context,
                                   selection, *, keypoint_conf, extractor, use_field_mask):
    """Rerank remaining actors when a later track fragment is the goalkeeper."""
    from src.goal_geometry import select_goal_occupant

    selected_id, summaries, qualified = selection
    evidence, excluded = [], []
    capture = cv2.VideoCapture(video_path)
    try:
        for _ in range(len(actor_tracks)):
            peak = summaries.get(selected_id, {}).get("kick_detection", {}).get("kick_frame")
            frames = [row["frame"] for row in (context or {}).get("samples", [])
                      if row["goal"]["status"] == "candidate_supported"
                      and peak is not None and row["frame"] <= peak]
            if selected_id is None or not frames:
                break
            frame_id = min(frames, key=lambda fid: abs(fid - (peak - round(.5 * shot["fps"]))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Cannot decode actor reranking context")
            role = select_goal_occupant(frame, list(tracks.values()), frame_id)
            evidence.append({"frame": frame_id, "selected_actor_before_check": selected_id,
                             "goal_line_evidence": role})
            if role.get("status") != "candidate_supported" or role.get("seed_track_id") != selected_id:
                break
            excluded.append(selected_id)
            actor_tracks = {key: value for key, value in actor_tracks.items() if key != selected_id}
            selected_id, summaries, qualified = _analyse_tracks(
                actor_tracks, shot_frame_count=shot["shot_end_frame"] - shot["shot_start_frame"],
                shot_start_frame=shot["shot_start_frame"], fps=shot["fps"],
                frame_height=shot["frame_height"], keypoint_conf=keypoint_conf,
                extractor=extractor, use_field_mask=use_field_mask)
    finally:
        capture.release()
    return (selected_id, summaries, qualified), excluded, evidence


def _process_shot_steps(
    video_path: str,
    model: Any,
    *,
    shot: Mapping[str, Any],
    clip_id: Optional[str] = None,
    extractor: Optional[KickerFeatureExtractor] = None,
    batch_size: int = 16,
    max_shot_sec: float = 8.0,
    scene_cut_threshold: float = 30.0,
    pose_conf: Optional[float] = 0.25,
    keypoint_conf: float = 0.15,
    roi_mode: str = "auto",
    use_field_mask: bool = False,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
    exclude_goal_line_actors: bool = False,
    goal_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Count run-up steps within one independently tracked camera shot.

    Parameters
    ----------
    video_path:
        Canonical penalty clip to analyse.
    model:
        Already-loaded Ultralytics pose model.  This function never owns or
        reloads it.
    roi_mode:
        ``"full"`` uses only full frames, ``"lower_tiles"`` uses two
        overlapping magnified lower-field crops, and ``"auto"`` runs the
        crops only when full-frame tracking has no qualified kicker.

    Returns
    -------
    dict
        Rich JSON-friendly diagnostics, including raw per-frame pose data,
        tracks, candidate scores, the selected kicker, kick/foot estimates,
        run-up start, step result, status, and QA warnings.
    """

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not (0.0 <= keypoint_conf <= 1.0):
        raise ValueError("keypoint_conf must be between 0 and 1")
    if roi_mode not in {"auto", "full", "lower_tiles"}:
        raise ValueError("roi_mode must be 'auto', 'full', or 'lower_tiles'")

    resolved_clip_id = clip_id or path.stem
    extractor = extractor or KickerFeatureExtractor()
    qa_warnings: List[str] = []
    shot_start = int(shot.get("shot_start_frame", 0))
    shot_count = int(shot["shot_end_frame"]) - shot_start
    if shot["shot_end_reason"] == "duration_cap":
        _append_warning(qa_warnings, "selected_shot_reached_duration_cap")

    initial_mode = "lower_tiles" if roi_mode == "lower_tiles" else "full"
    pose_frames, inference_warnings = _run_pose_in_batches(
        str(path),
        int(shot["shot_end_frame"]),
        model,
        batch_size=batch_size,
        pose_conf=pose_conf,
        predict_kwargs=predict_kwargs,
        roi_inference_mode=initial_mode,
        shot_start_frame=shot_start,
    )
    for warning in inference_warnings:
        _append_warning(qa_warnings, warning)

    tracks = _build_tracks(pose_frames)
    actor_tracks, role_exclusions = (_goal_line_actor_tracks(str(path), tracks, shot)
        if exclude_goal_line_actors else (tracks, None))
    selected_track_id, candidate_summaries, selected_is_qualified = _analyse_tracks(
        actor_tracks,
        shot_frame_count=shot_count,
        shot_start_frame=shot_start,
        fps=float(shot["fps"]),
        frame_height=int(shot["frame_height"]),
        keypoint_conf=keypoint_conf,
        extractor=extractor,
    )
    roi_mode_used = initial_mode
    roi_fallback_reasons: List[str] = []

    if roi_mode == "auto":
        if not selected_is_qualified:
            roi_fallback_reasons.append("no_qualified_full_frame_kicker")
        elif selected_track_id is not None:
            full_summary = candidate_summaries[selected_track_id]
            # A small or intermittent full-frame candidate is precisely the
            # regime where crop magnification helps.  It may otherwise be a
            # distant goalkeeper that happens to move substantially after the
            # kick, so do not let that weak candidate suppress the fallback.
            if float(full_summary["box_height_ratio"]) < 0.16:
                roi_fallback_reasons.append("selected_full_frame_track_is_small")
            if float(full_summary["pose_coverage"]) < 0.45:
                roi_fallback_reasons.append("selected_full_frame_track_is_intermittent")
            # A well-detected, mostly stationary official can beat the smaller
            # runner on coverage alone. Resolve competing action evidence with
            # magnified inference instead of trusting that coverage advantage.
            if any(
                other_id != selected_track_id
                and other["valid_pose_count"] >= 8
                and other["kick_peak_norm"] > max(0.12, 1.5 * full_summary["kick_peak_norm"])
                and other["lower_body_motion_q90_norm"] > 1.5 * full_summary["lower_body_motion_q90_norm"]
                for other_id, other in candidate_summaries.items()
            ):
                roi_fallback_reasons.append("competing_track_has_stronger_leg_action")

    if roi_mode == "auto" and roi_fallback_reasons:
        fallback_frames, fallback_warnings = _run_pose_in_batches(
            str(path),
            int(shot["shot_end_frame"]),
            model,
            batch_size=batch_size,
            pose_conf=pose_conf,
            predict_kwargs=predict_kwargs,
            roi_inference_mode="lower_tiles",
            shot_start_frame=shot_start,
        )
        for warning in fallback_warnings:
            _append_warning(qa_warnings, warning)
        pose_frames = _merge_pose_frames(pose_frames, fallback_frames)
        tracks = _build_tracks(pose_frames)
        actor_tracks, role_exclusions = (_goal_line_actor_tracks(str(path), tracks, shot)
            if exclude_goal_line_actors else (tracks, None))
        selected_track_id, candidate_summaries, selected_is_qualified = _analyse_tracks(
            actor_tracks,
            shot_frame_count=shot_count,
            shot_start_frame=shot_start,
            fps=float(shot["fps"]),
            frame_height=int(shot["frame_height"]),
            keypoint_conf=keypoint_conf,
            extractor=extractor,
        )
        roi_mode_used = "full+lower_tiles"
        _append_warning(qa_warnings, "lower_tile_pose_fallback_used")

    mask_diagnostics = None
    if use_field_mask:
        from src.field_masks import annotate_track_field_support
        mask_diagnostics = annotate_track_field_support(str(path), tracks)
        selected_track_id, candidate_summaries, selected_is_qualified = _analyse_tracks(
            actor_tracks, shot_frame_count=shot_count,
            shot_start_frame=shot_start, fps=float(shot["fps"]),
            frame_height=int(shot["frame_height"]), keypoint_conf=keypoint_conf,
            extractor=extractor, use_field_mask=True,
        )
    if exclude_goal_line_actors and goal_context:
        selection, late_excluded, late_samples = _rerank_goal_occupant_selection(
            str(path), tracks, actor_tracks, shot, goal_context,
            (selected_track_id, candidate_summaries, selected_is_qualified),
            keypoint_conf=keypoint_conf, extractor=extractor, use_field_mask=use_field_mask)
        selected_track_id, candidate_summaries, selected_is_qualified = selection
        role_exclusions["excluded_track_ids"] = sorted(set(role_exclusions["excluded_track_ids"] + late_excluded))
        role_exclusions["late_fragment_role_checks"] = late_samples
    total_detections = sum(len(item["detections"]) for item in pose_frames)
    if total_detections == 0:
        _append_warning(qa_warnings, "no_pose_detections")

    kick_detection: Dict[str, Any] = {
        "kick_frame": None,
        "peak_motion": 0.0,
        "peak_foot": "unknown",
        "frame_pair": None,
    }
    kicking_foot_result: Dict[str, Any] = {"kicking_foot": "unknown"}
    runup_start_result: Dict[str, Any] = {
        "runup_start_frame": None,
        "left_censored": False,
        "transition_found": False,
        "speed_threshold": None,
        "signal_frames": [],
        "hip_speed_norm": [],
    }
    step_result = _empty_step_result(warning="no_kicker_track")
    selected_sequence: Dict[str, Any] = {}
    processing_error: Optional[str] = None

    if selected_track_id is None:
        _append_warning(qa_warnings, "no_kicker_track")
    else:
        if not selected_is_qualified:
            _append_warning(qa_warnings, "kicker_candidate_does_not_meet_coverage_motion_gate")
        selected_track = tracks[selected_track_id]
        selected_sequence = _track_pose_sequence(selected_track, keypoint_conf)
        kick_detection = dict(candidate_summaries[selected_track_id]["kick_detection"])
        kick_frame_value = kick_detection.get("kick_frame")

        if kick_frame_value is None:
            _append_warning(qa_warnings, "kick_frame_not_detected")
            step_result = _empty_step_result(warning="kick_frame_not_detected")
        else:
            kick_frame = int(kick_frame_value)
            try:
                kicking_foot_result = extractor.classify_kicking_foot_from_motion(
                    frames=selected_sequence["frames"],
                    left_ankles=selected_sequence["left_ankles"],
                    right_ankles=selected_sequence["right_ankles"],
                    left_hips=selected_sequence["left_hips"],
                    right_hips=selected_sequence["right_hips"],
                    kick_frame=kick_frame,
                    window=3,
                )
            except (TypeError, ValueError) as exc:
                kicking_foot_result = {
                    "kicking_foot": "unknown",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _append_warning(qa_warnings, "kicking_foot_classification_failed")

            kicking_foot = str(kicking_foot_result.get("kicking_foot", "unknown"))
            if kicking_foot not in {"left", "right"}:
                fallback_foot = str(kick_detection.get("peak_foot", "unknown"))
                if fallback_foot in {"left", "right"}:
                    kicking_foot = fallback_foot
                    kicking_foot_result["kicking_foot"] = fallback_foot
                    kicking_foot_result["used_kick_peak_fallback"] = True
                else:
                    _append_warning(qa_warnings, "kicking_foot_unknown")

            runup_start_result = _estimate_runup_start(
                selected_sequence, kick_frame=kick_frame, fps=float(shot["fps"])
            )
            if runup_start_result["left_censored"]:
                _append_warning(qa_warnings, "track_starts_already_moving_left_censored")
            elif not runup_start_result["transition_found"]:
                _append_warning(qa_warnings, "runup_stationary_to_motion_transition_not_found")

            try:
                step_result, call_warnings = _call_step_counter(
                    extractor,
                    selected_sequence,
                    fps=float(shot["fps"]),
                    kick_frame=kick_frame,
                    runup_start_frame=runup_start_result["runup_start_frame"],
                    kicking_foot=kicking_foot,
                )
                for warning in call_warnings:
                    _append_warning(qa_warnings, warning)
            except (AttributeError, TypeError, ValueError) as exc:
                processing_error = f"{type(exc).__name__}: {exc}"
                step_result = _empty_step_result(
                    kick_frame=kick_frame,
                    runup_start_frame=runup_start_result["runup_start_frame"],
                    warning="step_counting_failed",
                )
                _append_warning(qa_warnings, "step_counting_failed")

    for warning in step_result.get("warnings", []) or []:
        _append_warning(qa_warnings, str(warning))

    kick_frame = kick_detection.get("kick_frame")
    kicking_foot = str(kicking_foot_result.get("kicking_foot", "unknown"))
    step_count = step_result.get("step_count")
    successful = selected_track_id is not None and kick_frame is not None and step_count is not None
    status = "ok" if successful else "failed"
    # Some warnings are useful provenance rather than reasons to reject a
    # result (for example, a successful generic tile fallback or deliberate
    # suppression of the final kicking-foot swing).  Keep the counter's own
    # review decision and elevate only pipeline warnings that compromise the
    # analysed interval, identity, or required signals.
    review_warning_codes = {
        "first_shot_reached_duration_cap",
        "selected_shot_reached_duration_cap",
        "keypoint_confidences_unavailable",
        "pose_box_keypoint_count_mismatch",
        "pose_result_count_mismatch",
        "video_decode_ended_before_shot_end",
        "no_pose_detections",
        "no_kicker_track",
        "kicker_candidate_does_not_meet_coverage_motion_gate",
        "kick_frame_not_detected",
        "kicking_foot_classification_failed",
        "kicking_foot_unknown",
        "track_starts_already_moving_left_censored",
        "runup_stationary_to_motion_transition_not_found",
        "step_counter_does_not_accept_keypoint_confidences",
        "step_counting_failed",
    }
    review_required = bool(
        step_result.get("review_required", True)
        or any(warning in review_warning_codes for warning in qa_warnings)
    )
    if not successful and processing_error is None:
        if selected_track_id is None:
            processing_error = "NoKickerTrack: no usable kicker pose track"
        elif kick_frame is None:
            processing_error = "KickFrameNotDetected: no consecutive pose motion peak"
        else:
            processing_error = "StepCountUnavailable: step counter returned no count"

    result = {
        "clip_id": str(resolved_clip_id),
        "video_path": str(path),
        "status": status,
        "error": processing_error,
        "fps": float(shot["fps"]),
        "source_frame_count": int(shot["source_frame_count"]),
        "frame_width": int(shot["frame_width"]),
        "frame_height": int(shot["frame_height"]),
        "shot_start_frame": shot_start,
        "shot_start_sec": float(shot_start / shot["fps"]),
        "shot_end_frame": int(shot["shot_end_frame"]),
        "shot_end_sec": float(shot["shot_end_sec"]),
        "shot_end_reason": str(shot["shot_end_reason"]),
        "roi_mode_requested": roi_mode,
        "field_mask_diagnostics": mask_diagnostics,
        "roi_mode_used": roi_mode_used,
        "roi_fallback_reasons": roi_fallback_reasons,
        "processed_frame_count": len(pose_frames),
        "pose_detection_count": total_detections,
        "selected_track_id": selected_track_id,
        "goal_line_role_exclusions": role_exclusions,
        "selected_track_qualified": selected_is_qualified,
        "track_candidates": list(candidate_summaries.values()),
        "kick_frame": int(kick_frame) if kick_frame is not None else None,
        "kick_time_sec": float(kick_frame / shot["fps"]) if kick_frame is not None else None,
        "kick_detection": kick_detection,
        "kicking_foot": kicking_foot,
        "kicker_footedness": kicking_foot,
        "kicking_foot_result": kicking_foot_result,
        "runup_start_frame": runup_start_result.get("runup_start_frame"),
        "runup_start_result": runup_start_result,
        "step_count": step_count,
        "kicker_steps_count": step_count,
        "step_frames": step_result.get("step_frames", []),
        "step_feet": step_result.get("step_feet", []),
        "step_result": step_result,
        "review_required": review_required,
        "auto_accepted": bool(status == "ok" and not review_required),
        "qa_warnings": qa_warnings,
        # Raw evidence is intentionally retained for reproducible QA.
        "pose_frames": pose_frames,
        "tracks": [tracks[track_id] for track_id in sorted(tracks)],
        "selected_pose_sequence": selected_sequence,
    }
    return _json_safe(result)


def _scan_adaptive_shots(video_path: str, max_shot_sec: float) -> List[Dict[str, Any]]:
    from src.camera_timing_v2 import scan_video

    if max_shot_sec <= 0:
        raise ValueError("Shot duration must be positive")
    scan = scan_video(video_path)
    limit = max(1, math.ceil(max_shot_sec * scan["fps"]))
    shots = []
    for segment in scan["segments"]:
        start, end = segment["start_frame"], segment["end_frame_exclusive"]
        analysis_end = min(end, start + limit)
        shots.append({"shot_index": segment["segment_id"], "shot_start_frame": start,
            "shot_start_sec": start / scan["fps"], "shot_end_frame": analysis_end,
            "shot_end_sec": analysis_end / scan["fps"], "camera_shot_end_frame": end,
            "shot_end_reason": "duration_cap" if analysis_end < end else
                "video_end" if end == scan["frame_count"] else "scene_cut",
            "fps": scan["fps"], "frame_width": scan["width"], "frame_height": scan["height"],
            "source_frame_count": scan["frame_count"], "boundary_method": scan["boundary_method"],
            "boundary_verified": False})
    return shots


def _shot_goal_context(video_path: str, shot: Mapping[str, Any]) -> Dict[str, Any]:
    """Check spatial goal evidence before accepting an approach camera shot."""
    from src.goal_geometry import detect_goal_frame

    start, end = int(shot["shot_start_frame"]), int(shot["shot_end_frame"])
    frame_ids = sorted({start, (start + end - 1) // 2, max(start, end - 1)})
    capture = cv2.VideoCapture(video_path)
    observations = []
    try:
        for frame_id in frame_ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode camera-context frame {frame_id}")
            goal = detect_goal_frame(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green = ((hsv[..., 0] >= 28) & (hsv[..., 0] <= 95)
                     & (hsv[..., 1] >= 45) & (hsv[..., 2] >= 35))
            observations.append({"frame": frame_id, "goal": goal,
                                 "green_fraction": float(np.mean(green))})
    finally:
        capture.release()
    supported = [row for row in observations if row["goal"]["status"] == "candidate_supported"
                 and row["green_fraction"] >= .18]
    return {"supported": bool(supported), "samples": observations,
            "rule": "visible_goal_geometry_and_field_context",
            "actor_identity_verified": False, "human_verified": False}


def process_video_steps(
    video_path: str,
    model: Any,
    *,
    clip_id: Optional[str] = None,
    extractor: Optional[KickerFeatureExtractor] = None,
    batch_size: int = 16,
    max_shot_sec: float = 8.0,
    scene_cut_threshold: float = 30.0,
    pose_conf: Optional[float] = 0.25,
    keypoint_conf: float = 0.15,
    roi_mode: str = "auto",
    use_field_mask: bool = False,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
    shot_selection: str = "first_viable",
    require_goal_context: bool = False,
    event_slice_interval: Optional[Tuple[int, int]] = None,
    shot_scan_method: str = "legacy",
) -> Dict[str, Any]:
    """Select the earliest qualifying pose shot, not blindly the opening shot.

    No mask or moving-ball gate is needed. Each shot gets its own tracker;
    all exported frame IDs refer to the original clip, including nested pose,
    kick and step diagnostics. ``first`` reproduces the legacy window policy
    for comparisons. A later shot is a candidate, not proof of live action:
    rescued and fallback results always require review.
    """
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if shot_selection not in {"first_viable", "first"}:
        raise ValueError("shot_selection must be 'first_viable' or 'first'")
    if shot_scan_method not in {"legacy", "adaptive"}:
        raise ValueError("shot_scan_method must be 'legacy' or 'adaptive'")
    if shot_selection == "first":
        shots = [dict(_detect_first_shot_end(video_path, max_shot_sec, scene_cut_threshold),
                      shot_start_frame=0, shot_start_sec=0.0, shot_index=0)]
    elif shot_scan_method == "adaptive":
        shots = _scan_adaptive_shots(video_path, max_shot_sec)
    else:
        shots = _scan_camera_shots(video_path, max_shot_sec, scene_cut_threshold)
    if event_slice_interval is not None:
        if (len(event_slice_interval) != 2 or any(isinstance(value, bool) or not isinstance(value, int)
                for value in event_slice_interval) or not 0 <= event_slice_interval[0]
                < event_slice_interval[1] <= shots[0]["source_frame_count"]):
            raise ValueError("Event slice must be a valid half-open source-frame interval")
    attempts = []
    best_result = None
    best_rank = (-1, -1.0)
    selected_index = None
    found_qualified = False
    shared_extractor = extractor or KickerFeatureExtractor()
    for shot in shots:
        if event_slice_interval is not None and (shot["shot_end_frame"] <= event_slice_interval[0]
                or shot["shot_start_frame"] >= event_slice_interval[1]):
            attempts.append({"shot_index": shot["shot_index"], "start_frame": shot["shot_start_frame"],
                "end_frame_exclusive": shot["shot_end_frame"], "status": "skipped", "qualifies": False,
                "error": None, "selected_track_id": None, "valid_pose_count": 0, "candidate_score": None,
                "scene_rejections": ["camera_shot_outside_manual_event_slice"],
                "reason": "camera_shot_outside_manual_event_slice"})
            continue
        context = _shot_goal_context(video_path, shot) if require_goal_context else None
        if context is not None and not context["supported"]:
            attempts.append({"shot_index": shot["shot_index"],
                "start_frame": shot["shot_start_frame"], "end_frame_exclusive": shot["shot_end_frame"],
                "status": "skipped", "qualifies": False, "error": None,
                "selected_track_id": None, "valid_pose_count": 0, "candidate_score": None,
                "scene_rejections": ["goal_and_field_context_not_established"],
                "reason": "goal_and_field_context_not_established", "goal_context": context})
            continue
        result = _process_shot_steps(
            video_path, model, shot=shot, clip_id=clip_id, extractor=shared_extractor,
            batch_size=batch_size, max_shot_sec=max_shot_sec,
            scene_cut_threshold=scene_cut_threshold, pose_conf=pose_conf,
            keypoint_conf=keypoint_conf, roi_mode=roi_mode,
            use_field_mask=use_field_mask, predict_kwargs=predict_kwargs,
            exclude_goal_line_actors=require_goal_context,
            goal_context=context,
        )
        candidate = next((item for item in result["track_candidates"]
                          if item["track_id"] == result["selected_track_id"]), {})
        scene_rejections = []
        role_context = None
        if context and result.get("kick_frame") is not None:
            from src.goal_geometry import select_goal_occupant
            supported = [row for row in context["samples"]
                         if row["goal"]["status"] == "candidate_supported"
                         and row["frame"] <= result["kick_frame"]]
            if supported:
                frame_id = min(supported, key=lambda row: abs(row["frame"] - (result["kick_frame"] - 15)))["frame"]
                capture = cv2.VideoCapture(video_path)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = capture.read()
                capture.release()
                if not ok:
                    raise ValueError("Cannot decode actor-role context")
                role_context = select_goal_occupant(frame, result.get("tracks", []), frame_id)
                if (role_context["status"] == "candidate_supported"
                        and role_context["seed_track_id"] == result["selected_track_id"]):
                    scene_rejections.append("selected_kicker_is_goal_line_occupant")
        if shot_selection == "first_viable":
            if (shot["shot_end_frame"] - shot["shot_start_frame"]) / shot["fps"] < 1.0:
                scene_rejections.append("shot_too_short_to_establish_runup")
            if float(candidate.get("box_height_ratio", 0.0)) > 0.60:
                scene_rejections.append("large_foreground_pose_likely_closeup")
        qualifies = bool(result["selected_track_qualified"] and result["status"] == "ok"
                         and not scene_rejections)
        attempts.append({
            "shot_index": shot["shot_index"], "start_frame": shot["shot_start_frame"],
            "end_frame_exclusive": shot["shot_end_frame"], "status": result["status"],
            "qualifies": qualifies, "error": result["error"],
            "selected_track_id": result["selected_track_id"],
            "valid_pose_count": candidate.get("valid_pose_count", 0),
            "candidate_score": candidate.get("candidate_score"),
            "scene_rejections": scene_rejections,
            "reason": "qualified_pose_shot" if qualifies else "no_qualified_kicker_in_shot",
            "goal_context": context,
            "role_context": role_context,
        })
        rank = (int(result["status"] == "ok"), float(candidate.get("candidate_score", -1)))
        if best_result is None or rank > best_rank or qualifies:
            best_result, best_rank, selected_index = result, rank, shot["shot_index"]
        if qualifies or shot_selection == "first":
            found_qualified = qualifies
            break
    if best_result is None:
        # Preserve an inspectable failure without falling back to a rejected scene.
        best_result = _manifest_error_result(clip_id=clip_id or Path(video_path).stem,
            video_path=video_path, error="GoalContext: no supported approach camera shot",
            warning="goal_and_field_context_not_established")
        best_result.update(fps=shots[0]["fps"], frame_width=shots[0]["frame_width"],
                          frame_height=shots[0]["frame_height"],
                          source_frame_count=shots[0]["source_frame_count"],
                          pose_frames=[], tracks=[], selected_pose_sequence={})
    best_result["shot_selection"] = {
        "method": shot_selection, "selected_shot_index": selected_index,
        "scan_method": shot_scan_method,
        "found_qualified_shot": found_qualified, "attempts": attempts,
        "camera_shots": shots, "frame_index_origin": "original_clip_zero_based",
        "human_verified": False,
    }
    if shot_selection == "first_viable" and (selected_index != 0 or not found_qualified):
        _append_warning(best_result["qa_warnings"], "shot_selection_requires_review")
        best_result["review_required"] = True
        best_result["auto_accepted"] = False
    return _json_safe(best_result)


_CSV_FIELDS = [
    "clip_id",
    "match_key",
    "match_title",
    "penalty_number",
    "kicker",
    "team",
    "video_path",
    "status",
    "error",
    "fps",
    "shot_start_frame",
    "shot_start_sec",
    "shot_end_frame",
    "shot_end_sec",
    "shot_end_reason",
    "roi_mode_used",
    "selected_track_id",
    "selected_track_qualified",
    "runup_start_frame",
    "kick_frame",
    "kick_time_sec",
    "kicking_foot",
    "step_count",
    "step_frames",
    "step_feet",
    "coverage",
    "confidence",
    "review_required",
    "auto_accepted",
    "qa_warnings",
]


def _manifest_error_result(
    *,
    clip_id: str,
    video_path: str,
    error: str,
    warning: str,
) -> Dict[str, Any]:
    """Create a stable failed record so every canonical manifest row is represented."""

    step_result = _empty_step_result(warning=warning)
    return {
        "clip_id": clip_id,
        "video_path": video_path,
        "status": "failed",
        "error": error,
        "fps": None,
        "shot_start_frame": None,
        "shot_start_sec": None,
        "shot_end_frame": None,
        "shot_end_sec": None,
        "shot_end_reason": None,
        "roi_mode_used": None,
        "selected_track_id": None,
        "selected_track_qualified": False,
        "runup_start_frame": None,
        "kick_frame": None,
        "kick_time_sec": None,
        "kicking_foot": "unknown",
        "kicker_footedness": "unknown",
        "step_count": None,
        "kicker_steps_count": None,
        "step_frames": [],
        "step_feet": [],
        "step_result": step_result,
        "review_required": True,
        "auto_accepted": False,
        "qa_warnings": [warning],
        "pose_frames": [],
        "tracks": [],
    }


def _csv_row(result: Mapping[str, Any]) -> Dict[str, Any]:
    step_result = result.get("step_result") or {}
    row: Dict[str, Any] = {}
    for field in _CSV_FIELDS:
        if field in {"coverage", "confidence"}:
            row[field] = step_result.get(field)
        else:
            row[field] = result.get(field)
    for field in ("step_frames", "step_feet", "qa_warnings"):
        row[field] = json.dumps(_json_safe(row[field] or []), ensure_ascii=False, separators=(",", ":"))
    return row


def _ensure_output_parent(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def process_manifest_steps(
    manifest_path: str,
    clips_dir: str,
    model: Any,
    output_csv: str,
    diagnostics_json: Optional[str] = None,
    *,
    batch_size: int = 16,
    max_shot_sec: float = 8.0,
    scene_cut_threshold: float = 30.0,
    pose_conf: Optional[float] = 0.25,
    keypoint_conf: float = 0.15,
    roi_mode: str = "auto",
    use_field_mask: bool = False,
    predict_kwargs: Optional[Mapping[str, Any]] = None,
    shot_selection: str = "first_viable",
) -> List[Dict[str, Any]]:
    """Process exactly the canonical clips named by the slicing manifest.

    The function never globs ``clips_dir``.  Legacy duplicate clips such as a
    shortened kicker-name variant are consequently ignored.  A missing or
    failed canonical clip still produces one failed output row, preserving a
    one-to-one relationship with manifest entries.

    The only files written are ``output_csv`` and, when requested,
    ``diagnostics_json``.  The latter contains the rich per-frame evidence;
    the CSV contains stable scalar/list summaries suitable for analysis.
    """

    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    output_file = Path(output_csv)
    diagnostics_file = Path(diagnostics_json) if diagnostics_json is not None else None
    if diagnostics_file is not None and os.path.abspath(str(diagnostics_file)) == os.path.abspath(
        str(output_file)
    ):
        raise ValueError("output_csv and diagnostics_json must be different paths")

    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest root must be a mapping of match keys")

    shared_extractor = KickerFeatureExtractor()
    results: List[Dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    clip_root = Path(clips_dir)

    for match_key, match_data_value in manifest.items():
        if not isinstance(match_data_value, Mapping):
            raise ValueError(f"Manifest match '{match_key}' must be a mapping")
        match_data = dict(match_data_value)
        penalties = match_data.get("penalties", [])
        if not isinstance(penalties, list):
            raise ValueError(f"Manifest match '{match_key}' penalties must be a list")

        for penalty_value in penalties:
            if not isinstance(penalty_value, Mapping):
                raise ValueError(f"Manifest penalty in '{match_key}' must be a mapping")
            penalty = dict(penalty_value)
            if "number" not in penalty:
                raise ValueError(f"Manifest penalty in '{match_key}' has no number")
            penalty_number = int(penalty["number"])
            kicker = str(penalty.get("kicker", f"kicker_{penalty_number}"))
            clip_id = f"{match_key}_p{penalty_number:02d}_{kicker}"
            if clip_id in seen_clip_ids:
                raise ValueError(f"Duplicate canonical manifest clip id: {clip_id}")
            seen_clip_ids.add(clip_id)
            canonical_path = clip_root / f"{clip_id}.mp4"

            if not canonical_path.is_file():
                result = _manifest_error_result(
                    clip_id=clip_id,
                    video_path=str(canonical_path),
                    error=f"FileNotFoundError: Video not found: {canonical_path}",
                    warning="canonical_manifest_clip_missing",
                )
            else:
                try:
                    result = process_video_steps(
                        str(canonical_path),
                        model,
                        clip_id=clip_id,
                        extractor=shared_extractor,
                        batch_size=batch_size,
                        max_shot_sec=max_shot_sec,
                        scene_cut_threshold=scene_cut_threshold,
                        pose_conf=pose_conf,
                        keypoint_conf=keypoint_conf,
                        roi_mode=roi_mode,
                        use_field_mask=use_field_mask,
                        predict_kwargs=predict_kwargs,
                        shot_selection=shot_selection,
                    )
                except Exception as exc:  # Keep the remaining canonical manifest rows processable.
                    result = _manifest_error_result(
                        clip_id=clip_id,
                        video_path=str(canonical_path),
                        error=f"{type(exc).__name__}: {exc}",
                        warning="clip_processing_exception",
                    )

            result.update(
                {
                    "match_key": str(match_key),
                    "match_title": match_data.get("match_title"),
                    "penalty_number": penalty_number,
                    "kicker": kicker,
                    "team": penalty.get("team"),
                    "manifest_start_sec": penalty.get("start_sec"),
                    "manifest_end_sec": penalty.get("end_sec"),
                    "manifest_raw_time": penalty.get("raw_time"),
                }
            )
            results.append(_json_safe(result))

    _ensure_output_parent(output_file)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_row(result) for result in results)

    if diagnostics_file is not None:
        _ensure_output_parent(diagnostics_file)
        with diagnostics_file.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2, allow_nan=False)

    return results


__all__ = ["process_video_steps", "process_manifest_steps"]
