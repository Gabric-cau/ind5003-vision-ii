"""HSV field priors. Masks restrict search/background, never assert object identity.

OpenCV sources: tutorial_threshold_inRange.html, tutorial_py_morphological_ops.html.
Thresholds and quality gates below are project hypotheses, not learned accuracy.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass
class FieldMask:
    grass: np.ndarray
    region: np.ndarray
    valid: bool
    coverage: float
    hue_center: float | None
    reason: str


def build_field_mask(frame: np.ndarray, *, adaptive: bool = True) -> FieldMask:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR frame")
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (25, 35, 30), (95, 255, 255))
    hue = None
    seed = grass[int(h * .35):] > 0
    if adaptive and np.count_nonzero(seed) >= max(20, h * w * .015):
        values = hsv[int(h * .35):, :, 0][seed]
        hue = float(np.argmax(np.bincount(values, minlength=180)))
        grass = cv2.inRange(hsv, (max(25, int(hue) - 22), 35, 30),
                            (min(95, int(hue) + 22), 255, 255))
    k = max(3, int(round(min(h, w) / 180)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(grass, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros((h, w), dtype=np.uint8)
    if contours:
        # Fill holes caused by players/white markings. Do not take a convex hull:
        # that could incorrectly include grandstands beyond the pitch boundary.
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(region, [largest], -1, 255, cv2.FILLED)
    coverage = float(np.mean(region > 0))
    lower_support = float(np.mean(cleaned[int(h * .55):] > 0))
    occupied = np.flatnonzero(np.any(region > 0, axis=0))
    horizontal_span = (occupied[-1] - occupied[0] + 1) / w if len(occupied) else 0.
    # A close-up green jersey is not a pitch. Require broad horizontal support;
    # this still is a heuristic and cannot distinguish every green object.
    wide_rows = np.mean(region > 0, axis=1) >= .75
    valid = (.10 <= coverage <= .98 and lower_support >= .08
             and horizontal_span >= .85 and np.mean(wide_rows) >= .12)
    return FieldMask(grass, region, bool(valid), coverage, hue,
                     "usable_field_prior" if valid else "insufficient_or_ambiguous_field")


def foot_region_support(field: FieldMask, box: Sequence[float]) -> float | None:
    if not field.valid:
        return None
    h, w = field.region.shape
    x1, y1, x2, y2 = map(float, box)
    bh = max(1., y2 - y1)
    # A band around the feet tolerates imperfect boxes and jumping poses.
    xa, xb = max(0, int(x1)), min(w, int(np.ceil(x2)))
    ya, yb = max(0, int(y2 - .10 * bh)), min(h, int(np.ceil(y2 + .10 * bh)))
    if xa >= xb or ya >= yb:
        return 0.
    return float(np.mean(field.region[ya:yb, xa:xb] > 0))


def background_feature_mask(frame: np.ndarray, boxes: Sequence[Sequence[float]] = ()):
    field = build_field_mask(frame)
    h, w = frame.shape[:2]
    allowed = np.full((h, w), 255, np.uint8)
    allowed[:int(h * .06)] = 0  # Scoreboard overlays are not scene motion.
    for box in boxes:
        x1, y1, x2, y2 = map(float, box)
        pad = max(3., .12 * (y2 - y1))
        xa, xb = max(0, int(x1 - pad)), min(w, int(np.ceil(x2 + pad)))
        ya, yb = max(0, int(y1 - pad)), min(h, int(np.ceil(y2 + pad)))
        if xa < xb and ya < yb:
            allowed[ya:yb, xa:xb] = 0
    masked = cv2.bitwise_and(allowed, field.region) if field.valid else allowed
    # Low field coverage falls back to the scene excluding known players.
    use_field = field.valid and np.mean(masked > 0) >= .04
    return (masked if use_field else allowed), {
        "field_used": bool(use_field), "field_coverage": field.coverage,
        "allowed_fraction": float(np.mean((masked if use_field else allowed) > 0)),
    }


def white_ball_proposals(frame: np.ndarray, field: FieldMask | None = None):
    """Small white connected components on the field, for candidate generation only.

    Shoes, line fragments and gloves can pass; these are never football labels.
    A ball against the net/sky can be absent, so never hard-gate YOLO with this.
    """
    field = field or build_field_mask(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 135), (179, 100, 255))
    if not field.valid:
        return np.zeros_like(white), []
    white = cv2.bitwise_and(white, field.region)
    n, labels, stats, centres = cv2.connectedComponentsWithStats(white)
    max_side = max(10, int(round(frame.shape[0] * .06)))
    candidates = []
    for index in range(1, n):
        x, y, width, height, area = map(int, stats[index])
        if 3 <= area <= max_side ** 2 and max(width, height) <= max_side:
            ratio = min(width, height) / max(width, height)
            if ratio >= .30:
                candidates.append({"x": float(centres[index, 0]), "y": float(centres[index, 1]),
                                   "width": width, "height": height, "area": area,
                                   "source": "hsv_white_candidate_not_verified"})
    return white, candidates


def annotate_track_field_support(video_path: str, tracks: Mapping, *, samples_per_track: int = 5):
    """Attach sparse on-field evidence to retained pose observations in place."""
    if samples_per_track < 1:
        raise ValueError("samples_per_track must be positive")
    by_frame = {}
    for track in tracks.values():
        observations = track.get("observations", [])
        # Clear previous annotations so an ablation cannot inherit stale scores.
        for obs in observations:
            obs.pop("field_support", None)
        if not observations:
            continue
        for index in np.unique(np.linspace(0, len(observations)-1, min(samples_per_track, len(observations))).astype(int)):
            obs = observations[index]
            by_frame.setdefault(int(obs["frame"]), []).append(obs)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")
    valid = 0
    decoded = 0
    coverages = []
    try:
        for frame_id, observations in sorted(by_frame.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            decoded += 1
            field = build_field_mask(frame)
            coverages.append(field.coverage)
            valid += int(field.valid)
            for obs in observations:
                support = foot_region_support(field, obs["bbox"])
                if support is not None:
                    obs["field_support"] = support
    finally:
        cap.release()
    return {"sampled_frames": len(by_frame), "decoded_frames": decoded,
            "usable_masks": valid, "median_field_coverage": float(np.median(coverages)) if coverages else None}


def pitch_line_candidates(frame: np.ndarray, boxes: Sequence[Sequence[float]] = ()):
    """Canny + white/field masks + Hough proposals, without semantic calibration.

    Canny runs BEFORE masking so the ROI border cannot create a fake field line.
    Goal-net segments may remain; a detected segment is not a known pitch marking.
    """
    field = build_field_mask(frame)
    if not field.valid:
        return np.zeros(frame.shape[:2], np.uint8), []
    background, _ = background_feature_mask(frame, boxes)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 135), (179, 100, 255))
    support = cv2.dilate(white, np.ones((5, 5), np.uint8))
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.bitwise_and(edges, cv2.bitwise_and(background, support))
    minimum = max(25, int(frame.shape[1] * .05))
    detected = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                              minLineLength=minimum, maxLineGap=12)
    lines = [] if detected is None else [list(map(int, segment)) for segment in detected.reshape(-1, 4)]
    return edges, lines
