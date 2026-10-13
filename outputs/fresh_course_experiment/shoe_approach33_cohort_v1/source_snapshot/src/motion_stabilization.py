"""Background-motion compensation for image-plane movement descriptions.

This is not pitch calibration and never produces metres or real-world speed.
People are excluded from feature sampling geometrically; no grass/HSV gate is
used. Transform quality must accompany any directional or curvature claim.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import cv2
import numpy as np


def apply_transforms(points, frames, transforms):
    points = np.asarray(points, dtype=float)
    if points.shape != (len(frames), 2):
        raise ValueError("Expected aligned Nx2 points and frame IDs")
    result = []
    for frame, point in zip(frames, points):
        raw = transforms.get(str(int(frame)), transforms.get(int(frame)))
        if raw is None:
            result.append([float("nan"), float("nan")])
            continue
        transform = np.asarray(raw, dtype=float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            result.append([float("nan"), float("nan")])
            continue
        homogeneous = transform @ np.r_[point, 1.0]
        result.append((homogeneous[:2] / homogeneous[2]).tolist() if abs(homogeneous[2]) > 1e-12 else [float("nan"), float("nan")])
    return result


def estimate_background_motion(
    video_path: str, start_frame: int, end_frame: int,
    boxes_by_frame: Mapping[int, Sequence[Sequence[float]]] | None = None,
    *, target_width: int = 640,
):
    """Return clip-absolute transforms into the selected shot's first frame.

    End is exclusive. A rejected pair permanently breaks this first-frame
    transform chain. Later matrices are null until a future relocalization
    method is implemented; missing displacement must never be silently skipped.
    """
    if not 0 <= start_frame < end_frame:
        raise ValueError("Invalid frame interval")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {video_path}")
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ratio = min(1.0, target_width / max(1, original_width))
    boxes_by_frame = boxes_by_frame or {}
    for _ in range(start_frame):
        if not cap.grab():
            cap.release()
            raise ValueError("Video ended before selected shot")
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Cannot decode shot start")
    previous = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=ratio, fy=ratio)
    cumulative = np.eye(3)
    transforms = {str(start_frame): cumulative.tolist()}
    records = []
    failure_run = max_failure_run = 0
    chain_valid = True
    scale_to_small = np.diag([ratio, ratio, 1.0])
    scale_to_full = np.linalg.inv(scale_to_small)
    try:
        for fid in range(start_frame + 1, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            current = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=ratio, fy=ratio)
            feature_region = np.full(previous.shape, 255, np.uint8)
            feature_region[:max(1, round(previous.shape[0] * .08))] = 0
            for box in boxes_by_frame.get(fid - 1, []):
                x1, y1, x2, y2 = np.asarray(box, float) * ratio
                pad = max(4, .15 * (y2 - y1))
                cv2.rectangle(feature_region, (round(x1 - pad), round(y1 - pad)),
                              (round(x2 + pad), round(y2 + pad)), 0, -1)
            corners = cv2.goodFeaturesToTrack(previous, 350, .01, 7, mask=feature_region)
            matrix, inlier_ratio, residual, count = None, 0.0, None, 0
            if corners is not None and len(corners) >= 12:
                tracked, forward_status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
                if tracked is not None:
                    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, tracked, None)
                    if backward is not None:
                        valid = (forward_status.ravel() > 0) & (backward_status.ravel() > 0)
                        valid &= np.linalg.norm(backward.reshape(-1, 2) - corners.reshape(-1, 2), axis=1) < 1.5
                        src, dst = corners.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid]
                        count = len(src)
                        if count >= 12:
                            matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                                        ransacReprojThreshold=1.5)
                            if matrix is not None and inliers is not None:
                                mask = inliers.ravel().astype(bool)
                                inlier_ratio = float(mask.mean())
                                prediction = src @ matrix[:, :2].T + matrix[:, 2]
                                residual = float(np.median(np.linalg.norm(prediction[mask] - dst[mask], axis=1)))
            accepted = False
            if matrix is not None and np.isfinite(matrix).all():
                scale = math.sqrt(abs(float(np.linalg.det(matrix[:, :2]))))
                angle = abs(math.atan2(matrix[1, 0], matrix[0, 0]))
                accepted = (count >= 12 and inlier_ratio >= .60 and residual is not None and residual <= 1.5
                            and .94 <= scale <= 1.06 and angle <= .06
                            and np.linalg.norm(matrix[:, 2]) <= previous.shape[1] * .12)
            if accepted:
                affine = np.eye(3)
                affine[:2] = matrix
                cumulative = affine @ cumulative
                failure_run = 0
            else:
                failure_run += 1
                max_failure_run = max(max_failure_run, failure_run)
                chain_valid = False
            inverse = scale_to_full @ np.linalg.inv(cumulative) @ scale_to_small
            transforms[str(fid)] = np.round(inverse, 8).tolist() if chain_valid else None
            records.append({"frame": fid, "accepted": bool(accepted), "tracked_features": count,
                            "reference_chain_valid": chain_valid,
                            "inlier_fraction": round(inlier_ratio, 4),
                            "median_residual_small_px": None if residual is None else round(residual, 4)})
            previous = current
    finally:
        cap.release()
    fraction = sum(r["accepted"] for r in records) / max(1, end_frame - start_frame - 1)
    complete = len(transforms) == end_frame - start_frame
    status = "supported" if complete and records and chain_valid else "partial" if fraction >= .6 else "unavailable"
    return {"status": status, "transforms": transforms, "pairs": records,
            "success_fraction": round(fraction, 4), "max_failed_pairs_in_a_row": max_failure_run,
            "decoded_complete": complete, "coordinate_system": "first_shot_frame_image_plane",
            "reference_chain_complete": bool(complete and chain_valid and records),
            "metric_calibrated": False, "human_verified": False,
            "method": "background_LK_forward_backward_RANSAC_partial_affine"}
