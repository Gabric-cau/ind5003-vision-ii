"""Scale-normalised goalkeeper features for penalty-kick video.

Coordinates use the source-image convention (x increases to the viewer's
right, y increases downwards). A trajectory sample is ``(time, x, y)`` or
``(time, x, y, body_height)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


Trajectory = Sequence[Sequence[float]]


class GoalkeeperFeatureExtractor:
    """Extract pre-kick activity and post-kick dive categories."""

    def __init__(
        self,
        goal_height_px: float = 150.0,
        goal_width_px: float = 400.0,
        *,
        movement_static_max: float = 0.18,
        movement_high_min: float = 0.60,
        direction_deadband: float = 0.22,
        high_rise_min: float = 0.32,
        low_drop_min: float = 0.18,
    ) -> None:
        self.goal_height_px = float(goal_height_px)
        self.goal_width_px = float(goal_width_px)
        self.movement_static_max = float(movement_static_max)
        self.movement_high_min = float(movement_high_min)
        self.direction_deadband = float(direction_deadband)
        self.high_rise_min = float(high_rise_min)
        self.low_drop_min = float(low_drop_min)

    def _arrays(self, samples: Trajectory) -> Tuple[np.ndarray, ...]:
        if not samples:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty, empty
        values = np.asarray(samples, dtype=float)
        if values.ndim != 2 or values.shape[1] not in (3, 4):
            raise ValueError("Trajectory samples must be (time,x,y) or (time,x,y,body_height)")
        values = values[np.all(np.isfinite(values[:, :3]), axis=1)]
        if not len(values):
            empty = np.empty(0, dtype=float)
            return empty, empty, empty, empty
        values = values[np.argsort(values[:, 0], kind="stable")]
        if values.shape[1] == 4:
            scale = values[:, 3]
            valid = np.isfinite(scale) & (scale > 1.0)
            fallback = float(np.median(scale[valid])) if np.any(valid) else self.goal_height_px
            scale = np.where(valid, scale, max(1.0, fallback))
        else:
            scale = np.full(len(values), max(1.0, self.goal_height_px), dtype=float)
        return values[:, 0], values[:, 1], values[:, 2], scale

    @staticmethod
    def _confidence(coverage: float, sample_count: int, displacement: float | None = None) -> str:
        score = min(1.0, max(0.0, coverage))
        if sample_count < 6:
            score *= 0.65
        if displacement is not None and displacement < 0.15:
            score *= 0.75
        if score >= 0.78:
            return "high"
        if score >= 0.48:
            return "medium"
        return "low"

    def quantify_pre_kick_movement(
        self,
        gk_positions: Trajectory,
        *,
        expected_duration_sec: float | None = None,
    ) -> Dict[str, Any]:
        """Classify movement using robust speed in body-heights per second."""

        ts, xs, ys, scales = self._arrays(gk_positions)
        if len(ts) < 2:
            return {
                "gk_prekick_movement_level": "unknown",
                "gk_prekick_mean_speed_px_s": None,
                "gk_prekick_speed_body_s": None,
                "gk_prekick_lateral_range_body": None,
                "gk_prekick_coverage": 0.0,
                "gk_prekick_confidence": "low",
                "gk_prekick_review_required": True,
            }

        dt = np.diff(ts)
        valid = (dt > 0) & (dt <= 0.25)
        speed_px = np.hypot(np.diff(xs), np.diff(ys)) / np.maximum(dt, 1e-9)
        speed_body = speed_px / np.maximum((scales[:-1] + scales[1:]) / 2.0, 1.0)
        usable = speed_body[valid]
        usable_px = speed_px[valid]
        robust_speed = float(np.median(usable)) if len(usable) else 0.0
        mean_speed_px = float(np.median(usable_px)) if len(usable_px) else 0.0
        body_scale = max(1.0, float(np.median(scales)))
        lateral_range = float((np.quantile(xs, 0.90) - np.quantile(xs, 0.10)) / body_scale)
        observed_duration = max(0.0, float(ts[-1] - ts[0]))
        activity = max(robust_speed, lateral_range / max(observed_duration, 0.25))
        if activity < self.movement_static_max and lateral_range < 0.18:
            level = "static"
        elif activity >= self.movement_high_min or lateral_range >= 0.55:
            level = "high_active"
        else:
            level = "moderate_jitter"

        duration = expected_duration_sec if expected_duration_sec and expected_duration_sec > 0 else observed_duration
        coverage = min(1.0, observed_duration / duration) if duration > 0 else 0.0
        confidence = self._confidence(coverage, len(ts))
        review = bool(confidence == "low" or len(usable) < 3)
        return {
            "gk_prekick_movement_level": level,
            "gk_prekick_mean_speed_px_s": round(mean_speed_px, 3),
            "gk_prekick_speed_body_s": round(robust_speed, 4),
            "gk_prekick_lateral_range_body": round(lateral_range, 4),
            "gk_prekick_coverage": round(coverage, 4),
            "gk_prekick_confidence": confidence,
            "gk_prekick_review_required": review,
        }

    def classify_dive_direction_and_height(
        self,
        gk_dive_trajectory: Trajectory,
        goal_center_x: float = 500.0,
        goal_line_y: float = 180.0,
        *,
        baseline_samples: int = 3,
        expected_duration_sec: float | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Classify image-left/right and low/mid/high relative to baseline."""

        ts, xs, ys, scales = self._arrays(gk_dive_trajectory)
        if len(ts) < 2:
            return {
                "dive_direction": "unknown",
                "dive_height": "unknown",
                "dive_lateral_displacement_body": None,
                "dive_vertical_displacement_body": None,
                "dive_coverage": 0.0,
                "dive_confidence": "low",
                "dive_review_required": True,
            }

        baseline_count = max(1, min(int(baseline_samples), len(ts) - 1))
        baseline_x = float(np.median(xs[:baseline_count]))
        baseline_y = float(np.median(ys[:baseline_count]))
        body_scale = max(1.0, float(np.median(scales[:baseline_count])))
        dx = (xs - baseline_x) / body_scale
        dy = (ys - baseline_y) / body_scale
        displacement = np.hypot(dx, dy)
        extreme_index = int(np.argmax(displacement))
        lateral = float(dx[extreme_index])
        vertical = float(dy[extreme_index])
        magnitude = float(displacement[extreme_index])

        if lateral < -self.direction_deadband:
            direction = "image_left"
        elif lateral > self.direction_deadband:
            direction = "image_right"
        else:
            direction = "center"
        if vertical <= -self.high_rise_min:
            height = "high"
        elif vertical >= self.low_drop_min:
            height = "low"
        else:
            height = "mid"

        observed_duration = max(0.0, float(ts[-1] - ts[0]))
        duration = expected_duration_sec if expected_duration_sec and expected_duration_sec > 0 else observed_duration
        coverage = min(1.0, observed_duration / duration) if duration > 0 else 0.0
        confidence = self._confidence(coverage, len(ts), magnitude)
        review = bool(confidence == "low" or len(ts) < 4)
        return {
            "dive_direction": direction,
            "dive_height": height,
            "dive_lateral_displacement_body": round(lateral, 4),
            "dive_vertical_displacement_body": round(vertical, 4),
            "dive_displacement_body": round(magnitude, 4),
            "dive_coverage": round(coverage, 4),
            "dive_confidence": confidence,
            "dive_review_required": review,
        }

    def analyze(
        self,
        pre_kick: Trajectory,
        post_kick: Trajectory,
        *,
        pre_window_sec: float = 1.5,
        post_window_sec: float = 1.0,
    ) -> Dict[str, Any]:
        movement = self.quantify_pre_kick_movement(pre_kick, expected_duration_sec=pre_window_sec)
        combined: List[Sequence[float]] = list(pre_kick[-3:]) + list(post_kick)
        dive = self.classify_dive_direction_and_height(
            combined,
            baseline_samples=min(3, max(1, len(pre_kick[-3:]))),
            expected_duration_sec=post_window_sec,
        )
        return {
            **movement,
            **dive,
            "review_required": bool(
                movement["gk_prekick_review_required"] or dive["dive_review_required"]
            ),
        }


__all__ = ["GoalkeeperFeatureExtractor"]
