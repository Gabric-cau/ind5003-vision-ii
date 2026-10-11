"""Ball trajectory and penalty-outcome feature extraction.

Coordinates are retained in image pixels for auditability and are also
normalised by frame size for comparison across videos. Outcome labels are kept
separate from tracking because a save and a goal can have similar geometry.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


BallSample = Tuple[float, float, float]


class BallOutcomeExtractor:
    def __init__(self, goal_bbox: Optional[Tuple[float, float, float, float]] = None):
        self.goal_bbox = goal_bbox or (300.0, 100.0, 700.0, 250.0)

    @staticmethod
    def _clean_samples(raw_ball_positions: Sequence[Sequence[float]]) -> List[BallSample]:
        by_time: Dict[float, BallSample] = {}
        for row in raw_ball_positions or []:
            if len(row) < 3:
                continue
            t, x, y = map(float, row[:3])
            if all(math.isfinite(value) for value in (t, x, y)):
                by_time[t] = (t, x, y)
        return [by_time[key] for key in sorted(by_time)]

    def extract_trajectory_and_speed(
        self,
        raw_ball_positions: Sequence[Sequence[float]],
        fps: float = 25.0,
        *,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        expected_frames: Optional[int] = None,
        confidences: Optional[Sequence[float]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Summarise an observed ball track without inventing missing points."""

        samples = self._clean_samples(raw_ball_positions)
        if not samples:
            return {
                "ball_speed_px_s": 0.0,
                "ball_median_speed_px_s": 0.0,
                "ball_peak_speed_px_s": 0.0,
                "ball_path_length_px": 0.0,
                "ball_displacement_px": 0.0,
                "ball_track_points": 0,
                "ball_track_coverage": 0.0,
                "ball_track_confidence": 0.0,
                "final_position": (0.0, 0.0),
                "final_position_normalized": (0.0, 0.0),
                "ball_review_required": True,
            }

        final_x, final_y = samples[-1][1:]
        segment_speeds: List[float] = []
        total_dist = 0.0
        for previous, current in zip(samples, samples[1:]):
            dt = current[0] - previous[0]
            if dt <= 0:
                continue
            distance = float(np.hypot(current[1] - previous[1], current[2] - previous[2]))
            total_dist += distance
            segment_speeds.append(distance / dt)
        duration = samples[-1][0] - samples[0][0]
        average_speed = total_dist / duration if duration > 0 else 0.0
        median_speed = float(np.median(segment_speeds)) if segment_speeds else 0.0
        peak_speed = float(np.quantile(segment_speeds, 0.90)) if segment_speeds else 0.0
        coverage = min(1.0, len(samples) / max(1, expected_frames or len(samples)))
        valid_confidences = [
            float(value) for value in (confidences or []) if value is not None and math.isfinite(float(value))
        ]
        track_confidence = float(np.median(valid_confidences)) if valid_confidences else 0.0
        norm_x = final_x / frame_width if frame_width else 0.0
        norm_y = final_y / frame_height if frame_height else 0.0
        displacement = float(np.hypot(final_x - samples[0][1], final_y - samples[0][2]))
        review_required = bool(len(samples) < 3 or coverage < 0.12 or displacement < 35.0)
        return {
            "ball_speed_px_s": round(average_speed, 2),
            "ball_median_speed_px_s": round(median_speed, 2),
            "ball_peak_speed_px_s": round(peak_speed, 2),
            "ball_path_length_px": round(total_dist, 2),
            "ball_displacement_px": round(displacement, 2),
            "ball_track_points": len(samples),
            "ball_track_coverage": round(coverage, 4),
            "ball_track_confidence": round(track_confidence, 4),
            "final_position": (round(final_x, 2), round(final_y, 2)),
            "final_position_normalized": (round(norm_x, 5), round(norm_y, 5)),
            "ball_review_required": review_required,
        }

    def determine_outcome(
        self,
        final_ball_pos: Tuple[float, float],
        gk_contact: bool = False,
        *,
        observed_outcome: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return an observed label when supplied, plus an auditable target zone."""

        xmin, ymin, xmax, ymax = self.goal_bbox
        bx, by = map(float, final_ball_pos)
        is_inside = xmin <= bx <= xmax and ymin <= by <= ymax
        if observed_outcome is not None:
            observed_outcome = observed_outcome.strip().lower()
            if observed_outcome not in {"goal", "no_goal"}:
                raise ValueError("observed_outcome must be 'goal' or 'no_goal'")
            outcome = observed_outcome
            source = "observed_annotation"
        else:
            outcome = "goal" if is_inside and not gk_contact else "no_goal"
            source = "geometry_fallback"
        rel_x = (bx - xmin) / (xmax - xmin) if xmax > xmin else 0.5
        rel_y = (by - ymin) / (ymax - ymin) if ymax > ymin else 0.5
        col = "left" if rel_x < 0.33 else ("right" if rel_x > 0.67 else "center")
        row = "top" if rel_y < 0.33 else ("bottom" if rel_y > 0.67 else "middle")
        return {
            "outcome": outcome,
            "outcome_source": source,
            "final_inside_goal_bbox": bool(is_inside),
            "target_zone": f"{row}_{col}",
        }
