import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Any, Tuple, Optional

from src.homography import PitchHomographyTransformer


class KickerFeatureExtractor:
    """
    Extract features related to the penalty kicker.

    Current features:
    1. Run-up trajectory
    2. Number of steps
    3. Kicking foot

    The main kicking-foot classifier uses temporal 2D ankle motion
    relative to the kicker's hip centre.

    A simpler ball-distance method is retained as a baseline.
    """

    def __init__(
        self,
        homography_transformer: Optional[
            PitchHomographyTransformer
        ] = None
    ):
        self.transformer = (
            homography_transformer
            or PitchHomographyTransformer()
        )

    # ============================================================
    # Run-up trajectory
    # ============================================================

    def analyze_runup_trajectory(
        self,
        pixel_trajectory: List[
            Tuple[float, float, float]
        ]
    ) -> Dict[str, Any]:
        """
        Analyse the kicker's run-up trajectory.

        Parameters
        ----------
        pixel_trajectory:
            Sequence of (time, x, y) observations.

        Returns
        -------
        dict
            Run-up starting position, angle, distance and speed.
        """

        if not pixel_trajectory:
            return {
                "runup_angle_deg": 0.0,
                "runup_distance_m": None,
                "runup_speed_mps": None,
                "runup_start_x": 0.0,
                "runup_start_y": 0.0
            }

        pts_2d = []

        for item in pixel_trajectory:
            t_val, x_val, y_val = item

            pts_2d.append(
                (
                    float(x_val),
                    float(y_val)
                )
            )

        top_down = (
            self.transformer
            .transform_trajectory(pts_2d)
        )

        if len(top_down) >= 2:

            start_x, start_y = top_down[0]
            end_x, end_y = top_down[-1]

            dx = end_x - start_x
            dy = end_y - start_y

            dist = np.sqrt(
                dx * dx + dy * dy
            )

            angle = np.degrees(
                np.arctan2(dx, dy)
            )

            t_start, _, _ = (
                pixel_trajectory[0]
            )

            t_end, _, _ = (
                pixel_trajectory[-1]
            )

            dt = t_end - t_start

            speed = (
                dist / dt
                if dt > 0
                else 0.0
            )

        else:

            start_x, start_y = (
                top_down[0]
                if top_down
                else (0.0, 0.0)
            )

            dist = 0.0
            angle = 0.0
            speed = 0.0

        return {
            "coordinate_unit": self.transformer.destination_unit if self.transformer.H is not None else "px",
            "metric_calibrated": self.transformer.metric_calibrated,
            "runup_displacement": round(float(dist), 2),
            "runup_mean_displacement_speed": round(float(speed), 2),
            "runup_start_x":
                round(float(start_x), 2),

            "runup_start_y":
                round(float(start_y), 2),

            "runup_angle_deg":
                round(float(angle), 2),

            "runup_distance_m":
                round(float(dist), 2) if self.transformer.metric_calibrated else None,

            "runup_speed_mps":
                round(float(speed), 2) if self.transformer.metric_calibrated else None
        }

    # ============================================================
    # Step counting
    # ============================================================

    def count_steps(
        self,
        left_foot_y: List[float],
        right_foot_y: List[float],
        fps: float = 25.0
    ) -> int:
        """
        Estimate the number of run-up steps from alternating
        left/right foot vertical motion.
        """

        if (
            not left_foot_y
            or not right_foot_y
            or len(left_foot_y) < 4
        ):
            return 0

        diff = (
            np.array(left_foot_y)
            - np.array(right_foot_y)
        )

        smoothed = np.convolve(
            diff,
            np.ones(3) / 3.0,
            mode="same"
        )

        min_peak_distance = int(
            max(1, fps * 0.1)
        )

        p_pos, _ = find_peaks(
            smoothed,
            distance=min_peak_distance,
            prominence=2.0
        )

        p_neg, _ = find_peaks(
            -smoothed,
            distance=min_peak_distance,
            prominence=2.0
        )

        return max(
            1,
            len(p_pos) + len(p_neg)
        )

    def count_steps_from_pose(
        self,
        frames: List[int],
        left_ankles: List[Tuple[float, float]],
        right_ankles: List[Tuple[float, float]],
        left_hips: List[Tuple[float, float]],
        right_hips: List[Tuple[float, float]],
        fps: float,
        kick_frame: int,
        runup_start_frame: Optional[int] = None,
        kicking_foot: Optional[str] = None,
        left_confidences: Optional[List[float]] = None,
        right_confidences: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Count run-up steps from a temporal 2D pose sequence.

        A step is represented by an alternating extremum of the
        left/right ankle phase along the player's run-up direction.
        Coordinates are expressed relative to the hip centre and
        normalised by leg length, making the signal insensitive to
        image translation and approximately invariant to scale.

        The first foot already planted at the run-up boundary is not
        counted. When the kicking foot is known, a final extremum from
        that foot close to ball contact is treated as the kicking swing,
        while a strong endpoint from the opposite support foot is kept.

        The returned diagnostics are intended for human review. In
        particular, ``review_required`` is set when pose coverage,
        temporal continuity, signal strength, or alternation is weak.
        """

        sequence_lengths = [
            len(frames),
            len(left_ankles),
            len(right_ankles),
            len(left_hips),
            len(right_hips)
        ]

        if len(set(sequence_lengths)) != 1:
            raise ValueError(
                "All pose sequences must have the same length."
            )

        sample_count = sequence_lengths[0]

        if left_confidences is not None:
            if len(left_confidences) != sample_count:
                raise ValueError(
                    "left_confidences must match the pose sequence length."
                )

        if right_confidences is not None:
            if len(right_confidences) != sample_count:
                raise ValueError(
                    "right_confidences must match the pose sequence length."
                )

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive finite number.")

        kick_frame = int(kick_frame)

        if kicking_foot is not None:
            kicking_foot = str(kicking_foot).lower()

            if kicking_foot not in {"left", "right"}:
                raise ValueError(
                    "kicking_foot must be 'left', 'right', or None."
                )

        warnings = []

        if sample_count == 0:
            return self._empty_step_result(
                kick_frame=kick_frame,
                runup_start_frame=runup_start_frame,
                warnings=["insufficient_pose_data"]
            )

        raw_frames = np.asarray(frames)

        if (
            raw_frames.ndim != 1
            or not np.all(np.isfinite(raw_frames))
            or not np.allclose(raw_frames, np.round(raw_frames))
        ):
            raise ValueError("frames must contain finite integer IDs.")

        frames_arr = raw_frames.astype(int)

        if len(np.unique(frames_arr)) != len(frames_arr):
            raise ValueError("frames must not contain duplicate IDs.")

        coordinate_arrays = []

        for name, values in [
            ("left_ankles", left_ankles),
            ("right_ankles", right_ankles),
            ("left_hips", left_hips),
            ("right_hips", right_hips)
        ]:
            arr = np.asarray(values, dtype=float)

            if arr.ndim != 2 or arr.shape != (sample_count, 2):
                raise ValueError(
                    f"{name} must have shape (N, 2)."
                )

            coordinate_arrays.append(arr)

        (
            left_ankles_arr,
            right_ankles_arr,
            left_hips_arr,
            right_hips_arr
        ) = coordinate_arrays

        confidence_was_supplied = (
            left_confidences is not None
            or right_confidences is not None
        )

        if left_confidences is None:
            left_conf_arr = np.ones(sample_count, dtype=float)
        else:
            left_conf_arr = np.asarray(
                left_confidences,
                dtype=float
            )

        if right_confidences is None:
            right_conf_arr = np.ones(sample_count, dtype=float)
        else:
            right_conf_arr = np.asarray(
                right_confidences,
                dtype=float
            )

        if (
            left_conf_arr.ndim != 1
            or right_conf_arr.ndim != 1
        ):
            raise ValueError(
                "Pose confidence sequences must be one-dimensional."
            )

        order = np.argsort(frames_arr)

        frames_arr = frames_arr[order]
        left_ankles_arr = left_ankles_arr[order]
        right_ankles_arr = right_ankles_arr[order]
        left_hips_arr = left_hips_arr[order]
        right_hips_arr = right_hips_arr[order]
        left_conf_arr = left_conf_arr[order]
        right_conf_arr = right_conf_arr[order]

        temporal_mask = frames_arr <= kick_frame

        if runup_start_frame is not None:
            runup_start_frame = int(runup_start_frame)
            temporal_mask &= frames_arr >= runup_start_frame

        finite_mask = np.ones(sample_count, dtype=bool)

        for arr in [
            left_ankles_arr,
            right_ankles_arr,
            left_hips_arr,
            right_hips_arr
        ]:
            finite_mask &= np.all(np.isfinite(arr), axis=1)
            finite_mask &= np.any(np.abs(arr) > 1e-6, axis=1)

        confidence_floor = 0.15
        confidence_mask = (
            np.isfinite(left_conf_arr)
            & np.isfinite(right_conf_arr)
            & (left_conf_arr >= confidence_floor)
            & (right_conf_arr >= confidence_floor)
        )

        if np.any(temporal_mask & ~confidence_mask):
            warnings.append("low_confidence_pose_filtered")

        valid_mask = temporal_mask & finite_mask & confidence_mask

        frames_valid = frames_arr[valid_mask]
        left_ankles_valid = left_ankles_arr[valid_mask]
        right_ankles_valid = right_ankles_arr[valid_mask]
        left_hips_valid = left_hips_arr[valid_mask]
        right_hips_valid = right_hips_arr[valid_mask]
        left_conf_valid = np.clip(
            left_conf_arr[valid_mask],
            0.0,
            1.0
        )
        right_conf_valid = np.clip(
            right_conf_arr[valid_mask],
            0.0,
            1.0
        )

        if len(frames_valid) < 5:
            start_value = (
                runup_start_frame
                if runup_start_frame is not None
                else (
                    int(frames_valid[0])
                    if len(frames_valid)
                    else None
                )
            )

            return self._empty_step_result(
                kick_frame=kick_frame,
                runup_start_frame=start_value,
                warnings=warnings + ["insufficient_pose_data"]
            )

        max_interpolated_missing = max(
            1,
            int(round(fps * 0.20))
        )

        expanded_frames = []
        expanded_left_ankles = []
        expanded_right_ankles = []
        expanded_left_hips = []
        expanded_right_hips = []
        expanded_left_conf = []
        expanded_right_conf = []

        pose_groups = [
            left_ankles_valid,
            right_ankles_valid,
            left_hips_valid,
            right_hips_valid
        ]

        expanded_groups = [
            expanded_left_ankles,
            expanded_right_ankles,
            expanded_left_hips,
            expanded_right_hips
        ]

        long_gap_detected = False

        for idx in range(len(frames_valid)):
            current_frame = int(frames_valid[idx])

            if idx > 0:
                previous_frame = int(frames_valid[idx - 1])
                missing_count = current_frame - previous_frame - 1

                if 0 < missing_count <= max_interpolated_missing:
                    frame_span = current_frame - previous_frame

                    for offset in range(1, frame_span):
                        alpha = offset / frame_span
                        expanded_frames.append(
                            previous_frame + offset
                        )

                        for source, target in zip(
                            pose_groups,
                            expanded_groups
                        ):
                            interpolated = (
                                (1.0 - alpha) * source[idx - 1]
                                + alpha * source[idx]
                            )
                            target.append(interpolated)

                        expanded_left_conf.append(
                            min(
                                left_conf_valid[idx - 1],
                                left_conf_valid[idx]
                            )
                        )
                        expanded_right_conf.append(
                            min(
                                right_conf_valid[idx - 1],
                                right_conf_valid[idx]
                            )
                        )

                elif missing_count > max_interpolated_missing:
                    long_gap_detected = True

            expanded_frames.append(current_frame)

            for source, target in zip(
                pose_groups,
                expanded_groups
            ):
                target.append(source[idx])

            expanded_left_conf.append(left_conf_valid[idx])
            expanded_right_conf.append(right_conf_valid[idx])

        expanded_frames = np.asarray(
            expanded_frames,
            dtype=int
        )
        expanded_left_ankles = np.asarray(
            expanded_left_ankles,
            dtype=float
        )
        expanded_right_ankles = np.asarray(
            expanded_right_ankles,
            dtype=float
        )
        expanded_left_hips = np.asarray(
            expanded_left_hips,
            dtype=float
        )
        expanded_right_hips = np.asarray(
            expanded_right_hips,
            dtype=float
        )
        expanded_left_conf = np.asarray(
            expanded_left_conf,
            dtype=float
        )
        expanded_right_conf = np.asarray(
            expanded_right_conf,
            dtype=float
        )

        hip_centres = (
            expanded_left_hips
            + expanded_right_hips
        ) / 2.0

        left_leg_lengths = np.linalg.norm(
            expanded_left_ankles - expanded_left_hips,
            axis=1
        )
        right_leg_lengths = np.linalg.norm(
            expanded_right_ankles - expanded_right_hips,
            axis=1
        )
        body_scales = (
            left_leg_lengths + right_leg_lengths
        ) / 2.0

        positive_scales = body_scales[
            np.isfinite(body_scales)
            & (body_scales > 1e-6)
        ]

        if len(positive_scales) < 3:
            return self._empty_step_result(
                kick_frame=kick_frame,
                runup_start_frame=runup_start_frame,
                warnings=warnings + ["invalid_body_scale"]
            )

        median_body_scale = float(
            np.median(positive_scales)
        )

        body_scales = np.clip(
            body_scales,
            median_body_scale * 0.50,
            median_body_scale * 2.00
        )

        edge_count = max(
            2,
            min(5, len(hip_centres) // 4)
        )
        start_centre = np.median(
            hip_centres[:edge_count],
            axis=0
        )
        end_centre = np.median(
            hip_centres[-edge_count:],
            axis=0
        )
        runup_vector = end_centre - start_centre
        runup_distance = float(np.linalg.norm(runup_vector))

        if runup_distance < median_body_scale * 0.05:
            return self._empty_step_result(
                kick_frame=kick_frame,
                runup_start_frame=runup_start_frame,
                warnings=warnings + ["runup_direction_unreliable"]
            )

        runup_direction = runup_vector / runup_distance

        if runup_start_frame is None:
            inferred_start, onset_warning = (
                self._infer_runup_start_frame(
                    frames=expanded_frames,
                    hip_centres=hip_centres,
                    body_scale=median_body_scale,
                    runup_direction=runup_direction,
                    fps=fps
                )
            )
            runup_start_frame = inferred_start

            if onset_warning is not None:
                warnings.append(onset_warning)

        analysis_mask = (
            (expanded_frames >= runup_start_frame)
            & (expanded_frames <= kick_frame)
        )

        expanded_frames = expanded_frames[analysis_mask]
        expanded_left_ankles = expanded_left_ankles[analysis_mask]
        expanded_right_ankles = expanded_right_ankles[analysis_mask]
        expanded_left_hips = expanded_left_hips[analysis_mask]
        expanded_right_hips = expanded_right_hips[analysis_mask]
        expanded_left_conf = expanded_left_conf[analysis_mask]
        expanded_right_conf = expanded_right_conf[analysis_mask]
        body_scales = body_scales[analysis_mask]

        original_interval_mask = (
            valid_mask
            & (frames_arr >= runup_start_frame)
            & (frames_arr <= kick_frame)
        )
        expected_frame_count = max(
            1,
            kick_frame - runup_start_frame + 1
        )
        coverage = min(
            1.0,
            float(np.sum(original_interval_mask))
            / expected_frame_count
        )

        if len(expanded_frames) < 5:
            return self._empty_step_result(
                kick_frame=kick_frame,
                runup_start_frame=runup_start_frame,
                coverage=coverage,
                warnings=warnings + ["insufficient_runup_data"]
            )

        hip_centres = (
            expanded_left_hips
            + expanded_right_hips
        ) / 2.0
        left_relative = expanded_left_ankles - hip_centres
        right_relative = expanded_right_ankles - hip_centres
        ankle_phase = left_relative - right_relative

        raw_phase_signal = (
            ankle_phase @ runup_direction
        ) / body_scales

        phase_signal = self._smooth_pose_phase_signal(
            frames=expanded_frames,
            values=raw_phase_signal,
            fps=fps
        )

        signal_range = float(
            np.percentile(phase_signal, 95)
            - np.percentile(phase_signal, 5)
        )
        peak_prominence = max(
            0.04,
            min(0.18, signal_range * 0.08)
        )
        minimum_peak_distance = max(
            2,
            int(round(fps * 0.16))
        )

        candidates = []
        segment_start = 0

        for idx in range(1, len(expanded_frames) + 1):
            is_segment_end = (
                idx == len(expanded_frames)
                or expanded_frames[idx]
                - expanded_frames[idx - 1]
                != 1
            )

            if not is_segment_end:
                continue

            segment_values = phase_signal[segment_start:idx]

            if len(segment_values) >= 3:
                left_peaks, left_props = find_peaks(
                    segment_values,
                    distance=minimum_peak_distance,
                    prominence=peak_prominence
                )
                right_peaks, right_props = find_peaks(
                    -segment_values,
                    distance=minimum_peak_distance,
                    prominence=peak_prominence
                )

                for peak_idx, prominence in zip(
                    left_peaks,
                    left_props["prominences"]
                ):
                    absolute_idx = segment_start + int(peak_idx)
                    candidates.append({
                        "frame": int(expanded_frames[absolute_idx]),
                        "foot": "left",
                        "prominence": float(prominence),
                        "source": "extremum"
                    })

                for peak_idx, prominence in zip(
                    right_peaks,
                    right_props["prominences"]
                ):
                    absolute_idx = segment_start + int(peak_idx)
                    candidates.append({
                        "frame": int(expanded_frames[absolute_idx]),
                        "foot": "right",
                        "prominence": float(prominence),
                        "source": "extremum"
                    })

            segment_start = idx

        if kicking_foot is not None:
            support_foot = (
                "left"
                if kicking_foot == "right"
                else "right"
            )
            support_sign = 1.0 if support_foot == "left" else -1.0
            support_score = support_sign * phase_signal
            support_window = max(
                3,
                int(round(fps * 0.33))
            )
            support_mask = (
                expanded_frames
                >= kick_frame - support_window
            )

            if np.any(support_mask):
                support_indices = np.flatnonzero(support_mask)
                local_best = int(
                    np.argmax(support_score[support_indices])
                )
                best_idx = int(support_indices[local_best])
                best_frame = int(expanded_frames[best_idx])
                lookback_start_frame = (
                    best_frame
                    - max(4, int(round(fps * 0.40)))
                )
                lookback_mask = (
                    (expanded_frames >= lookback_start_frame)
                    & (expanded_frames <= best_frame)
                )
                baseline = float(
                    np.min(support_score[lookback_mask])
                )
                endpoint_prominence = float(
                    support_score[best_idx] - baseline
                )
                strong_support_endpoint = (
                    endpoint_prominence >= peak_prominence
                    and support_score[best_idx]
                    >= peak_prominence * 0.5
                    and best_frame - runup_start_frame
                    >= max(2, int(round(fps * 0.08)))
                )

                if strong_support_endpoint:
                    already_present = any(
                        candidate["foot"] == support_foot
                        and abs(candidate["frame"] - best_frame)
                        <= max(1, int(round(fps * 0.10)))
                        for candidate in candidates
                    )

                    if not already_present:
                        candidates.append({
                            "frame": best_frame,
                            "foot": support_foot,
                            "prominence": endpoint_prominence,
                            "source": "support_endpoint"
                        })

        boundary_distance = max(
            2,
            int(round(fps * 0.08))
        )
        candidates = [
            candidate
            for candidate in candidates
            if (
                candidate["frame"] - runup_start_frame
                >= boundary_distance
                and candidate["frame"] <= kick_frame
            )
        ]

        if kicking_foot is not None:
            final_swing_window = max(
                2,
                int(round(fps * 0.18))
            )
            before_suppression = len(candidates)
            candidates = [
                candidate
                for candidate in candidates
                if not (
                    candidate["foot"] == kicking_foot
                    and 0
                    <= kick_frame - candidate["frame"]
                    <= final_swing_window
                )
            ]

            if len(candidates) < before_suppression:
                warnings.append(
                    "suppressed_final_kicking_foot_candidate"
                )

        candidates.sort(
            key=lambda item: (
                item["frame"],
                -item["prominence"]
            )
        )
        merge_distance = max(
            1,
            int(round(fps * 0.10))
        )
        merged_candidates = []

        for candidate in candidates:
            if (
                merged_candidates
                and candidate["frame"]
                - merged_candidates[-1]["frame"]
                <= merge_distance
            ):
                if (
                    candidate["prominence"]
                    > merged_candidates[-1]["prominence"]
                ):
                    merged_candidates[-1] = candidate
                continue

            merged_candidates.append(candidate)

        non_alternating = any(
            merged_candidates[idx]["foot"]
            == merged_candidates[idx - 1]["foot"]
            for idx in range(1, len(merged_candidates))
        )

        minimum_plausible_interval = max(2, int(np.ceil(fps * 0.15)))
        implausibly_short_interval = any(
            merged_candidates[idx]["frame"]
            - merged_candidates[idx - 1]["frame"]
            < minimum_plausible_interval
            for idx in range(1, len(merged_candidates))
        )

        if non_alternating:
            warnings.append("non_alternating_step_candidates")

        if implausibly_short_interval:
            warnings.append("implausibly_short_step_interval")

        if long_gap_detected:
            warnings.append("long_pose_gap")

        if not confidence_was_supplied:
            warnings.append("keypoint_confidence_not_supplied")

        if not merged_candidates:
            warnings.append("no_step_extrema_detected")

        pose_confidence = (
            float(np.mean(np.minimum(
                expanded_left_conf,
                expanded_right_conf
            )))
            if confidence_was_supplied
            else 0.85
        )

        if merged_candidates:
            median_prominence = float(np.median([
                candidate["prominence"]
                for candidate in merged_candidates
            ]))
            signal_quality = min(
                1.0,
                median_prominence
                / max(2.0 * peak_prominence, 1e-6)
            )
        elif signal_range < peak_prominence:
            signal_quality = 0.90
        else:
            signal_quality = 0.35

        alternation_quality = 0.65 if non_alternating else 1.0
        confidence = float(np.clip(
            coverage
            * (0.5 + 0.5 * pose_confidence)
            * signal_quality
            * alternation_quality,
            0.0,
            1.0
        ))

        review_required = bool(
            coverage < 0.70
            or pose_confidence < 0.40
            or long_gap_detected
            or non_alternating
            or implausibly_short_interval
            or confidence < 0.55
        )

        return {
            "step_count": len(merged_candidates),
            "step_frames": [
                int(candidate["frame"])
                for candidate in merged_candidates
            ],
            "step_feet": [
                candidate["foot"]
                for candidate in merged_candidates
            ],
            "runup_start_frame": int(runup_start_frame),
            "kick_frame": kick_frame,
            "coverage": round(coverage, 3),
            "confidence": round(confidence, 3),
            "review_required": review_required,
            "warnings": list(dict.fromkeys(warnings)),
            "signal_frames": [
                int(frame)
                for frame in expanded_frames
            ],
            "phase_signal": [
                round(float(value), 6)
                for value in phase_signal
            ]
        }

    @staticmethod
    def _infer_runup_start_frame(
        frames: np.ndarray,
        hip_centres: np.ndarray,
        body_scale: float,
        runup_direction: np.ndarray,
        fps: float
    ) -> Tuple[int, Optional[str]]:
        """
        Find the last sustained stationary-to-forward transition.

        Small pauses are closed before selecting the last movement
        run that is sustained close to the kick boundary.
        """

        if len(frames) < 5:
            return int(frames[0]), "runup_start_fallback_first_pose"

        progress = (
            hip_centres @ runup_direction
        ) / max(body_scale, 1e-6)
        frame_deltas = np.diff(frames).astype(float)
        progress_deltas = np.diff(progress)
        forward_speed = np.zeros(len(frames), dtype=float)
        forward_speed[1:] = progress_deltas / frame_deltas

        smooth_window = max(3, int(round(fps * 0.10)))

        if smooth_window % 2 == 0:
            smooth_window += 1

        smooth_window = min(
            smooth_window,
            len(forward_speed)
            if len(forward_speed) % 2 == 1
            else len(forward_speed) - 1
        )

        if smooth_window >= 3:
            pad = smooth_window // 2
            padded = np.pad(
                forward_speed,
                (pad, pad),
                mode="edge"
            )
            forward_speed = np.convolve(
                padded,
                np.ones(smooth_window) / smooth_window,
                mode="valid"
            )

        positive_speed = np.maximum(forward_speed, 0.0)
        high_speed = float(np.percentile(positive_speed, 90))

        if high_speed < 0.003:
            return int(frames[0]), "runup_start_fallback_first_pose"

        noise_level = float(
            np.percentile(positive_speed, 20)
        )
        movement_threshold = max(
            0.003,
            min(noise_level * 2.5, high_speed * 0.50),
            high_speed * 0.18
        )
        moving = positive_speed >= movement_threshold

        max_pause = max(1, int(round(fps * 0.18)))
        idx = 0

        while idx < len(moving):
            if moving[idx]:
                idx += 1
                continue

            gap_start = idx

            while idx < len(moving) and not moving[idx]:
                idx += 1

            gap_end = idx

            if (
                gap_start > 0
                and gap_end < len(moving)
                and gap_end - gap_start <= max_pause
                and frames[gap_end] - frames[gap_start - 1]
                <= max_pause + 1
            ):
                moving[gap_start:gap_end] = True

        sustained_length = max(3, int(round(fps * 0.12)))
        movement_runs = []
        idx = 0

        while idx < len(moving):
            if not moving[idx]:
                idx += 1
                continue

            run_start = idx

            while (
                idx + 1 < len(moving)
                and moving[idx + 1]
                and frames[idx + 1] - frames[idx] == 1
            ):
                idx += 1

            run_end = idx

            if run_end - run_start + 1 >= sustained_length:
                movement_runs.append((run_start, run_end))

            idx += 1

        if not movement_runs:
            return int(frames[0]), "runup_start_fallback_first_pose"

        near_kick_tolerance = max(
            sustained_length,
            int(round(fps * 0.30))
        )
        near_kick_runs = [
            item
            for item in movement_runs
            if frames[-1] - frames[item[1]] <= near_kick_tolerance
        ]
        selected_run = (
            near_kick_runs[-1]
            if near_kick_runs
            else movement_runs[-1]
        )

        return int(frames[selected_run[0]]), None

    @staticmethod
    def _smooth_pose_phase_signal(
        frames: np.ndarray,
        values: np.ndarray,
        fps: float
    ) -> np.ndarray:
        """Smooth consecutive phase segments without crossing gaps."""

        smoothed = np.asarray(values, dtype=float).copy()
        target_window = max(3, int(round(fps * 0.10)))

        if target_window % 2 == 0:
            target_window += 1

        segment_start = 0

        for idx in range(1, len(frames) + 1):
            segment_end = (
                idx == len(frames)
                or frames[idx] - frames[idx - 1] != 1
            )

            if not segment_end:
                continue

            segment = smoothed[segment_start:idx]
            window = min(
                target_window,
                len(segment)
                if len(segment) % 2 == 1
                else len(segment) - 1
            )

            if window >= 3:
                pad = window // 2
                padded = np.pad(
                    segment,
                    (pad, pad),
                    mode="edge"
                )
                smoothed[segment_start:idx] = np.convolve(
                    padded,
                    np.ones(window) / window,
                    mode="valid"
                )

            segment_start = idx

        return smoothed

    @staticmethod
    def _empty_step_result(
        kick_frame: int,
        runup_start_frame: Optional[int],
        warnings: List[str],
        coverage: float = 0.0
    ) -> Dict[str, Any]:
        """Return the stable schema used for uncountable pose clips."""

        return {
            "step_count": 0,
            "step_frames": [],
            "step_feet": [],
            "runup_start_frame": (
                int(runup_start_frame)
                if runup_start_frame is not None
                else None
            ),
            "kick_frame": int(kick_frame),
            "coverage": round(float(coverage), 3),
            "confidence": 0.0,
            "review_required": True,
            "warnings": list(dict.fromkeys(warnings)),
            "signal_frames": [],
            "phase_signal": []
        }

    # ============================================================
    # Kicking foot - baseline method
    # ============================================================

    def classify_kicking_foot(
        self,
        left_ankle: Tuple[float, float],
        right_ankle: Tuple[float, float],
        ball: Tuple[float, float]
    ) -> str:
        """
        Baseline kicking-foot classifier.

        Selects the ankle closest to the detected football.

        Note
        ----
        This method can fail when the football is partially
        occluded or when ball detection is unstable. It is
        retained as a baseline for comparison with the temporal
        motion-based method.
        """

        lx, ly = left_ankle
        rx, ry = right_ankle
        bx, by = ball

        d_left = np.hypot(
            lx - bx,
            ly - by
        )

        d_right = np.hypot(
            rx - bx,
            ry - by
        )

        return (
            "right"
            if d_right <= d_left
            else "left"
        )

    # ============================================================
    # Kicking foot - temporal 2D motion method
    # ============================================================

    def classify_kicking_foot_from_motion(
        self,
        frames: List[int],
        left_ankles: List[
            Tuple[float, float]
        ],
        right_ankles: List[
            Tuple[float, float]
        ],
        left_hips: List[
            Tuple[float, float]
        ],
        right_hips: List[
            Tuple[float, float]
        ],
        kick_frame: int,
        window: int = 3
    ) -> Dict[str, Any]:
        """
        Classify the kicking foot using temporal 2D ankle motion.

        Ankle coordinates are first expressed relative to the
        midpoint of the two hips. This reduces the effect of the
        player's overall translation during the run-up.

        The algorithm then measures frame-to-frame 2D displacement
        of each ankle around the kick event. The foot with the
        larger maximum displacement is classified as the kicking
        foot.

        Missing frames are allowed. Motion is calculated only
        between consecutive available frames.

        Parameters
        ----------
        frames:
            Frame IDs corresponding to pose observations.

        left_ankles:
            Sequence of (x, y) left ankle coordinates.

        right_ankles:
            Sequence of (x, y) right ankle coordinates.

        left_hips:
            Sequence of (x, y) left hip coordinates.

        right_hips:
            Sequence of (x, y) right hip coordinates.

        kick_frame:
            Estimated or manually labelled kick/contact frame.

        window:
            Number of frames before and after the kick used for
            motion analysis.

        Returns
        -------
        dict
            Kicking-foot prediction and motion statistics.
        """

        lengths = [
            len(frames),
            len(left_ankles),
            len(right_ankles),
            len(left_hips),
            len(right_hips)
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                "All pose sequences must have the same length."
            )

        if len(frames) < 2:
            return self._empty_motion_result()

        # --------------------------------------------------------
        # Convert to NumPy
        # --------------------------------------------------------

        frames_arr = np.asarray(
            frames,
            dtype=int
        )

        left_ankles_arr = np.asarray(
            left_ankles,
            dtype=float
        )

        right_ankles_arr = np.asarray(
            right_ankles,
            dtype=float
        )

        left_hips_arr = np.asarray(
            left_hips,
            dtype=float
        )

        right_hips_arr = np.asarray(
            right_hips,
            dtype=float
        )

        # --------------------------------------------------------
        # Validate coordinate shapes
        # --------------------------------------------------------

        coordinate_arrays = [
            left_ankles_arr,
            right_ankles_arr,
            left_hips_arr,
            right_hips_arr
        ]

        for arr in coordinate_arrays:

            if (
                arr.ndim != 2
                or arr.shape[1] != 2
            ):
                raise ValueError(
                    "Pose coordinates must have shape (N, 2)."
                )

        # --------------------------------------------------------
        # Hip-relative coordinates
        # --------------------------------------------------------

        hip_centres = (
            left_hips_arr
            + right_hips_arr
        ) / 2.0

        left_relative = (
            left_ankles_arr
            - hip_centres
        )

        right_relative = (
            right_ankles_arr
            - hip_centres
        )

        # --------------------------------------------------------
        # Select frames around kick
        # --------------------------------------------------------

        start_frame = (
            kick_frame - window
        )

        end_frame = (
            kick_frame + window
        )

        mask = (
            (frames_arr >= start_frame)
            & (frames_arr <= end_frame)
        )

        selected_frames = (
            frames_arr[mask]
        )

        selected_left = (
            left_relative[mask]
        )

        selected_right = (
            right_relative[mask]
        )

        if len(selected_frames) < 2:
            return self._empty_motion_result()

        # Ensure temporal ordering
        order = np.argsort(
            selected_frames
        )

        selected_frames = (
            selected_frames[order]
        )

        selected_left = (
            selected_left[order]
        )

        selected_right = (
            selected_right[order]
        )

        # --------------------------------------------------------
        # Frame-to-frame 2D displacement
        # --------------------------------------------------------

        left_motion = []
        right_motion = []
        used_frame_pairs = []

        for i in range(
            len(selected_frames) - 1
        ):

            current_frame = (
                selected_frames[i]
            )

            next_frame = (
                selected_frames[i + 1]
            )

            # Do not calculate motion across missing-frame gaps
            if (
                next_frame
                - current_frame
                != 1
            ):
                continue

            left_delta = (
                selected_left[i + 1]
                - selected_left[i]
            )

            right_delta = (
                selected_right[i + 1]
                - selected_right[i]
            )

            left_distance = float(
                np.linalg.norm(
                    left_delta
                )
            )

            right_distance = float(
                np.linalg.norm(
                    right_delta
                )
            )

            left_motion.append(
                left_distance
            )

            right_motion.append(
                right_distance
            )

            used_frame_pairs.append(
                (
                    int(current_frame),
                    int(next_frame)
                )
            )

        if not left_motion:
            return self._empty_motion_result()

        # --------------------------------------------------------
        # Motion statistics
        # --------------------------------------------------------

        left_max = float(
            np.max(left_motion)
        )

        right_max = float(
            np.max(right_motion)
        )

        left_total = float(
            np.sum(left_motion)
        )

        right_total = float(
            np.sum(right_motion)
        )

        # Maximum short-term movement is the primary feature.
        if right_max > left_max:
            kicking_foot = "right"
        elif left_max > right_max:
            kicking_foot = "left"
        else:
            kicking_foot = "unknown"

        smaller = min(
            left_max,
            right_max
        )

        larger = max(
            left_max,
            right_max
        )

        if smaller > 1e-6:
            motion_ratio = (
                larger / smaller
            )
        elif larger > 0:
            motion_ratio = float("inf")
        else:
            motion_ratio = 0.0

        return {
            "kicking_foot":
                kicking_foot,

            "left_max_motion":
                round(left_max, 2),

            "right_max_motion":
                round(right_max, 2),

            "left_total_motion":
                round(left_total, 2),

            "right_total_motion":
                round(right_total, 2),

            "motion_ratio":
                round(motion_ratio, 2),

            "used_frame_pairs":
                used_frame_pairs
        }

    # ============================================================
    # Helper
    # ============================================================
    def detect_kick_frame_from_pose(
            self,
            frames: List[int],
            left_ankles: List[Tuple[float, float]],
            right_ankles: List[Tuple[float, float]],
            left_hips: List[Tuple[float, float]],
            right_hips: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Estimate the kick frame from temporal pose motion.

        The method converts both ankles into coordinates relative
        to the hip centre, then finds the largest frame-to-frame
        2D ankle displacement.

        The frame at which the largest motion ends is returned as
        the estimated kick frame.

        Missing frame gaps are ignored.
        """

        lengths = [
            len(frames),
            len(left_ankles),
            len(right_ankles),
            len(left_hips),
            len(right_hips)
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                "All pose sequences must have the same length."
            )

        if len(frames) < 2:
            return {
                "kick_frame": None,
                "peak_motion": 0.0,
                "peak_foot": "unknown",
                "frame_pair": None
            }

        frames_arr = np.asarray(
            frames,
            dtype=int
        )

        left_ankles_arr = np.asarray(
            left_ankles,
            dtype=float
        )

        right_ankles_arr = np.asarray(
            right_ankles,
            dtype=float
        )

        left_hips_arr = np.asarray(
            left_hips,
            dtype=float
        )

        right_hips_arr = np.asarray(
            right_hips,
            dtype=float
        )

        # Sort observations by frame number
        order = np.argsort(frames_arr)

        frames_arr = frames_arr[order]
        left_ankles_arr = left_ankles_arr[order]
        right_ankles_arr = right_ankles_arr[order]
        left_hips_arr = left_hips_arr[order]
        right_hips_arr = right_hips_arr[order]

        # Hip centre
        hip_centres = (
                              left_hips_arr
                              + right_hips_arr
                      ) / 2.0

        # Remove overall body translation
        left_relative = (
                left_ankles_arr
                - hip_centres
        )

        right_relative = (
                right_ankles_arr
                - hip_centres
        )

        best_motion = -1.0
        best_frame = None
        best_foot = "unknown"
        best_pair = None

        for i in range(
                len(frames_arr) - 1
        ):

            current_frame = int(
                frames_arr[i]
            )

            next_frame = int(
                frames_arr[i + 1]
            )

            # Never calculate motion across a missing-frame gap
            if (
                    next_frame
                    - current_frame
                    != 1
            ):
                continue

            left_delta = (
                    left_relative[i + 1]
                    - left_relative[i]
            )

            right_delta = (
                    right_relative[i + 1]
                    - right_relative[i]
            )

            left_motion = float(
                np.linalg.norm(
                    left_delta
                )
            )

            right_motion = float(
                np.linalg.norm(
                    right_delta
                )
            )

            if left_motion >= right_motion:
                frame_motion = left_motion
                frame_foot = "left"
            else:
                frame_motion = right_motion
                frame_foot = "right"

            if frame_motion > best_motion:
                best_motion = frame_motion
                best_frame = next_frame
                best_foot = frame_foot

                best_pair = (
                    current_frame,
                    next_frame
                )

        if best_frame is None:
            return {
                "kick_frame": None,
                "peak_motion": 0.0,
                "peak_foot": "unknown",
                "frame_pair": None
            }

        return {
            "kick_frame": int(
                best_frame
            ),

            "peak_motion": round(
                float(best_motion),
                2
            ),

            "peak_foot": best_foot,

            "frame_pair": best_pair
        }
    @staticmethod
    def _empty_motion_result() -> Dict[str, Any]:
        """
        Return a standard result when insufficient pose data
        are available.
        """

        return {
            "kicking_foot": "unknown",
            "left_max_motion": 0.0,
            "right_max_motion": 0.0,
            "left_total_motion": 0.0,
            "right_total_motion": 0.0,
            "motion_ratio": 0.0,
            "used_frame_pairs": []
        }
