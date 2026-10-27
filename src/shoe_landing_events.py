"""Visible image-plane swing-to-stance proposals from paired shoe landmarks.

Landmark stability near an actor box's bottom is not proof of ground contact.
These proposals never describe total run-up steps or use reference labels.
"""
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from collections.abc import Mapping

import numpy as np


SHOES = {"left": [29, 31], "right": [30, 32]}
HIPS = [23, 24]


@dataclass(frozen=True)
class LandingConfig:
    confidence_min: float = .5
    minimum_body_height_px: float = 30.
    maximum_mirror_error_body: float = .06
    maximum_pair_gap_seconds: float = .12
    swing_speed_min_body_per_second: float = .5
    stance_speed_max_body_per_second: float = .25
    minimum_swing_observations: int = 3
    minimum_swing_displacement_body: float = .04
    minimum_stance_observations: int = 3
    minimum_stance_seconds: float = .06
    maximum_stance_spread_body: float = .03
    maximum_swing_to_stance_seconds: float = .35
    minimum_hip_progress_body: float = .02
    maximum_shoe_speed_body_per_second: float = 6.
    maximum_hip_speed_body_per_second: float = 3.
    maximum_scale_ratio: float = 1.3
    near_bottom_fraction: float = .25
    below_box_tolerance_fraction: float = .08
    channel_swap_margin_body: float = .04
    channel_swap_minimum_direct_body: float = .08
    minimum_same_foot_separation_seconds: float = .15

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_swing_observations", "minimum_stance_observations"):
            if not isinstance(getattr(self, name), Integral) or getattr(self, name) < 3:
                raise ValueError(f"{name} must be an integer of at least three")
        if (self.confidence_min > 1 or self.near_bottom_fraction >= 1
                or self.stance_speed_max_body_per_second >= self.swing_speed_min_body_per_second
                or self.maximum_shoe_speed_body_per_second <= self.swing_speed_min_body_per_second
                or self.maximum_scale_ratio < 1):
            raise ValueError("Inconsistent landing configuration")


def _frame(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Source frame IDs must be nonnegative integers")
    return int(value)


def _lookup(mapping, frame):
    return mapping.get(str(frame), mapping.get(frame))


def _project(points, matrix):
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) <= 1e-9):
        return None
    result = homogeneous[:, :2] / homogeneous[:, 2:]
    return result if np.isfinite(result).all() else None


def _transform_map(transforms):
    if not isinstance(transforms, Mapping):
        raise ValueError("Camera transforms must be a frame-keyed map or camera-motion payload")
    if "transforms" in transforms:
        mapping = transforms["transforms"]
        if not isinstance(mapping, Mapping):
            raise ValueError("Camera-motion transforms must be a mapping")
        pairs = transforms.get("pairs", [])
        if not isinstance(pairs, list) or any(not isinstance(row, Mapping) or "frame" not in row for row in pairs):
            raise ValueError("Camera-motion pairs must be a list")
        rejected = {_frame(row["frame"]) for row in pairs
                    if row.get("accepted") is not True or row.get("reference_chain_valid") is not True}
    else:
        mapping, rejected = transforms, set()
    output = {}
    for key, raw in mapping.items():
        try:
            frame = int(key)
        except (ValueError, TypeError):
            raise ValueError("Invalid transform frame key") from None
        if isinstance(key, bool) or frame < 0 or str(frame) != str(key):
            raise ValueError("Invalid transform frame key")
        if frame in output:
            raise ValueError("Duplicate transform frame key")
        if raw is None or frame in rejected:
            output[frame] = None
            continue
        matrix = np.asarray(raw, float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or abs(np.linalg.det(matrix)) <= 1e-12):
            output[frame] = None
        else:
            output[frame] = matrix
    return output


def _read_records(records, first, last):
    if not isinstance(records, (list, tuple)):
        raise ValueError("Pose records must be a sequence")
    output = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("Pose records must be mappings")
        frame = _frame(row.get("frame"))
        if not isinstance(row.get("reflected"), bool):
            raise ValueError("Each pose record needs a boolean reflected flag")
        if not first <= frame <= last:
            continue
        key = (frame, row["reflected"])
        if key in output:
            raise ValueError("Duplicate same-frame pose view")
        points, confidence, box = (row.get("landmarks_source_xy"),
                                   row.get("landmark_confidence"), row.get("anchor_bbox"))
        if points is None or confidence is None or box is None:
            output[key] = None
            continue
        points, confidence, box = np.asarray(points, float), np.asarray(confidence, float), np.asarray(box, float)
        if (points.shape != (33, 2) or confidence.shape != (33,) or box.shape != (4,)
                or not np.isfinite(points).all() or not np.isfinite(confidence).all()
                or not np.isfinite(box).all() or np.any(confidence < 0) or np.any(confidence > 1)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Expected finite 33-point xy, 33 confidences and positive anchor box")
        output[key] = dict(points=points, confidence=confidence, box=box,
                           pixel_hash=row.get("frame_pixel_sha256"),
                           accepted=row.get("roi_identity_status") == "accepted")
    return output


def _near_bottom(row, indices, cfg):
    box, points = row["box"], row["points"][indices]
    height = box[3] - box[1]
    return bool(np.all(points[:, 1] >= box[3] - cfg.near_bottom_fraction * height)
                and np.all(points[:, 1] <= box[3] + cfg.below_box_tolerance_fraction * height)
                and np.all(points[:, 0] >= box[0] - cfg.below_box_tolerance_fraction * height)
                and np.all(points[:, 0] <= box[2] + cfg.below_box_tolerance_fraction * height))


def _series(records, first, last, matrices, hashes, cfg):
    views = {False: [], True: []}
    diagnostics, seen_hashes = [], set()
    identity_segment = 0
    for frame in range(first, last + 1):
        pair = [records.get((frame, flag)) for flag in (False, True)]
        missing_actor = any(row is None or not row["accepted"] for row in pair)
        if missing_actor:
            identity_segment += 1
        digest = _lookup(hashes, frame)
        if not isinstance(digest, str) or not digest:
            diagnostics.append(dict(frame=frame, reason="missing_decoded_pixel_hash"))
            continue
        if digest in seen_hashes:
            diagnostics.append(dict(frame=frame, reason="repeated_decoded_pixels"))
            continue
        seen_hashes.add(digest)
        matrix = matrices.get(frame)
        if matrix is None:
            diagnostics.append(dict(frame=frame, reason="missing_measurable_camera_transform"))
            continue
        if missing_actor:
            diagnostics.append(dict(frame=frame, reason="missing_paired_accepted_actor_pose"))
            continue
        if any(row["pixel_hash"] is not None and row["pixel_hash"] != digest for row in pair):
            diagnostics.append(dict(frame=frame, reason="pose_decoded_pixel_hash_disagreement"))
            continue
        scale = float(np.median([row["box"][3] - row["box"][1] for row in pair]))
        if scale < cfg.minimum_body_height_px:
            diagnostics.append(dict(frame=frame, reason="actor_too_small"))
            continue
        if any(np.min(row["confidence"][HIPS]) < cfg.confidence_min for row in pair):
            diagnostics.append(dict(frame=frame, reason="missing_confident_paired_hips"))
            continue
        if any(np.any(row["points"][HIPS] < row["box"][:2])
               or np.any(row["points"][HIPS] > row["box"][2:]) for row in pair):
            diagnostics.append(dict(frame=frame, reason="paired_hips_outside_actor_box"))
            continue
        if np.max(np.linalg.norm(pair[0]["points"][HIPS] - pair[1]["points"][HIPS], axis=1)) / scale > cfg.maximum_mirror_error_body:
            diagnostics.append(dict(frame=frame, reason="normal_mirror_hip_disagreement"))
            continue
        projected = [_project(row["points"], matrix) for row in pair]
        corners = [np.array([[row["box"][0], row["box"][1]], [row["box"][0], row["box"][3]]]) for row in pair]
        projected_corners = [_project(points, matrix) for points in corners]
        if any(points is None for points in projected + projected_corners):
            diagnostics.append(dict(frame=frame, reason="invalid_camera_projection"))
            continue
        common_scale = float(np.median([np.linalg.norm(points[1] - points[0]) for points in projected_corners]))
        if common_scale <= 1e-6:
            diagnostics.append(dict(frame=frame, reason="degenerate_camera_compensated_body_scale"))
            continue
        side_valid = {}
        for side, indices in SHOES.items():
            reason = None
            if any(np.min(row["confidence"][indices]) < cfg.confidence_min for row in pair):
                reason = "missing_confident_paired_shoe"
            elif np.max(np.linalg.norm(pair[0]["points"][indices] - pair[1]["points"][indices], axis=1)) / scale > cfg.maximum_mirror_error_body:
                reason = "normal_mirror_shoe_disagreement"
            side_valid[side] = reason is None
            if reason:
                diagnostics.append(dict(frame=frame, foot=side, reason=reason))
        for flag, row, points in zip((False, True), pair, projected):
            views[flag].append(dict(frame=frame, points=points, hip=points[HIPS].mean(axis=0),
                scale=common_scale, valid=side_valid.copy(), identity_segment=identity_segment,
                near_bottom={side: _near_bottom(row, indices, cfg) for side, indices in SHOES.items()}))
    return views, diagnostics


def _connected(previous, current, matrices, fps, cfg):
    return ((current["frame"] - previous["frame"]) / fps <= cfg.maximum_pair_gap_seconds
            and previous.get("identity_segment") == current.get("identity_segment")
            and all(matrices.get(frame) is not None for frame in range(previous["frame"], current["frame"] + 1)))


def _channel_swaps(rows, matrices, fps, cfg):
    ambiguous = []
    for previous, current in zip(rows, rows[1:]):
        if (not _connected(previous, current, matrices, fps, cfg)
                or not all(previous["valid"].values()) or not all(current["valid"].values())):
            continue
        scale = (previous["scale"] + current["scale"]) / 2
        previous_feet = np.array([previous["points"][indices].mean(axis=0) for indices in SHOES.values()])
        current_feet = np.array([current["points"][indices].mean(axis=0) for indices in SHOES.values()])
        direct = float(np.linalg.norm(current_feet - previous_feet, axis=1).mean() / scale)
        swapped = float(np.linalg.norm(current_feet[::-1] - previous_feet, axis=1).mean() / scale)
        if direct >= cfg.channel_swap_minimum_direct_body and direct - swapped >= cfg.channel_swap_margin_body:
            ambiguous.append(current["frame"])
    return ambiguous


def _propose_view(rows, side, matrices, fps, cfg, support_indices=None):
    indices = SHOES[side] if support_indices is None else support_indices
    previous, fast, armed, stance = None, [], None, []
    events, diagnostics, analyzable_segments, segment = [], [], [], []
    last_event_frame = active_stance_event = None
    for row in rows:
        if not row["valid"][side]:
            previous, fast, armed, stance = None, [], None, []
            active_stance_event = None
            if segment:
                analyzable_segments.append(segment)
                segment = []
            continue
        connected = previous is not None and _connected(previous, row, matrices, fps, cfg)
        if not connected:
            if segment:
                analyzable_segments.append(segment)
            segment = [row["frame"]]
            previous, fast, armed, stance = row, [], None, []
            active_stance_event = None
            continue
        scale = (previous["scale"] + row["scale"]) / 2
        dt = (row["frame"] - previous["frame"]) / fps
        speeds = np.linalg.norm(row["points"][indices] - previous["points"][indices], axis=1) / scale / dt
        hip_speed = np.linalg.norm(row["hip"] - previous["hip"]) / scale / dt
        scale_ratio = max(previous["scale"], row["scale"]) / min(previous["scale"], row["scale"])
        if (max(speeds) > cfg.maximum_shoe_speed_body_per_second
                or hip_speed > cfg.maximum_hip_speed_body_per_second
                or scale_ratio > cfg.maximum_scale_ratio):
            diagnostics.append(dict(frame=row["frame"], foot=side, reason="pose_jump_or_scale_discontinuity"))
            analyzable_segments.append(segment)
            segment = [row["frame"]]
            previous, fast, armed, stance = row, [], None, []
            active_stance_event = None
            continue
        segment.append(row["frame"])
        # Keep later support from the same stance: paired views can detect its
        # onset one frame apart without describing different landing events.
        if active_stance_event is not None:
            extended = stance + [row]
            cloud = np.array([item["points"][indices] for item in extended])
            spread = np.max(np.linalg.norm(cloud[:, None] - cloud[None, :], axis=3)) / scale
            if (max(speeds) <= cfg.stance_speed_max_body_per_second
                    and row["near_bottom"][side] and spread <= cfg.maximum_stance_spread_body):
                stance = extended
                active_stance_event.update(stance_frames=[item["frame"] for item in stance],
                    stance_seconds=float((stance[-1]["frame"] - stance[0]["frame"]) / fps),
                    stance_spread_body=float(spread))
                previous = row
                continue
            active_stance_event, stance = None, []
        if min(speeds) >= cfg.swing_speed_min_body_per_second:
            stance = []
            fast = fast + [row] if fast else [previous, row]
            if len(fast) >= cfg.minimum_swing_observations:
                displacement = np.linalg.norm(fast[-1]["points"][indices].mean(axis=0)
                                              - fast[0]["points"][indices].mean(axis=0)) / scale
                if displacement >= cfg.minimum_swing_displacement_body:
                    armed = dict(first=fast[0], last=row, frames=[item["frame"] for item in fast])
        elif max(speeds) <= cfg.stance_speed_max_body_per_second and armed:
            fast = []
            if (row["frame"] - armed["last"]["frame"]) / fps > cfg.maximum_swing_to_stance_seconds:
                armed, stance = None, []
            else:
                stance = stance + [row] if stance else [previous, row]
                duration = (stance[-1]["frame"] - stance[0]["frame"]) / fps
                cloud = np.array([item["points"][indices] for item in stance])
                spread = np.max(np.linalg.norm(cloud[:, None] - cloud[None, :], axis=3)) / scale
                hip_progress = np.linalg.norm(row["hip"] - armed["first"]["hip"]) / scale
                separation = float("inf") if last_event_frame is None else (stance[0]["frame"] - last_event_frame) / fps
                if (len(stance) >= cfg.minimum_stance_observations and duration >= cfg.minimum_stance_seconds
                        and all(item["near_bottom"][side] for item in stance)
                        and spread <= cfg.maximum_stance_spread_body
                        and hip_progress >= cfg.minimum_hip_progress_body
                        and separation >= cfg.minimum_same_foot_separation_seconds):
                    active_stance_event = dict(frame=stance[0]["frame"], foot=side,
                        swing_frames=armed["frames"], stance_frames=[item["frame"] for item in stance],
                        stance_seconds=float(duration), stance_spread_body=float(spread),
                        hip_progress_body=float(hip_progress))
                    events.append(active_stance_event)
                    last_event_frame = stance[0]["frame"]
                    armed = None
        else:
            fast, stance = [], []
            if armed and (row["frame"] - armed["last"]["frame"]) / fps > cfg.maximum_swing_to_stance_seconds:
                armed = None
        previous = row
    if segment:
        analyzable_segments.append(segment)
    return events, diagnostics, analyzable_segments


def propose_shoe_landings(records, fps, interval_frames, transforms, decoded_frame_hashes, config=None):
    """Propose paired transitions within inclusive clip-local source bounds.

    ``transforms`` maps every source frame into one common shot image plane;
    null/missing intermediate matrices break the evidence chain. An existing
    ``estimate_background_motion`` payload can be supplied directly. Restored
    normal/mirror anatomical source landmarks must already be inverse mapped.
    Hashes identify decoded full-frame pixels, not model crops or timestamps.
    """
    cfg = LandingConfig() if config is None else config
    if not isinstance(cfg, LandingConfig):
        raise ValueError("config must be LandingConfig")
    if isinstance(fps, bool) or not isinstance(fps, Real) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(interval_frames, (list, tuple)) or len(interval_frames) != 2:
        raise ValueError("An inclusive [approach_start,last_stationary] frame interval is required")
    first, last = map(_frame, interval_frames)
    if last < first:
        raise ValueError("Frame interval is reversed")
    if not isinstance(decoded_frame_hashes, Mapping):
        raise ValueError("Decoded pixel hashes must be a frame-keyed mapping")
    matrices = _transform_map(transforms)
    rows = _read_records(records, first, last)
    views, diagnostics = _series(rows, first, last, matrices, decoded_frame_hashes, cfg)
    base = dict(schema_version="shoe-landing-proposals-v1", method_version="complete_stance_support_v2", fps=float(fps),
        observable_window=dict(start_frame=first, end_frame=last),
        full_approach_visible=False, count_complete=False, total_run_up_steps=None,
        striking_foot="unknown", candidate_count=None, observed_proposal_count=0, landings=None,
        settings=asdict(cfg), human_verified=False, accuracy_validated=False,
        reference_labels_used_for_prediction=False, metric_calibrated=False,
        scope="visible image-plane swing-to-stance proposals; not physical foot contacts or total run-up steps",
        diagnostics=diagnostics, paired_unique_pose_frames=[row["frame"] for row in views[False]])
    swaps = {name: _channel_swaps(view, matrices, fps, cfg)
             for name, view in (("normal", views[False]), ("mirrored_inverse_mapped", views[True]))}
    base["temporal_channel_ambiguous_frames"] = swaps
    if any(swaps.values()):
        base["status"] = "temporal_anatomical_channel_ambiguity"
        return base
    proposals, segments = {}, {}
    for flag, name in ((False, "normal"), (True, "mirrored_inverse_mapped")):
        proposals[name], segments[name] = {}, {}
        for side in SHOES:
            events, rejected, support = _propose_view(views[flag], side, matrices, fps, cfg)
            proposals[name][side], segments[name][side] = events, support
            base["diagnostics"].extend(dict(item, view=name) for item in rejected)
    base["view_proposals"], base["measurable_segments"] = proposals, segments
    minimum = max(cfg.minimum_swing_observations, cfg.minimum_stance_observations)
    measurable = any(len(segment) >= minimum
                     and (segment[-1] - segment[0]) / fps >= cfg.minimum_stance_seconds
                     for side in SHOES for segment in segments["normal"][side])
    if not measurable:
        base["status"] = "insufficient_paired_distinct_measurable_observations"
        return base
    matched = []
    for side in SHOES:
        mirrors = proposals["mirrored_inverse_mapped"][side]
        used = set()
        for normal in proposals["normal"][side]:
            nearby = [(abs(normal["frame"] - mirror["frame"]), index, mirror)
                      for index, mirror in enumerate(mirrors) if index not in used
                      and abs(normal["frame"] - mirror["frame"]) / fps <= cfg.maximum_pair_gap_seconds]
            if not nearby:
                base["diagnostics"].append(dict(frame=normal["frame"], foot=side,
                                                 reason="normal_proposal_without_mirror_support"))
                continue
            candidates = []
            for difference, index, mirror in nearby:
                overlap = sorted(set(normal["stance_frames"]) & set(mirror["stance_frames"]))
                if (len(overlap) >= cfg.minimum_stance_observations
                        and (overlap[-1] - overlap[0]) / fps >= cfg.minimum_stance_seconds
                        and all((last - first) / fps <= cfg.maximum_pair_gap_seconds
                                for first, last in zip(overlap, overlap[1:]))):
                    candidates.append((difference, index, mirror, overlap))
            if not candidates:
                base["diagnostics"].append(dict(frame=normal["frame"], foot=side,
                                                 reason="insufficient_common_stance_support"))
                continue
            _, index, mirror, overlap = min(candidates, key=lambda item: (item[0], item[1]))
            used.add(index)
            matched.append(dict(frame=max(normal["frame"], mirror["frame"]), foot=side,
                stance_frames=overlap, normal=normal, mirrored_inverse_mapped=mirror,
                distinct_decoded_stance_observations=len(overlap),
                common_stance_seconds=float((overlap[-1] - overlap[0]) / fps),
                evidence_type="paired_image_plane_swing_to_stance_candidate"))
    matched.sort(key=lambda event: (event["frame"], event["foot"]))
    base.update(landings=matched, candidate_count=len(matched), observed_proposal_count=len(matched),
                status="paired_visible_transition_proposals" if matched else "no_paired_visible_transition_proposal")
    return base
