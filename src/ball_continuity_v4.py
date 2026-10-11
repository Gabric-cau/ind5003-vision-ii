"""Experimental v4 association: full motion residual plus local direction continuity.

Only actual detections are returned as observations. Missing-frame predictions
are separate, bounded search proposals, never evidence of ball contact/crossing.
All dynamics use original source-frame intervals (including repeated frames).
"""
from dataclasses import asdict, dataclass
import numpy as np

from src.event_ball_tracking import point, select_stationary_seed
from src.goal_event_tracker_v2 import usable
from src.ball_direction_corridor import (assess_candidate, corridor_geometry, established_radius)


@dataclass(frozen=True)
class ContinuityConfig:
    max_gap_frames: int = 6
    beam_width: int = 60
    measurement_slack_px: float = 10.0
    residual_speed_fraction: float = .12
    gap_uncertainty_px: float = .8
    reachable_speed_ratio: float = 1.5
    turn_support_points: int = 3
    max_turns: int = 2
    miss_cost: float = .35
    require_collision_for_turn: bool = True
    lock_flight_direction: bool = True
    direction_tolerance_degrees: float = 22.0
    collision_margin_px: float = 22.0
    association_lag_frames: int = 2
    prediction_residual_weight: float = 0.4
    departure_min_displacement_px: float = 60.0
    departure_max_displacement_px: float = 300.0
    departure_goal_distance_fraction: float = .60
    departure_contact_uncertainty_frames: int = 8
    departure_max_gap_frames: int = 8
    departure_min_speed_px_per_frame: float = 15.0
    small_motion_component_penalty: float = .45
    stationary_release_confidence: float = .30
    stationary_release_before_contact_frames: int = 2


def _segment_distance(a, b, c, d):
    """Minimum distance between two 2-D segments, including intersection."""
    a, b, c, d = (np.asarray(p, float) for p in (a, b, c, d))
    u, v = b - a, d - c
    matrix = np.column_stack((u, -v))
    if abs(np.linalg.det(matrix)) > 1e-9:
        t, s = np.linalg.solve(matrix, c - a)
        if 0 <= t <= 1 and 0 <= s <= 1:
            return 0.0
    def distance_to_segment(p, x, y):
        direction = y - x
        t = np.clip(np.dot(p - x, direction) / max(1e-9, np.dot(direction, direction)), 0, 1)
        return float(np.linalg.norm(p - x - t * direction))
    return min(distance_to_segment(a, c, d), distance_to_segment(b, c, d),
               distance_to_segment(c, a, b), distance_to_segment(d, a, b))


def _inside_convex(p, polygon):
    p, polygon = np.asarray(p, float), np.asarray(polygon, float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = p - polygon
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(cross >= -1e-9) or np.all(cross <= 1e-9))


def collision_proximity(a, b, regions, margin):
    """Check the established ball's position, not a divergent destination.

    The candidate must not manufacture its own collision permission by landing
    on a player. Proximity still does not establish physical contact.
    """
    matches = []
    for f in range(a["frame"], b["frame"] + 1):
        for region in (regions or {}).get(str(f), (regions or {}).get(f, [])):
            if region.get("strength", "strong") != "strong":
                continue
            if region.get("segment"):
                separation = _segment_distance(point(a), point(a), *region["segment"])
            elif region.get("polygon"):
                polygon = region["polygon"]
                separation = (0. if _inside_convex(point(a), polygon)
                              else min(_segment_distance(point(a), point(a), x, y)
                                       for x, y in zip(polygon, polygon[1:] + polygon[:1])))
            else:
                continue
            if separation <= margin:
                matches.append({"kind": region["kind"], "geometry_frame": f,
                                "distance_px": round(separation, 3),
                                "source": region.get("source", "unverified_geometry"),
                                "permission_basis": "previous_ball_observation_only",
                                "physical_contact_established": False})
    return min(matches, key=lambda q: q["distance_px"]) if matches else None


def _reward(d, seed_height=None, cfg=None, corridor_cfg=None):
    # A motion blob is an observation candidate, not detector confidence.
    if "motion_blob" in d.get("sources", []):
        size_penalty = 0.
        if seed_height and cfg:
            size_penalty = cfg.small_motion_component_penalty * max(0., 1. -
                d.get('height', seed_height) / max(1., .50 * seed_height))
        pixel_penalty = (corridor_cfg.unsupported_current_pixel_penalty
                         if corridor_cfg is not None and d.get('current_pixel_mask_support') is False else 0.)
        return .65 - size_penalty - pixel_penalty
    return .65 + min(.25, .5 * d.get("confidence", 0.0))


def _tolerance(speed, dt, cfg):
    return (cfg.measurement_slack_px + cfg.residual_speed_fraction * speed * dt
            + cfg.gap_uncertainty_px * dt * dt)


def bootstrap_departure(seed, observations, goal, kick, cfg):
    """Infer release from two real moving observations and the stationary spot.

    Foot occlusion can delay the first detected flight point. Late boot/spot
    observations must not move the inferred release time forward. The first
    moving observation still needs a nearby earlier source observation and a
    backward line intersecting the measured spot; a keeper jump is insufficient.
    """
    origin = np.median([point(p) for p in seed], axis=0)
    direction = goal - origin
    reach = np.linalg.norm(direction)
    strong_stationary = [p for p in seed if p['frame'] <= kick - cfg.stationary_release_before_contact_frames and
                         p.get('confidence', 0.) >= cfg.stationary_release_confidence and
                         .5 <= p.get('width', 12) / max(1., p.get('height', 12)) <= 2.]
    # A moving background object must not imply that the ball left while two
    # credible observations still show it at the stationary spot afterwards.
    earliest_release = (max(p['frame'] for p in strong_stationary) - 1
                        if len(strong_stationary) >= 2 else seed[0]['frame'] - 1)
    cosine = lambda a, b: float(np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)))
    starts = []
    for i, a in enumerate(observations):
        if not seed[0]['frame'] < a['frame'] <= kick + 12:
            continue
        offset = point(a) - origin
        length = np.linalg.norm(offset)
        if not cfg.departure_min_displacement_px <= length <= min(cfg.departure_max_displacement_px,
                cfg.departure_goal_distance_fraction * reach) or cosine(offset, direction) < .7:
            continue
        for b in observations[i + 1:]:
            dt = b['frame'] - a['frame']
            if not 0 < dt <= 4:
                continue
            velocity = (point(b) - point(a)) / dt
            speed = np.linalg.norm(velocity)
            if not cfg.departure_min_speed_px_per_frame <= speed <= 160 or cosine(velocity, direction) < .7:
                continue
            back_time = np.dot(offset, velocity) / (speed * speed)
            miss = np.linalg.norm(offset - back_time * velocity)
            onset = a['frame'] - back_time
            if onset < earliest_release:
                continue
            if miss > 22 or back_time < .2 or not kick - cfg.departure_contact_uncertainty_frames <= onset <= kick + 12:
                continue
            if cosine(velocity, offset) < .92:
                continue
            prefix = [p for p in seed if p['frame'] <= onset]
            if not prefix or not 0 < a['frame'] - prefix[-1]['frame'] <= cfg.departure_max_gap_frames:
                continue
            starts.append({'path': [prefix[-1], a, b], 'velocity': velocity, 'score': 0.,
                           'turns': [], 'onset_motion_time': float(onset),
                           'departure_backward_miss_px': float(miss)})
    return starts


def track_ball_continuity(detections, kick, anchors, goal_center, *,
                          frame_metadata=None, config=None, initial_path=None,
                          collision_regions=None, corridor_config=None):
    """Continue existing observations with bounded, local motion association.

    Camera-transform loss breaks coordinate comparability; it ends the search.
    Person/kit flags do not exclude a ball overlapping a player at impact.
    An initial path is immutable: later candidate density cannot replace it.
    New associations become immutable after a short source-frame decision lag.
    """
    cfg = config or ContinuityConfig()
    if cfg.max_gap_frames < 1 or cfg.turn_support_points < 2 or cfg.association_lag_frames < 0:
        raise ValueError("Positive gap and at least two turn support points required")
    base = {"tracker_version": "local_motion_continuity_v4_residual", "config": asdict(cfg),
            "outcome": "unknown", "path_is_verified": False,
            "whole_ball_crossing_established": False, "human_verified": False,
            "semantic_masks_used": False, "time_basis": "original_source_frames",
            "collision_policy": "prior_ball_proximity_required" if cfg.require_collision_for_turn else "motion_only"}
    if corridor_config is not None:
        base.update(tracker_version='local_motion_continuity_v4_narrow_corridor',
                    direction_corridor_config=asdict(corridor_config),
                    corridor_policy='finite_forward_lane_or_supported_collision_turn')
    # Use frame numbers for dynamics rather than silently accelerating over
    # near-duplicate encoded frames. The original candidates remain unmodified.
    obs = sorted([{**d, "motion_time": float(d["frame"])}
                  for ds in detections.values() for d in ds if usable(d)],
                 key=lambda d: (d["frame"], d["detection_id"]))
    if initial_path is not None:
        if len(initial_path) < 3:
            raise ValueError("A fixed incoming prefix requires at least three observations")
        ids = {(d["frame"], d["detection_id"]) for ds in detections.values() for d in ds}
        if any((d["frame"], d["detection_id"]) not in ids for d in initial_path):
            raise ValueError("Incoming prefix must consist of input observations")
        if any(b["frame"] <= a["frame"] for a, b in zip(initial_path, initial_path[1:])):
            raise ValueError("Incoming prefix must have increasing source-frame numbers")
        seed = [{**d, "motion_time": float(d["frame"])} for d in initial_path]
        a, b = seed[-3], seed[-1]
        velocity = (point(b) - point(a)) / (b["frame"] - a["frame"])
        starts = [{"path": seed, "velocity": velocity, "turns": [], "score": 0.}]
        base["prefix_policy"] = "fixed_previous_incoming_candidate_not_ground_truth"
        base["fixed_prefix_observations"] = len(seed)
    else:
        seed, seed_info = select_stationary_seed(detections, kick, anchors)
        base["seed"] = seed_info
        if not seed:
            return [], {**base, "tracking_status": "abstain_no_seed"}
        seed = [{**d, "motion_time": float(d["frame"])} for d in seed]
        starts = bootstrap_departure(seed, obs, np.asarray(goal_center, float), kick, cfg)
        base["prefix_policy"] = "automatic_departure_bootstrap"
        if corridor_config is not None and corridor_config.require_current_pixels_for_motion_bootstrap:
            before_count = len(starts)
            # An absent current-frame object cannot establish the first flight
            # line. Once motion is established, low mask support is only a
            # soft penalty; neural detections are never masked out.
            starts = [s for s in starts if all(not (
                'motion_blob' in q.get('sources', []) and q.get('current_pixel_mask_support') is False)
                for q in s['path'][1:])]
            base['departure_hypotheses_rejected_without_current_pixels'] = before_count - len(starts)
    if not starts:
        return seed, {**base, "tracking_status": "abstain_no_departure"}
    seed_height = float(np.median([d.get('height', 12) for d in seed]))
    by_frame = {}
    for d in obs:
        by_frame.setdefault(d["frame"], []).append(d)
    frames = sorted(set(int(f) for f in detections) |
                    set(int(f) for f in (frame_metadata or {})))
    pending = {}
    for s in starts:
        direction = s["velocity"] / max(1e-9, np.linalg.norm(s["velocity"]))
        prefix_states = []
        for i, b in enumerate(s["path"][1:], 1):
            a = s["path"][max(0, i - 2)]
            prefix_states.append({"frame": b["frame"], "stabilized_xy": point(b).tolist(),
                "velocity_px_per_source_frame": ((point(b) - point(a)) /
                    (b["frame"] - a["frame"])).tolist(), "turn_pending": False})
        s.update(turn_pending=0, speed_cap=max(30., cfg.reachable_speed_ratio *
                                             float(np.linalg.norm(s["velocity"]))),
                 score=sum(_reward(d, seed_height, cfg, corridor_config) for d in s["path"][1:]) -
                       .25 * s.get('departure_backward_miss_px', 0.) / 22., direction=direction,
                 segment_origin=point(s["path"][-1]), segment_frame=s["path"][-1]["frame"],
                 collision_cues=[], model_states=prefix_states)
        if corridor_config is not None:
            for state in prefix_states:
                state['reachable_speed_cap_px_per_source_frame'] = s['speed_cap']
        pending.setdefault(s["path"][-1]["frame"], []).append(s)
    states, finished = [], []
    committed = []
    committed_cutoff = None
    rejected_jumps = 0
    rejected_unsupported_turns = 0
    rejected_corridor = 0
    camera_break = None
    for f in frames:
        def preserves_committed(state):
            return committed_cutoff is None or [
                (d['frame'], d['detection_id']) for d in state['path']
                if d['frame'] <= committed_cutoff] == committed
        # A bootstrap scheduled for a later frame is another birth hypothesis,
        # not permission to reset observations already selected at earlier times.
        expanded = [s for s in pending.get(f, []) if preserves_committed(s)]
        meta = (frame_metadata or {}).get(str(f), (frame_metadata or {}).get(f, {}))
        if states and meta.get("coordinate_supported") is False:
            finished.extend(states)
            states = []
            camera_break = f
            break
        for s in states:
            last = s["path"][-1]
            dt = f - last["frame"]
            if dt <= 0:
                expanded.append(s)
                continue
            if dt > cfg.max_gap_frames:
                finished.append(s)
                continue
            expanded.append({**s, "score": s["score"] - cfg.miss_cost})
            velocity = s["velocity"]
            speed = float(np.linalg.norm(velocity))
            tol = _tolerance(speed, dt, cfg)
            continuations, turn_candidates = [], []
            geometry = (corridor_geometry(point(last), velocity, dt, established_radius(s['path']),
                                         s['speed_cap'], config=corridor_config)
                        if corridor_config is not None and speed > 1e-8 else None)
            for d in by_frame.get(f, []):
                delta = point(d) - point(last)
                # Even a rebound must be reachable from the last observation.
                if np.linalg.norm(delta) > s["speed_cap"] * dt + cfg.measurement_slack_px:
                    rejected_jumps += 1
                    continue
                instantaneous = delta / dt
                residual = float(np.linalg.norm(delta - velocity * dt))
                lane = assess_candidate(point(d), geometry, config=corridor_config) if geometry else None
                if lane and not lane['inside_forward_corridor'] and not lane['large_turn_hypothesis']:
                    rejected_corridor += 1
                    continue
                turns = list(s["turns"])
                support = s["turn_pending"]
                direction = s["direction"]
                cross_track = abs(float(delta[0] * direction[1] - delta[1] * direction[0]))
                corridor = (cfg.measurement_slack_px +
                            np.sin(np.deg2rad(cfg.direction_tolerance_degrees)) * speed * dt +
                            cfg.gap_uncertainty_px * dt * dt)
                movement_length = float(np.linalg.norm(delta))
                cosine = float(np.dot(instantaneous, direction) / max(1e-9, np.linalg.norm(instantaneous)))
                # Compare with recent motion. Gravity and perspective can curve
                # a continuous image trajectory without a sudden collision.
                consistent_direction = (movement_length <= cfg.measurement_slack_px or
                    cosine >= np.cos(np.deg2rad(cfg.direction_tolerance_degrees)))
                # Apparent speed errors do not imply impact. Compare the new
                # heading with recent motion rather than the first flight line.
                requires_turn = ((not consistent_direction or cross_track > corridor)
                                 if cfg.lock_flight_direction else residual > tol)
                if lane and lane['large_turn_hypothesis']:
                    requires_turn = True
                cues = list(s["collision_cues"])
                segment_origin, segment_frame = s["segment_origin"], s["segment_frame"]
                if not requires_turn:
                    # Encoded repeats and perspective alter apparent speed.
                    # Do not prefer a missing observation just because a real
                    # ball travelled less far than the previous speed forecast.
                    lateral_cost = .6 * (cross_track / corridor) ** 2
                    speed_cost = .15 * min(1., abs(np.linalg.norm(instantaneous) - speed) /
                                           max(1., speed))
                    prediction_cost = cfg.prediction_residual_weight * min(3., (residual / max(1., tol)) ** 2)
                    cost = lateral_cost + speed_cost + prediction_cost
                    if support:
                        if dt > 2:
                            continue
                        support += 1
                        if support >= cfg.turn_support_points:
                            support = 0
                    new_velocity = .25 * velocity + .75 * instantaneous
                    direction = new_velocity / max(1e-9, np.linalg.norm(new_velocity))
                    segment_origin, segment_frame = point(d), f
                else:
                    # Keep a new direction as a hypothesis until several
                    # observations support it; never bridge a long hidden turn.
                    if support or dt > 2 or len(turns) >= cfg.max_turns:
                        continue
                    if not 4 <= np.linalg.norm(instantaneous) <= s["speed_cap"]:
                        continue
                    cue = collision_proximity(last, d, collision_regions, cfg.collision_margin_px)
                    if cfg.require_collision_for_turn and cue is None:
                        rejected_unsupported_turns += 1
                        continue
                    turns.append(f)
                    support = 1
                    cost = 3.0
                    new_velocity = instantaneous
                    direction = instantaneous / max(1e-9, np.linalg.norm(instantaneous))
                    segment_origin, segment_frame = point(last), last["frame"]
                    if cue:
                        cues.append({**cue, "turn_start_frame": f,
                                     "observation_interval_frames": [last["frame"], f]})
                candidate = {**s, "path": s["path"] + [d],
                                 "velocity": new_velocity, "turns": turns,
                                 "turn_pending": support,
                                 "direction": direction, "segment_origin": segment_origin,
                                 "segment_frame": segment_frame, "collision_cues": cues,
                                 "model_states": s["model_states"] + [{"frame": f,
                                     "stabilized_xy": point(d).tolist(),
                                     "velocity_px_per_source_frame": new_velocity.tolist(),
                                     "turn_pending": bool(support),
                                     **({'association_corridor': geometry, 'corridor_candidate': lane}
                                        if lane is not None else {})}],
                                 "score": s["score"] + _reward(d, seed_height, cfg, corridor_config) - cost
                                          - cfg.miss_cost * (dt - 1)}
                if corridor_config is not None:
                    candidate['model_states'][-1]['reachable_speed_cap_px_per_source_frame'] = s['speed_cap']
                (turn_candidates if requires_turn else continuations).append(candidate)
            # Proximity alone cannot override a still-supported flight. A long
            # stream of player/clothing detections must not retroactively erase
            # the continuing ball just because it yields more later rewards.
            expanded.extend(continuations if continuations else turn_candidates)
        expanded.sort(key=lambda s: s["score"], reverse=True)
        states, seen = [], set()
        for s in expanded:
            sig = (s["path"][-1]["detection_id"],
                   tuple(np.rint(s["velocity"] / 6).astype(int)),
                   tuple(s["turns"]), s["turn_pending"],
                   tuple(np.rint(s["direction"] * 20).astype(int)), s["segment_frame"])
            if sig in seen:
                continue
            seen.add(sig)
            states.append(s)
            if len(states) >= cfg.beam_width:
                break
        # A two-point release bootstrap is provisional. Apply the same minimum
        # support as final acceptance before freezing it; otherwise a short
        # background path can block the real ball's slightly later departure.
        lockable = [s for s in states if len(s['path']) >= 5]
        if lockable:
            leader = lockable[0]
            cutoff = f - cfg.association_lag_frames
            if leader["turn_pending"]:
                cutoff = min(cutoff, leader["turns"][-1] - 1)
            if committed_cutoff is not None:
                cutoff = max(cutoff, committed_cutoff)
            committed = [(d["frame"], d["detection_id"]) for d in leader["path"]
                         if d["frame"] <= cutoff]
            def agrees(s):
                return [(d["frame"], d["detection_id"]) for d in s["path"]
                        if d["frame"] <= cutoff] == committed
            states = [s for s in states if agrees(s)]
            finished = [s for s in finished if agrees(s)]
            committed_cutoff = cutoff
    finished.extend(states)
    # Drop tentative unsupported turns, not merely their labels.
    viable = [s for s in finished if len(s["path"]) >= 5 and not s["turn_pending"]]
    if not viable:
        return seed, {**base, "tracking_status": "abstain_short_flight"}
    best = max(viable, key=lambda s: s["score"])
    path = best["path"]
    gaps = [{"from_frame": a["frame"], "to_frame": b["frame"],
             "missing_frames": b["frame"] - a["frame"] - 1}
            for a, b in zip(path, path[1:]) if b["frame"] - a["frame"] > 1]
    return path, {**base, "tracking_status": "candidate", "point_count": len(path),
                  "beam_score": round(best["score"], 3), "turn_frames": best["turns"],
                  "max_gap_frames": max(b["frame"] - a["frame"] for a, b in zip(path, path[1:])),
                  "observed_gaps": gaps, "rejected_unreachable_associations": rejected_jumps,
                  "rejected_turns_without_collision_cue": rejected_unsupported_turns,
                  **({'rejected_outside_forward_corridor': rejected_corridor}
                     if corridor_config is not None else {}),
                  "collision_turn_candidates": best["collision_cues"],
                  "committed_observation_ids": [i for _, i in committed],
                  "association_policy": "preserve_prefix_then_fixed_lag_continuation",
                  "motion_states": best["model_states"], "coordinate_break_frame": camera_break}


def search_predictions(path, start_frame, end_frame_exclusive, *, config=None, model_states=None):
    """Causal search centres for missing frames; never interpolation/observations."""
    return [p for p in motion_forecasts(path, start_frame, end_frame_exclusive,
                                       config=config, model_states=model_states)
            if not p["observation_available"]]


def motion_forecasts(path, start_frame, end_frame_exclusive, *, config=None, model_states=None):
    """Forecast before each frame's association, including frames with observations.

    The state must precede the target frame. Future observations or future
    collisions never alter forecasts already made.
    """
    cfg = config or ContinuityConfig()
    observed = {d["frame"]: d for d in path}
    history, predictions = [], []
    states = {s["frame"]: s for s in (model_states or [])}
    model = None
    for f in range(start_frame, end_frame_exclusive):
        origin = velocity = None
        if model is not None:
            origin = np.array(model["stabilized_xy"], float)
            velocity = np.array(model["velocity_px_per_source_frame"], float)
            dt = f - model["frame"]
        elif model_states is None and len(history) >= 2:
            a, b = history[-2:]
            origin = point(b)
            velocity = (point(b) - point(a)) / (b["frame"] - a["frame"])
            dt = f - b["frame"]
        if origin is not None and 0 < dt <= cfg.max_gap_frames:
            xy = origin + velocity * dt
            predictions.append({"frame": f, "predicted_stabilized_xy": xy.tolist(),
                                "velocity_px_per_source_frame": velocity.tolist(),
                                "search_radius_px": _tolerance(float(np.linalg.norm(velocity)), dt, cfg),
                                "observation_available": f in observed,
                                "source": "motion_prediction_only", "is_observation": False})
        if f in observed:
            history.append(observed[f])
        if f in states:
            model = states[f]
    return predictions
