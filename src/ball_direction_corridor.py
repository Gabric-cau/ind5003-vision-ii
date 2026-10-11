"""A causal, finite-width image corridor, separate from ball/goal evidence."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class DirectionCorridorConfig:
    minimum_half_width_px: float = 18.0
    radius_multiplier: float = 1.25
    measurement_margin_px: float = 5.0
    gap_growth_px_per_frame: float = 1.0
    maximum_half_width_px: float = 32.0
    backward_slack_px: float = 3.0
    large_turn_degrees: float = 55.0
    unsupported_current_pixel_penalty: float = 0.15
    require_current_pixels_for_motion_bootstrap: bool = False


def established_radius(path):
    # The proposed destination must not enlarge its own acceptance corridor.
    # Use the smaller box side so a long motion blur does not widen the lane.
    sizes = [min(float(q.get('width', 14)), float(q.get('height', 14))) / 2
             for q in path[-5:]]
    return float(np.clip(np.median(sizes), 3., 18.))


def corridor_geometry(origin, velocity, dt, radius, speed_cap, *, config=None):
    cfg = config or DirectionCorridorConfig()
    if dt <= 0 or cfg.minimum_half_width_px <= 0 or cfg.maximum_half_width_px < cfg.minimum_half_width_px:
        raise ValueError('Positive interval and valid corridor widths required')
    o, v = np.asarray(origin, float), np.asarray(velocity, float)
    speed = float(np.linalg.norm(v))
    if o.shape != (2,) or v.shape != (2,) or not np.isfinite([*o, *v]).all() or speed < 1e-8:
        raise ValueError('Finite origin and nonzero velocity required')
    axis = v / speed
    normal = np.array([-axis[1], axis[0]])
    width = min(cfg.maximum_half_width_px,
                max(cfg.minimum_half_width_px, cfg.measurement_margin_px + cfg.radius_multiplier * radius)
                + cfg.gap_growth_px_per_frame * (dt - 1))
    forward = float(speed_cap * dt + cfg.measurement_margin_px)
    a = o - cfg.backward_slack_px * axis
    b = o + forward * axis
    polygon = [a + width * normal, b + width * normal,
               b - width * normal, a - width * normal]
    return {'origin_stabilized_xy': o.tolist(), 'axis': axis.tolist(),
            'half_width_px': float(width), 'backward_slack_px': cfg.backward_slack_px,
            'forward_limit_px': forward, 'source_frame_gap': dt,
            'established_radius_px': radius, 'polygon_stabilized_xy': [p.tolist() for p in polygon],
            'prediction_only': True, 'is_observation': False}


def assess_candidate(candidate_xy, geometry, *, config=None):
    cfg = config or DirectionCorridorConfig()
    delta = np.asarray(candidate_xy, float) - np.asarray(geometry['origin_stabilized_xy'])
    axis = np.asarray(geometry['axis'])
    along = float(np.dot(delta, axis))
    lateral = abs(float(delta[0] * axis[1] - delta[1] * axis[0]))
    length = float(np.linalg.norm(delta))
    angle = float(np.degrees(np.arccos(np.clip(along / length, -1, 1)))) if length > 1e-8 else 0.
    inside = (-geometry['backward_slack_px'] <= along <= geometry['forward_limit_px']
              and lateral <= geometry['half_width_px'])
    return {'inside_forward_corridor': bool(inside), 'along_px': along,
            'lateral_px': lateral, 'heading_change_degrees': angle,
            'large_turn_hypothesis': bool(not inside and angle >= cfg.large_turn_degrees),
            'destination_size_used_for_width': False}


def corridor_forecasts(path, model_states, start_frame, end_frame_exclusive, *,
                       config=None, max_gap_frames=6, reachable_speed_ratio=1.5):
    """Pre-association geometry; never use the current/future detection to draw it."""
    observed = {q['frame']: q for q in path}
    states = {q['frame']: q for q in model_states}
    history, model, result = [], None, []
    for f in range(start_frame, end_frame_exclusive):
        if model is not None and history:
            gap = f - model['frame']
            if 0 < gap <= max_gap_frames:
                velocity = model['velocity_px_per_source_frame']
                if np.linalg.norm(velocity) > 1e-8:
                    g = corridor_geometry(model['stabilized_xy'], velocity, gap,
                         established_radius(history), model.get('reachable_speed_cap_px_per_source_frame',
                             max(30., reachable_speed_ratio * np.linalg.norm(velocity))),
                         config=config)
                    result.append({'frame': f, **g, 'last_observation_frame': model['frame']})
        if f in observed:
            history.append(observed[f])
        if f in states:
            model = states[f]
    return result
