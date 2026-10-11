"""Conservative end-state evidence on one source-bound ball identity and shot."""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EndpointPolicy:
    stationary_seconds: float = .50
    minimum_stationary_observations: int = 8
    maximum_gap_seconds: float = .12
    maximum_stationary_speed_px_per_frame: float = .75
    stationary_radius_diameters: float = .75
    minimum_flight_distance_px: float = 35.
    minimum_flight_speed_px_per_frame: float = 4.
    edge_margin_diameters: float = 2.
    competing_center_diameters: float = 1.5


def classify_endpoint(path, pool, transforms, kick, fps, end_frame_exclusive, width, height,
                      *, boundary_reason, identity_supported, ambiguous_seed_frames=(),
                      people=None, duplicate_frames=(), policy=None):
    """A settled ball is a candidate state, never verified final match position."""
    cfg = policy or EndpointPolicy()
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError('Endpoint analysis requires a positive finite fps')
    if min(width, height, end_frame_exclusive) <= 0:
        raise ValueError('Endpoint source dimensions and interval must be positive')
    out = {'status': 'unknown', 'reason': 'insufficient_same_identity_observations',
           'final_position': None, 'final_position_verified': False,
           'stationary_end_candidate': None, 'human_verified': False,
           'accuracy_validated': False, 'physical_ball_rest_established': False,
           'boundary_reason': boundary_reason, 'thresholds': asdict(cfg),
           'coordinate_system': 'source_pixels_with_background_stabilized_image_evidence'}
    if not identity_supported or ambiguous_seed_frames:
        out['reason'] = 'ball_identity_not_supported_or_seed_ambiguous'
        return out
    if not path:
        return out
    ids = {(int(row['frame']), row['detection_id']): row for rows in pool.values() for row in rows}
    frames = [int(row['frame']) for row in path]
    if any(b <= a for a, b in zip(frames, frames[1:])) or any(f >= end_frame_exclusive for f in frames):
        raise ValueError('Endpoint path crosses its ordered source shot interval')
    if any((int(row['frame']), row['detection_id']) not in ids for row in path):
        raise ValueError('Endpoint path contains an unmeasured observation')
    if any(not np.allclose([row['x'], row['y']], [ids[(int(row['frame']), row['detection_id'])]['x'],
                                                        ids[(int(row['frame']), row['detection_id'])]['y']]) for row in path):
        raise ValueError('Endpoint path changed measured source coordinates')
    first = min(kick, frames[0])
    def usable_transform(frame):
        raw = transforms.get(str(frame), transforms.get(frame))
        matrix = np.asarray(raw, float) if raw is not None else None
        return (matrix is not None and matrix.shape == (3, 3) and np.isfinite(matrix).all()
                and abs(np.linalg.det(matrix)) > 1e-10)
    lost = next((f for f in range(first, frames[-1]+1) if not usable_transform(f)), None)
    if lost is not None:
        out.update(status='camera_reference_lost', reason='incomplete_contact_to_observation_camera_prefix',
                   camera_prefix_stop_frame=lost)
        return out
    duplicates = set(duplicate_frames)
    observed = [row for row in path if row['frame'] not in duplicates]
    out.update(last_observed_frame=frames[-1], last_observed_position=[path[-1]['x'], path[-1]['y']],
               observed_ball_identity_candidate=True, observed_point_count=len(path),
               independent_motion_observation_count=len(observed))
    max_gap = max(1, round(cfg.maximum_gap_seconds*fps))
    if any(np.asarray(row.get('stabilized_xy'), float).shape != (2,) or
           not np.isfinite(np.asarray(row.get('stabilized_xy'), float)).all() for row in observed):
        out.update(status='camera_reference_lost', reason='unsupported_stabilized_ball_observation')
        return out
    diameter = float(np.median([(row.get('width', 5)+row.get('height', 5))/2 for row in observed])) if observed else 5.
    for index, first_row in enumerate(observed):
        tail = observed[index:]
        if first_row['frame'] <= kick or len(tail) < cfg.minimum_stationary_observations:
            continue
        if (tail[-1]['frame']-tail[0]['frame'])/fps < cfg.stationary_seconds:
            continue
        if any(b['frame']-a['frame'] > max_gap for a, b in zip(tail, tail[1:])):
            continue
        if any('motion_blob' in row.get('sources', []) or row.get('confidence', 0) < .04 for row in tail):
            continue
        xy = np.asarray([row['stabilized_xy'] for row in tail], float)
        center = np.median(xy, axis=0)
        radius = max(3., cfg.stationary_radius_diameters*diameter)
        speeds = [np.linalg.norm(b-a)/(qb['frame']-qa['frame'])
                  for a, b, qa, qb in zip(xy, xy[1:], tail, tail[1:])]
        if np.max(np.linalg.norm(xy-center, axis=1)) > radius or max(speeds, default=math.inf) > cfg.maximum_stationary_speed_px_per_frame:
            continue
        before = observed[:index+1]
        fast = any(np.linalg.norm(np.asarray(b['stabilized_xy'])-a['stabilized_xy'])/(b['frame']-a['frame'])
                   >= cfg.minimum_flight_speed_px_per_frame for a, b in zip(before, before[1:]))
        if not fast or np.linalg.norm(np.asarray(before[0]['stabilized_xy'])-center) < max(cfg.minimum_flight_distance_px, 5*diameter):
            continue
        competitors = []
        for row in tail:
            for candidate in pool.get(str(row['frame']), pool.get(row['frame'], [])):
                if candidate['detection_id'] == row['detection_id'] or candidate.get('stabilized_xy') is None:
                    continue
                separation = np.linalg.norm(np.asarray(candidate['stabilized_xy'])-row['stabilized_xy'])
                if candidate.get('confidence', 0) >= .04 and separation < cfg.competing_center_diameters*diameter:
                    competitors.append({'frame': row['frame'], 'detection_id': candidate['detection_id']})
        if competitors:
            out.update(status='unknown', reason='competing_ball_identity_at_stationary_end', competitors=competitors)
            return out
        representative = min(tail, key=lambda row: np.linalg.norm(np.asarray(row['stabilized_xy'])-center))
        if end_frame_exclusive-1-tail[-1]['frame'] > max_gap:
            out.update(reason='observed_settling_followed_by_unobserved_interval',
                       observed_stationary_state={'supporting_frames': [row['frame'] for row in tail],
                           'duration_sec': (tail[-1]['frame']-tail[0]['frame'])/fps},
                       following_unobserved_frames=end_frame_exclusive-1-tail[-1]['frame'])
            return out
        out.update(status='stationary_end_candidate', reason='same_identity_flight_followed_by_sustained_observed_settling',
            stationary_end_candidate={'frame': representative['frame'], 'detection_id': representative['detection_id'],
                'source_xy': [representative['x'], representative['y']],
                'source_position_is_actual_observation': True,
                'stable_center_xy': center.tolist(), 'supporting_frames': [row['frame'] for row in tail],
                'supporting_detection_ids': [row['detection_id'] for row in tail],
                'duration_sec': (tail[-1]['frame']-tail[0]['frame'])/fps,
                'event_end_verified': False, 'physical_rest_verified': False})
        return out
    last = path[-1]
    gap = end_frame_exclusive-1-last['frame']
    if gap <= max_gap:
        status = {'scene_cut': 'camera_cut_censored', 'replay_cut_candidate': 'replay_cut_candidate',
                  'duration_cap': 'analysis_window_censored', 'video_end': 'source_clip_censored'}.get(boundary_reason, 'source_boundary_censored')
        out.update(status=status, reason='ball_end_state_not_observed_before_source_boundary')
        return out
    margin = max(8., cfg.edge_margin_diameters*diameter)
    if min(last['x'], width-last['x'], last['y'], height-last['y']) <= margin and len(path) >= 2:
        previous = path[-2]
        delta = np.asarray([last['x']-previous['x'], last['y']-previous['y']])
        outward = ((last['x'] < margin and delta[0] < 0) or (last['x'] > width-margin and delta[0] > 0)
                   or (last['y'] < margin and delta[1] < 0) or (last['y'] > height-margin and delta[1] > 0))
        if outward:
            out.update(status='out_of_view_candidate', reason='last_same_identity_observations_reach_frame_edge_outwards')
            return out
    for row in (people or {}).get(str(last['frame']), (people or {}).get(last['frame'], [])):
        x0, y0, x1, y1 = row['bbox']
        if x0-margin <= last['x'] <= x1+margin and y0-margin <= last['y'] <= y1+margin:
            out.update(status='occluded_candidate', reason='ball_observations_end_near_observed_person_box',
                       occluder_bbox=row['bbox'], occlusion_verified=False)
            return out
    out.update(reason='interior_detection_loss_cannot_distinguish_occlusion_from_tracking_failure')
    return out
