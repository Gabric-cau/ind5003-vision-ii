"""Source-only kinematic evidence around a stationary-ball reference spot.

This is a conservative diagnostic. It never calls nearest-foot distance a
striking-foot label and never uses the departing ball position as a distance
target. The last actual stationary source detection is a fixed counterfactual
spot; departure observations only delimit the follow-through interval.
"""
from dataclasses import asdict

import numpy as np

from src.kicking_foot_evidence import EvidenceParameters, _temporal_channel_check, _validate_samples


V3_POLICY = dict(min_stationary_samples=3, confidence_min=.4,
                 minimum_approach_reduction_body=.03, maximum_final_distance_body=.35,
                 minimum_approach_support=.67, movement_tolerance_body=.015,
                 minimum_pass_through_body=.10, support_stability_body=.12,
                 pass_through_ratio=1.25, minimum_post_samples=2,
                 maximum_pair_gap_seconds=.12)


def _project(point, matrix):
    if matrix is None:
        return np.asarray(point, float)
    matrix = np.asarray(matrix, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Camera transform must be a finite 3x3 matrix')
    value = matrix @ np.r_[point, 1.]
    if abs(value[2]) <= 1e-9:
        return None
    return value[:2]/value[2]


def _rows_by_frame(rows, required, matrices):
    lookup = {row['frame']: row for row in rows}
    output = []
    for frame in required:
        row = lookup.get(frame)
        if row is None:
            return None
        confidence = np.asarray(row['confidence'], float)
        points = np.asarray(row['points'], float)
        if (points.shape != (17, 2) or confidence.shape != (17,)
                or not np.isfinite(points[[11, 12, 13, 14, 15, 16]]).all()
                or np.min(confidence[[11, 12, 13, 14, 15, 16]]) < V3_POLICY['confidence_min']):
            return None
        matrix = matrices.get(frame) if matrices is not None else None
        projected = np.array([_project(points[index], matrix) for index in range(17)], dtype=object)
        if any(point is None for point in projected[[11, 12, 13, 14, 15, 16]]):
            return None
        projected = np.asarray(projected.tolist(), float)
        output.append(dict(frame=frame, points=projected, confidence=confidence,
                           height=float(row['height'])))
    return output


def _series(rows, target, scale, first_stationary, post_rows):
    distances = np.asarray([np.linalg.norm(row['points'][index]-target)/scale
                            for row in rows for index in (15, 16)], float).reshape(-1, 2)
    approach = distances[:len(first_stationary)]
    post_distances = distances[len(first_stationary):]
    reductions = approach[:-1]-approach[1:]
    approaches = [dict(frame=first_stationary[i]['frame'], left_distance_body=float(approach[i, 0]),
                       right_distance_body=float(approach[i, 1])) for i in range(len(approach))]
    candidates = {}
    for side, name in enumerate(('left', 'right')):
        alt = 1-side
        reduction = float(approach[0, side]-approach[-1, side])
        support = float(np.mean(reductions[:, side] >= -V3_POLICY['movement_tolerance_body'])) if len(reductions) else 0
        final_distance = float(approach[-1, side])
        pass_through = None
        support_motion = None
        if len(post_distances):
            pass_through = float(np.max(np.linalg.norm(
                np.asarray([row['points'][15+side] for row in post_rows])
                - rows[len(first_stationary)-1]['points'][15+side], axis=1))/scale)
            support_motion = float(np.max(np.linalg.norm(
                np.asarray([row['points'][15+alt] for row in post_rows])
                - rows[len(first_stationary)-1]['points'][15+alt], axis=1))/scale)
        ratio = pass_through/max(support_motion, 1e-9) if pass_through is not None else None
        candidates[name] = dict(approach_reduction_body=float(reduction),
            approach_support_fraction=support, final_stationary_distance_body=final_distance,
            pass_through_body=pass_through, alternate_support_motion_body=support_motion,
            pass_through_support_motion_ratio=ratio,
            approach_observations=approaches)
    return candidates


def _view(rows, stationary, departure, fps, *, matrices=None, parameters=None):
    parameters = parameters or EvidenceParameters()
    stationary_frames = [int(row['frame']) for row in stationary]
    first_frame, departure_frame = int(departure[0]), int(departure[1])
    if len(stationary_frames) < V3_POLICY['min_stationary_samples']:
        return dict(candidate='unknown', status='fewer_than_three_stationary_source_observations',
                    stationary_frames=stationary_frames)
    if stationary_frames != sorted(set(stationary_frames)) or stationary_frames[-1] != first_frame:
        return dict(candidate='unknown', status='stationary_observations_do_not_end_at_last_stationary_frame',
                    stationary_frames=stationary_frames)
    post_frames = sorted({row['frame'] for row in rows
                          if departure_frame < row['frame'] <= departure_frame + round(.20*fps)})
    if len(post_frames) < V3_POLICY['minimum_post_samples']:
        return dict(candidate='unknown', status='fewer_than_two_post_departure_pose_observations',
                    stationary_frames=stationary_frames, post_frames=post_frames)
    stationary_rows = _rows_by_frame(rows, stationary_frames, matrices)
    post_rows = _rows_by_frame(rows, post_frames, matrices)
    if stationary_rows is None or post_rows is None:
        return dict(candidate='unknown', status='missing_confident_stationary_or_followthrough_leg_chain',
                    stationary_frames=stationary_frames, post_frames=post_frames)
    target = _project(np.asarray(stationary[-1]['xy'], float),
                      matrices.get(stationary_frames[-1]) if matrices is not None else None)
    if target is None:
        return dict(candidate='unknown', status='stationary_reference_projection_invalid')
    rows_all = stationary_rows + [row for row in post_rows if row['frame'] > stationary_frames[-1]]
    candidates = _series(rows_all, target, float(np.median([row['height'] for row in stationary_rows])),
                         stationary_rows, post_rows)
    supported = []
    for side, value in candidates.items():
        if (value['approach_reduction_body'] >= V3_POLICY['minimum_approach_reduction_body']
                and value['approach_support_fraction'] >= V3_POLICY['minimum_approach_support']
                and value['final_stationary_distance_body'] <= V3_POLICY['maximum_final_distance_body']
                and value['pass_through_body'] is not None
                and value['pass_through_body'] >= V3_POLICY['minimum_pass_through_body']
                and value['alternate_support_motion_body'] is not None
                and value['alternate_support_motion_body'] <= V3_POLICY['support_stability_body']
                and value['pass_through_support_motion_ratio'] >= V3_POLICY['pass_through_ratio']):
            supported.append(side)
    if len(supported) != 1:
        status = 'no_unique_stationary_spot_approach_pass_through_pattern' if not supported else 'both_sides_fit_stationary_spot_pattern'
        return dict(candidate='unknown', status=status, stationary_frames=stationary_frames,
                    post_frames=post_frames, stationary_target_xy=target.tolist(), side_metrics=candidates)
    return dict(candidate=supported[0], status='stationary_spot_approach_and_pass_through_candidate',
                stationary_frames=stationary_frames, post_frames=post_frames,
                stationary_target_xy=target.tolist(), side_metrics=candidates,
                label_scope='source-coordinate kinematic candidate; no shoe contact or anatomical-side verification')


def kicking_foot_evidence_v3(normal_samples, mirrored_samples, contact_bracket, fps,
                              *, stationary_ball_observations, departure_ball_observations,
                              camera_transforms=None, parameters=None):
    parameters = parameters or EvidenceParameters()
    first, departure = contact_bracket
    stationary_window = sorted([row for row in stationary_ball_observations
                                if isinstance(row.get('frame'), int)
                                and first-round(.15*fps) <= row['frame'] <= first],
                               key=lambda row: row['frame'])
    if len(stationary_window) < V3_POLICY['min_stationary_samples']:
        return dict(schema_version='fresh-foot-kinematic-v3', candidate='unknown',
                    status='fewer_than_three_stationary_source_observations',
                    new_neural_inference=False, human_verified=False, accuracy_validated=False)
    if len(departure_ball_observations) < 2:
        return dict(schema_version='fresh-foot-kinematic-v3', candidate='unknown',
                    status='missing_two_source_departure_observations', new_neural_inference=False,
                    human_verified=False, accuracy_validated=False)
    def _coerce(rows):
        return [dict(row, keypoints=row.get('keypoints', row.get('points')),
                     keypoint_confidences=row.get('keypoint_confidences', row.get('confidence')))
                for row in rows]
    normal, mirror = _validate_samples(_coerce(normal_samples)), _validate_samples(_coerce(mirrored_samples))
    low, high = max(0, first-round(.15*fps)), departure+round(.20*fps)
    checks = {name: _temporal_channel_check(rows, low, high, fps, parameters)
              for name, rows in (('normal', normal), ('mirrored_inverse_mapped', mirror))}
    base = dict(schema_version='fresh-foot-kinematic-v3', candidate='unknown', status='unknown',
                contact_bracket_frames=list(contact_bracket), source_fps=float(fps),
                stationary_ball_observations=stationary_ball_observations,
                stationary_window_observations=stationary_window,
                causal_departure_observations=departure_ball_observations,
                stationary_target_definition='last actual stationary source detection, fixed counterfactual spot',
                departure_used_as_distance_target=False, nearest_side_used_as_label=False,
                temporal_channel_checks=checks, new_neural_inference=False,
                human_verified=False, accuracy_validated=False, ground_calibrated=False)
    if any(check['ambiguous_pairs'] for check in checks.values()):
        base['status'] = 'temporal_anatomical_channel_ambiguity'
        return base
    bracket = [int(first), int(departure)]
    views = {name: _view(rows, stationary_window, bracket,
                         fps, matrices=camera_transforms, parameters=parameters)
             for name, rows in (('normal', normal), ('mirrored_inverse_mapped', mirror))}
    base['views'] = views
    labels = [view['candidate'] for view in views.values()]
    if labels[0] != 'unknown' and labels == [labels[0], labels[0]]:
        base.update(candidate=labels[0], status='paired_stationary_spot_kinematic_candidate')
    else:
        base['status'] = 'normal_mirror_stationary_spot_disagreement' if labels[0] != labels[1] else 'no_paired_stationary_spot_candidate'
    return base
