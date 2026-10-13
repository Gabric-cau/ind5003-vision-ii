"""Paired anatomical motion confined to the ball-departure bracket.

The later support-leg recoil is deliberately excluded. A nearby, approaching
ankle is a relevance gate, not a nearest-foot label or observed shoe contact.
"""
from dataclasses import asdict, dataclass

import numpy as np

from src.kicking_foot_evidence import (
    EvidenceParameters, _temporal_channel_check, _validate_samples,
)
from src.kicking_foot_evidence_v2 import _motion


@dataclass(frozen=True)
class StrikeWindowParameters:
    precontact_seconds: float = .13
    maximum_bracket_seconds: float = .15
    maximum_ankle_distance_body: float = .35
    minimum_approach_body: float = .03
    minimum_stationary_samples: int = 2


def _observations(rows, name):
    lookup = {}
    for row in rows:
        frame, xy = row.get('frame'), np.asarray(row.get('xy'), float)
        if (not isinstance(frame, int) or isinstance(frame, bool) or frame < 0
                or xy.shape != (2,) or not np.isfinite(xy).all()
                or not isinstance(row.get('detection_id'), str)):
            raise ValueError(f'{name} requires actual integer source frames, xy and detection IDs')
        if frame in lookup:
            raise ValueError(f'{name} must contain one selected ball per source frame')
        lookup[frame] = row
    return lookup


def _target_relevance(rows, target, first, departure, scale, policy):
    early = [row for row in rows if row['frame'] < first - 1]
    local = [row for row in rows if first - 1 <= row['frame'] <= departure]
    output = {}
    for index, side in enumerate(('left', 'right')):
        before = [np.linalg.norm(row['points'][15 + index] - target) / scale for row in early]
        near = [np.linalg.norm(row['points'][15 + index] - target) / scale for row in local]
        minimum = float(min(near)) if near else None
        reduction = float(np.median(before) - minimum) if before and near else None
        output[side] = dict(minimum_bracket_ankle_distance_body=minimum,
            preparation_to_bracket_reduction_body=reduction,
            source_early_frames=[row['frame'] for row in early],
            source_bracket_frames=[row['frame'] for row in local],
            relevant=bool(minimum is not None and reduction is not None
                          and minimum <= policy.maximum_ankle_distance_body
                          and reduction >= policy.minimum_approach_body))
    return output


def kicking_foot_strike_window(normal_samples, mirrored_samples, contact_bracket, fps, *,
                              stationary_ball_observations, departure_ball_observations,
                              parameters=None, policy=None):
    parameters = parameters or EvidenceParameters()
    policy = policy or StrikeWindowParameters()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Source FPS must be positive and finite')
    if (len(contact_bracket) != 2 or any(not isinstance(value, int) or isinstance(value, bool)
                                       for value in contact_bracket)
            or not 0 <= contact_bracket[0] <= contact_bracket[1]):
        raise ValueError('Contact bracket must contain ordered nonnegative source frames')
    first, departure = contact_bracket
    low = max(0, first - round(policy.precontact_seconds * fps))
    out = dict(schema_version='fresh-foot-strike-window-v1', candidate='unknown',
        status='insufficient_paired_strike_evidence', contact_bracket_frames=[first, departure],
        analysis_window_frames=[low, departure], source_fps=float(fps),
        parameters=asdict(parameters), policy=asdict(policy), new_neural_inference=False,
        human_verified=False, accuracy_validated=False, held_out_test=False,
        reference_labels_used_for_prediction=False, shoe_ball_contact_verified=False,
        nearest_ankle_used_as_label=False, post_departure_pose_used=False,
        departure_coordinates_used_as_distance_target=False,
        anatomical_channel_permutation_applied=False,
        limitations=['paired pose agreement is not anatomical ground truth',
                     'ankle proximity does not establish shoe-ball contact',
                     'source camera motion and departure identity remain unverified'])
    stationary = _observations(stationary_ball_observations, 'Stationary evidence')
    departing = _observations(departure_ball_observations, 'Departure evidence')
    if (len(stationary) < policy.minimum_stationary_samples or not stationary
            or max(stationary) != first or len(departing) < 2 or min(departing) != departure
            or any(frame > first for frame in stationary)
            or departure <= first or (departure - first) / fps > policy.maximum_bracket_seconds):
        out['status'] = 'actual_stationary_departure_bracket_not_supported'
        return out
    normal, mirror = _validate_samples(normal_samples), _validate_samples(mirrored_samples)
    prep = [row['height'] for row in normal + mirror
            if first - round(.30 * fps) <= row['frame'] < first]
    if len(prep) < 6:
        out['status'] = 'insufficient_observed_preparation_scale'
        return out
    scale = float(np.median(prep))
    out['preparation_scale_source_px'] = scale
    checks = {name: _temporal_channel_check(rows, low, departure, fps, parameters)
              for name, rows in (('normal', normal), ('mirrored_inverse_mapped', mirror))}
    out['strike_window_channel_checks'] = checks
    if any(check['ambiguous_pairs'] for check in checks.values()):
        out['status'] = 'temporal_anatomical_channel_ambiguity'
        return out
    paired_lookup = {row['frame']: row for row in mirror}
    kept_normal, kept_mirror, fused, accepted, rejected = [], [], [], [], []
    indices = [11, 12, 13, 14, 15, 16]
    for row in normal:
        if not low <= row['frame'] <= departure:
            continue
        paired = paired_lookup.get(row['frame'])
        reason = None
        if paired is None:
            reason = 'missing_mirrored_source_frame'
        elif any(np.min(view['confidence'][indices]) < parameters.confidence_min
                 or not np.isfinite(view['points'][indices]).all() for view in (row, paired)):
            reason = 'low_confidence_or_missing_bilateral_leg_chain'
        else:
            errors = np.linalg.norm(row['points'][indices] - paired['points'][indices], axis=1) / scale
            if errors.max() > parameters.mirror_relative_error_max:
                reason = 'source_coordinate_leg_chain_disagrees_between_views'
        if reason:
            rejected.append(dict(frame=row['frame'], reason=reason))
            continue
        kept_normal.append(row)
        kept_mirror.append(paired)
        fused.append(dict(frame=row['frame'], points=(row['points'] + paired['points']) / 2,
                          confidence=np.minimum(row['confidence'], paired['confidence']), height=scale))
        accepted.append(dict(frame=row['frame'], maximum_leg_joint_error_body=float(errors.max())))
    out.update(accepted_paired_frames=accepted, rejected_paired_frames=rejected)
    target = np.asarray(stationary[first]['xy'], float)
    out['stationary_distance_reference'] = dict(frame=first, xy=target.tolist(),
                                               detection_id=stationary[first]['detection_id'])
    views = {}
    for name, rows in (('normal', kept_normal), ('mirrored_inverse_mapped', kept_mirror), ('paired', fused)):
        views[name] = dict(motion=_motion(rows, low, departure, fps, scale, parameters),
                          target_relevance=_target_relevance(rows, target, first, departure, scale, policy))
    out['views'] = views
    labels = [view['motion']['candidate'] for view in views.values()]
    if 'unknown' in labels:
        out['status'] = 'strike_motion_not_supported_in_all_paired_views'
    elif len(set(labels)) != 1:
        out['status'] = 'normal_mirror_strike_motion_disagreement'
    elif not all(view['target_relevance'][labels[0]]['relevant'] for view in views.values()):
        out['status'] = 'motion_side_not_approaching_stationary_ball_region'
    else:
        out.update(candidate=labels[0], status='paired_predeparture_strike_motion_candidate')
    return out
