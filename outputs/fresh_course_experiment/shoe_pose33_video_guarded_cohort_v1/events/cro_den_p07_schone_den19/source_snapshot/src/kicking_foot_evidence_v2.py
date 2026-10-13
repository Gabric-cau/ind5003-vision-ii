"""Paired pose motion with a fixed scale and stationary-only ball diagnostics."""
from dataclasses import asdict

import numpy as np

from src.kicking_foot_evidence import (EvidenceParameters, _validate_samples,
                                      _temporal_channel_check)


def _motion(rows, first, last, fps, scale, parameters):
    selected = [row for row in rows if first <= row['frame'] <= last]
    pairs = []
    for a, b in zip(selected, selected[1:]):
        elapsed = (b['frame']-a['frame'])/fps
        if elapsed > parameters.maximum_pair_gap_seconds:
            continue
        hips_a, hips_b = a['points'][[11, 12]].mean(0), b['points'][[11, 12]].mean(0)
        ankle = np.linalg.norm((b['points'][[15, 16]]-hips_b)
                              -(a['points'][[15, 16]]-hips_a), axis=1)/scale/elapsed
        knee = np.linalg.norm((b['points'][[13, 14]]-hips_b)
                             -(a['points'][[13, 14]]-hips_a), axis=1)/scale/elapsed
        pairs.append(dict(frames=[a['frame'], b['frame']], ankle_speed=ankle.tolist(),
                          knee_speed=knee.tolist()))
    out = dict(candidate='unknown', reason='insufficient_connected_paired_leg_chain',
               window_frames=[first, last], valid_pairs=len(pairs), pairs=pairs)
    if len(pairs) < parameters.minimum_pairs:
        return out
    medians = np.median([row['ankle_speed'] for row in pairs], axis=0)
    side = int(np.argmax(medians))
    ratio = float(medians[side]/max(medians[1-side], 1e-9))
    votes = float(np.mean([row['ankle_speed'][side] >= parameters.motion_ratio_min
                           * row['ankle_speed'][1-side] for row in pairs]))
    out.update(median_ankle_speeds=medians.tolist(), median_motion_ratio=ratio,
               median_direction='left' if side == 0 else 'right', pair_support_fraction=votes,
               reason='weak_or_inconsistent_relative_ankle_motion')
    if (ratio >= parameters.motion_ratio_min and votes >= parameters.pair_support_fraction_min
            and medians[side] >= parameters.minimum_motion_body_heights_per_second):
        out.update(candidate=out['median_direction'], reason='sustained_paired_relative_ankle_motion')
    return out


def _stationary_distances(rows, balls, first, scale):
    lookup = {}
    for observation in balls:
        frame, xy = observation.get('frame'), np.asarray(observation.get('xy'), float)
        if (isinstance(frame, bool) or not isinstance(frame, int) or frame < 0
                or xy.shape != (2,) or not np.isfinite(xy).all()
                or not isinstance(observation.get('detection_id'), str)):
            raise ValueError('Stationary evidence needs actual source frame, detection ID and xy')
        if frame in lookup:
            raise ValueError('Stationary evidence must select one actual ball per frame')
        lookup[frame] = observation
    matched = []
    for row in rows:
        if not first-5 <= row['frame'] <= first or row['frame'] not in lookup:
            continue
        ball = lookup[row['frame']]
        distances = np.linalg.norm(row['points'][[15, 16]]-ball['xy'], axis=1)/scale
        matched.append(dict(frame=row['frame'], ball_detection_id=ball['detection_id'],
            ball_source_xy=ball['xy'], left_distance_body= float(distances[0]),
            right_distance_body=float(distances[1])))
    return dict(observations=matched, matched_stationary_frames=len(matched),
                label_chosen_from_distance=False, departure_observations_used_for_distance=False,
                interpretation='nearest stationary-ball ankle can be the support foot')


def kicking_foot_evidence_v2(normal_samples, mirrored_samples, contact_bracket, fps, *,
                              stationary_ball_observations, departure_ball_observations,
                              parameters=None):
    parameters = parameters or EvidenceParameters()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Source FPS must be positive and finite')
    if (len(contact_bracket) != 2 or any(isinstance(value, bool) or not isinstance(value, int)
                                        for value in contact_bracket)
            or not 0 <= contact_bracket[0] <= contact_bracket[1]):
        raise ValueError('Contact bracket must contain ordered nonnegative source frames')
    first, departure = contact_bracket
    normal, mirror = _validate_samples(normal_samples), _validate_samples(mirrored_samples)
    out = dict(schema_version='fresh-foot-paired-motion-v2', candidate='unknown',
        status='insufficient_paired_leg_evidence', contact_bracket_frames=list(contact_bracket),
        source_fps=float(fps), parameters=asdict(parameters), new_neural_inference=False,
        human_verified=False, accuracy_validated=False, shoe_ball_contact_verified=False,
        reference_labels_used_for_prediction=False, anatomical_channel_permutation_applied=False,
        distance_used_to_choose_anatomical_label=False, stationary_ball_distance={},
        limitations=['paired pose agreement is not anatomical ground truth',
                     'relative ankle motion is not directly observed shoe-ball contact',
                     'source-clock motion can include slow motion'])
    stationary_frames = [row.get('frame') for row in stationary_ball_observations]
    departure_frames = [row.get('frame') for row in departure_ball_observations]
    for observation in departure_ball_observations:
        xy = np.asarray(observation.get('xy'), float)
        if (xy.shape != (2,) or not np.isfinite(xy).all()
                or not isinstance(observation.get('detection_id'), str)):
            raise ValueError('Departure gate needs actual source detection ID and xy')
    out['causal_departure_observations'] = departure_ball_observations
    if (not stationary_frames or any(not isinstance(frame, int) or isinstance(frame, bool)
                                    for frame in stationary_frames+departure_frames)
            or max(stationary_frames) != first or len(set(departure_frames)) < 2
            or min(departure_frames) != departure):
        out['status'] = 'actual_stationary_departure_bracket_not_supported'
        return out
    low, high = max(0, first-round(parameters.precontact_seconds*fps)), departure+round(parameters.postdeparture_seconds*fps)
    prep = [row['height'] for row in normal+mirror
            if first-round(.30*fps) <= row['frame'] < first]
    if len(prep) < 6:
        out['status'] = 'insufficient_observed_preparation_scale'
        return out
    scale = float(np.median(prep))
    out.update(preparation_scale_source_px=scale,
               preparation_scale_definition='median observed normal/mirrored preparation box height, held constant across the motion windows')
    other = {row['frame']: row for row in mirror}
    kept_normal, kept_mirror, fused, accepted, rejected = [], [], [], [], []
    indices = [11, 12, 13, 14, 15, 16]
    for row in normal:
        if not low <= row['frame'] <= high:
            continue
        paired = other.get(row['frame'])
        reason = None
        if paired is None:
            reason = 'missing_mirrored_source_frame'
        elif any(np.min(view['confidence'][indices]) < parameters.confidence_min
                 or not np.isfinite(view['points'][indices]).all() for view in (row, paired)):
            reason = 'low_confidence_or_missing_bilateral_leg_chain'
        else:
            errors = np.linalg.norm(row['points'][indices]-paired['points'][indices], axis=1)/scale
            if errors.max() > parameters.mirror_relative_error_max:
                reason = 'source_coordinate_leg_chain_disagrees_between_views'
        if reason:
            rejected.append(dict(frame=row['frame'], reason=reason))
            continue
        kept_normal.append(row)
        kept_mirror.append(paired)
        fused.append(dict(frame=row['frame'], points=(row['points']+paired['points'])/2,
                          confidence=np.minimum(row['confidence'], paired['confidence']), height=scale))
        accepted.append(dict(frame=row['frame'], maximum_leg_joint_error_body=float(errors.max()),
                             minimum_leg_joint_confidence=float(min(row['confidence'][indices].min(),
                                                                    paired['confidence'][indices].min()))))
    out.update(accepted_paired_frames=accepted, rejected_paired_frames=rejected)
    checks = {name: _temporal_channel_check(rows, max(0, first-2), departure+1, fps, parameters)
              for name, rows in (('normal', normal), ('mirrored_inverse_mapped', mirror))}
    out['contact_window_channel_checks'] = checks
    out['stationary_ball_distance'] = {name: _stationary_distances(rows, stationary_ball_observations, first, scale)
        for name, rows in (('normal', kept_normal), ('mirrored_inverse_mapped', kept_mirror), ('paired', fused))}
    if any(check['ambiguous_pairs'] for check in checks.values()):
        out['status'] = 'temporal_anatomical_channel_ambiguity'
        return out
    windows = dict(bracket_context=[low, high], departure_followthrough=[departure, high])
    views = {name: {window: _motion(rows, *bounds, fps, scale, parameters)
                   for window, bounds in windows.items()}
             for name, rows in (('normal', kept_normal), ('mirrored_inverse_mapped', kept_mirror), ('paired', fused))}
    out['views'] = views
    supported = []
    for window, fused_window in views['paired'].items():
        label = fused_window['candidate']
        if label == 'unknown':
            continue
        directions = [views[name][window].get('median_direction')
                      for name in ('normal', 'mirrored_inverse_mapped')]
        if directions != [label, label]:
            out['status'] = 'normal_mirror_median_direction_disagreement'
            return out
        supported.append(label)
    out['supported_window_labels'] = supported
    if len(set(supported)) > 1:
        out['status'] = 'conflicting_supported_paired_windows'
    elif supported:
        out.update(candidate=supported[0], status='paired_constant_scale_temporal_motion_candidate')
    return out
