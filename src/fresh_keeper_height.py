"""Source-bound torso clearance candidates, never metric goalkeeper height.

The reference is an observed planted-ankle line parallel to the visible goal
baseline. Perspective and forward/backward motion remain explicit assumptions.
Names, outcomes, legacy image-motion heights and human labels are not inputs.
"""
import math

import cv2
import numpy as np

from src.goal_plane_rectification import project_image_point
from src.goalkeeper_dive_phase import sustained_start
from src.goalkeeper_mask_experiment import pose_seed
from src.goalkeeper_role_association import select_goalkeeper_role
from src.goalkeeper_sam2 import mask_qc, points_inside


POLICY = dict(min_pose_confidence=.4, pre_reference_seconds=.5,
              min_reference_samples=3, max_pose_gap_seconds=.15,
              max_reference_ground_scatter_body=.08,
              max_reference_scale_scatter_fraction=.25,
              max_ground_line_angle_scatter_radians=.04,
              onset_shift_body=.15, onset_lean_degrees=30,
              extension_lean_degrees=45, landing_hip_clearance_body=.30,
              landing_bottom_clearance_body=.05, max_ground_depth_change_body=.20,
              landing_min_torso_lean_degrees=60, landing_max_body_extent_body=.65,
              landing_sustained_samples=5, landing_max_clearance_scatter_body=.08,
              low_torso_clearance_below_body=.35, high_torso_clearance_above_body=.75,
              category_boundary_margin_body=.04, min_connected_mask_component_fraction=.85)


def _points(row, indices):
    points = np.asarray(row.get('keypoints', []), float)
    confidence = np.asarray(row.get('keypoint_confidences', []), float)
    if (points.shape != (17, 2) or confidence.shape != (17,)
            or not np.isfinite(points[indices]).all()
            or not np.isfinite(confidence[indices]).all()
            or np.any(confidence[indices] < POLICY['min_pose_confidence'])):
        return None
    return points, confidence


def connected_camera_prefix(camera, start, end):
    """Coordinates after a missing/rejected camera link never reconnect."""
    if camera.get('shot_interval') != [start, end]:
        raise ValueError('Height needs camera transforms for the full selected shot')
    pairs = {row['frame']: row for row in camera.get('pairs', [])}
    transforms, stopped = {}, None
    for frame in range(start, end):
        raw = camera.get('transforms', {}).get(str(frame))
        matrix = np.asarray(raw, float) if raw is not None else None
        pair = pairs.get(frame)
        good = (matrix is not None and matrix.shape == (3, 3)
                and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-10
                and not (pair and (pair.get('accepted') is False
                                   or pair.get('reference_chain_valid') is False)))
        if stopped is None and not good:
            stopped = frame
        transforms[str(frame)] = matrix.tolist() if stopped is None else None
    return transforms, stopped if stopped is not None else end


def ground_reference(observations, goals, transforms, kick, fps):
    rows = [row for row in observations if kick-round(.5*fps) <= row['frame'] < kick]
    directions, uprights, scales, feet = [], [], [], []
    for frame, goal in goals.items():
        if not kick-round(.5*fps) <= int(frame) < kick or not goal.get('selected'):
            continue
        corners = np.asarray(goal['selected'].get('corners'), float)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            continue
        projected = [project_image_point(transforms.get(str(frame)), point) for point in corners]
        if any(point is None for point in projected):
            continue
        line = np.asarray(projected[2])-projected[3]
        if np.linalg.norm(line) > 20:
            if line[0] < 0:
                line = -line
            directions.append(line/np.linalg.norm(line))
            upright = (np.asarray(projected[0])-projected[3]
                       +np.asarray(projected[1])-projected[2])/2
            if np.linalg.norm(upright) > 10 and upright[1] < 0:
                uprights.append(upright/np.linalg.norm(upright))
    if len(directions) < 2:
        return None, 'missing_stable_visible_goal_groundline'
    angles = np.unwrap([math.atan2(line[1], line[0]) for line in directions])
    if np.std(angles) > POLICY['max_ground_line_angle_scatter_radians']:
        return None, 'inconsistent_stabilized_goal_groundline'
    direction = np.array([math.cos(np.median(angles)), math.sin(np.median(angles))])
    down = np.array([-direction[1], direction[0]])
    if len(uprights) < 2:
        return None, 'missing_projected_vertical_goal_posts'
    upright = np.median(uprights, axis=0)
    upright /= np.linalg.norm(upright)
    for row in rows:
        observed = _points(row, [5, 6, 11, 12, 15, 16])
        if observed is None:
            continue
        points, _ = observed
        matrix = transforms.get(str(row['frame']))
        ankles = [project_image_point(matrix, point) for point in points[[15, 16]]]
        torso_points = [project_image_point(matrix, point) for point in points[[5, 6, 11, 12]]]
        box = np.asarray(row.get('bbox'), float)
        if (box.shape != (4,) or not np.isfinite(box).all()
                or any(p is None for p in ankles+torso_points)):
            continue
        torso = np.mean(torso_points[:2], axis=0)-np.mean(torso_points[2:], axis=0)
        if np.linalg.norm(torso) < 3:
            continue
        lean = float(np.degrees(np.arccos(np.clip(abs(torso @ upright)/np.linalg.norm(torso), 0, 1))))
        # Crouching/diving changes box height without changing body size. Use
        # only the observed upright preparation, before this deformation.
        if lean >= POLICY['onset_lean_degrees']:
            continue
        center_x = float((box[0]+box[2])/2)
        corners = [project_image_point(matrix, [center_x, y]) for y in (box[1], box[3])]
        if any(point is None for point in corners):
            continue
        scale = float(np.ptp(np.asarray(corners) @ down))
        if scale <= 20:
            continue
        scales.append(scale)
        foot = max(ankles, key=lambda point: np.dot(point, down))
        feet.append((row['frame'], np.asarray(foot)))
    if len(feet) < POLICY['min_reference_samples']:
        return None, 'missing_confident_pre_kick_planted_ankles_and_torso'
    scale = float(np.median(scales))
    if np.ptp(scales)/scale > POLICY['max_reference_scale_scatter_fraction']:
        return None, 'unstable_pre_kick_body_scale'
    offsets = np.array([point @ down for _, point in feet])
    if np.ptp(offsets)/scale > POLICY['max_reference_ground_scatter_body']:
        return None, 'pre_kick_ankles_not_a_stable_ground_reference'
    return dict(ground_direction=direction.tolist(), ground_down_normal=down.tolist(),
                projected_upright_direction=upright.tolist(),
                ground_offset_px=float(np.median(offsets)), body_scale_px=scale,
                reference_frames=[frame for frame, _ in feet],
                ground_scatter_body=float(np.ptp(offsets)/scale),
                definition='observed pre-contact ankle line parallel to visible goal baseline',
                body_scale_reference='confident upright preparation only, not crouched or sideways bbox height',
                coordinate_system='camera_stabilized_selected_shot_image_plane',
                physical_ground_plane_calibrated=False,
                assumption='keeper stays approximately at the reference depth'), None


def pose_only_measurement(row, matrix, reference, mask=None):
    observed = _points(row, [5, 6, 11, 12])
    if observed is None:
        return None
    points, confidence = observed
    if mask is not None and not all(points_inside(mask, points)[[5, 6, 11, 12]]):
        return None
    stable = [project_image_point(matrix, point) for point in points]
    if any(stable[j] is None for j in [5, 6, 11, 12]):
        return None
    shoulder, hip = np.mean([stable[5], stable[6]], 0), np.mean([stable[11], stable[12]], 0)
    torso = shoulder-hip
    scale, down = reference['body_scale_px'], np.asarray(reference['ground_down_normal'])
    if np.linalg.norm(torso) < max(3, .08*scale):
        return None
    height = lambda point: float((reference['ground_offset_px']-np.dot(point, down))/scale)
    center = (shoulder+hip)/2
    ankles = [height(stable[j]) for j in (15, 16)
              if np.isfinite(confidence[j]) and confidence[j] >= .4 and stable[j] is not None
              and (mask is None or points_inside(mask, points[[j]])[0])]
    box = np.asarray(row.get('bbox'), float)
    body_extent = None
    if box.shape == (4,) and np.isfinite(box).all():
        x = float((box[0]+box[2])/2)
        ends = [project_image_point(matrix, [x, y]) for y in (box[1], box[3])]
        if all(point is not None for point in ends):
            body_extent = float(abs((np.asarray(ends[1])-ends[0]) @ down)/scale)
    return dict(frame=int(row['frame']), hip=hip.tolist(), shoulder=shoulder.tolist(),
                torso_center=center.tolist(), hip_clearance_body=height(hip),
                torso_clearance_body=height(center),
                torso_lean_degrees=float(np.degrees(np.arccos(np.clip(
                    abs(np.dot(torso, reference['projected_upright_direction']))/np.linalg.norm(torso), 0, 1)))),
                visible_body_extent_body=body_extent,
                bottom_clearance_body=min(ankles) if ankles else None,
                landing_bottom_source='observed_confident_ankle' if ankles else None,
                min_confidence=float(confidence[[5, 6, 11, 12]].min()),
                association_source_track_id=row.get('association_source_track_id'),
                locked_body_mask_available=mask is not None)


def _category(value):
    if not -.05 <= value <= 1.6:
        return 'unknown', 'implausible_projected_torso_clearance'
    if any(abs(value-boundary) < POLICY['category_boundary_margin_body'] for boundary in [.35, .75]):
        return 'unknown', 'torso_clearance_near_category_boundary'
    return ('low' if value < .35 else 'high' if value > .75 else 'mid'), None


def analyse_height_phases(samples, kick, fps, reference):
    """A connected onset, extension and grounded landing must precede a peak."""
    out = dict(height_candidate='unknown', supported=False, reasons=[], phases=[],
               complete_first_dive_observed=False, samples=samples,
               physical_height_metres=None, thresholds=POLICY.copy(),
               height_definition='maximum three-sample median torso-center ground clearance / pre-kick body height',
               category_thresholds_are_engineering_choices=True,
               grounded_ankles_are_pose_candidates_not_contact_truth=True)
    def reject(reason):
        out['reasons'].append(reason)
        return out
    samples = sorted(samples, key=lambda sample: sample['frame'])
    near = [s for s in samples if kick-round(.5*fps) <= s['frame'] < kick]
    pre = [s for s in near if s['frame'] < kick-round(.2*fps)] or near
    if len(pre) < 3 or not near or kick-near[-1]['frame'] > round(.15*fps):
        return reject('missing_near_contact_confident_torso_reference')
    ref = np.median([s['torso_center'] for s in pre[:5]], axis=0)
    direction, scale = np.asarray(reference['ground_direction']), reference['body_scale_px']
    event_start = max(pre[-1]['frame']+1, kick-round(.25*fps))
    connected, previous = [], event_start
    for sample in [s for s in samples if s['frame'] >= event_start]:
        if (sample['frame']-previous)/fps > POLICY['max_pose_gap_seconds']:
            out['pose_prefix_stop_frame'] = sample['frame']
            break
        connected.append(sample)
        previous = sample['frame']
    out['connected_pose_frames'] = [s['frame'] for s in connected]
    shift = lambda s: abs(float((np.asarray(s['torso_center'])-ref) @ direction))/scale
    onset = sustained_start(connected, lambda s: s['torso_lean_degrees'] >= 30 or shift(s) >= .15, fps)
    if onset is None:
        return reject('no_sustained_first_dive_onset')
    active = [s for s in connected if s['frame'] >= onset]
    landing = None
    for index in range(len(active)-POLICY['landing_sustained_samples']+1):
        run = active[index:index+POLICY['landing_sustained_samples']]
        grounded = all(s['frame'] >= onset+round(.1*fps)
            and s['torso_lean_degrees'] >= POLICY['landing_min_torso_lean_degrees']
            and s['hip_clearance_body'] <= POLICY['landing_hip_clearance_body']
            and s.get('bottom_clearance_body') is not None
            and -.20 <= s['bottom_clearance_body'] <= POLICY['landing_bottom_clearance_body']
            and s.get('visible_body_extent_body') is not None
            and s['visible_body_extent_body'] <= POLICY['landing_max_body_extent_body'] for s in run)
        connected_run = all((b['frame']-a['frame'])/fps <= .15 for a, b in zip(run, run[1:]))
        settled = (grounded and np.ptp([s['torso_clearance_body'] for s in run]) <= .08
                   and np.ptp([s['bottom_clearance_body'] for s in run]) <= .08)
        if grounded and connected_run and settled and (run[-1]['frame']-run[0]['frame'])/fps >= .10:
            landing = run[0]['frame']
            break
    out.update(onset_frame=onset, landing_frame=landing,
               phases=[dict(name='preparation', start_frame=pre[0]['frame'], end_frame=onset-1),
                       dict(name='first_extension', start_frame=onset,
                            end_frame=landing-1 if landing is not None else previous)])
    if landing is None:
        return reject('grounded_landing_not_observed_in_connected_first_dive')
    out['complete_first_dive_observed'] = True
    out['phases'].append(dict(name='grounded_landing_candidate', start_frame=landing, end_frame=previous))
    extension = [s for s in active if s['frame'] < landing and s['torso_lean_degrees'] >= 45]
    if len(extension) < 3 or (extension[-1]['frame']-extension[0]['frame'])/fps < .06:
        return reject('too_few_observed_prelanding_torso_samples')
    if max(shift(s) for s in extension) < .15:
        return reject('no_supported_lateral_dive_motion')
    bottom = [s['bottom_clearance_body'] for s in active if s.get('bottom_clearance_body') is not None]
    if bottom and min(bottom) < -POLICY['max_ground_depth_change_body']:
        return reject('depth_motion_or_ankle_error_invalidates_fixed_ground_reference')
    peaks = [(float(np.median([s['torso_clearance_body'] for s in extension[i:i+3]])),
              [s['frame'] for s in extension[i:i+3]]) for i in range(len(extension)-2)
             if (extension[i+2]['frame']-extension[i]['frame'])/fps <= .15]
    if not peaks:
        return reject('no_connected_three_sample_extension_peak')
    peak, frames = max(peaks, key=lambda value: value[0])
    category, reason = _category(peak)
    out.update(peak_torso_clearance_body=peak, supporting_frames=frames,
               extension_frames=[s['frame'] for s in extension], landing_frames_excluded_from_height=True)
    if reason:
        return reject(reason)
    out.update(height_candidate=category, supported=True)
    return out


def guard_observed_height_reference(signal):
    """Retain measured diagnostics while rejecting an invalid reference line."""
    out = dict(signal, reasons=list(signal.get('reasons', [])))
    if out.get('reference_depth_assumption_failed') is True:
        out.update(candidate='unknown', supported=False)
        reason = 'depth_motion_or_ankle_error_invalidates_fixed_ground_reference'
        if reason not in out['reasons']:
            out['reasons'].append(reason)
    return out


def observed_image_plane_signal(samples, phase, kick, fps, reference):
    """Summarize measured torso samples without asserting an entire dive peak."""
    out = dict(candidate='unknown', supported=False, reasons=[],
               observed_maximum_torso_clearance_body=None, supporting_frames=[],
               continuous_samples=[], complete_first_dive_observed=phase.get(
                   'complete_first_dive_observed', False),
               censored=not phase.get('complete_first_dive_observed', False),
               censoring_reasons=phase.get('reasons', []).copy(),
               physical_height_metres=None, physical_ground_plane_calibrated=False,
               human_verified=False, accuracy_validated=False,
               definition='maximum three-sample median observed torso midpoint clearance from the pre-contact image reference line, normalized by preparation body height',
               coordinate_system=reference['coordinate_system'],
               reference_line_offset_px=reference['ground_offset_px'],
               reference_scale_px=reference['body_scale_px'],
               clearance_axis_ground_down_normal=reference['ground_down_normal'],
               projected_vertical_direction=reference['projected_upright_direction'],
               category_is_observed_signal_not_complete_event_peak=True,
               category_thresholds_are_engineering_choices=True,
               lower_bound_scope='sampled image-plane signal in the connected observed first extension; not physical height',
               fixed_reference_depth_assumption=reference['assumption'])
    connected_frames = set(phase.get('connected_pose_frames', []))
    connected = [s for s in sorted(samples, key=lambda s: s['frame'])
                 if s['frame'] in connected_frames]
    out['continuous_samples'] = [dict(frame=s['frame'],
        torso_clearance_body=s['torso_clearance_body'],
        torso_lean_degrees=s['torso_lean_degrees'],
        min_torso_confidence=s['min_confidence'],
        bottom_clearance_body=s.get('bottom_clearance_body')) for s in connected]
    onset = phase.get('onset_frame')
    out.update(onset_frame=onset, landing_candidate_frame=phase.get('landing_frame'),
               pose_prefix_stop_frame=phase.get('pose_prefix_stop_frame'))
    if onset is None:
        out['reasons'].append('no_connected_observed_first_extension_onset')
        return out
    near = [s for s in samples if kick-round(.5*fps) <= s['frame'] < kick]
    pre = [s for s in near if s['frame'] < kick-round(.2*fps)] or near
    if len(pre) < 3:
        out['reasons'].append('missing_confident_preparation_torso_reference')
        return out
    preparation = np.median([s['torso_center'] for s in pre[:5]], axis=0)
    extension = [s for s in connected if s['frame'] >= onset
                 and s['torso_lean_degrees'] >= POLICY['extension_lean_degrees']
                 and (phase.get('landing_frame') is None or s['frame'] < phase['landing_frame'])]
    direction = np.asarray(reference['ground_direction'])
    shifts = [abs(float((np.asarray(s['torso_center'])-preparation) @ direction))
              / reference['body_scale_px'] for s in extension]
    if not shifts or max(shifts) < POLICY['onset_shift_body']:
        out['reasons'].append('no_supported_lateral_extension_motion')
        return out
    peaks = [(float(np.median([s['torso_clearance_body'] for s in run])),
              [s['frame'] for s in run])
             for i in range(len(extension)-2) if (run := extension[i:i+3])
             and (run[-1]['frame']-run[0]['frame'])/fps <= POLICY['max_pose_gap_seconds']]
    if not peaks:
        out['reasons'].append('fewer_than_three_connected_confident_extension_samples')
        return out
    value, frames = max(peaks, key=lambda item: item[0])
    category, reason = _category(value)
    bottom = [s['bottom_clearance_body'] for s in connected
              if s.get('bottom_clearance_body') is not None]
    out.update(observed_maximum_torso_clearance_body=value, supporting_frames=frames,
               reference_depth_assumption_failed=bool(bottom and min(bottom) <
                   -POLICY['max_ground_depth_change_body']),
               candidate=category, supported=category != 'unknown')
    if reason:
        out['reasons'].append(reason)
    return guard_observed_height_reference(out)


def _mask_guard(mask, row, *, first, previous, scale, initial_area, excluded, roi=None):
    mask = np.asarray(mask, bool)
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return dict(valid=False, reason='body_mask_truncated_at_source_edge')
    if roi is not None:
        x0, y0, x1, y1 = roi
        local = mask[y0:y1, x0:x1]
        if local[0].any() or local[-1].any() or local[:, 0].any() or local[:, -1].any():
            return dict(valid=False, reason='body_mask_truncated_at_inference_roi_edge')
    components, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if components <= 1 or stats[1:, cv2.CC_STAT_AREA].max()/max(1, mask.sum()) < .85:
        return dict(valid=False, reason='body_mask_disconnected_components')
    if first:
        observed = _points(row or {}, [5, 6, 11, 12])
        if observed is None or not all(points_inside(mask, observed[0])[[5, 6, 11, 12]]):
            return dict(valid=False, reason='first_body_mask_does_not_cover_confident_keeper_torso')
    return mask_qc(mask, previous, scale, initial_area, excluded)


def analyze_height(pose, cache, *, camera, association=None, masks=None, role=None, mask_roi_xyxy=None):
    """The runner binds hashes; this function enforces source-frame guards."""
    kick, fps = cache.get('kick_frame'), float(pose['fps'])
    start, end = pose['shot_start_frame'], pose['shot_end_frame']
    out = dict(dive_height='unknown', strict_height_candidate='unknown',
               ordinal_pose_height_candidate='unknown', landing_candidate_frame=None,
               observed_image_plane_height_candidate='unknown',
               physical_height_metres=None, physical_ground_plane_calibrated=False,
               human_verified=False, accuracy_validated=False, identity_verified=False,
               new_keeper_neural_inference=False, requires_review=True,
               height_definition='camera-stabilized torso midpoint clearance / pre-kick body height',
               coordinate_system='selected_shot_image_plane_not_3d', reasons=[])
    def finish(reason):
        out['reason'] = reason
        if reason not in ('pose_only_ordinal_candidate', 'source_bound_mask_pose_torso_height_candidate'):
            out['reasons'].append(reason)
        return out
    contact = cache.get('contact') or {}
    if (isinstance(kick, bool) or not isinstance(kick, int) or not start <= kick < end
            or contact.get('contact_frame') != kick or not contact.get('ball_departure_frames')
            or contact.get('camera_chain_measured') is not True):
        return finish('independent_causal_ball_contact_not_supported')
    selection = pose.get('shot_selection') or {}
    attempts = [row for row in selection.get('attempts', [])
                if row.get('shot_index') == selection.get('selected_shot_index')]
    if (selection.get('found_qualified_shot') is not True or len(attempts) != 1
            or attempts[0].get('qualifies') is not True or attempts[0].get('scene_rejections') != []
            or pose.get('selected_track_qualified') is not True):
        return finish('selected_shot_scene_not_qualified')
    transforms, stop = connected_camera_prefix(camera, start, end)
    out['camera_prefix_stop_frame'] = stop if stop < end else None
    if stop <= kick:
        return finish('camera_reference_lost_before_contact')
    geometries = {int(frame): goal for frame, goal in cache['goal_proposals'].items() if goal.get('selected')}
    role = role or select_goalkeeper_role(pose, geometries, kick)
    out['role'] = role
    if role.get('status') != 'candidate_supported' or role.get('seed_track_id') is None:
        return finish('fresh_pre_contact_goal_line_role_not_supported')
    if role['seed_track_id'] == contact.get('contact_associated_track_id'):
        return finish('goal_line_role_conflicts_with_independent_striker_actor')
    track = next((track for track in pose['tracks'] if track['track_id'] == role['seed_track_id']), None)
    if track is None:
        return finish('keeper_role_track_missing')
    rows = association.get('observations', []) if association else track['observations']
    rows = sorted([row for row in rows if start <= row['frame'] < min(stop, end)], key=lambda row: row['frame'])
    if len({row['frame'] for row in rows}) != len(rows):
        raise ValueError('Duplicate role-locked keeper observations at one source frame')
    out.update(keeper_track_id_candidate=role['seed_track_id'], association=association,
               observed_keeper_frame_count=len(rows))
    reference, reason = ground_reference(rows, cache['goal_proposals'], transforms, kick, fps)
    if reference is None:
        return finish(reason)
    samples = [sample for row in rows if (sample := pose_only_measurement(
        row, transforms[str(row['frame'])], reference)) is not None]
    phase = analyse_height_phases(samples, kick, fps, reference)
    signal = observed_image_plane_signal(samples, phase, kick, fps, reference)
    out.update(reference=reference, pose_only_phase=phase,
               observed_image_plane_signal=signal,
               observed_image_plane_height_candidate=signal['candidate'],
               ordinal_pose_height_candidate=phase['height_candidate'],
               landing_candidate_frame=phase.get('landing_frame'))
    if not masks:
        return finish('pose_only_ordinal_candidate' if phase['supported'] else 'pose_only_first_dive_support_inadequate')
    seed, by_frame = min(masks), {row['frame']: row for row in rows}
    if not kick-round(.5*fps) <= seed < kick:
        return finish('body_mask_seed_not_in_pre_contact_keeper_reference')
    previous, initial_area, qcs, locked = None, int(np.count_nonzero(masks[seed])), {}, []
    own_ids = set(role.get('same_person_seed_ids', [role['seed_track_id']]))
    for frame in range(seed, min(stop, end)):
        mask = masks.get(frame)
        if mask is None:
            out['mask_prefix_stop_reason'] = 'missing_body_mask_breaks_identity_prefix'
            break
        if mask.shape != (pose['frame_height'], pose['frame_width']):
            raise ValueError('Fresh height masks must use full source pixel coordinates')
        row = by_frame.get(frame)
        selected_id = row.get('association_source_track_id') if row else None
        excluded = np.zeros_like(mask, dtype=np.uint8)
        for other in pose['tracks']:
            if other['track_id'] in own_ids or other['track_id'] == selected_id:
                continue
            for observed in other['observations']:
                if observed['frame'] == frame:
                    excluded |= pose_seed(observed, mask.shape)
        qc = _mask_guard(mask, row, first=previous is None, previous=previous,
                         scale=reference['body_scale_px'], initial_area=initial_area, excluded=excluded,
                         roi=mask_roi_xyxy)
        qcs[str(frame)] = qc
        if not qc['valid']:
            out['mask_prefix_stop_reason'] = qc['reason']
            break
        previous = mask
        if row is None:
            continue
        sample = pose_only_measurement(row, transforms[str(frame)], reference, mask)
        if sample is None:
            continue
        ys, xs = np.nonzero(mask)
        homogeneous = np.c_[xs, ys, np.ones(len(xs))] @ np.asarray(transforms[str(frame)]).T
        stable = homogeneous[:, :2]/homogeneous[:, 2:]
        lower = float(np.percentile(stable @ np.asarray(reference['ground_down_normal']), 99))
        sample['bottom_clearance_body'] = (reference['ground_offset_px']-lower)/reference['body_scale_px']
        sample['landing_bottom_source'] = 'observed_locked_body_silhouette_lower_edge'
        locked.append(sample)
    locked_phase = analyse_height_phases(locked, kick, fps, reference)
    comparable = locked_phase['supported'] and phase['supported']
    agreement = (locked_phase['height_candidate'] == phase['height_candidate']
                 and abs(locked_phase['landing_frame']-phase['landing_frame'])/fps <= .15) if comparable else None
    supported = locked_phase['supported'] and agreement is not False
    out.update(mask_qc=qcs, locked_pose_phase=locked_phase,
               pose_only_and_mask_landing_agreement=agreement,
               strict_height_candidate=locked_phase['height_candidate'] if supported else 'unknown',
               dive_height=locked_phase['height_candidate'] if supported else 'unknown')
    return finish('source_bound_mask_pose_torso_height_candidate' if supported else 'strict_mask_pose_phase_support_inadequate')
