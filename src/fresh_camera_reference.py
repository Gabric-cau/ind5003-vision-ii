"""Reuse exact live-shot neural observations without cross-shot identity guesses."""

from collections import defaultdict

import numpy as np

from src.camera_partial_alignment import contiguous, partial_angles, usable
from src.goalkeeper_pipeline import _centre_and_scale


def source_role_observations(pose, contact, ball_result):
    candidate = (contact.get('independent_contact_search') or {}).get('selected_candidate') or {}
    actor = candidate.get('contact_associated_track_id')
    bracket = candidate.get('contact_bracket_frames')
    if (candidate.get('contact_timing_status') != 'bounded_stationary_departure_candidate' or
            not isinstance(bracket, list) or len(bracket) != 2 or
            not all(isinstance(frame, int) for frame in bracket) or
            not pose['shot_start_frame'] <= bracket[0] < bracket[1] < pose['shot_end_frame']):
        raise ValueError('A bounded source-shot contact is required')
    tracks = {track['track_id']: track for track in pose.get('tracks', [])}
    if actor not in tracks or actor in (pose.get('goal_line_role_exclusions') or {}).get('excluded_track_ids', []):
        raise ValueError('Contact actor is absent or goal-line excluded')
    roles = {'kicker': {}, 'keeper': {}}
    diagnostics = {'kicker_track_id': actor, 'keeper_ambiguous_frames': [],
                   'keeper_unmatched_frames': [], 'keeper_identity_verified': False}
    for row in tracks[actor].get('observations', []):
        if pose['shot_start_frame'] <= row['frame'] < pose['shot_end_frame']:
            if row['frame'] in roles['kicker']:
                raise ValueError('Duplicate actor observations at one source frame')
            roles['kicker'][row['frame']] = {**row, 'source_track_id': actor}
    keeper = ball_result.get('keeper') or {}
    selected = keeper.get('selected_candidate') or {}
    role = keeper.get('role') or {}
    if role.get('status') != 'candidate_supported' or not selected.get('qualifies'):
        diagnostics['keeper_rejection'] = 'source_keeper_role_or_association_not_supported'
        return roles, diagnostics
    ids = set(selected.get('source_track_ids') or [])
    if role.get('seed_track_id') is not None:
        ids.add(role['seed_track_id'])
    ids.discard(actor)
    pool = defaultdict(list)
    for tid in ids:
        for row in tracks.get(tid, {}).get('observations', []):
            pool[row['frame']].append((tid, row))
    fps = float(pose['fps'])
    # Recover keypoints only from the actual pixel centre retained by the
    # existing locked keeper association. A nearby person cannot take over.
    for target in selected.get('pre_trajectory', []) + selected.get('post_trajectory', []):
        frame = round(float(target[0]) * fps)
        if frame in roles['keeper'] or not pose['shot_start_frame'] <= frame < pose['shot_end_frame']:
            continue
        matches = [(tid, row) for tid, row in pool[frame]
                   if np.linalg.norm(np.asarray(_centre_and_scale(row)[:2])-target[1:3]) <= .5]
        if not matches:
            diagnostics['keeper_unmatched_frames'].append(frame)
            continue
        if len(matches) > 1:
            centres = np.array([_centre_and_scale(row)[:2] for _, row in matches])
            if np.max(np.linalg.norm(centres-centres[0], axis=1)) > .1:
                diagnostics['keeper_ambiguous_frames'].append(frame)
                continue
        tid, row = max(matches, key=lambda item: (
            item[0] == role.get('seed_track_id'), item[1].get('box_confidence', 0)))
        roles['keeper'][frame] = {**row, 'source_track_id': tid}
    diagnostics['keeper_source_track_ids'] = sorted(ids, key=str)
    return roles, diagnostics


def live_reference_action(pose, contact, ball_result, start, end):
    roles, diagnostics = source_role_observations(pose, contact, ball_result)
    if start < pose['shot_start_frame'] or end > pose['shot_end_frame']:
        raise ValueError('Reference action cannot cross the selected source shot')
    fps = float(pose['fps'])
    result = {'roles': {'kicker': [], 'keeper': []}, 'raw_pose_observations': [],
              'sampling_fps_target': 8, 'new_neural_inference_this_action': False,
              'measurement_source': 'exact_source_pose_producer_observations',
              'source_pose_sha256': pose['_pose_file_sha256'],
              'source_pose_inference_identity': pose['inference_identity'],
              'association_diagnostics': diagnostics, 'identity_verified': False,
              'source_shot_bounds': [pose['shot_start_frame'], pose['shot_end_frame']],
              'action_bounds': [start, end]}
    for frame in range(start, end, max(1, round(fps/8))):
        people = []
        for role in roles:
            row = roles[role].get(frame)
            if row is None:
                continue
            people.append({**row, 'role': role+'_candidate',
                           'role_source': 'same_shot_bound_source_track'})
            feature = partial_angles(row)
            if usable({'feature': feature}):
                result['roles'][role].append({'frame': frame, 'time_sec': frame/fps,
                    'feature': feature, 'source_track_id': row['source_track_id'],
                    'source_pose_sha256': pose['_pose_file_sha256']})
        result['raw_pose_observations'].append({'frame': frame, 'people': people})
    result['contiguous_support'] = {}
    for role, rows in result['roles'].items():
        blocks = contiguous(rows, .35)
        result['contiguous_support'][role] = {
            'usable_samples': len(rows), 'longest_block_samples': max(map(len, blocks), default=0),
            'longest_block_span_sec': max((b[-1]['time_sec']-b[0]['time_sec'] for b in blocks), default=0.)}
    return result
