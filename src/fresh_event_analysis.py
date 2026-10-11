"""Adapt fresh course-clip evidence to the retained ball and goalkeeper rules."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np

from src.ball_pipeline import _extract_detections, _to_numpy
from src.ball_continuity_v4 import ContinuityConfig, track_ball_continuity, bootstrap_departure
from src.ball_direction_corridor import DirectionCorridorConfig, corridor_forecasts
from src.ball_mask_seed import MaskSeedConfig, associate_with_seed, choose_seed, measure_mask_support, white_mask
from src.ball_shot_focus import ShotFocusConfig, focus_shot
from src.ball_keeper_rebound import associate_keeper_rebound
from src.ball_goal_decision import classify_goal_evidence
from src.goal_plane_rectification import project_image_point
from src.goalkeeper_role_association import select_goalkeeper_role, _same_person
from src.goalkeeper_pipeline import analyze_goalkeeper_result
from src.goalkeeper_observation_guard import guard_postkick_observations
from src.kick_contact import _all_track_samples, score_contact_candidates, prefer_causal_first_departure
from src.motion_stabilization import estimate_background_motion
from scripts.probe_goal_event_tracking import motion_blobs, source_time_indices
from scripts.run_ball_continuity_probe import collision_geometry
from scripts.run_ball_shot_focus_review import goal_polygon
from scripts.review_ball_keeper_rebound import keeper_regions
from scripts.detect_all_ball_goals import arrival_evidence, pixel_evidence


def contact_window(pose, source):
    fps = float(pose.get('fps') or 0)
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError('Fresh pose requires a positive finite fps')
    if not math.isclose(fps, float(source['fps']), rel_tol=1e-5):
        raise ValueError('Source/pose fps mismatch')
    if pose.get('selected_track_id') is None:
        return None, 'no_selected_kicker_track'
    if pose.get('selected_track_qualified') is not True:
        return None, 'unqualified_automatic_kicker_track'
    start, end = int(pose['shot_start_frame']), int(pose['shot_end_frame'])
    if not 0 <= start < end <= int(source['frame_count']):
        return None, 'selected_shot_outside_source_clip'
    selection = pose.get('shot_selection', {})
    selected = next((attempt for attempt in selection.get('attempts', [])
                     if attempt['shot_index'] == selection.get('selected_shot_index')), {})
    if selection.get('found_qualified_shot') is not True or selected.get('qualifies') is not True or selected.get('scene_rejections') != []:
        return None, 'selected_shot_qualification_not_established'
    if (selected.get('goal_context') or {}).get('supported') is not True:
        return None, 'selected_shot_goal_context_not_established'
    actor = next((row for row in pose.get('track_candidates', [])
                  if row['track_id'] == pose['selected_track_id']), {})
    if actor.get('field_support') is None or actor['field_support'] < .20:
        return None, 'selected_actor_field_support_not_established'
    if end-start < 3:
        return None, 'insufficient_selected_shot'
    return (start, end), 'entire_selected_shot_independent_ball_contact_search'


def search_fresh_contact(cache, pose, frames):
    """Use the existing stationary/approaching-ankle/departure probe globally.

    Same-frame ankle proximity nominates independent camera windows; it cannot
    establish contact. Camera chains are estimated afresh within every window.
    No prior kick frame constrains this search.
    """
    fps = cache['fps']
    all_samples = _all_track_samples(pose)
    excluded = set((pose.get('goal_line_role_exclusions') or {}).get('excluded_track_ids', []))
    field_support = {row['track_id']: row.get('field_support', 0) for row in pose.get('track_candidates', [])}
    tracks = {track['track_id']: track for track in pose.get('tracks', [])}
    eligible_tracks = {tid: samples for tid, samples in all_samples.items()
                       if tid not in excluded and float(field_support.get(tid) or 0) >= .20 and tid in tracks}
    samples = [sample for rows in eligible_tracks.values() for sample in rows]
    measured = {}
    flight = {}
    for fid, detections in cache['detections'].items():
        flight[int(fid)] = [q for q in detections if q['confidence'] >= .04]
        mask = white_mask(frames[int(fid)], MaskSeedConfig(maximum_saturation=95, maximum_green_excess=25))
        measured[int(fid)] = [q for q in flight[int(fid)] if
                 measure_mask_support(mask, q)['mask_support']]
    near = []
    for sample in samples:
        fid = sample['frame']
        if fid not in measured:
            continue
        for detection in measured[fid]:
            distance = min(np.linalg.norm(np.asarray(point)-[detection['x'], detection['y']]) for point in sample['ankles'])
            if distance <= .28*sample['height']:
                near.append(fid)
                break
    groups = []
    for fid in sorted(set(near)):
        if not groups or fid-groups[-1][-1] > max(3, round(.20*fps)):
            groups.append([fid])
        else:
            groups[-1].append(fid)
    windows, candidates = [], []
    people = {int(f): [q['bbox'] for q in rows if q['confidence'] >= .10] for f, rows in cache['people'].items()}
    for track in pose.get('tracks', []):
        for observation in track.get('observations', []):
            people.setdefault(observation['frame'], []).append(observation['bbox'])
    for group in groups:
        start = max(cache['start_frame'], group[0]-math.ceil(.80*fps))
        end = min(cache['end_frame_exclusive'], group[-1]+math.ceil(.80*fps)+1)
        camera = estimate_background_motion(cache['video_path'], start, end, boxes_by_frame=people)
        transforms = {int(f): np.asarray(matrix) for f, matrix in camera['transforms'].items() if matrix is not None}
        observations = [{'frame': fid, 'xy': [q['x'], q['y']],
                         'bbox': [q['x']-q['width']/2, q['y']-q['height']/2, q['x']+q['width']/2, q['y']+q['height']/2],
                         'confidence': q['confidence'], 'diameter': (q['width']+q['height'])/2,
                         'detection_id': q['detection_id']}
                        for fid in range(start, end) if fid in transforms for q in flight[fid]]
        seed_ids = {q['detection_id'] for fid in range(start, end) for q in measured[fid]}
        proposals = []
        for tid, actor_samples in eligible_tracks.items():
            supported_samples = [s for s in actor_samples if s['frame'] in transforms]
            actor_proposals = score_contact_candidates(observations, supported_samples, transforms, fps, 1,
                                                       contact_window_policy='causal_first_departure',
                                                       stationary_seed_detection_ids=seed_ids)
            for proposal in actor_proposals:
                proposal['goalward_departure_support'] = goalward_departure_support(proposal, observations, transforms, cache)
                verified = proposal['goalward_departure_support'].get('selected_hypothesis')
                canonicalize_contact_timing(proposal, verified, observations, fps)
                proposal.update(contact_associated_track_id=tid,
                                original_pose_selected_track_id=pose['selected_track_id'],
                                actor_reselected_by_ball_evidence=tid != pose['selected_track_id'],
                                actor_field_support=field_support[tid], actor_goal_line_excluded=False,
                                seed_ambiguous_frames=proposal.get('stationary_identity_ambiguous_frames', []),
                                search_window_index=len(windows), camera_chain_measured=True)
                candidates.append(proposal)
            proposals.extend(actor_proposals)
        windows.append({'start_frame': start, 'end_frame_exclusive': end, 'ankle_near_ball_frames': group,
                        'camera': camera, 'contact_candidates': proposals})
    unique = {}
    for proposal in candidates:
        key = proposal['contact_associated_track_id'], proposal['contact_frame']
        if key not in unique or proposal['score'] > unique[key]['score']:
            unique[key] = proposal
    ranked = sorted(unique.values(), key=lambda q: (-q['score'], q['contact_frame'], str(q['contact_associated_track_id'])))
    eligible = [q for q in ranked if q['score'] >= .60 and q.get('ball_departure_frames') and
                q.get('goalward_departure_support', {}).get('status') == 'supported_goalward_departure_candidate' and
                q.get('contact_timing_status') == 'bounded_stationary_departure_candidate' and
                q.get('ankle_approach_supported') is True and
                'stationary_ball_persists_after_candidate' not in q.get('warnings', [])]
    actor_candidates = []
    for tid in eligible_tracks:
        actor = [q for q in eligible if q['contact_associated_track_id'] == tid]
        if actor:
            first = prefer_causal_first_departure(actor)[0]
            actor_candidates.append({**first, 'actor_support_score': max(q['score'] for q in actor)})
    actor_candidates.sort(key=lambda q: (-q['actor_support_score'], q['contact_frame'], str(q['contact_associated_track_id'])))
    representatives = []
    for candidate in actor_candidates:
        tid = candidate['contact_associated_track_id']
        if any(tid == other['contact_associated_track_id'] or
               _same_person(tracks[tid], tracks[other['contact_associated_track_id']]) for other in representatives):
            continue
        representatives.append(candidate)
    ambiguous_actor = len(representatives) > 1 and representatives[0]['actor_support_score']-representatives[1]['actor_support_score'] < .08
    selected = representatives[0] if representatives and not ambiguous_actor else None
    return {'status': 'candidate_supported' if selected else 'abstain',
            'reason': 'different_actor_contact_candidates_have_close_scores' if ambiguous_actor else None,
            'policy': 'global_onfield_actor_stationary_ball_causal_first_departure',
            'eligible_actor_ids': sorted(eligible_tracks), 'goal_line_excluded_actor_ids': sorted(excluded),
            'actor_score_margin_required': .08, 'actor_identity_ambiguous': ambiguous_actor,
            'distinct_actor_candidates': representatives,
            'pose_kick_proposal_for_comparison_only': pose.get('kick_frame'),
            'full_selected_shot_search': [cache['start_frame'], cache['end_frame_exclusive']],
            'candidate_windows': windows, 'ranked_candidates': ranked, 'eligible_candidates': eligible,
            'selected_candidate': selected,
            'old_contact_or_outcome_labels_used': False, 'human_verified': False, 'accuracy_validated': False}


def canonicalize_contact_timing(proposal, hypothesis, observations, fps):
    """Bound timing by observed identity, not ankle proximity or backprojection."""
    proposal.update(scored_ankle_proximity_frame=proposal['contact_frame'],
                    scored_ankle_proximity_interval_frames=proposal.get('interval_frames'),
                    scored_ankle_proximity_warnings=list(proposal.get('warnings', [])),
                    contact_timing_status='unknown', contact_bracket_frames=None,
                    exact_contact_frame_established=False, contact_frame_is_estimated=True,
                    maximum_contact_bracket_frames=max(2, math.floor(.15*fps)),
                    timing_interval_interpretation='last_observed_stationary_to_first_observed_goalward_departure')
    if not hypothesis:
        proposal['contact_timing_reason'] = 'no_causally_supported_departure'
        return
    lookup = {(row['frame'], row['detection_id']): row for row in observations}
    moving = [lookup[(row['frame'], row['detection_id'])] for row in hypothesis['moving_observations']]
    if not moving:
        proposal['contact_timing_reason'] = 'no_actual_departure_observations'
        return
    first_departure = min(row['frame'] for row in moving)
    stationary = proposal.get('stationary_ball_observations', [])
    before = [row['frame'] for row in stationary if row['frame'] < first_departure]
    proposal.update(ball_departure_frames=[row['frame'] for row in moving],
                    departure_ball_observations=moving,
                    departure_backprojection_frame_for_comparison_only=hypothesis['release_frame_prediction'])
    if len(set(before)) < 3:
        proposal['contact_timing_reason'] = 'insufficient_same_identity_stationary_history'
        return
    last_stationary = max(before)
    proposal['last_stationary_near_contact_frame'] = last_stationary
    proposal['departure_backprojection_precedes_actual_stationary_observation'] = hypothesis['release_frame_prediction'] < last_stationary
    if any(row['frame'] >= first_departure for row in stationary):
        proposal['contact_timing_reason'] = 'stationary_identity_observed_after_departure'
        return
    bracket = [last_stationary, first_departure]
    proposal['observed_stationary_departure_bracket_frames'] = bracket
    if first_departure-last_stationary > proposal['maximum_contact_bracket_frames']:
        proposal['contact_timing_reason'] = 'stationary_departure_bracket_too_wide'
        return
    proposal.update(contact_timing_status='bounded_stationary_departure_candidate',
                    contact_timing_reason=None, contact_bracket_frames=bracket,
                    interval_frames=bracket, contact_frame=(last_stationary+first_departure)//2)


def goalward_departure_support(proposal, observations, transforms, cache):
    """Use retained v4 bootstrap constraints, without masking neural flight."""
    from dataclasses import asdict
    cfg = ContinuityConfig()
    result = {'status': 'unknown', 'policy': 'retained_v4_goalward_speed_backward_miss_bootstrap',
              'config': asdict(cfg), 'selected_hypothesis': None, 'hypotheses': [],
              'neural_flight_detections_white_mask_filtered': False, 'human_verified': False}
    contact = proposal['contact_frame']
    goals = [(abs(int(frame)-contact), int(frame), row['selected']) for frame, row in cache.get('goal_proposals', {}).items()
             if row.get('selected') and int(frame) in transforms]
    if not goals:
        result['reason'] = 'missing_same_window_goal_geometry'
        return result
    _, goal_frame, goal = min(goals, key=lambda row: (row[0], row[1]))
    center = project_image_point(transforms[goal_frame], np.mean(goal['corners'], axis=0))
    lookup = {(row['frame'], row['detection_id']): row for row in observations}
    seed_ids = {(row['frame'], row.get('detection_id')) for row in proposal['stationary_ball_observations']}
    candidates = [{**row, 'x': row['xy'][0], 'y': row['xy'][1],
                   'width': row['bbox'][2]-row['bbox'][0], 'height': row['bbox'][3]-row['bbox'][1],
                   'stabilized_xy': project_image_point(transforms[row['frame']], row['xy']),
                   'motion_time': row['frame'], 'sources': ['fresh_neural_detection']}
                  for row in observations]
    seed = [row for row in candidates if (row['frame'], row['detection_id']) in seed_ids]
    if len(seed) < 3 or center is None:
        result['reason'] = 'insufficient_same_identity_stationary_seed'
        return result
    hypotheses = bootstrap_departure(seed, candidates, np.asarray(center), contact, cfg)
    for row in hypotheses:
        moving = [lookup[(item['frame'], item['detection_id'])] for item in row['path'][1:]]
        result['hypotheses'].append({'release_frame_prediction': row['onset_motion_time'],
                                    'backward_miss_px': row['departure_backward_miss_px'],
                                    'moving_observations': moving})
    if result['hypotheses']:
        selected = min(result['hypotheses'], key=lambda row: (row['moving_observations'][0]['frame'],
                                                            row['backward_miss_px'], row['release_frame_prediction']))
        result.update(status='supported_goalward_departure_candidate', selected_hypothesis=selected)
    else:
        result['reason'] = 'no_retained_v4_goalward_departure_hypothesis'
    return result


def contact_pose(pose, contact):
    """Use the independently associated contact actor without changing source pose."""
    actor = contact['contact_associated_track_id']
    candidate = next((row for row in pose.get('track_candidates', []) if row['track_id'] == actor), {})
    if candidate.get('field_support') is None or candidate['field_support'] < .20:
        raise ValueError('Contact actor is not supported on the field')
    if actor in (pose.get('goal_line_role_exclusions') or {}).get('excluded_track_ids', []):
        raise ValueError('Contact actor was excluded as a goal-line occupant')
    return {**pose, 'selected_track_id': actor,
            'original_pose_selected_track_id': pose['selected_track_id'],
            'actor_reselected_by_ball_evidence': actor != pose['selected_track_id'],
            'kick_frame': int(contact['contact_frame'])}


def local_event_cache(cache, pose, contact, before_sec, after_sec):
    frame = int(contact['contact_frame'])
    start = max(cache['start_frame'], frame-math.ceil(before_sec*cache['fps']))
    end = min(cache['end_frame_exclusive'], frame+math.ceil(after_sec*cache['fps'])+1)
    result = {**cache, 'start_frame': start, 'end_frame_exclusive': end, 'kick_frame': frame,
              'contact': {**contact, 'source': 'fresh_global_ball_and_ankle_probe', 'human_verified': False}}
    for name in ('detections', 'people', 'goal_proposals', 'raw_neural_detections', 'source_times'):
        result[name] = {f: deepcopy(rows) for f, rows in cache[name].items() if start <= int(f) < end}
    boxes = {int(f): [q['bbox'] for q in rows if q['confidence'] >= .10] for f, rows in result['people'].items()}
    for track in pose.get('tracks', []):
        for observation in track.get('observations', []):
            if start <= observation['frame'] < end:
                boxes.setdefault(observation['frame'], []).append(observation['bbox'])
    result['camera'] = estimate_background_motion(cache['video_path'], start, end, boxes_by_frame=boxes)
    for fid, detections in result['detections'].items():
        matrix = result['camera']['transforms'].get(fid)
        for detection in detections:
            detection['stabilized_xy'] = project_image_point(matrix, [detection['x'], detection['y']]) if matrix is not None else None
            detection['motion_mean_abs_gray'] = None
    return result


def partition_predictions(ids, predictions, frames, *, offset=(0, 0), mode='full'):
    """Keep all raw boxes; apply the existing ball candidate size/visual gates."""
    if len(ids) != len(predictions):
        raise ValueError('Detector batch output length mismatch')
    balls, people, raw = {}, {}, {}
    for fid, result, frame in zip(ids, predictions, frames):
        boxes = getattr(result, 'boxes', None)
        xyxy = _to_numpy(boxes.xyxy).astype(float).copy() if boxes is not None else np.empty((0, 4))
        confidence = _to_numpy(boxes.conf).reshape(-1) if boxes is not None else np.empty(0)
        classes = _to_numpy(boxes.cls).reshape(-1).astype(int) if boxes is not None else np.empty(0, dtype=int)
        if not len(xyxy) == len(confidence) == len(classes):
            raise ValueError('Unaligned detection tensors')
        xyxy[:, [0, 2]] += offset[0]
        xyxy[:, [1, 3]] += offset[1]
        raw[str(fid)] = [{'bbox': box.tolist(), 'confidence': float(cf), 'class_id': int(cls),
                          'mode': mode, 'crop_translation_xy': list(offset)}
                         for box, cf, cls in zip(xyxy, confidence, classes)]
        selected = classes == 32
        adapted = SimpleNamespace(boxes=SimpleNamespace(xyxy=xyxy[selected], conf=confidence[selected]))
        balls[str(fid)] = [{**q, 'sources': [mode]} for q in _extract_detections([fid], [adapted], [frame])[fid]]
        people[str(fid)] = [{'bbox': box.tolist(), 'confidence': float(cf), 'sources': [mode]}
                            for box, cf, cls in zip(xyxy, confidence, classes) if cls == 0]
    return balls, people, raw


def merge_ball_candidates(candidates, fid):
    merged = []
    for item in sorted(candidates, key=lambda q: -q['confidence']):
        hit = next((q for q in merged if np.hypot(item['x']-q['x'], item['y']-q['y']) <
                    max(4., .4*min(item['width'], q['width']))), None)
        if hit is not None:
            hit['sources'] = sorted(set(hit['sources']+item['sources']))
        else:
            merged.append({**item, 'sources': list(item['sources'])})
    return [{**q, 'detection_id': f'{fid}:{i}'} for i, q in enumerate(merged)]


def pose_anchors(pose, cache):
    track = next((t for t in pose.get('tracks', []) if t['track_id'] == pose['selected_track_id']), {})
    anchors = []
    for obs in track.get('observations', []):
        transform = cache['camera']['transforms'].get(str(obs['frame']))
        if abs(obs['frame']-cache['kick_frame']) > 3 or transform is None:
            continue
        for j in (15, 16):
            if len(obs.get('keypoint_confidences', [])) > j and obs['keypoint_confidences'][j] >= .25:
                point = project_image_point(transform, obs['keypoints'][j])
                if point is not None:
                    anchors.append(point)
    options = [(abs(int(f)-cache['kick_frame']), int(f), g['selected'])
               for f, g in cache['goal_proposals'].items() if g.get('selected') and
               cache['camera']['transforms'].get(str(f)) is not None]
    if not options:
        return anchors, None
    _, frame, goal = min(options, key=lambda q: (q[0], q[1]))
    center = project_image_point(cache['camera']['transforms'][str(frame)], np.mean(goal['corners'], axis=0))
    return anchors, center


def image_fingerprints(frames):
    result, previous, last = {}, None, None
    for fid, frame in sorted(frames.items()):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        duplicate = previous is not None and float(np.mean(cv2.absdiff(gray, previous))) < .35
        last = last if duplicate else hashlib.sha256(frame.tobytes()).hexdigest()
        result[str(fid)] = last
        previous = gray
    return result


def analyze_fresh_ball(cache, pose, frames, input_dir):
    anchors, goal = pose_anchors(pose, cache)
    timing, blobs = source_time_indices(cache), motion_blobs(cache)
    mask_config = MaskSeedConfig(maximum_saturation=95, maximum_green_excess=25, seed_after_kick_frames=3)
    mask_evidence, pool = {}, {}
    for fid, frame in frames.items():
        key = str(fid)
        mask = white_mask(frame, mask_config)
        for detection in cache['detections'][key]:
            mask_evidence[detection['detection_id']] = measure_mask_support(mask, detection, mask_config)
        pool[key] = []
        for detection in cache['detections'][key]+blobs.get(key, []):
            item = {**detection, **timing[fid]}
            if 'motion_blob' in item.get('sources', []):
                current = measure_mask_support(mask, item, mask_config)
                item['current_pixel_mask_support'] = current['mask_support']
            pool[key].append(pixel_evidence(item, frame))
    fingerprints = image_fingerprints(frames)
    candidate_hash = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()
    seed_pool = cache['detections']
    contact = cache.get('contact') or {}
    if 'stationary_ball_observations' in contact:
        seed_ids = {(row['frame'], row.get('detection_id')) for row in contact['stationary_ball_observations']}
        seed_pool = {fid: [q for q in rows if (q['frame'], q['detection_id']) in seed_ids]
                     for fid, rows in cache['detections'].items()}
    seed, seed_diagnostics = choose_seed(seed_pool, cache['kick_frame'], anchors, mask_evidence,
                                      use_mask=True, radius=35., config=mask_config)
    seed_diagnostics['independent_contact_seed_identity_used'] = 'stationary_ball_observations' in contact
    metadata = {f: {'coordinate_supported': cache['camera']['transforms'].get(f) is not None} for f in pool}
    cfg, corridor_cfg, focus_cfg = ContinuityConfig(), DirectionCorridorConfig(), ShotFocusConfig()
    regions = collision_geometry(cache, {})
    track_arguments = dict(tracker=track_ball_continuity, config=cfg, frame_metadata=metadata,
                           collision_regions=regions, corridor_config=corridor_cfg)
    if goal is None or not anchors:
        path, tracking = [], {'tracking_status': 'abstain_missing_goal_or_ankles'}
    else:
        path, tracking = associate_with_seed(pool, cache['kick_frame'], anchors, goal, seed, seed_diagnostics, **track_arguments)
        repeat, repeated = associate_with_seed(pool, cache['kick_frame'], anchors, goal, seed, seed_diagnostics, **track_arguments)
        if path != repeat or tracking != repeated:
            raise AssertionError('Fresh association replay differs')
    departure = {'status': 'unknown', 'pose_contact_frame': cache['kick_frame'],
                 'inferred_release_frame': None, 'release_time_is_prediction': True,
                 'human_verified': False}
    departure_hypotheses = []
    if seed and goal is not None:
        candidates = [q for ds in pool.values() for q in ds if q.get('stabilized_xy') is not None]
        candidates.sort(key=lambda q: (q['frame'], q['detection_id']))
        for proposal in bootstrap_departure(seed, candidates, np.asarray(goal), cache['kick_frame'], cfg):
            ids = [q['detection_id'] for q in proposal['path']]
            departure_hypotheses.append({'observation_ids': ids, 'release_frame_prediction': proposal['onset_motion_time'],
                           'backward_miss_px': proposal['departure_backward_miss_px']})
            if tracking.get('tracking_status') == 'candidate' and ids == [q['detection_id'] for q in path[:3]]:
                departure.update(status='supported_association_candidate', inferred_release_frame=proposal['onset_motion_time'],
                         offset_from_pose_contact_frames=proposal['onset_motion_time']-cache['kick_frame'],
                         supporting_observation_ids=ids, backward_miss_px=proposal['departure_backward_miss_px'])
    effective_seed = [q for q in seed if q['frame'] <= path[0]['frame']] if path else seed
    if goal is None:
        kept, phase = [], {'reason': 'missing_goal_geometry', 'main_end_frame': None,
                           'focus_end_frame': cache['kick_frame'], 'goal_area_gate_progress_px': None}
    else:
        try:
            kept, phase = focus_shot(path, effective_seed, goal, cache['kick_frame'], cache['fps'],
                                     cache['end_frame_exclusive'], goal_polygon=goal_polygon(cache), config=focus_cfg)
        except ValueError as exc:
            kept, phase = [], {'reason': 'unsupported_departure_goal_geometry', 'detail': str(exc),
                               'main_end_frame': None, 'focus_end_frame': cache['kick_frame'],
                               'goal_area_gate_progress_px': None}
    people = {'start_frame': cache['start_frame'], 'end_frame_exclusive': cache['end_frame_exclusive'],
              'frames': {f: {'people': cache['people'][f], 'balls': cache['detections'][f]} for f in pool}}
    keeper_geometry, keeper_boxes = keeper_regions(cache, people)
    boundary = phase['main_end_frame']
    incoming = [q for q in path if boundary is not None and q['frame'] <= boundary]
    if len(incoming) >= 3:
        continued, rebound = associate_keeper_rebound(incoming, pool, keeper_geometry, cache['fps'], frame_fingerprints=fingerprints)
    else:
        continued, rebound = incoming, {'turn_confirmed': False, 'reason': 'insufficient_incoming_prefix'}
    observed = continued if rebound.get('turn_confirmed') else kept
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir/f'{cache["clip_id"]}.json').write_text(json.dumps({'anchors': anchors, 'goal_center': goal,
                    'source_sha256': cache['source_sha256'], 'candidate_hash': candidate_hash,
                    'pose_sha256': cache['pose_sha256'], 'candidates': pool}, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    arrivals = []
    if effective_seed and goal is not None and boundary is not None:
        valid = [q for q in observed if cache['camera']['transforms'].get(str(q['frame'])) is not None]
        arrivals = arrival_evidence(valid, cache, frames, fingerprints,
                                    {'seed': effective_seed, 'phase': phase}, input_dir=input_dir)
    decision = classify_goal_evidence(arrivals, boundary if boundary is not None else cache['kick_frame'], rebound)
    allowed = {(q['frame'], q['detection_id']) for ds in pool.values() for q in ds}
    if any((q['frame'], q['detection_id']) not in allowed for q in observed):
        raise AssertionError('Associated point is not a measured candidate')
    if hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest() != candidate_hash:
        raise AssertionError('Association changed the measured candidates')
    forecasts = corridor_forecasts(path, tracking.get('motion_states', []), cache['start_frame'],
                          phase['focus_end_frame']+1, config=corridor_cfg, max_gap_frames=cfg.max_gap_frames,
                          reachable_speed_ratio=cfg.reachable_speed_ratio)
    return {'clip_id': cache['clip_id'], 'anchors': anchors, 'goal_reference_center': goal,
            'automatic_seed': seed, 'seed': effective_seed, 'seed_diagnostics': seed_diagnostics,
            'ball_departure': departure, 'departure_hypotheses': departure_hypotheses,
            'mask_evidence': mask_evidence, 'motion_candidates': blobs, 'candidate_hash': candidate_hash,
            'trajectory_before_rebound': path, 'trajectory_in_focus': kept, 'trajectory': observed,
            'tracking': tracking, 'phase': phase, 'rebound': rebound, 'rule_decision': decision,
            'arrival_evidence': arrivals, 'keeper_regions': keeper_geometry, 'keeper_boxes': keeper_boxes,
            'frame_fingerprints': fingerprints, 'source_time_indices': timing, 'corridor_forecasts': forecasts,
            'last_observed_position': [observed[-1]['x'], observed[-1]['y']] if observed else None,
            'last_observed_frame': observed[-1]['frame'] if observed else None,
            'final_position': None, 'final_position_status': 'not_established_from_short_event_window',
            'old_ball_trajectory_input': False, 'reference_labels_used_for_prediction': False,
            'manual_ball_points_used': False, 'human_verified': False, 'accuracy_validated': False,
            'coordinate_system': 'image_pixels_and_background_stabilized_image_pixels',
            'ground_plane_trajectory_established': False, 'whole_ball_crossing_established': False,
            'deterministic_replay': goal is not None and bool(anchors), 'candidate_membership_verified': True}


def analyze_fresh_keeper(pose, cache):
    geometry = {int(f): value for f, value in cache['goal_proposals'].items() if value.get('selected')}
    role = select_goalkeeper_role(pose, geometry, cache['kick_frame'])
    if role['seed_track_id'] is None:
        return {'clip_id': pose['clip_id'], 'role': role, 'status': 'abstain',
                'gk_prekick_movement_level': 'unknown', 'dive_direction': 'unknown', 'dive_height': 'unknown',
                'reason': 'fresh_pose_does_not_support_goal_line_identity',
                'human_verified': False, 'accuracy_validated': False}
    result = analyze_goalkeeper_result(pose, cache['kick_frame'], seed_track_id=role['seed_track_id'],
                    pre_window_sec=1.5, post_window_sec=1., tracking_policy='appearance_bounded',
                    use_field_mask=True, kick_frame_confidence='unknown', kick_frame_source='fresh_source_bound_stationary_ball_and_departure_candidate')
    result['goal_identity_evidence'] = role
    guarded = guard_postkick_observations(result, policy='pose_direction_v3', pose=pose)
    guarded.update(role=role, identity_verified=False, new_keeper_neural_inference=False,
                   prediction_scope='candidate_features_from_fresh_pose_tracks',
                   camera_compensation='fresh_source_decode_complete_prefix_chain_with_field_and_person_masks',
                   direction_coordinate_system='source_image_x_right', goalkeeper_relative_direction='unknown',
                   human_verified=False, accuracy_validated=False)
    return guarded
