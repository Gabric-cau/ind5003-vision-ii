"""Abstain when projected goal evidence depends on a keeper-occluded identity."""
from copy import deepcopy


def guard_keeper_occluded_goal(result, pose):
    keeper = result.get('keeper') or {}
    seed_id = (keeper.get('role') or {}).get('seed_track_id')
    source_ids = set((keeper.get('selected_candidate') or {}).get('source_track_ids') or [])
    if seed_id is not None:
        source_ids.add(seed_id)
    contact = (result.get('independent_contact_search') or {}).get('selected_candidate') or {}
    source_ids.discard(contact.get('contact_associated_track_id'))
    boxes = {}
    for track in pose.get('tracks', []):
        if track['track_id'] in source_ids:
            for row in track.get('observations', []):
                boxes.setdefault(row['frame'], []).append({'track_id': track['track_id'], 'bbox': row['bbox']})
    path = {(row['frame'], row['detection_id']): row for row in result.get('trajectory', [])}
    intersections = []
    for key, row in path.items():
        radius = max(1., min(row.get('width', 2), row.get('height', 2))/2)
        for keeper_box in boxes.get(row['frame'], []):
            x0, y0, x1, y1 = keeper_box['bbox']
            if x0-radius <= row['x'] <= x1+radius and y0-radius <= row['y'] <= y1+radius:
                intersections.append({'frame': row['frame'], 'detection_id': row['detection_id'],
                    'source_xy': [row['x'], row['y']], 'source_ball_radius_px': radius,
                    'keeper_track_id': keeper_box['track_id'], 'keeper_source_bbox': keeper_box['bbox'],
                    'ball_or_clothing_identity_not_resolved': True,
                    'physical_keeper_contact_established': False})
    first_occlusion = min((row['frame'] for row in intersections), default=None)
    original = result['rule_decision']
    decision = deepcopy(original)
    entry = original.get('projected_entry_support_frame')
    invalid_entry = (original['rule_outcome'] == 'goal_likely' and first_occlusion is not None and
                     (entry is None or entry >= first_occlusion))
    support_ids = set((result.get('rebound') or {}).get('support_ids') or [])
    occluded_rebound = (original['rule_outcome'] == 'no_goal_likely' and
                       any(row['detection_id'] in support_ids for row in intersections))
    if invalid_entry or occluded_rebound:
        decision.update(rule_outcome='unknown', evidence_frames=[],
            reason='projected_entry_depends_on_keeper_occluded_identity' if invalid_entry else
                   'rebound_support_depends_on_keeper_occluded_identity',
            original_automatic_rule_outcome=original['rule_outcome'],
            projected_entry_support_frame=None)
    decision.update(goal_occlusion_guard_applied=True, whole_ball_crossing_established=False,
                    keeper_occlusion_used_as_no_goal_evidence=False, human_verified=False)
    evidence = {'policy': 'same_source_frame_ball_footprint_keeper_pose_overlap_before_projected_entry',
        'keeper_source_track_ids': sorted(source_ids, key=str), 'keeper_identity_verified': False,
        'first_keeper_overlap_frame': first_occlusion, 'overlap_observations': intersections,
        'original_projected_entry_support_frame': entry, 'original_rule_decision': deepcopy(original),
        'decision_changed': decision['rule_outcome'] != original['rule_outcome'],
        'trajectory_changed': False, 'contact_changed': False,
        'occlusion_is_not_proof_of_save_or_goal': True,
        'reference_labels_used_for_prediction': False, 'human_verified': False, 'accuracy_validated': False}
    return decision, evidence
