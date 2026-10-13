"""Validate dense source observations before supplementing frozen foot candidates."""
from __future__ import annotations

from collections import Counter
import re

import numpy as np

from src.kicking_foot_strike_window import kicking_foot_strike_window
from src.mediapipe_shoe_evidence import coco_samples, restore_landmarks, shoe_ball_distances, source_body_box
from src.pose_refinement import select_anchored_detection
from src.shoe_pose33_fusion import select_candidate


SCHEMA = 'fresh-foot-dense-approach-supplement-v1'
REGION_SCALE_DEFINITION = 'median exact source actor box height over all unique normal approach observations before last stationary ball frame'
REASONS = {
    'v2_and_shoe_region_agree': 'existing_fusion_and_dense_region_agree',
    'v2_and_shoe_region_conflict': 'existing_fusion_and_dense_region_conflict',
    'v2_retained_shoe_unavailable_or_abstained': 'existing_fusion_retained_dense_abstained',
    'shoe_region_fallback_for_v2_unknown': 'dense_region_fallback_for_existing_unknown',
    'both_methods_unavailable_or_abstained': 'both_methods_unavailable_or_abstained',
}


def _same_array(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=0., atol=1e-9)


def validate_dense_observations(records, inputs, roi_plan, bound, decoded_hashes):
    """Recheck the original actor geometry and independently decoded pixel clock."""
    keys = ('source_mapping', 'fps', 'scene_interval_frames', 'interval_frames',
            'inference_interval_frames', 'contact_bracket_frames', 'actor_track_id',
            'original_pose_selected_track_id', 'original_camera_motion', 'original_phase_result', 'contact')
    if (any(inputs.get(key) != bound[key] for key in keys)
            or inputs.get('coordinate_space') != 'original_source_image_pixels'
            or inputs.get('interval_endpoints') != 'inclusive'
            or inputs.get('new_neural_inference') is not True
            or inputs.get('reference_labels_read') is not False
            or inputs.get('human_verified') is not False
            or roi_plan != bound['plan']):
        raise ValueError('Dense observations do not reproduce their source actor, phase and camera plan')
    frames = list(range(bound['inference_interval_frames'][0], bound['inference_interval_frames'][1] + 1))
    if (not isinstance(decoded_hashes, dict) or set(decoded_hashes) != {str(frame) for frame in frames}
            or any(not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
                   for digest in decoded_hashes.values())
            or inputs.get('decoded_frame_hashes') != decoded_hashes):
        raise ValueError('Dense decoded hashes disagree with independently decoded source pixels')
    expected_pairs = [(frame, reflected) for frame in frames for reflected in (False, True)]
    if (len(records) != len(expected_pairs)
            or any(type(row.get('frame')) is not int or type(row.get('reflected')) is not bool for row in records)
            or [(row['frame'], row['reflected']) for row in records] != expected_pairs):
        raise ValueError('Dense records must contain exactly ordered normal and mirrored source-frame pairs')
    plan = {row['frame']: row for row in roi_plan}
    first_seen = {}
    calls = accepted = 0
    for frame in frames:
        digest = decoded_hashes[str(frame)]
        duplicate = first_seen.get(digest)
        first_seen.setdefault(digest, frame)
        task = plan[frame]
        for row in records[2 * (frame - frames[0]):2 * (frame - frames[0]) + 2]:
            if (row.get('crop_xyxy') != bound['fixed_roi'] or row.get('anchor_bbox') != task['anchor_bbox']
                    or row.get('anchor_source') != task['anchor_source']
                    or row.get('rival_source_boxes') != task['rival_source_boxes']
                    or row.get('timestamp_ms') != round(frame / inputs['fps'] * 1000)
                    or not isinstance(row.get('decoded_timestamp_ms'), (float, int))
                    or not np.isfinite(row['decoded_timestamp_ms'])
                    or abs(row['decoded_timestamp_ms'] - frame / inputs['fps'] * 1000) > max(1., 100 / inputs['fps'])
                    or row.get('frame_pixel_sha256') != digest
                    or row.get('duplicate_of_frame') != duplicate
                    or row.get('unique_decoded_support') is not (duplicate is None)
                    or row.get('neural_call') is not (task['anchor_bbox'] is not None)):
                raise ValueError('Dense source clock, original actor, rivals or duplicate support changed')
            calls += int(row['neural_call'])
            if type(row.get('pose_count')) is not int or row['pose_count'] not in (0, 1):
                raise ValueError('Dense single-pose model must explicitly report zero or one output')
            if task['anchor_bbox'] is None or row['pose_count'] == 0:
                expected_status = ('original_actor_observation_unavailable' if task['anchor_bbox'] is None
                                   else 'no_detected_pose')
                if (row['roi_identity_status'] != expected_status or row['landmarks_source_xy'] is not None
                        or row['landmark_confidence'] is not None or row['pose_count'] != 0):
                    raise ValueError('Missing source actor or model observation cannot acquire landmarks')
                continue
            visibility, presence = np.asarray(row['original_visibility'], float), np.asarray(row['original_presence'], float)
            if (visibility.shape != (33,) or presence.shape != (33,)
                    or any(not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                           for values in (visibility, presence))):
                raise ValueError('Raw visibility and presence must be finite separate 33-point confidences')
            points, confidence = restore_landmarks(row['original_normalized_xy'], np.minimum(visibility, presence),
                                                  bound['fixed_roi'], row['reflected'])
            body = source_body_box(points, confidence)
            selected, identity = (select_anchored_detection([dict(bbox=body)], task['anchor_bbox'],
                                  rivals=task['rival_source_boxes']) if body else
                                  (None, 'insufficient_confident_body_for_roi_identity'))
            if row['roi_identity_status'] != identity or row.get('predicted_source_body_bbox') != body:
                raise ValueError('Dense output body identity does not reproduce from raw landmarks')
            if selected is None:
                if row['landmarks_source_xy'] is not None or row['landmark_confidence'] is not None:
                    raise ValueError('Rejected source actor cannot supply restored landmarks')
            elif (not _same_array(points, row['landmarks_source_xy'])
                  or not _same_array(confidence, row['landmark_confidence'])):
                raise ValueError('Dense normalized landmark restoration changed')
            else:
                accepted += 1
    return dict(actual_inference_calls=calls, accepted_roi_pose_calls=accepted,
                decoded_interval_frames=len(frames), distinct_decoded_pixel_frames=len(first_seen),
                repeated_decoded_pixel_frames=len(frames) - len(first_seen),
                roi_identity_status_counts=dict(Counter(row['roi_identity_status'] for row in records)))


def recompute_dense_foot(records, inputs, export, *, method_label='paired_shoe_strike_with_longer_VIDEO_history_and_approach_median_region_scale'):
    """Reproduce the frozen v1 diagnostic, including its whole-approach scale."""
    if method_label not in {'unchanged_paired_shoe_strike_rule_with_longer_VIDEO_history',
                            'paired_shoe_strike_with_longer_VIDEO_history_and_approach_median_region_scale'}:
        raise ValueError('Unknown dense shoe diagnostic method label')
    payload = export.get('contact', export)
    identity = export.get('inference_identity', {})
    source, bracket = inputs['source_mapping'], inputs['contact_bracket_frames']
    if (export.get('source_mapping') != source or export.get('clip_id') != source['clip_id']
            or identity.get('source_sha256') != source['clip_sha256']
            or identity.get('pose_sha256') != inputs['input_receipts']['pose']['sha256']
            or identity.get('source_manifest_sha256') != inputs['source_manifest_sha256']
            or payload.get('contact_bracket_frames') != bracket
            or payload.get('observed_stationary_departure_bracket_frames') != bracket
            or payload.get('contact_associated_track_id') != inputs['actor_track_id']
            or export.get('reference_labels_used_for_prediction') is not False):
        raise ValueError('Dense contact export does not retain the bound source, causal bracket and actor')
    normal = [row for row in records if not row['reflected'] and row['unique_decoded_support']]
    mirror = [row for row in records if row['reflected'] and row['unique_decoded_support']]
    stationary, departure = payload['stationary_ball_observations'], payload['departure_ball_observations']
    evidence = kicking_foot_strike_window(coco_samples(normal, shoe_centroids=True),
        coco_samples(mirror, shoe_centroids=True), bracket, inputs['fps'],
        stationary_ball_observations=stationary, departure_ball_observations=departure)
    candidate = evidence['candidate']
    target = next(row['xy'] for row in stationary if row['frame'] == bracket[0])
    scales = [row['anchor_bbox'][3] - row['anchor_bbox'][1] for row in normal
              if row.get('anchor_bbox') and row['frame'] < bracket[0]]
    scale = float(np.median(scales)) if scales else None
    support = []
    if candidate != 'unknown' and scale:
        distances = [shoe_ball_distances(rows, target, scale) for rows in (normal, mirror)]
        near = [{row['frame'] for row in rows if bracket[0] - 1 <= row['frame'] <= bracket[1]
                 and row[candidate]['confident'] and row[candidate]['stationary_point_distance_body'] <= .08}
                for rows in distances]
        paired = {row['frame'] for row in evidence.get('accepted_paired_frames', [])}
        support = sorted(near[0] & near[1] & paired)
    ambiguous = payload.get('seed_ambiguous_frames', []) + export.get('seed_ambiguous_frames', [])
    if any(bracket[0] - 5 <= frame <= bracket[1] for frame in ambiguous):
        candidate, support = 'unknown', []
    return dict(striking_foot=candidate if support else 'unknown', swing_candidate=candidate,
                joint_region_frames=support, evidence=evidence, physical_contact_verified=False,
                method=method_label)


def guarded_selection(existing, dense, legacy_neural, bracket):
    ambiguous = legacy_neural.get('local_seed_identity_ambiguous_frames', [])
    if not isinstance(ambiguous, list) or any(type(frame) is not int for frame in ambiguous):
        raise ValueError('Legacy contact identity ambiguity must retain explicit integer frames')
    local = sorted(set(frame for frame in ambiguous if bracket and bracket[0] - 5 <= frame <= bracket[1]))
    guarded = 'unknown' if local else dense
    selected, selector_reason = select_candidate(existing, guarded)
    return dict(kicking_foot_candidate=selected, dense_region_candidate=guarded,
                selection_reason=REASONS[selector_reason], selector_reason=selector_reason,
                legacy_contact_identity_ambiguous_frames=local)
