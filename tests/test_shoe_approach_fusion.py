from copy import deepcopy

import numpy as np
import pytest

from src.mediapipe_shoe_evidence import MIRROR_PAIRS, restore_landmarks, source_body_box
from src.shoe_approach_fusion import guarded_selection, recompute_dense_foot, validate_dense_observations


def observation_fixture():
    source = dict(clip_id='event', clip_sha256='source', fps=30., frame_count=40)
    bound = dict(source_mapping=source, fps=30., scene_interval_frames=[0, 40],
                 interval_frames=[16, 20], inference_interval_frames=[16, 24],
                 contact_bracket_frames=[20, 22], actor_track_id=1, original_pose_selected_track_id=1,
                 original_camera_motion={'transforms': {'16': np.eye(3).tolist()}},
                 original_phase_result={'approach_start_frame': 16}, contact={'contact_associated_track_id': 1},
                 fixed_roi=[0, 0, 200, 240])
    plan = [dict(frame=frame, crop_xyxy=bound['fixed_roi'], anchor_bbox=[80, 20, 120, 220],
                 anchor_source='observed', rival_source_boxes=[]) for frame in range(16, 25)]
    bound['plan'] = plan
    hashes = {str(frame): f'{frame:064x}' for frame in range(16, 25)}
    hashes['17'] = hashes['16']
    inputs = {key: deepcopy(bound[key]) for key in bound if key not in {'plan', 'fixed_roi'}}
    inputs.update(coordinate_space='original_source_image_pixels', interval_endpoints='inclusive',
                  new_neural_inference=True, reference_labels_read=False, human_verified=False,
                  decoded_frame_hashes=hashes)
    points = np.full((33, 2), [100., 40.])
    points[[0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]] = [
        [100, 20], [80, 60], [120, 60], [85, 130], [115, 130], [85, 170], [115, 170],
        [90, 220], [110, 220], [88, 220], [108, 220], [95, 220], [115, 220]]
    records = []
    for item in plan:
        frame = item['frame']
        for reflected in (False, True):
            raw = points.copy()
            if reflected:
                for left, right in MIRROR_PAIRS:
                    raw[[left, right]] = raw[[right, left]]
                raw[:, 0] = 199 - raw[:, 0]
            raw /= [200, 240]
            restored, confidence = restore_landmarks(raw, np.ones(33), bound['fixed_roi'], reflected)
            records.append(dict(item, reflected=reflected, neural_call=True, pose_count=1,
                timestamp_ms=round(frame / 30. * 1000), decoded_timestamp_ms=frame / 30. * 1000,
                frame_pixel_sha256=hashes[str(frame)], duplicate_of_frame=16 if frame == 17 else None,
                unique_decoded_support=frame != 17, original_normalized_xy=raw.tolist(),
                original_visibility=[1.] * 33, original_presence=[1.] * 33,
                predicted_source_body_bbox=source_body_box(restored, confidence), roi_identity_status='accepted',
                landmarks_source_xy=restored.tolist(), landmark_confidence=confidence.tolist()))
    return records, inputs, plan, bound, hashes


def test_restore_original_actor_and_reconstructed_duplicate_support():
    value = observation_fixture()
    quality = validate_dense_observations(*value)
    assert quality['actual_inference_calls'] == quality['accepted_roi_pose_calls'] == 18
    assert quality['decoded_interval_frames'] == 9
    assert quality['distinct_decoded_pixel_frames'] == 8
    assert quality['repeated_decoded_pixel_frames'] == 1


@pytest.mark.parametrize('change', ['restored_xy', 'visibility', 'presence', 'anchor', 'rival', 'duplicate', 'clock', 'body'])
def test_tampered_observation_cannot_supply_supplement(change):
    records, inputs, plan, bound, hashes = observation_fixture()
    if change == 'restored_xy':
        records[0]['landmarks_source_xy'][29][0] += 1
    elif change == 'visibility':
        records[0]['original_visibility'][29] = -1
    elif change == 'presence':
        records[0]['original_presence'][29] = 2
    elif change == 'anchor':
        records[0]['anchor_bbox'] = [100, 20, 140, 220]
    elif change == 'rival':
        records[0]['rival_source_boxes'] = [[100, 20, 140, 220]]
    elif change == 'duplicate':
        records[2]['unique_decoded_support'] = True
    elif change == 'clock':
        records[0]['decoded_timestamp_ms'] += 33
    else:
        records[0]['predicted_source_body_bbox'][0] += 1
    with pytest.raises(ValueError):
        validate_dense_observations(records, inputs, plan, bound, hashes)


def test_changed_camera_payload_rejected_even_with_unchanged_record_geometry():
    records, inputs, plan, bound, hashes = observation_fixture()
    inputs['original_camera_motion']['transforms']['16'][0][2] = 10
    with pytest.raises(ValueError, match='camera plan'):
        validate_dense_observations(records, inputs, plan, bound, hashes)


def test_relabelled_source_pixel_hashes_are_rejected():
    records, inputs, plan, bound, hashes = observation_fixture()
    independently_decoded = dict(hashes, **{'16': 'f' * 64})
    with pytest.raises(ValueError, match='source pixels'):
        validate_dense_observations(records, inputs, plan, bound, independently_decoded)


def test_missing_paired_view_cannot_be_filled_by_another_source_frame():
    records, inputs, plan, bound, hashes = observation_fixture()
    del records[1]
    with pytest.raises(ValueError, match='source-frame pairs'):
        validate_dense_observations(records, inputs, plan, bound, hashes)


@pytest.mark.parametrize('old,dense,expected,reason', [
    ('unknown', 'right', 'right', 'dense_region_fallback_for_existing_unknown'),
    ('unknown', 'left', 'left', 'dense_region_fallback_for_existing_unknown'),
    ('right', 'unknown', 'right', 'existing_fusion_retained_dense_abstained'),
    ('left', 'unknown', 'left', 'existing_fusion_retained_dense_abstained'),
    ('right', 'right', 'right', 'existing_fusion_and_dense_region_agree'),
    ('left', 'right', 'unknown', 'existing_fusion_and_dense_region_conflict'),
    ('right', 'left', 'unknown', 'existing_fusion_and_dense_region_conflict'),
    ('unknown', 'unknown', 'unknown', 'both_methods_unavailable_or_abstained'),
])
def test_existing_selection_rule_retains_abstention_and_conflicts(old, dense, expected, reason):
    value = guarded_selection(old, dense, {'local_seed_identity_ambiguous_frames': []}, [20, 22])
    assert value['kicking_foot_candidate'] == expected
    assert value['selection_reason'] == reason


def test_legacy_contact_ambiguity_blocks_only_new_dense_candidate():
    legacy = {'local_seed_identity_ambiguous_frames': [19]}
    value = guarded_selection('unknown', 'right', legacy, [20, 22])
    assert value['kicking_foot_candidate'] == value['dense_region_candidate'] == 'unknown'
    assert value['legacy_contact_identity_ambiguous_frames'] == [19]
    retained = guarded_selection('left', 'right', legacy, [20, 22])
    assert retained['kicking_foot_candidate'] == 'left'


def test_recomputation_preserves_static_foot_abstention_and_supports_both_frozen_labels():
    records, inputs, _, _, _ = observation_fixture()
    inputs['source_manifest_sha256'] = 'manifest'
    inputs['input_receipts'] = {'pose': {'sha256': 'pose'}}
    export = dict(clip_id='event', source_mapping=inputs['source_mapping'], reference_labels_used_for_prediction=False,
        inference_identity=dict(source_sha256='source', pose_sha256='pose', source_manifest_sha256='manifest'),
        contact=dict(contact_bracket_frames=[20, 22], observed_stationary_departure_bracket_frames=[20, 22],
                     contact_associated_track_id=1, seed_ambiguous_frames=[],
                     stationary_ball_observations=[dict(frame=frame, xy=[100., 220.], detection_id=f'{frame}:ball')
                                                   for frame in (18, 19, 20)],
                     departure_ball_observations=[dict(frame=frame, xy=[60., 200.], detection_id=f'{frame}:ball')
                                                  for frame in (22, 23)]))
    new = recompute_dense_foot(records, inputs, export)
    old = recompute_dense_foot(records, inputs, export, method_label='unchanged_paired_shoe_strike_rule_with_longer_VIDEO_history')
    assert new['striking_foot'] == 'unknown' and new['joint_region_frames'] == []
    assert new['physical_contact_verified'] is False
    assert {key: value for key, value in new.items() if key != 'method'} == {key: value for key, value in old.items() if key != 'method'}


@pytest.mark.parametrize('change', ['source', 'actor', 'bracket', 'pose_hash', 'manifest_hash', 'labels'])
def test_recomputed_contact_requires_exact_source_and_original_actor(change):
    records, inputs, _, _, _ = observation_fixture()
    inputs['source_manifest_sha256'] = 'manifest'
    inputs['input_receipts'] = {'pose': {'sha256': 'pose'}}
    export = dict(clip_id='event', source_mapping=inputs['source_mapping'], reference_labels_used_for_prediction=False,
                  inference_identity=dict(source_sha256='source', pose_sha256='pose', source_manifest_sha256='manifest'),
                  contact=dict(contact_bracket_frames=[20, 22], observed_stationary_departure_bracket_frames=[20, 22],
                               contact_associated_track_id=1))
    if change == 'source':
        export['inference_identity']['source_sha256'] = 'extra-video'
    elif change == 'actor':
        export['contact']['contact_associated_track_id'] = 2
    elif change == 'bracket':
        export['contact']['observed_stationary_departure_bracket_frames'] = [21, 23]
    elif change == 'pose_hash':
        export['inference_identity']['pose_sha256'] = 'new-pose'
    elif change == 'manifest_hash':
        export['inference_identity']['source_manifest_sha256'] = 'new-manifest'
    else:
        export['reference_labels_used_for_prediction'] = True
    with pytest.raises(ValueError, match='contact export'):
        recompute_dense_foot(records, inputs, export)
