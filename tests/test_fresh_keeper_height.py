import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.run_fresh_keeper_height_experiment import load_bound_masks
from src.fresh_keeper_height import (analyze_height, analyse_height_phases,
                                     connected_camera_prefix, ground_reference,
                                     pose_only_measurement, observed_image_plane_signal, _mask_guard)


def observation(frame, peak=.6):
    points = np.tile([100., 140.], (17, 1))
    if frame < 15:
        points[[5, 6]] = [[126, 110], [134, 110]]
        points[[11, 12]] = [[126, 150], [134, 150]]
        points[[15, 16]] = [[126, 200], [134, 200]]
        box = [115, 100, 145, 200]
    else:
        height = .20 if frame >= 26 else peak
        y = 200-height*100
        points[[5, 6]] = [[175, y-2], [175, y+2]]
        points[[11, 12]] = [[145, y-2], [145, y+2]]
        points[[15, 16]] = [[140, 200 if frame >= 26 else 190], [146, 200 if frame >= 26 else 190]]
        box = [135, y-10, 185, 200]
    return dict(frame=frame, bbox=box, keypoints=points.tolist(),
                keypoint_confidences=[.95]*17, association_source_track_id=3)


def inputs(peak=.6):
    rows = [observation(frame, peak) for frame in range(36)]
    pose = dict(fps=30., frame_width=400, frame_height=300,
                shot_start_frame=0, shot_end_frame=36, selected_track_id=0,
                selected_track_qualified=True, tracks=[dict(track_id=3, observations=rows)],
                shot_selection=dict(found_qualified_shot=True, selected_shot_index=0,
                                    attempts=[dict(shot_index=0, qualifies=True, scene_rejections=[])]))
    goal = dict(selected=dict(corners=[[0, 50], [300, 50], [300, 200], [0, 200]], bbox=[0, 50, 300, 200]))
    cache = dict(kick_frame=15, fps=30., goal_proposals={str(frame): copy.deepcopy(goal) for frame in range(36)},
                 contact=dict(contact_frame=15, contact_associated_track_id=0,
                              ball_departure_frames=[17, 18], camera_chain_measured=True))
    camera = dict(shot_interval=[0, 36], transforms={str(frame): np.eye(3).tolist() for frame in range(36)})
    role = dict(status='candidate_supported', seed_track_id=3, same_person_seed_ids=[3])
    return pose, cache, camera, role


def measurements(peak=.6):
    pose, cache, camera, _ = inputs(peak)
    reference, reason = ground_reference(pose['tracks'][0]['observations'], cache['goal_proposals'],
                                         camera['transforms'], 15, 30.)
    assert reason is None
    rows = [pose_only_measurement(row, camera['transforms'][str(row['frame'])], reference)
            for row in pose['tracks'][0]['observations']]
    return rows, reference


class HeightPhaseTests(unittest.TestCase):
    def test_torso_midpoint_is_not_hip_height(self):
        rows, _ = measurements()
        self.assertAlmostEqual(rows[0]['hip_clearance_body'], .5)
        self.assertAlmostEqual(rows[0]['torso_clearance_body'], .7)

    def test_low_mid_high_are_supported_before_landing(self):
        for peak, expected in ((.25, 'low'), (.6, 'mid'), (.9, 'high')):
            rows, reference = measurements(peak)
            result = analyse_height_phases(rows, 15, 30., reference)
            self.assertEqual(result['height_candidate'], expected)
            self.assertLess(max(result['supporting_frames']), result['landing_frame'])
            self.assertTrue(result['complete_first_dive_observed'])
            self.assertIsNone(result['physical_height_metres'])

    def test_cut_before_landing_does_not_claim_low_or_high(self):
        rows, reference = measurements(.9)
        result = analyse_height_phases(rows[:24], 15, 30., reference)
        self.assertFalse(result['supported'])
        self.assertIn('grounded_landing_not_observed_in_connected_first_dive', result['reasons'])

    def test_missing_pose_cannot_jump_to_recovery(self):
        rows, reference = measurements()
        rows = [row for row in rows if not 19 <= row['frame'] <= 28]
        self.assertFalse(analyse_height_phases(rows, 15, 30., reference)['supported'])

    def test_one_outlier_is_not_height_peak(self):
        rows, reference = measurements(.6)
        rows[21]['torso_clearance_body'] = 1.4
        self.assertEqual(analyse_height_phases(rows, 15, 30., reference)['height_candidate'], 'mid')

    def test_unknown_ankles_are_not_ground_contact_zero(self):
        rows, reference = measurements()
        for row in rows[26:]:
            row['bottom_clearance_body'] = None
        self.assertFalse(analyse_height_phases(rows, 15, 30., reference)['supported'])

    def test_forward_depth_or_wrong_ankle_breaks_fixed_ground_reference(self):
        rows, reference = measurements()
        rows[20]['bottom_clearance_body'] = -.3
        result = analyse_height_phases(rows, 15, 30., reference)
        self.assertFalse(result['supported'])
        self.assertIn('depth_motion_or_ankle_error_invalidates_fixed_ground_reference', result['reasons'])

    def test_boundary_remains_unknown(self):
        rows, reference = measurements(.35)
        self.assertFalse(analyse_height_phases(rows, 15, 30., reference)['supported'])

    def test_collapse_in_place_is_not_a_lateral_dive(self):
        rows, reference = measurements()
        for row in rows[15:]:
            row['torso_center'][0] = 130
        self.assertFalse(analyse_height_phases(rows, 15, 30., reference)['supported'])

    def test_squatting_preparation_is_not_grounded_landing(self):
        rows, reference = measurements()
        for row in rows[15:]:
            row.update(torso_lean_degrees=50, visible_body_extent_body=.9)
        self.assertFalse(analyse_height_phases(rows, 15, 30., reference)['supported'])

    def test_scale_and_measurements_share_camera_coordinates(self):
        pose, cache, camera, _ = inputs()
        matrix = [[.5, 0, 8], [0, .5, 12], [0, 0, 1]]
        camera['transforms'] = {str(frame): matrix for frame in range(36)}
        reference, reason = ground_reference(pose['tracks'][0]['observations'], cache['goal_proposals'],
                                             camera['transforms'], 15, 30.)
        self.assertIsNone(reason)
        self.assertAlmostEqual(reference['body_scale_px'], 50)
        sample = pose_only_measurement(pose['tracks'][0]['observations'][16], matrix, reference)
        self.assertAlmostEqual(sample['torso_clearance_body'], .6)


class SourceGuardTests(unittest.TestCase):
    def test_full_shot_camera_required(self):
        _, _, camera, _ = inputs()
        camera['shot_interval'] = [5, 25]
        with self.assertRaisesRegex(ValueError, 'full selected shot'):
            connected_camera_prefix(camera, 0, 36)

    def test_camera_never_reconnects_after_failed_pair(self):
        _, _, camera, _ = inputs()
        camera['pairs'] = [dict(frame=20, accepted=False)]
        transforms, stop = connected_camera_prefix(camera, 0, 36)
        self.assertEqual(stop, 20)
        self.assertIsNone(transforms['35'])

    def test_pose_only_candidate_is_distinct_from_strict_mask_candidate(self):
        pose, cache, camera, role = inputs(.9)
        result = analyze_height(pose, cache, camera=camera, role=role)
        self.assertEqual(result['ordinal_pose_height_candidate'], 'high')
        self.assertEqual(result['strict_height_candidate'], 'unknown')
        self.assertFalse(result['accuracy_validated'])

    def test_rejected_fallback_scene_cannot_produce_height(self):
        pose, cache, camera, role = inputs()
        pose['shot_selection']['attempts'][0]['scene_rejections'] = ['missing_goal_context']
        result = analyze_height(pose, cache, camera=camera, role=role)
        self.assertEqual(result['reason'], 'selected_shot_scene_not_qualified')

    def test_keeper_cannot_be_independent_striker(self):
        pose, cache, camera, role = inputs()
        cache['contact']['contact_associated_track_id'] = 3
        self.assertEqual(analyze_height(pose, cache, camera=camera, role=role)['reason'],
                         'goal_line_role_conflicts_with_independent_striker_actor')

    def test_weak_torso_never_becomes_measurement(self):
        rows, reference = measurements()
        row = observation(18)
        row['keypoint_confidences'][12] = .1
        self.assertIsNone(pose_only_measurement(row, np.eye(3), reference))


class ObservedSignalTests(unittest.TestCase):
    def test_cut_observation_has_signal_and_explicit_censoring(self):
        rows, reference = measurements(.6)
        rows = rows[:24]
        phase = analyse_height_phases(rows, 15, 30., reference)
        signal = observed_image_plane_signal(rows, phase, 15, 30., reference)
        self.assertEqual(signal['candidate'], 'mid')
        self.assertTrue(signal['censored'])
        self.assertFalse(signal['complete_first_dive_observed'])
        self.assertFalse(phase['supported'])
        self.assertIsNone(signal['physical_height_metres'])

    def test_later_recovery_cannot_supply_observed_peak_after_gap(self):
        rows, reference = measurements(.6)
        rows = [row for row in rows if not 16 <= row['frame'] <= 24]
        phase = analyse_height_phases(rows, 15, 30., reference)
        signal = observed_image_plane_signal(rows, phase, 15, 30., reference)
        self.assertEqual(signal['candidate'], 'unknown')
        self.assertIsNone(signal['observed_maximum_torso_clearance_body'])

    def test_depth_failure_is_exposed_not_reclassified_as_ground_truth(self):
        rows, reference = measurements(.6)
        rows[20]['bottom_clearance_body'] = -.3
        phase = analyse_height_phases(rows, 15, 30., reference)
        signal = observed_image_plane_signal(rows, phase, 15, 30., reference)
        self.assertTrue(signal['reference_depth_assumption_failed'])
        self.assertEqual(signal['candidate'], 'mid')
        self.assertFalse(phase['supported'])
        self.assertTrue(signal['category_is_observed_signal_not_complete_event_peak'])


class BodyMaskTests(unittest.TestCase):
    def guard(self, mask, row=None, first=False):
        return _mask_guard(mask, row, first=first, previous=None, scale=100,
                           initial_area=int(mask.sum()), excluded=None)

    def test_wrong_first_person_seed_is_rejected(self):
        mask = np.zeros((300, 400), bool)
        mask[50:90, 200:230] = True
        self.assertEqual(self.guard(mask, observation(0), True)['reason'],
                         'first_body_mask_does_not_cover_confident_keeper_torso')

    def test_source_edge_truncation_is_rejected(self):
        mask = np.zeros((300, 400), bool)
        mask[100:201, 0:30] = True
        self.assertFalse(self.guard(mask)['valid'])

    def test_roi_edge_is_checked_even_inside_full_source(self):
        mask = np.zeros((300, 400), bool)
        mask[100:201, 50:80] = True
        result = _mask_guard(mask, None, first=False, previous=None, scale=100,
            initial_area=int(mask.sum()), excluded=None, roi=[50, 50, 250, 250])
        self.assertEqual(result['reason'], 'body_mask_truncated_at_inference_roi_edge')

    def test_disconnected_other_object_is_rejected(self):
        mask = np.zeros((300, 400), bool)
        mask[100:201, 85:116] = True
        mask[100:180, 200:225] = True
        self.assertEqual(self.guard(mask)['reason'], 'body_mask_disconnected_components')

    def test_strict_candidate_requires_contiguous_masks_through_landing(self):
        pose, cache, camera, role = inputs()
        masks = {}
        for frame, row in enumerate(pose['tracks'][0]['observations']):
            mask = np.zeros((300, 400), bool)
            if frame < 15:
                mask[100:201, 115:146] = True
            elif frame < 26:
                mask[135:192, 135:186] = True
            else:
                mask[175:201, 135:186] = True
            masks[frame] = mask
        result = analyze_height(pose, cache, camera=camera, role=role, masks=masks)
        self.assertEqual(result['strict_height_candidate'], 'mid')
        del masks[23]
        result = analyze_height(pose, cache, camera=camera, role=role, masks=masks)
        self.assertEqual(result['strict_height_candidate'], 'unknown')
        self.assertEqual(result['mask_prefix_stop_reason'], 'missing_body_mask_breaks_identity_prefix')

    def test_mask_manifest_binds_source_role_and_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            clip = base/'clip'
            clip.mkdir()
            image_path = clip/'mask.png'
            cv2.imwrite(str(image_path), np.zeros((30, 40), np.uint8))
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            manifest = dict(schema_version='fresh-keeper-height-mask-v1', clip_id='clip',
                source_sha256='a'*64, pose_sha256='b'*64, event_cache_sha256='c'*64,
                keeper_track_id=3, coordinate_system='full_source_pixels', frame_height=30, frame_width=40,
                new_neural_inference=True, reference_labels_used_for_prediction=False,
                model_sha256='d'*64, seed_frame=2, masks=[dict(frame=2, path='mask.png', sha256=sha)])
            path = clip/'mask_manifest.json'
            path.write_text(json.dumps(manifest))
            kwargs = dict(clip_id='clip', source_sha='a'*64, pose_sha='b'*64,
                          event_sha='c'*64, keeper_id=3, shape=(30, 40))
            self.assertEqual(list(load_bound_masks(base, **kwargs)[0]), [2])
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_bound_masks(base, **{**kwargs, 'keeper_id': 9})
            cv2.imwrite(str(image_path), np.ones((30, 40), np.uint8))
            with self.assertRaisesRegex(ValueError, 'file hash mismatch'):
                load_bound_masks(base, **kwargs)


if __name__ == '__main__':
    unittest.main()
