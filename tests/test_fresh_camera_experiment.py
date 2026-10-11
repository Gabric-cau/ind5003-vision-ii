from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import run_fresh_camera_experiment as camera


class FreshCameraExperimentTest(unittest.TestCase):
    def setUp(self):
        self.pose = {'clip_id': 'clip', 'fps': 25., 'kick_frame': 50, 'selected_track_id': 1,
                     'selected_track_qualified': True, 'shot_start_frame': 0, 'shot_end_frame': 100,
                     'shot_selection': {'selected_shot_index': 2, 'found_qualified_shot': True,
                         'attempts': [{'shot_index': 2, 'qualifies': True, 'scene_rejections': [],
                                       'goal_context': {'supported': True}}]},
                     'track_candidates': [{'track_id': tid, 'field_support': .8} for tid in (1, 2, 3)],
                     'inference_identity': {'source_sha256': 'source'}, '_pose_file_sha256': 'pose',
                     'tracks': []}
        for tid, x in ((1, 50), (2, 50), (3, 200)):
            self.pose['tracks'].append({'track_id': tid, 'observations': [
                {'frame': frame, 'bbox': [x, 0, x+30, 100],
                 'keypoints': [[x+15, 50]]*17, 'keypoint_confidences': [.9]*17}
                for frame in (20, 30, 40)]})

    def test_missing_or_rejected_scene_never_creates_appearance_references(self):
        for field in ('found_qualified_shot',):
            pose = deepcopy(self.pose)
            del pose['shot_selection'][field]
            with patch.object(camera, 'get_frame') as decode, patch.object(camera, 'jersey_hist') as histogram:
                references = camera.appearance_references(Mock(), pose)
            self.assertEqual(references['kicker_histograms'], [])
            self.assertEqual(references['keeper_histograms'], [])
            decode.assert_not_called()
            histogram.assert_not_called()
        self.pose['shot_selection']['attempts'][0]['scene_rejections'] = ['wrong_goal_occupant']
        self.assertFalse(camera.qualified_scene(self.pose))

    def test_camera_rejects_detection_and_wrong_keypoint_models_before_inference(self):
        for task, shape in (('detect', [17, 3]), ('pose', [21, 3]), ('pose', [17, 2])):
            with self.subTest(task=task, shape=shape), self.assertRaisesRegex(ValueError, 'COCO pose model'):
                camera.validate_pose_model(Mock(task=task, model=Mock(yaml={'kpt_shape': shape})))
        checked = camera.validate_pose_model(Mock(task='pose', model=Mock(yaml={'kpt_shape': [17, 3]})))
        self.assertTrue(checked['validated'])

    def test_same_actor_and_duplicate_pose_tracks_cannot_be_keeper_reference(self):
        frame = np.ones((140, 300, 3), np.uint8)
        seen = []
        def occupant(_frame, tracks, _fid):
            seen.append([row['track_id'] for row in tracks])
            return {'seed_track_id': 3}
        with patch.object(camera, 'get_frame', return_value=frame), \
                patch.object(camera, 'jersey_hist', return_value=[1.]), \
                patch.object(camera, 'select_goal_occupant', side_effect=occupant):
            references = camera.appearance_references(Mock(), self.pose, 50)
        self.assertEqual(references['excluded_same_actor_ids'], [1, 2])
        self.assertEqual(seen, [[3]]*3)
        self.assertEqual(len(references['keeper_histograms']), 3)

    def test_empty_reference_window_does_not_start_pose_inference(self):
        with patch.object(camera, 'infer') as infer:
            result = camera.action_sequence(Mock(), Mock(), {'kicker_histograms': [], 'keeper_histograms': []},
                                            0, 50, 25, 'cpu')
        infer.assert_not_called()
        self.assertEqual(result['status'], 'unknown')

    def test_wrong_role_alignment_cannot_label_goalkeeper_view_slow_motion(self):
        action = {'roles': {'kicker': ['kicker'], 'keeper': ['keeper']}}
        with patch.object(camera, 'align', side_effect=[{'status': 'relative_slowdown_candidate'}, {'status': 'unknown'}]):
            result = camera.speed_evidence(action, action, 'back_of_goalkeeper')
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['target_role'], 'keeper')
        self.assertIsNone(result['absolute_speed_factor'])

    def contact_file(self, path, **changes):
        value = {'clip_id': 'clip', 'inference_identity': {'source_sha256': 'source', 'pose_sha256': 'pose'},
                 'new_neural_inference': True, 'reference_labels_used_for_prediction': False,
                 'independent_contact_search': {'selected_candidate': {
                     'contact_frame': 35, 'contact_bracket_frames': [34, 37],
                     'contact_timing_status': 'bounded_stationary_departure_candidate',
                     'contact_associated_track_id': 3}},
                 'ball_departure': {'status': 'supported_association_candidate', 'inferred_release_frame': 37}}
        value.update(changes)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_fresh_contact_replaces_late_pose_time_and_actor_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'clip.json'
            self.contact_file(path)
            anchor = camera.contact_anchor(self.pose, path)
        self.assertEqual(anchor['frame'], 35)
        self.assertEqual(anchor['contact_bracket_frames'], [34, 37])
        self.assertEqual(anchor['pose_proposal_for_comparison_only'], 50)
        self.assertEqual(anchor['contact_associated_track_id'], 3)
        self.assertTrue(anchor['actor_reselected_by_ball_evidence'])
        self.assertFalse(anchor['human_verified'])

    def test_departure_backprojection_cannot_replace_unknown_canonical_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'clip.json'
            self.contact_file(path, independent_contact_search={'selected_candidate': {
                'contact_frame': 35, 'contact_associated_track_id': 3,
                'contact_timing_status': 'unknown', 'contact_bracket_frames': None}})
            self.assertIsNone(camera.contact_anchor(self.pose, path)['frame'])

    def test_missing_or_abstained_fresh_contact_does_not_fall_back_to_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'clip.json'
            self.assertIsNone(camera.contact_anchor(self.pose, path)['frame'])
            self.contact_file(path, independent_contact_search={'selected_candidate': None})
            self.assertIsNone(camera.contact_anchor(self.pose, path)['frame'])

    def test_contact_from_different_pose_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'clip.json'
            self.contact_file(path, inference_identity={'source_sha256': 'source', 'pose_sha256': 'stale-pose'})
            with self.assertRaisesRegex(ValueError, 'different pose'):
                camera.contact_anchor(self.pose, path)

    def test_complete_camera_timeline_rejects_gap(self):
        scan = {'frame_count': 100, 'decoded_frames': 100, 'segments': [
            {'start_frame': 0, 'end_frame_exclusive': 40, 'sample_frame_ids': [10]},
            {'start_frame': 41, 'end_frame_exclusive': 100, 'sample_frame_ids': [50]}]}
        with self.assertRaisesRegex(ValueError, 'gap or overlap'):
            camera.validate_timeline(scan, {'frame_count': 100})


if __name__ == '__main__':
    unittest.main()
