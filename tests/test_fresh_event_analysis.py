from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src import fresh_event_analysis as fresh
from scripts.run_fresh_ball_experiment import detect_event, propose_crops, validate_detection_model


class FreshEventAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.pose = {'clip_id': 'clip', 'fps': 25., 'kick_frame': 40, 'selected_track_id': 1,
                     'selected_track_qualified': True, 'shot_start_frame': 10, 'shot_end_frame': 100,
                     'shot_selection': {'selected_shot_index': 1, 'found_qualified_shot': True, 'attempts': [
                         {'shot_index': 1, 'qualifies': True, 'scene_rejections': [], 'goal_context': {'supported': True}}]},
                     'track_candidates': [{'track_id': 1, 'field_support': .75}], 'tracks': []}
        self.source = {'fps': 25., 'frame_count': 200, 'original_slice_start_frame_in_clip': 110,
                       'original_slice_end_frame_in_clip': 180}

    def test_wrong_model_task_is_rejected_before_source_inference(self):
        validate_detection_model(SimpleNamespace(task='detect', names={0: 'person', 32: 'sports ball'}))
        with self.assertRaisesRegex(ValueError, 'COCO detection model'):
            validate_detection_model(SimpleNamespace(task='pose', names={0: 'person'}))
        with self.assertRaisesRegex(ValueError, 'COCO detection model'):
            validate_detection_model(SimpleNamespace(task='detect', names={0: 'person', 32: 'wrong object'}))

    def test_contact_window_respects_camera_cut_without_using_nominal_slice_as_truth(self):
        interval, reason = fresh.contact_window(self.pose, self.source)
        self.assertEqual(interval, (10, 100))
        self.assertEqual(reason, 'entire_selected_shot_independent_ball_contact_search')
        self.pose['kick_frame'] = 98
        self.assertEqual(fresh.contact_window(self.pose, self.source)[0], (10, 100))
        self.pose['kick_frame'] = None
        self.assertEqual(fresh.contact_window(self.pose, self.source)[0], (10, 100))

    def test_abstains_on_wrong_scene_or_actor(self):
        cases = [('selected_track_id', None, 'no_selected_kicker_track'),
                 ('selected_track_qualified', False, 'unqualified_automatic_kicker_track'),
                 ('shot_end_frame', 201, 'selected_shot_outside_source_clip')]
        for field, value, reason in cases:
            pose = {**self.pose, field: value}
            with self.subTest(field=field, value=value):
                self.assertEqual(fresh.contact_window(pose, self.source), (None, reason))
        self.pose['shot_selection']['attempts'][0]['goal_context']['supported'] = False
        self.assertEqual(fresh.contact_window(self.pose, self.source)[1], 'selected_shot_goal_context_not_established')
        self.pose['shot_selection']['attempts'][0]['goal_context']['supported'] = True
        self.pose['track_candidates'][0]['field_support'] = .01
        self.assertEqual(fresh.contact_window(self.pose, self.source)[1], 'selected_actor_field_support_not_established')

    def test_fallback_pose_is_rejected_even_if_its_track_qualified(self):
        self.pose['shot_selection']['found_qualified_shot'] = False
        self.assertEqual(fresh.contact_window(self.pose, self.source)[1], 'selected_shot_qualification_not_established')

    def test_source_pose_fps_mismatch_is_integrity_failure(self):
        with self.assertRaisesRegex(ValueError, 'fps mismatch'):
            fresh.contact_window(self.pose, {**self.source, 'fps': 50.})

    def test_unqualified_actor_does_not_start_detector_or_decode(self):
        model = Mock()
        pose = {**self.pose, 'selected_track_qualified': False}
        with patch('scripts.run_fresh_ball_experiment._read_window') as decode:
            cache, frames, reason = detect_event(Path('/unused.mp4'), pose, self.source, model,
                                    {'pre_sec': .45, 'post_sec': 1.5})
        self.assertIsNone(cache)
        self.assertEqual(frames, {})
        self.assertEqual(reason, 'unqualified_automatic_kicker_track')
        decode.assert_not_called()
        model.predict.assert_not_called()

    def test_ball_and_person_classes_and_crop_offsets_are_separate(self):
        boxes = SimpleNamespace(xyxy=np.array([[1, 2, 7, 8], [10, 10, 30, 40], [2, 2, 102, 102]], dtype=float),
                                conf=np.array([.8, .9, .2]), cls=np.array([32, 0, 32]))
        frame = np.full((200, 240, 3), 255, np.uint8)
        before = boxes.xyxy.copy()
        balls, people, raw = fresh.partition_predictions([7], [SimpleNamespace(boxes=boxes)], [frame],
                                                         offset=(100, 50), mode='goal_roi')
        self.assertEqual(len(raw['7']), 3)
        self.assertEqual(len(balls['7']), 1)
        self.assertEqual((balls['7'][0]['x'], balls['7'][0]['y']), (104., 55.))
        self.assertEqual(people['7'][0]['bbox'], [110., 60., 130., 90.])
        np.testing.assert_array_equal(boxes.xyxy, before)

    def test_duplicate_ball_views_keep_source_modes_without_changing_input(self):
        rows = [{'x': 10., 'y': 20., 'width': 6., 'confidence': .7, 'sources': ['full']},
                {'x': 11., 'y': 20., 'width': 6., 'confidence': .5, 'sources': ['goal_roi']}]
        before = deepcopy(rows)
        merged = fresh.merge_ball_candidates(rows, 9)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['sources'], ['full', 'goal_roi'])
        self.assertEqual(rows, before)

    def test_automatic_crops_use_supported_goal_and_selected_ankles(self):
        pose = deepcopy(self.pose)
        pose['tracks'] = [{'track_id': 1, 'observations': [{'frame': 40,
                           'keypoints': [[350, 600]]*17, 'keypoint_confidences': [.9]*17}]}]
        goals = {'40': {'selected': {'bbox': [700, 150, 1000, 400]}}}
        crops = dict(propose_crops(pose, goals, 40, 1280, 720))
        self.assertIn('goal_roi', crops)
        self.assertIn('automatic_ankle_roi', crops)
        for bbox in crops.values():
            self.assertTrue(0 <= bbox[0] < bbox[2] <= 1280)
            self.assertTrue(0 <= bbox[1] < bbox[3] <= 720)

    def test_missing_geometry_produces_unknown_not_a_last_detection_endpoint(self):
        frames = {fid: np.full((80, 120, 3), 255, np.uint8) for fid in range(3)}
        cache = {'clip_id': 'clip', 'source_sha256': 'video', 'pose_sha256': 'pose',
                 'kick_frame': 1, 'start_frame': 0, 'end_frame_exclusive': 3, 'fps': 25.,
                 'detections': {str(f): [] for f in frames}, 'people': {str(f): [] for f in frames},
                 'camera': {'transforms': {str(f): np.eye(3).tolist() for f in frames}},
                 'goal_proposals': {str(f): {'selected': None} for f in frames}}
        timing = {f: {'motion_time': f, 'duplicate_frame': False} for f in frames}
        with tempfile.TemporaryDirectory() as temporary, patch.object(fresh, 'source_time_indices', return_value=timing), \
                patch.object(fresh, 'motion_blobs', return_value={}):
            result = fresh.analyze_fresh_ball(cache, self.pose, frames, Path(temporary))
        self.assertEqual(result['rule_decision']['rule_outcome'], 'unknown')
        self.assertIsNone(result['final_position'])
        self.assertEqual(result['trajectory'], [])
        self.assertFalse(result['ground_plane_trajectory_established'])
        self.assertFalse(result['accuracy_validated'])

    def test_missing_pose_goal_role_does_not_fall_back_to_a_different_actor(self):
        with patch.object(fresh, 'analyze_goalkeeper_result') as analyze:
            result = fresh.analyze_fresh_keeper(self.pose, {'goal_proposals': {}, 'kick_frame': 40})
        self.assertEqual(result['dive_direction'], 'unknown')
        self.assertEqual(result['dive_height'], 'unknown')
        analyze.assert_not_called()

    def test_global_ball_contact_search_can_precede_wrong_pose_kick_by_more_than_a_second(self):
        frames = {f: np.full((160, 300, 3), 255, np.uint8) for f in range(40)}
        detections = {}
        for fid in frames:
            x = 50. if fid <= 8 else 50.+25*(fid-8)
            detections[str(fid)] = [{'frame': fid, 'x': x, 'y': 80., 'width': 6., 'height': 6.,
                          'confidence': .8, 'sources': ['full'], 'detection_id': f'{fid}:0'}] if x < 290 else []
        pose = deepcopy(self.pose)
        pose['kick_frame'] = 38
        observations = []
        for fid in frames:
            ankle_x = max(50., 100.-6*fid)
            observations.append({'frame': fid, 'bbox': [ankle_x-20, 0, ankle_x+20, 100],
                           'keypoints': [[ankle_x, 80.]]*17, 'keypoint_confidences': [.9]*17})
        pose['tracks'] = [{'track_id': 1, 'observations': observations}]
        cache = {'fps': 25., 'start_frame': 0, 'end_frame_exclusive': 40,
                 'video_path': '/unused.mp4', 'detections': detections,
                 'goal_proposals': {str(f): {'selected': {'corners': [[270, 30], [295, 30], [295, 100], [270, 100]]}} for f in frames},
                 'people': {str(f): [] for f in frames}}
        def camera(_video, start, end, **_kwargs):
            return {'transforms': {str(f): np.eye(3).tolist() for f in range(start, end)}}
        with patch.object(fresh, 'estimate_background_motion', side_effect=camera):
            search = fresh.search_fresh_contact(cache, pose, frames)
        selected = search['selected_candidate']
        self.assertIsNotNone(selected)
        self.assertLess(selected['contact_frame'], pose['kick_frame']-25)
        self.assertTrue(selected['ball_departure_frames'])
        self.assertEqual(search['pose_kick_proposal_for_comparison_only'], 38)
        self.assertEqual(search['full_selected_shot_search'], [0, 40])

    def actor_search(self, scores, same_person=False):
        pose = deepcopy(self.pose)
        pose['track_candidates'] = [{'track_id': tid, 'field_support': .8 if tid != 4 else .01}
                                    for tid in (1, 2, 3, 4)]
        pose['tracks'] = [{'track_id': tid, 'observations': []} for tid in (1, 2, 3, 4)]
        pose['goal_line_role_exclusions'] = {'excluded_track_ids': [3]}
        frames = {f: np.full((100, 100, 3), 255, np.uint8) for f in range(12)}
        samples = {tid: [{'frame': f, 'ankles': [[50, 50]], 'height': tid*100.} for f in frames]
                   for tid in (1, 2, 3, 4)}
        cache = {'fps': 25., 'start_frame': 0, 'end_frame_exclusive': 12, 'video_path': '/unused',
                 'people': {str(f): [] for f in frames},
                 'detections': {str(f): [{'x': 50., 'y': 50., 'width': 6., 'height': 6.,
                                         'confidence': .8, 'detection_id': str(f)}] for f in frames}}
        def score(_balls, actor, _transforms, *_args, **_kwargs):
            tid = int(actor[0]['height']/100)
            values = scores[tid] if isinstance(scores[tid], list) else [(5, scores[tid])]
            return [{'contact_frame': frame, 'score': value, 'ball_departure_frames': [frame+1, frame+2],
                     'stationary_ball_observations': [{'frame': f, 'detection_id': str(f)} for f in range(frame-2, frame+1)],
                     'ankle_approach_supported': True, 'warnings': []} for frame, value in values]
        def departure(proposal, *_args):
            frame = proposal['contact_frame']
            return {'status': 'supported_goalward_departure_candidate',
                    'selected_hypothesis': {'release_frame_prediction': frame,
                         'moving_observations': [{'frame': f, 'detection_id': str(f)} for f in (frame+1, frame+2)]}}
        camera = {'transforms': {str(f): np.eye(3).tolist() for f in frames}}
        with patch.object(fresh, '_all_track_samples', return_value=samples), \
                patch.object(fresh, 'estimate_background_motion', return_value=camera), \
                patch.object(fresh, 'score_contact_candidates', side_effect=score), \
                patch.object(fresh, 'goalward_departure_support', side_effect=departure), \
                patch.object(fresh, '_same_person', return_value=same_person):
            return fresh.search_fresh_contact(cache, pose, frames), pose

    def test_ball_contact_can_reselect_onfield_actor_and_excludes_goal_line_and_offfield(self):
        search, pose = self.actor_search({1: .75, 2: .95})
        selected = search['selected_candidate']
        self.assertEqual(search['eligible_actor_ids'], [1, 2])
        self.assertEqual(selected['contact_associated_track_id'], 2)
        self.assertTrue(selected['actor_reselected_by_ball_evidence'])
        event_pose = fresh.contact_pose(pose, selected)
        self.assertEqual(event_pose['selected_track_id'], 2)
        self.assertEqual(event_pose['kick_frame'], 5)
        self.assertEqual(pose['selected_track_id'], 1)
        self.assertEqual(pose['kick_frame'], 40)

    def test_different_people_with_close_contact_scores_abstain(self):
        search, _ = self.actor_search({1: .93, 2: .95})
        self.assertIsNone(search['selected_candidate'])
        self.assertTrue(search['actor_identity_ambiguous'])

    def test_duplicate_tracks_of_same_person_do_not_create_false_identity_ambiguity(self):
        search, _ = self.actor_search({1: .93, 2: .95}, same_person=True)
        self.assertIsNotNone(search['selected_candidate'])
        self.assertFalse(search['actor_identity_ambiguous'])

    def test_stronger_late_walking_contact_does_not_replace_same_actor_first_departure(self):
        search, _ = self.actor_search({1: [(3, .67), (8, .98)], 2: .70})
        self.assertEqual(search['selected_candidate']['contact_frame'], 3)
        self.assertEqual(search['selected_candidate']['actor_support_score'], .98)

    def test_canonical_contact_uses_actual_stationary_observation_after_early_proximity_and_backprojection(self):
        proposal = {'contact_frame': 228, 'interval_frames': [225, 233], 'warnings': [],
                    'stationary_ball_observations': [{'frame': f, 'detection_id': str(f)} for f in (225, 228, 230, 231)]}
        moving = [{'frame': f, 'detection_id': str(f), 'xy': [f, 0]} for f in (234, 237)]
        fresh.canonicalize_contact_timing(proposal, {'release_frame_prediction': 230, 'moving_observations': moving}, moving, 30.)
        self.assertEqual(proposal['contact_frame'], 232)
        self.assertEqual(proposal['interval_frames'], [231, 234])
        self.assertEqual(proposal['contact_bracket_frames'], [231, 234])
        self.assertEqual(proposal['scored_ankle_proximity_frame'], 228)
        self.assertTrue(proposal['departure_backprojection_precedes_actual_stationary_observation'])
        self.assertFalse(proposal['exact_contact_frame_established'])

    def test_missing_or_wide_transition_abstains_instead_of_promoting_proximity_frame(self):
        for stationary in ([1, 2], [1, 2, 3], [1, 2, 8]):
            proposal = {'contact_frame': 3, 'stationary_ball_observations': [
                {'frame': f, 'detection_id': str(f)} for f in stationary]}
            moving = [{'frame': f, 'detection_id': str(f)} for f in (8, 10)]
            fresh.canonicalize_contact_timing(proposal, {'release_frame_prediction': 3, 'moving_observations': moving}, moving, 30.)
            self.assertEqual(proposal['contact_timing_status'], 'unknown')
            self.assertIsNone(proposal['contact_bracket_frames'])

    def test_rebounded_contact_time_uses_final_candidate_frame_and_source_fps(self):
        for fps in (50., 30000/1001):
            with self.subTest(fps=fps):
                proposal = {'contact_frame': 205, 'contact_time_sec': 205/fps,
                            'stationary_ball_observations': [
                                {'frame': f, 'detection_id': str(f)} for f in (211, 212, 213)]}
                moving = [{'frame': f, 'detection_id': str(f)} for f in (217, 218)]
                fresh.canonicalize_contact_timing(proposal,
                    {'release_frame_prediction': 216.4, 'moving_observations': moving}, moving, fps)
                self.assertEqual(proposal['contact_bracket_frames'], [213, 217])
                self.assertEqual(proposal['contact_frame'], 215)
                self.assertAlmostEqual(proposal['contact_time_sec'], 215/fps, places=12)
                self.assertEqual(proposal['scored_ankle_proximity_frame'], 205)
                self.assertEqual(proposal['scored_ankle_proximity_time_sec'], 205/fps)
                self.assertNotEqual(proposal['contact_time_sec'], proposal['scored_ankle_proximity_time_sec'])

    def test_continuity_seed_cannot_switch_to_another_nearby_ball_detection(self):
        frames = {fid: np.full((80, 120, 3), 255, np.uint8) for fid in range(3)}
        detections = {str(fid): [
            {'frame': fid, 'detection_id': f'{fid}:seed', 'x': 50., 'y': 50., 'width': 6., 'height': 6.,
             'confidence': .7, 'stabilized_xy': [50., 50.], 'sources': ['full']},
            {'frame': fid, 'detection_id': f'{fid}:other', 'x': 52., 'y': 50., 'width': 6., 'height': 6.,
             'confidence': .95, 'stabilized_xy': [52., 50.], 'sources': ['full']}] for fid in frames}
        cache = {'clip_id': 'clip', 'source_sha256': 'video', 'pose_sha256': 'pose',
                 'kick_frame': 1, 'start_frame': 0, 'end_frame_exclusive': 3, 'fps': 25.,
                 'detections': detections, 'people': {str(f): [] for f in frames},
                 'camera': {'transforms': {str(f): np.eye(3).tolist() for f in frames}},
                 'goal_proposals': {str(f): {'selected': None} for f in frames},
                 'contact': {'stationary_ball_observations': [{'frame': f, 'detection_id': f'{f}:seed'} for f in frames]}}
        timing = {f: {'motion_time': f, 'duplicate_frame': False} for f in frames}
        with tempfile.TemporaryDirectory() as temporary, patch.object(fresh, 'source_time_indices', return_value=timing), \
                patch.object(fresh, 'motion_blobs', return_value={}), \
                patch.object(fresh, 'choose_seed', return_value=([], {'pixel_mask_used_for_seed': True})) as choose:
            fresh.analyze_fresh_ball(cache, self.pose, frames, Path(temporary))
        selected = choose.call_args.args[0]
        self.assertEqual([row['detection_id'] for rows in selected.values() for row in rows],
                         ['0:seed', '1:seed', '2:seed'])


if __name__ == '__main__':
    unittest.main()
