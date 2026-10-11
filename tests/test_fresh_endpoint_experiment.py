from pathlib import Path
import tempfile
import unittest

from scripts.run_fresh_ball_experiment import save
from scripts.run_fresh_endpoint_experiment import boundary_evidence, fixed_source_prefix


class FreshEndpointExperimentTest(unittest.TestCase):
    def test_fixed_prefix_cannot_switch_to_a_closer_higher_confidence_ball(self):
        candidate = {'contact_bracket_frames': [10, 13],
                     'stationary_ball_observations': [{'frame': f, 'detection_id': f'{f}:seed'} for f in (8, 9, 10)],
                     'departure_ball_observations': [{'frame': f, 'detection_id': f'{f}:ball'} for f in (13, 15)]}
        detections = {}
        for frame, label in ((10, 'seed'), (13, 'ball'), (15, 'ball')):
            detections[str(frame)] = [
                {'frame': frame, 'detection_id': f'{frame}:{label}', 'stabilized_xy': [20, 20], 'confidence': .2},
                {'frame': frame, 'detection_id': f'{frame}:other', 'stabilized_xy': [21, 20], 'confidence': .99}]
        prefix = fixed_source_prefix({'detections': detections}, candidate)
        self.assertEqual([row['detection_id'] for row in prefix], ['10:seed', '13:ball', '15:ball'])
        detections['13'][0]['stabilized_xy'] = None
        self.assertIsNone(fixed_source_prefix({'detections': detections}, candidate))

    def test_view_change_alone_cannot_turn_scene_cut_into_replay(self):
        source = {'clip_id': 'clip', 'clip_sha256': 'source'}
        pose = {'shot_end_frame': 50, 'shot_end_reason': 'scene_cut'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'clip.json'
            camera = {'clip_id': 'clip', 'source_mapping': source, 'new_neural_inference': True,
                      'segments': [{'start_frame': 50, 'end_frame_exclusive': 100,
                                    'view_candidate': 'back_of_goalkeeper', 'playback_evidence': {'status': 'unknown'}}]}
            save(path, camera)
            self.assertEqual(boundary_evidence(pose, path, source)['reason'], 'scene_cut')
            camera['segments'][0]['playback_evidence']['status'] = 'slow_motion_candidate'
            save(path, camera)
            evidence = boundary_evidence(pose, path, source)
            self.assertEqual(evidence['reason'], 'replay_cut_candidate')
            self.assertFalse(evidence['replay_verified'])

    def test_camera_from_another_source_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'clip.json'
            save(path, {'clip_id': 'clip', 'source_mapping': {'clip_sha256': 'other'},
                        'new_neural_inference': True})
            with self.assertRaisesRegex(ValueError, 'source-bound'):
                boundary_evidence({'shot_end_reason': 'scene_cut'}, path, {'clip_id': 'clip', 'clip_sha256': 'source'})


if __name__ == '__main__':
    unittest.main()
