from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.run_fresh_ball_experiment import save
from src.final_results import sha256
from src.fresh_cache_provenance import validate_cached_neural_producer


class FreshCacheProvenanceTest(unittest.TestCase):
    def bundle(self, root):
        identity = {key: key for key in ('source_sha256', 'pose_sha256', 'source_manifest_sha256', 'model_sha256')}
        cache = {'clip_id': 'clip', 'inference_identity': identity,
                 'source_sha256': identity['source_sha256'], 'fresh_model_inference': True}
        cache_path = root/'detections/clip.json'
        save(cache_path, cache)
        receipt = {'clip_id': 'clip', 'inference_identity': identity,
                   'new_neural_inference': True, 'reference_labels_used_for_prediction': False,
                   'source_cache_sha256': sha256(cache_path)}
        receipt_path = root/'producer_results/clip.json'
        save(receipt_path, receipt)
        return {'clip_id': 'clip', 'inference_identity': {**identity, 'code_sha256': {'new_rule': 'new'}},
                'new_neural_inference': False, 'new_neural_inference_this_run': False,
                'fresh_neural_cache_used': True, 'source_cache_sha256': sha256(cache_path),
                'original_neural_inference_producer': {
                    'inference_identity': identity, 'receipt_file': 'producer_results/clip.json',
                    'receipt_sha256': sha256(receipt_path), 'detections_file': 'detections/clip.json',
                    'detections_sha256': sha256(cache_path), 'new_neural_inference': True,
                    'fresh_model_inference': True}}

    def test_unchanged_neural_producer_can_support_changed_rules_without_claiming_new_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.bundle(root)
            self.assertTrue(validate_cached_neural_producer(payload, root))
            self.assertFalse(payload['new_neural_inference'])

    def test_changed_source_pose_model_manifest_or_detection_bytes_are_integrity_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.bundle(root)
            for key in ('source_sha256', 'pose_sha256', 'model_sha256', 'source_manifest_sha256'):
                changed = deepcopy(payload)
                changed['inference_identity'][key] = 'other'
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    validate_cached_neural_producer(changed, root)
            (root/'detections/clip.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                validate_cached_neural_producer(payload, root)

    def test_replay_cannot_claim_cache_was_newly_inferred_or_leave_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.bundle(root)
            changed = deepcopy(payload)
            changed['new_neural_inference_this_run'] = True
            with self.assertRaisesRegex(ValueError, 'no new neural inference'):
                validate_cached_neural_producer(changed, root)
            changed = deepcopy(payload)
            changed['original_neural_inference_producer']['receipt_file'] = '../clip.json'
            with self.assertRaisesRegex(ValueError, 'outside its bundle'):
                validate_cached_neural_producer(changed, root)


if __name__ == '__main__':
    unittest.main()
