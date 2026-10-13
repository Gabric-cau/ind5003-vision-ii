from copy import deepcopy
import unittest

from scripts.replay_fresh_height_reference_guard import guard_cached_height
from src.fresh_keeper_height import guard_observed_height_reference


def cached_height(*, depth_failed=True):
    source = dict(clip_id='penalty', clip_sha256='a'*64)
    result = dict(clip_id='penalty', inference_identity=dict(
        source_clip_sha256='a'*64, source_manifest_sha256='b'*64),
        human_verified=False, accuracy_validated=False, reference_labels_read=False,
        physical_height_metres=None, strict_height_candidate='unknown',
        ordinal_pose_height_candidate='unknown', observed_image_plane_height_candidate='low',
        observed_image_plane_signal=dict(candidate='low', supported=True, reasons=[],
            reference_depth_assumption_failed=depth_failed,
            observed_maximum_torso_clearance_body=.2, supporting_frames=[10, 11, 12],
            continuous_samples=[dict(frame=10, torso_clearance_body=.2)]),
        reference=dict(body_scale_px=100), pose_only_phase=dict(supported=False, samples=[1, 2]))
    return result, source


class HeightReferenceGuardTests(unittest.TestCase):
    def test_invalid_reference_withdraws_only_category_and_support(self):
        original, source = cached_height()
        snapshot = deepcopy(original)
        result = guard_cached_height(original, source, 'b'*64)
        self.assertEqual(original, snapshot)
        self.assertEqual(result['observed_image_plane_height_candidate'], 'unknown')
        signal = result['observed_image_plane_signal']
        self.assertEqual(signal['candidate'], 'unknown')
        self.assertFalse(signal['supported'])
        self.assertEqual(signal['observed_maximum_torso_clearance_body'], .2)
        self.assertEqual(signal['supporting_frames'], [10, 11, 12])
        self.assertEqual(signal['continuous_samples'], original['observed_image_plane_signal']['continuous_samples'])
        for key in ('reference', 'pose_only_phase', 'strict_height_candidate',
                    'ordinal_pose_height_candidate', 'physical_height_metres'):
            self.assertEqual(result[key], original[key])

    def test_valid_reference_retains_original_candidate(self):
        original, source = cached_height(depth_failed=False)
        self.assertEqual(guard_cached_height(original, source, 'b'*64), original)

    def test_guard_is_idempotent(self):
        original, _ = cached_height()
        signal = guard_observed_height_reference(original['observed_image_plane_signal'])
        self.assertEqual(guard_observed_height_reference(signal), signal)
        self.assertEqual(len(signal['reasons']), 1)

    def test_unknown_without_signal_stays_unknown(self):
        original, source = cached_height()
        del original['observed_image_plane_signal']
        original['observed_image_plane_height_candidate'] = 'unknown'
        self.assertEqual(guard_cached_height(original, source, 'b'*64), original)

    def test_replay_rejects_source_or_manifest_mismatch(self):
        original, source = cached_height()
        with self.assertRaisesRegex(ValueError, 'source or manifest mismatch'):
            guard_cached_height(original, dict(source, clip_sha256='c'*64), 'b'*64)
        with self.assertRaisesRegex(ValueError, 'source or manifest mismatch'):
            guard_cached_height(original, source, 'c'*64)

    def test_replay_cannot_change_verified_or_reference_labelled_measurements(self):
        original, source = cached_height()
        for flag in ('human_verified', 'accuracy_validated', 'reference_labels_read'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'source-only'):
                guard_cached_height(dict(original, **{flag: True}), source, 'b'*64)

    def test_replay_rejects_category_signal_disagreement(self):
        original, source = cached_height()
        original['observed_image_plane_height_candidate'] = 'high'
        with self.assertRaisesRegex(ValueError, 'category disagrees'):
            guard_cached_height(original, source, 'b'*64)


if __name__ == '__main__':
    unittest.main()
