import unittest
import numpy as np
from src.phase_step_events import propose_step_events
from src.motion_stabilization import apply_transforms


class EventTests(unittest.TestCase):
    def test_fractional_frames_rejected(self):
        args = self.data()
        args[0] = args[0].astype(float) + .4
        with self.assertRaises(ValueError):
            propose_step_events(*args)

    def test_unanchored_scope_disclosed(self):
        result = propose_step_events(*self.data())
        self.assertIn("unanchored", result["direction_source"])
        self.assertIn("contact_anchor_unavailable_scope_not_known_precontact", result["warnings"])

    def data(self):
        frames = np.arange(180)
        hips = np.c_[frames / 2, np.full(180, 50)]
        wave = 15 * np.sin(frames * 2 * np.pi / 24)
        left = hips + np.c_[wave, np.full(180, 60)]
        right = hips + np.c_[-wave, np.full(180, 60)]
        return [frames, 30., hips, left, right, np.full(180, 150.), np.ones((180, 3))]

    def test_translation_scale_invariance(self):
        args = self.data()
        a = propose_step_events(*args, kick_frame=170)
        for i in (2, 3, 4):
            args[i] = args[i] * 2 + [74, 30]
        args[5] *= 2
        b = propose_step_events(*args, kick_frame=170)
        self.assertEqual([e['frame'] for e in a['events']], [e['frame'] for e in b['events']])
        self.assertGreater(len(a['events']), 8)

    def test_long_gap_not_bridged(self):
        args = self.data()
        args[6][40:90] = 0
        result = propose_step_events(*args, kick_frame=170)
        self.assertFalse(any(40 <= x['frame'] < 90 for x in result['events']))
        self.assertIsNone(result['signal']['values'][70])

    def test_absolute_frame_offset(self):
        args = self.data()
        a = propose_step_events(*args, kick_frame=170)
        args[0] += 60
        b = propose_step_events(*args, kick_frame=230)
        self.assertEqual([e['frame'] + 60 for e in a['events']], [e['frame'] for e in b['events']])

    def test_no_postcontact_event(self):
        result = propose_step_events(*self.data(), kick_frame=90)
        self.assertTrue(all(e['frame'] <= 90 for v in result['variants'].values() for e in v))
        self.assertFalse(result['human_verified'])
        self.assertFalse(result['counts_are_contacts'])

    def test_constant_no_events(self):
        args = self.data()
        args[3] = args[2] + [4, 60]
        args[4] = args[2] + [-4, 60]
        self.assertEqual(propose_step_events(*args)['events'], [])

    def test_confidence_abstention(self):
        args = self.data()
        args[6][:] = .1
        self.assertEqual(propose_step_events(*args)['events'], [])

    def test_camera_inverse_transform(self):
        actual = apply_transforms([[10, 20], [12, 23]], [50, 51],
                                  {'50': np.eye(3).tolist(), '51': [[1, 0, -2], [0, 1, -3], [0, 0, 1]]})
        np.testing.assert_allclose(actual, [[10, 20], [10, 20]])


if __name__ == '__main__':
    unittest.main()
