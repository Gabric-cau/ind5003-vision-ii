import unittest
import numpy as np
from src.ball_mask_seed import MaskSeedConfig, white_mask
from src.ball_continuity_v4 import ContinuityConfig, track_ball_continuity
from tests.test_ball_continuity import detection, flight
from unittest.mock import patch


class MaskContinuityTests(unittest.TestCase):
    def test_pastel_grass_does_not_pass_a_white_threshold(self):
        frame = np.full((20, 20, 3), (142, 185, 130), np.uint8)
        frame[5:10, 5:10] = 230
        mask = white_mask(frame, MaskSeedConfig(maximum_saturation=95, maximum_green_excess=25))
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[7, 7]), 255)

    def test_delayed_departure_is_recovered_without_a_keeper_jump(self):
        ds = flight()
        for f in range(11, 17): ds[str(f)] = []
        for f in range(11, 26):
            ds[str(f)].append(detection(f, 440, 240, confidence=.99))
        path, diag = track_ball_continuity(ds, 10, [[1000, 400]], [550, 250])
        self.assertEqual(diag['tracking_status'], 'candidate')
        self.assertEqual(path[-1]['frame'], 25)
        self.assertEqual(path[1]['frame'], 17)
        self.assertTrue(all(p['x'] >= 550 for p in path))
        self.assertFalse(any(11 <= p['frame'] < 17 for p in path))

    def test_latest_spot_observation_does_not_hide_an_earlier_release(self):
        ds = flight()
        # A boot appears near the old spot after the real ball has left.
        for f in (11, 12, 13): ds[str(f)].append(detection(f, 1000, 400, confidence=.99))
        path, diag = track_ball_continuity(ds, 12, [[1000, 400]], [550, 250])
        self.assertEqual(path[-1]['frame'], 25)
        self.assertTrue(all(p['x'] < 1000 for p in path[1:]))
        self.assertLess(path[0]['frame'], 13)

    def test_prediction_and_size_prevent_a_tiny_mesh_point_replacing_ball(self):
        ds = flight()
        for f in range(18, 26):
            actual = ds[str(f)][0]
            ds[str(f)].append(detection(f, actual['x'] + 24, actual['y'] + 8,
                sources=['motion_blob'], width=5, height=3, confidence=0))
        path, _ = track_ball_continuity(ds, 10, [[1000, 400]], [550, 250])
        self.assertEqual(path[-1]['frame'], 25)
        self.assertTrue(all('motion_blob' not in p['sources'] for p in path))

    def test_later_departure_birth_cannot_replace_committed_flight(self):
        ds=flight()
        for f in range(20,36):
            ds.setdefault(str(f),[]).append(detection(f,900-12*(f-20),370,confidence=.99))
        early=[ds['10'][0],ds['12'][0],ds['13'][0]]
        later=[ds['10'][0],ds['20'][-1],ds['21'][-1]]
        births=[{'path':early,'velocity':np.array([-30.,-10.]),'score':0.,'turns':[]},
                {'path':later,'velocity':np.array([-12.,0.]),'score':0.,'turns':[]}]
        with patch('src.ball_continuity_v4.bootstrap_departure',return_value=births):
            path,_=track_ball_continuity(ds,10,[[1000,400]],[550,250])
        self.assertEqual(path[-1]['frame'],25)
        self.assertEqual(path[1]['detection_id'],ds['12'][0]['detection_id'])

    def test_release_cannot_precede_later_strong_stationary_ball_observations(self):
        from src.ball_continuity_v4 import bootstrap_departure
        seed=[detection(f,1000,400) for f in range(11)]
        # This moving background line back-projects to a release at frame 5,
        # while credible ball observations still occupy the spot through 10.
        obs=[detection(11,820,340),detection(12,790,330)]
        starts=bootstrap_departure(seed,obs,np.array([550,250]),10,ContinuityConfig())
        self.assertFalse(starts)

    def test_short_unconfirmed_birth_does_not_lock_out_supported_departure(self):
        ds=flight()
        a=detection(12,940,350);b=detection(13,930,360)
        ds['12'].append(a);ds['13'].append(b)
        weak=[ds['10'][0],a,b]
        actual=[ds['10'][0],ds['17'][0],ds['18'][0]]
        births=[{'path':weak,'velocity':np.array([-10.,10.]),'score':0.,'turns':[]},
                {'path':actual,'velocity':np.array([-30.,-10.]),'score':0.,'turns':[]}]
        with patch('src.ball_continuity_v4.bootstrap_departure',return_value=births):
            path,_=track_ball_continuity(ds,10,[[1000,400]],[550,250])
        self.assertEqual(path[-1]['frame'],25)
        self.assertEqual(path[1]['frame'],17)

    def test_running_clothing_cannot_lock_out_sparse_fast_departure(self):
        ds=flight()
        for f in range(11,17):ds[str(f)]=[]
        # A dense slow object leaves the spot before the first visible ball.
        # Direction alone cannot distinguish the two departures.
        for f in range(16,36):
            ds.setdefault(str(f),[]).append(detection(f,1000-12*(f-10),400-4*(f-10),
                confidence=0,sources=['motion_blob']))
        path,diag=track_ball_continuity(ds,10,[[1000,400]],[550,250])
        self.assertEqual(diag['tracking_status'],'candidate')
        self.assertIn(ds['17'][0]['detection_id'],[q['detection_id'] for q in path])
        self.assertEqual(path[-1]['frame'],25)
        self.assertTrue(all('motion_blob' not in q['sources'] for q in path if q['frame']>=17))


if __name__ == '__main__': unittest.main()
