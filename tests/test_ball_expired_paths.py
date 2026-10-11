import unittest
from src.ball_pipeline import select_ball_track


class BallExpiredPathTests(unittest.TestCase):
    def test_no_crash_after_all_paths_expire(self):
        detections={f:[] for f in range(70)}
        detections[0]=[{"frame":0,"x":100.,"y":100.,"confidence":.9}]
        track,diag=select_ball_track(detections,1,[(100,100)])
        self.assertEqual(track,[])
        self.assertEqual(diag["reason"],"all_motion_paths_expired")

    def test_completed_mode_keeps_observations_not_extrapolations(self):
        detections={f:[] for f in range(70)}
        detections[0]=[{"frame":0,"x":100.,"y":100.,"confidence":.9}]
        track,_=select_ball_track(detections,1,[(100,100)],preserve_completed=True)
        self.assertEqual([p["frame"] for p in track],[0])
