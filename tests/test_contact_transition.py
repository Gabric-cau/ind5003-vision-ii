"""Synthetic causal-timing regressions, not an accuracy benchmark."""
import unittest
import numpy as np
from src.kick_contact import score_contact_candidates, prefer_causal_first_departure


class ContactTransitionTests(unittest.TestCase):
    def test_earlier_plausible_event_of_same_actor(self):
        a={"score":.85,"contact_frame":90,"contact_associated_track_id":2,
           "ankle_approach_supported":True,"ball_departure_frames":[92,95]}
        b={**a,"score":.65,"contact_frame":37,"ball_departure_frames":[39,42]}
        self.assertIs(prefer_causal_first_departure([a,b])[0],b)
        other={**b,"contact_associated_track_id":8}
        self.assertIs(prefer_causal_first_departure([a,other])[0],a)

    def test_weak_earlier_event_is_not_promoted(self):
        a={"score":.85,"contact_frame":90,"contact_associated_track_id":2,
           "ankle_approach_supported":True,"ball_departure_frames":[92,95]}
        b={**a,"score":.49,"contact_frame":37}
        self.assertIs(prefer_causal_first_departure([a,b])[0],a)
    def make_case(self):
        balls = [{"frame": f, "xy": [100, 100], "diameter": 6} for f in range(0, 31, 3)]
        balls += [{"frame": f, "xy": [100+(f-30)*4, 100], "diameter": 6} for f in range(33, 49, 3)]
        poses = [{"frame": f, "ankles": [[100+abs(f-30)*3, 108]], "height": 100} for f in range(49)]
        # A post-strike support foot is closer than the actual strike ankle.
        poses[44]["ankles"] = [[100, 100]]
        return balls, poses, {f: np.eye(3) for f in range(49)}

    def test_returning_foot_does_not_replace_contact(self):
        b, p, t = self.make_case()
        legacy = score_contact_candidates(b, p, t, 30, 3)
        new = score_contact_candidates(b, p, t, 30, 3, contact_window_policy="stationary_transition")
        self.assertEqual(legacy[0]["contact_frame"], 44)
        self.assertEqual(new[0]["contact_frame"], 30)
        self.assertTrue(new[0]["ankle_approach_supported"])
        self.assertTrue(new[0]["ball_departure_frames"])

    def test_disappearance_alone_still_not_departure(self):
        b, p, t = self.make_case()
        new = score_contact_candidates([v for v in b if v["frame"]<=30], p, t, 30, 3,
                                       contact_window_policy="stationary_transition")
        self.assertIsNone(new[0]["ball_departure_frames"])

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            score_contact_candidates([], [], {}, 30, contact_window_policy="anything")


if __name__ == "__main__":
    unittest.main()
