import unittest
from src.goalkeeper_observation_guard import guard_postkick_observations


def record(frames):
    return {"fps":30,"kick_frame":60,"selected_candidate":{
        "post_trajectory":[[f/30,100,100,100] for f in frames]},
        "dive_direction":"center","dive_height":"low","auto_accepted":True,
        "review_required":False,"qa_warnings":[]}


class ObservationGuardTests(unittest.TestCase):
    def test_complete_window_preserved(self):
        out=guard_postkick_observations(record(range(60,91)))
        self.assertTrue(out["auto_accepted"])
        self.assertTrue(out["dive_observability"]["window_supported"])

    def test_short_prefix_not_center(self):
        original=record(range(60,76));out=guard_postkick_observations(original)
        self.assertEqual(out["dive_direction"],"unknown")
        self.assertEqual(out["candidate_dive_direction"],"center")
        self.assertEqual(original["dive_direction"],"center")
        self.assertIn("right_censored_dive_window",out["qa_warnings"])

    def test_big_interior_gap_rejected(self):
        out=guard_postkick_observations(record(list(range(60,73))+list(range(80,91))))
        self.assertIn("postkick_observation_gap",out["qa_warnings"])

    def test_duplicate_points_do_not_raise_coverage(self):
        out=guard_postkick_observations(record([60,90]*30))
        self.assertEqual(out["dive_observability"]["observed_frames"],2)
        self.assertFalse(out["auto_accepted"])

    def test_missing_and_prior_rejection_not_promoted(self):
        self.assertFalse(guard_postkick_observations(record([]))["auto_accepted"])
        r=record(range(60,91));r["auto_accepted"]=False;r["review_required"]=True
        self.assertFalse(guard_postkick_observations(r)["auto_accepted"])
