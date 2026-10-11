import unittest
import numpy as np
import cv2
from src.goal_geometry import detect_goal_frame,select_goal_occupant


class GoalGeometryTests(unittest.TestCase):
    def test_blank_image_abstains(self):
        r=detect_goal_frame(np.zeros((360,640,3),np.uint8))
        self.assertEqual(r["status"],"abstain")

    def test_no_person_never_becomes_keeper(self):
        frame=np.full((360,640,3),(40,120,40),np.uint8)
        cv2.rectangle(frame,(180,120),(450,245),(255,255,255),5)
        r=select_goal_occupant(frame,[],10)
        self.assertIsNone(r["seed_track_id"])

    def test_geometry_is_not_metric_calibration(self):
        frame=np.full((360,640,3),(40,120,40),np.uint8)
        cv2.rectangle(frame,(180,120),(450,245),(255,255,255),5)
        for x in range(185,450,10):cv2.line(frame,(x,125),(x,240),(210,210,210),1)
        r=detect_goal_frame(frame)
        self.assertFalse(r.get("metric_calibration",False))
