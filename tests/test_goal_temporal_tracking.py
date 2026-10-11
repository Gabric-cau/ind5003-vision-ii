import unittest
import cv2
import numpy as np
from src.goal_temporal_tracking import track_goal_pair


def fixture():
    frame=np.full((360,640,3),(20,110,20),np.uint8)
    corners=np.array([[120,80],[520,80],[520,250],[120,250]],np.float32)
    cv2.polylines(frame,[corners.astype(np.int32)],False,(255,255,255),5)
    cv2.line(frame,(120,250),(120,80),(255,255,255),5)
    rng=np.random.default_rng(17)
    for side in range(3):
        start,end=[(0,1),(1,2),(3,0)][side]
        for t in np.linspace(.02,.98,35):
            p=(corners[start]*(1-t)+corners[end]*t).astype(int)
            p+=rng.integers(-7,8,2)
            cv2.circle(frame,tuple(p),2,(180,180,180),-1)
    return frame,corners


class GoalTemporalTests(unittest.TestCase):
    def test_translation_supported(self):
        a,c=fixture();m=np.array([[1,0,4],[0,1,2]],float)
        b=cv2.warpAffine(a,m,(640,360))
        r=track_goal_pair(a,b,c)
        self.assertEqual(r['status'],'tracked_candidate',r)
        self.assertLess(np.max(np.abs(np.array(r['corners'])-(c+[4,2]))),1)
        self.assertFalse(r['human_verified'])
    def test_blank_has_no_features(self):
        a,c=fixture();a[:]=20
        self.assertEqual(track_goal_pair(a,a,c)['status'],'abstain')
    def test_disappearing_goal_abstains(self):
        a,c=fixture()
        self.assertEqual(track_goal_pair(a,np.zeros_like(a),c)['status'],'abstain')
    def test_excluded_region_not_used(self):
        a,c=fixture()
        self.assertEqual(track_goal_pair(a,a,c,[(0,0,640,360)])['status'],'abstain')
    def test_invalid_corners(self):
        a,c=fixture();c[0,0]=np.nan
        self.assertEqual(track_goal_pair(a,a,c)['reason'],'invalid_corners')
    def test_shape_mismatch(self):
        a,c=fixture()
        self.assertEqual(track_goal_pair(a,a[:300],c)['reason'],'frame_shape_mismatch')


if __name__=='__main__':unittest.main()
