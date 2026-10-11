import unittest
import cv2
import numpy as np
from src.goalkeeper_mask_experiment import (box_mask, nearest_component, warp_mask,
    lk_tracks, mask_info, color_score, grabcut_mask)


class MaskExperimentTests(unittest.TestCase):
    def test_empty_prior_abstains(self):
        m, reason=nearest_component(np.ones((30,30),np.uint8),np.zeros((30,30),np.uint8))
        self.assertFalse(m.any());self.assertEqual(reason,'no_temporal_prior')

    def test_disconnected_far_person_not_selected(self):
        raw=np.zeros((100,180),np.uint8);raw[10:20,10:20]=1;raw[60:90,130:160]=1
        prior=np.zeros_like(raw);prior[10:20,10:20]=1
        m,_=nearest_component(raw,prior)
        self.assertEqual(int(m.sum()),100);self.assertFalse(m[70,140])

    def test_no_candidate_returns_empty_not_stale_prior(self):
        m,_=nearest_component(np.zeros((30,30),np.uint8),np.ones((30,30),np.uint8))
        self.assertFalse(m.any())

    def test_box_clip_not_negative_index(self):
        m=box_mask([-10,-10,5,5],(20,20))
        self.assertEqual(m.sum(),36);self.assertFalse(m[-1,-1])

    def test_backward_warp_translation(self):
        m=np.zeros((20,20),np.uint8);m[5:10,5:10]=1
        back=np.zeros((20,20,2),np.float32);back[:,:,0]=-2
        w=warp_mask(m,back);self.assertEqual(mask_info(w,1)['bbox'],[7,5,12,10])

    def test_flow_empty_is_empty(self):
        a,b=lk_tracks(np.zeros((20,20),np.uint8),np.zeros((20,20),np.uint8),None)
        self.assertEqual(len(a),0);self.assertEqual(len(b),0)

    def test_no_grabcut_foreground_abstains(self):
        m,r=grabcut_mask(np.zeros((30,30,3),np.uint8),np.zeros((30,30),np.uint8),np.zeros((30,30)))
        self.assertFalse(m.any());self.assertEqual(r,'no_reliable_foreground_seed')

    def test_color_score_joint_lab(self):
        img=np.full((3,3,3),[0,220,0],np.uint8)
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)[0,0].astype(float)
        score=color_score(img,[(lab,np.ones(3)*36),(lab+30,np.ones(3)*36)])
        self.assertTrue(np.all(score>0))


if __name__=='__main__':unittest.main()
