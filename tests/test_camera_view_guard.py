import unittest
import numpy as np
from src.camera_view_guard import closeup_guard, closeup_context, classify_guarded


def person(role='kicker_candidate',area=.2,height=.7):
    return {'role':role,'area_fraction':area,'height_fraction':height,'bbox':[10,10,50,95],
            'face_visible_fraction':1.}


class CameraViewGuardTests(unittest.TestCase):
    def test_single_same_kit_not_identity_proof(self):
        r=closeup_guard([person()],{'green_outside_people_fraction':.3})
        self.assertTrue(r['allowed']);self.assertIn('unverified',r['reason'])
    def test_same_kit_competitors_abstain(self):
        r=closeup_guard([person(),person(area=.15)],{'green_outside_people_fraction':.5})
        self.assertFalse(r['allowed'])
    def test_no_field_context_abstains(self):
        self.assertFalse(closeup_guard([person()],{'green_outside_people_fraction':.01})['allowed'])
    def test_grass_inside_person_does_not_count(self):
        frame=np.full((100,100,3),(0,130,0),np.uint8)
        p=person();p['bbox']=[0,0,100,100]
        self.assertEqual(closeup_context(frame,[p])['green_outside_people_fraction'],0)
    def test_failed_keeper_back_test_falls_through_to_geometry(self):
        kicker=person(height=.25,area=.01);kicker['bbox']=[40,70,55,95]
        keeper=person('keeper_candidate',height=.4,area=.04);keeper['bbox']=[45,30,55,70]
        frame={'width':100,'height':100,'people':[kicker,keeper],
               'goal':{'bbox':[20,20,80,70],'corners':[[20,20],[80,20],[80,70],[20,70]]},
               'goal_camera':{'median_degrees':0,'min_degrees':-3,'max_degrees':3}}
        r=classify_guarded([frame,frame])
        self.assertEqual(r['old_sample_votes'],['unknown','unknown'])
        self.assertEqual(r['sample_votes'],['back_of_kicker','back_of_kicker'])
    def test_missing_goal_not_fabricated(self):
        f={'width':100,'height':100,'people':[person('keeper_candidate')], 'goal':None}
        self.assertNotEqual(classify_guarded([f,f])['sample_votes'][0],'back_of_kicker')


if __name__=='__main__':unittest.main()
