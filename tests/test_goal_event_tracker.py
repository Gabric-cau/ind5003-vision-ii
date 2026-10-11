import unittest
import numpy as np
from src.goal_event_tracker_v2 import bootstrap,track_event_segments,usable
from src.event_ball_tracking import summarize_event_cues
from src.keeper_kit_evidence import learn_kit_profile,kit_mask,annotate_kit_ambiguity


def detection(f,x,y,**extra):
    return {"frame":f,"x":x,"y":y,"stabilized_xy":[x,y],"confidence":.8,
            "width":14,"height":14,"white_ratio":.8,"sources":["full"],
            "motion_mean_abs_gray":50.,"detection_id":f"{f}:{x}:{y}",**extra}


class EventTrackerTests(unittest.TestCase):
    def test_reject_first_jump_to_keeper(self):
        seed=[detection(i,1000,400) for i in range(7)]
        distant=[detection(7,450,320),detection(8,455,320)]
        self.assertEqual(bootstrap(seed,distant,np.array([450,320.])),[])

    def test_repeated_encoded_frame_not_new_motion(self):
        self.assertFalse(usable(detection(2,100,100,duplicate_frame=True)))

    def test_stationary_field_artifact_not_flight(self):
        self.assertFalse(usable(detection(2,100,100,motion_mean_abs_gray=0)))

    def test_no_seed_abstains(self):
        path,diag=track_event_segments({str(i):[detection(i,10*i,0)] for i in range(10)},
                                      4,[[1000,400]],[400,300])
        self.assertFalse(path)
        self.assertEqual(diag['outcome'],'unknown')

    def test_rebound_retained_without_outcome_claim(self):
        ds={str(i):[detection(i,1000,400)] for i in range(11)}
        for j in range(1,6):ds[str(10+j)]=[detection(10+j,1000-60*j,400-20*j)]
        for j in range(1,7):ds[str(15+j)]=[detection(15+j,700+20*j,300-25*j)]
        path,diag=track_event_segments(ds,10,[[1000,400]],[700,300],300)
        self.assertEqual(diag['outgoing_status'],'candidate')
        self.assertGreater(diag['outgoing_observations'],3)
        self.assertGreater(path[-1]['x'],700)
        self.assertGreater(diag['turn_angle_degrees'],70)
        self.assertEqual(diag['outcome'],'unknown')
        self.assertFalse(diag['whole_ball_crossing_established'])
        self.assertTrue(all(d in ds[str(d['frame'])] for d in path))
        self.assertFalse(summarize_event_cues(path,[700,300],300)['crossing_established'])

    def test_kit_mask_is_appearance_not_contact_label(self):
        im=np.zeros((100,100,3),np.uint8);im[20:80,30:70]=(180,100,0)
        profile=learn_kit_profile(im,[30,20,70,80])
        self.assertIsNotNone(profile)
        mask=kit_mask(im,profile,[0,0,100,100])
        self.assertGreater(np.count_nonzero(mask),1000)
        points=[detection(0,72,50),detection(0,95,95)]
        evidence=annotate_kit_ambiguity(im,profile,[0,0,100,100],points,[50,50])
        self.assertTrue(points[0]['person_overlap'])
        self.assertFalse(points[1].get('person_overlap',False))
        self.assertFalse(evidence['identity_verified'])

    def test_invalid_kit_seed_abstains(self):
        self.assertIsNone(learn_kit_profile(np.zeros((20,20,3),np.uint8),[2,2,10,10]))


if __name__=='__main__':unittest.main()
