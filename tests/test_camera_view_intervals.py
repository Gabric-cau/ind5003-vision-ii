import unittest
from src.camera_view_intervals import supported_intervals

class ViewIntervalsTests(unittest.TestCase):
    def test_partial_shot_retained(self):
        f=[0,9,18,27,36,45];v=['back_of_kicker']*4+['unknown']*2
        r=supported_intervals(f,v,0,60,30)
        self.assertEqual(len(r),1);self.assertEqual(r[0]['end_frame_exclusive'],28)
    def test_lone_cue_not_enough(self):
        self.assertEqual(supported_intervals([0,9,18],['back_of_kicker']*3,0,40,30),[])
    def test_unknown_breaks_run(self):
        self.assertEqual(supported_intervals([0,9,18,27,36],['back_of_kicker']*2+['unknown']+['back_of_kicker']*2,0,50,30),[])
    def test_large_gap_not_bridged(self):
        self.assertEqual(supported_intervals([0,9,60,69],['back_of_kicker']*4,0,80,30),[])
    def test_no_slowmo_claim(self):
        r=supported_intervals([0,9,18,27],['kicker_closeup']*4,0,40,30)
        self.assertEqual(r[0]['playback_status'],'unknown');self.assertFalse(r[0]['human_verified'])
    def test_boundary_checked(self):
        with self.assertRaises(ValueError):supported_intervals([0,9,18,40],['back_of_kicker']*4,0,40,30)

if __name__=='__main__':unittest.main()
