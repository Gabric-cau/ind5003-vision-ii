import unittest
from src.goalkeeper_height_fusion import contact_bracket_hip_height


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.silhouette=[dict(frame=0,lower_edge=[0,100]),dict(frame=10,lower_edge=[0,120])]
        self.phase=dict(complete_first_dive_candidate=True,extension_frames=list(range(2,9)),contact_anchor_frames=[0,10])
        self.reference=dict(ground_line=[0,100],body_scale_px=100)
        self.samples=[dict(frame=f,hip=[0,104+2*(f-2)-25],min_confidence=.8) for f in [2,4,6,8]]

    def run_height(self):
        return contact_bracket_hip_height(self.samples,self.silhouette,self.phase,self.reference,30)

    def test_low_body_before_landing(self):
        self.assertEqual(self.run_height()['height_candidate'],'low')
        self.assertTrue(self.run_height()['requires_review'])

    def test_high_body_then_low_landing(self):
        for s in self.samples:s['hip'][1]-=65
        self.assertEqual(self.run_height()['height_candidate'],'high')

    def test_missing_peak(self):
        self.samples=self.samples[:1]+self.samples[-1:]
        self.assertFalse(self.run_height()['supported'])

    def test_large_depth_change(self):
        self.silhouette[-1]['lower_edge'][1]=200
        self.assertFalse(self.run_height()['supported'])

    def test_missing_stage(self):
        self.phase['complete_first_dive_candidate']=False
        self.assertFalse(self.run_height()['supported'])

    def test_contact_boundaries_disagree(self):
        self.phase['landing_frame']=20
        result=contact_bracket_hip_height(self.samples,self.silhouette,self.phase,self.reference,30,pose_landing_frame=10)
        self.assertFalse(result['supported'])
