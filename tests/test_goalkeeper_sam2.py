import unittest
import numpy as np
from src.goalkeeper_sam2 import mask_geometry,points_inside,mask_qc,sam2_direction_evidence,anatomy_sample,head_part_sample,head_part_direction_evidence


class SAM2EvidenceTests(unittest.TestCase):
    def test_unsigned_mask_never_proves_direction(self):
        q={f:dict(valid=True) for f in range(8,41)}
        self.assertFalse(sam2_direction_evidence([],q,10,30)['supported'])

    def test_no_head_no_anatomy(self):
        m=np.ones((50,80),np.uint8)
        o=dict(frame=10,keypoints=[[30,20]]*17,keypoint_confidences=[.9]*17)
        o['keypoint_confidences'][0]=.1
        self.assertIsNone(anatomy_sample(o,m))

    def test_points_outside_and_nan(self):
        m=np.ones((10,10),np.uint8)
        self.assertEqual(points_inside(m,[[4,4],[20,20],[float('nan'),2]]).tolist(),[True,False,False])

    def test_empty_mask(self):
        self.assertIsNone(mask_geometry(np.zeros((20,20),np.uint8)))

    def test_referee_torso_overlap_rejected(self):
        m=np.zeros((40,40),np.uint8);m[10:30,10:30]=1
        self.assertFalse(mask_qc(m,None,30,400,m)['valid'])

    def sample(self,f,sign=0,upright=False):return dict(frame=f,sign=sign,upright=upright)

    def test_connected_transition(self):
        s=[self.sample(f,upright=True) for f in [7,8,9]]+[self.sample(f,-1) for f in [11,12,13]]
        q={f:dict(valid=True) for f in range(10,41)}
        out=sam2_direction_evidence(s,q,10,30);self.assertTrue(out['supported']);self.assertEqual(out['direction'],'image_left')

    def test_mask_gap_before_late_pose_not_bridged(self):
        s=[self.sample(f,upright=True) for f in [7,8,9]]+[self.sample(f,-1) for f in [20,21,22]]
        q={f:dict(valid=f!=15,reason='lost') for f in range(10,41)}
        self.assertFalse(sam2_direction_evidence(s,q,10,30)['supported'])

    def test_conflicting_runs_abstain(self):
        s=[self.sample(f,upright=True) for f in [7,8,9]]+[self.sample(f,-1) for f in [11,12,13]]+[self.sample(f,1) for f in [15,16,17]]
        q={f:dict(valid=True) for f in range(10,41)}
        self.assertEqual(sam2_direction_evidence(s,q,10,30)['reasons'],['conflicting_anatomical_directions'])

    def test_two_samples_not_enough(self):
        s=[self.sample(f,upright=True) for f in [7,8,9]]+[self.sample(f,-1) for f in [11,12]]
        q={f:dict(valid=True) for f in range(10,41)}
        self.assertFalse(sam2_direction_evidence(s,q,10,30)['supported'])

    def test_head_part_expanding_to_whole_person_is_rejected(self):
        body=np.zeros((80,120),np.uint8);body[20:60,10:100]=1
        self.assertIsNone(head_part_sample(body,body,10,100))

    def test_head_part_outside_body_is_rejected(self):
        body=np.zeros((80,120),np.uint8);body[20:60,10:100]=1
        head=np.zeros_like(body);head[0:10,0:10]=1
        self.assertIsNone(head_part_sample(head,body,10,100))

    def test_no_body_qc_cannot_promote_head_direction(self):
        m=np.zeros((80,120),np.uint8);m[20:40,20:100]=1
        h=np.zeros_like(m);h[23:33,23:33]=1
        masks={f:m for f in range(1,20)};heads={f:h for f in masks}
        self.assertFalse(head_part_direction_evidence(heads,masks,{},10,30,80)['supported'])


if __name__=='__main__':unittest.main()
