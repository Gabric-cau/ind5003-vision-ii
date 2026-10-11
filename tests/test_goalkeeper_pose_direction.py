import unittest
from copy import deepcopy
from tests.test_goalkeeper_direction_evidence import record
from src.goalkeeper_pose_direction import pose_direction_evidence
from src.goalkeeper_observation_guard import guard_postkick_observations


def fixture(sign=-1):
    r=record([100]*13);r['clip_id']='synthetic';r['selected_candidate']['source_track_ids']=[7]
    obs=[]
    for f in range(55,73):
        kp=[[100,100] for _ in range(17)]
        shoulder=[100,70];hip=[100,90];ankle=[100,120]
        if f>=69:shoulder=[100+sign*30,90];ankle=[100-sign*30,95]
        for js,xy in (((5,6),shoulder),((11,12),hip),((15,16),ankle)):
            for j in js:kp[j]=xy[:]
        obs.append(dict(frame=f,keypoints=kp,keypoint_confidences=[.95]*17))
    return r,dict(clip_id='synthetic',fps=30,tracks=[dict(track_id=7,observations=obs)])


class PoseDirectionTests(unittest.TestCase):
    def test_left_with_zero_box_translation(self):
        r,p=fixture();e=pose_direction_evidence(r,p)
        self.assertTrue(e['supported']);self.assertEqual(e['direction'],'image_left')
        self.assertEqual(e['supporting_frames'],[69,70,71,72])
    def test_right(self):
        r,p=fixture(1);self.assertEqual(pose_direction_evidence(r,p)['direction'],'image_right')
    def test_does_not_mutate_inputs(self):
        r,p=fixture();old=deepcopy((r,p));pose_direction_evidence(r,p);self.assertEqual((r,p),old)
    def test_one_lateral_frame_insufficient(self):
        r,p=fixture();p['tracks'][0]['observations']=p['tracks'][0]['observations'][:-3]
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_bad_keypoint_confidence(self):
        r,p=fixture()
        for o in p['tracks'][0]['observations']:o['keypoint_confidences'][5]=.1
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_already_lying_is_not_a_new_dive(self):
        r,p=fixture();lateral=deepcopy(p['tracks'][0]['observations'][-1]['keypoints'])
        for o in p['tracks'][0]['observations']:o['keypoints']=deepcopy(lateral)
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_missing_onset(self):
        r,p=fixture();p['tracks'][0]['observations']=[o for o in p['tracks'][0]['observations'] if o['frame']!=60]
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_no_bridge_over_long_gap(self):
        r,p=fixture();r['selected_candidate']['post_trajectory']=[x for x in r['selected_candidate']['post_trajectory'] if round(x[0]*30)<63 or round(x[0]*30)>69]
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_no_source_identity_switch(self):
        r,p=fixture();r['selected_candidate']['source_track_ids']=[7,8];obs=p['tracks'][0]['observations'];p['tracks'][0]['observations']=[o for o in obs if o['frame']<69]
        p['tracks'].append(dict(track_id=8,observations=[o for o in obs if o['frame']>=69]))
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_duplicate_source_tracks_rejected(self):
        r,p=fixture();r['selected_candidate']['source_track_ids']=[7,8]
        p['tracks'].append(dict(track_id=8,observations=deepcopy(p['tracks'][0]['observations'])))
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_unqualified_identity(self):
        r,p=fixture();r['selected_candidate']['qualifies']=False
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_ambiguous_identity(self):
        r,p=fixture();r['goal_identity_evidence']['ambiguous_identity']=True
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_body_axes_must_agree(self):
        r,p=fixture()
        for o in p['tracks'][0]['observations']:
            if o['frame']>=69:
                o['keypoints'][15]=o['keypoints'][16]=[20,95]
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_pose_overrides_wrong_center_motion(self):
        r,p=fixture();r['selected_candidate']['post_trajectory']=record()['selected_candidate']['post_trajectory']
        out=guard_postkick_observations(r,policy='pose_direction_v3',pose=p)
        self.assertEqual(out['pre_pose_dive_direction'],'image_right')
        self.assertEqual(out['dive_direction'],'image_left');self.assertTrue(out['pose_vs_motion_conflict'])
    def test_direction_does_not_invent_height_or_accuracy(self):
        r,p=fixture();out=guard_postkick_observations(r,policy='pose_direction_v3',pose=p)
        self.assertEqual(out['dive_height'],'unknown');self.assertFalse(out['auto_accepted'])
        self.assertFalse(out['accuracy_validated']);self.assertFalse(out['human_verified'])
    def test_missing_pose_retains_v2(self):
        r=record();out=guard_postkick_observations(r,policy='pose_direction_v3')
        self.assertEqual(out['dive_direction'],'image_right')
    def test_wrong_clip_rejected(self):
        r,p=fixture();p['clip_id']='another';self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_wrong_fps_rejected(self):
        r,p=fixture();p['fps']=25;self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_nan_keypoint_rejected(self):
        r,p=fixture()
        for o in p['tracks'][0]['observations']:o['keypoints'][5][0]=float('nan')
        self.assertFalse(pose_direction_evidence(r,p)['supported'])
    def test_two_sustained_opposite_sides_rejected(self):
        r,p=fixture();_,right=fixture(1)
        for f in range(65,69):
            o=deepcopy(right['tracks'][0]['observations'][-1]);o['frame']=f
            p['tracks'][0]['observations']=[o if old['frame']==f else old for old in p['tracks'][0]['observations']]
        self.assertFalse(pose_direction_evidence(r,p)['supported'])


if __name__=='__main__':unittest.main()
