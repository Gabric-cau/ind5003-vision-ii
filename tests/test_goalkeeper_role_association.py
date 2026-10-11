import unittest
from copy import deepcopy
import numpy as np
from src.goalkeeper_role_association import select_goalkeeper_role,line_features,associate_locked_goalkeeper,_same_person,deduplicate_pose_people,unrotate_pose


def observation(frame,x=300,foot_y=200):
    kp=[[x,foot_y-60] for _ in range(17)]
    for js,xy in [((5,6),[x,foot_y-90]),((11,12),[x,foot_y-50]),((15,16),[x,foot_y]),((9,),[x-20,foot_y-60]),((10,),[x+20,foot_y-60])]:
        for j in js:kp[j]=xy[:]
    return dict(frame=frame,bbox=[x-40,foot_y-100,x+40,foot_y+3],keypoints=kp,keypoint_confidences=[.95]*17,box_confidence=.9)


def fixture():
    frames=list(range(45,65));g={f:dict(selected=dict(corners=[[100,50],[500,50],[500,200],[100,200]])) for f in [48,50,52,54,56,58]}
    p=dict(fps=30,tracks=[dict(track_id=2,observations=[observation(f) for f in frames]),
                          dict(track_id=1,observations=[observation(f,305,275) for f in frames])])
    return p,g


class RoleAssociationTests(unittest.TestCase):
    def test_nearby_referee_excluded_by_ankles(self):
        p,g=fixture();r=select_goalkeeper_role(p,g,60)
        self.assertEqual(r['seed_track_id'],2);self.assertIn(1,r['excluded_pre_person_ids'])
        ref=next(c for c in r['candidates'] if c['track_id']==1)
        self.assertIn('feet_far_from_goal_line',ref['rejection_reasons'])
    def test_missing_wrists_do_not_beat_goal_line_evidence(self):
        p,g=fixture()
        for o in p['tracks'][0]['observations']:o['keypoint_confidences'][9]=o['keypoint_confidences'][10]=0
        self.assertEqual(select_goalkeeper_role(p,g,60)['seed_track_id'],2)
    def test_missing_ankles_abstains_not_bbox_fallback(self):
        p,g=fixture();p['tracks']=p['tracks'][:1]
        for o in p['tracks'][0]['observations']:o['keypoint_confidences'][15]=0
        self.assertIsNone(select_goalkeeper_role(p,g,60)['seed_track_id'])
    def test_same_line_two_people_abstain(self):
        p,g=fixture();p['tracks'][1]['observations']=[observation(f,370,200) for f in range(45,65)]
        self.assertEqual(select_goalkeeper_role(p,g,60)['reason'],'ambiguous_goal_line_role')
    def test_same_pose_duplicates_are_not_two_people(self):
        p,g=fixture();p['tracks'].append(dict(track_id=3,observations=deepcopy(p['tracks'][0]['observations'])))
        r=select_goalkeeper_role(p,g,60);self.assertEqual(r['seed_track_id'],2)
        self.assertEqual(set(r['same_person_seed_ids']),{2,3});self.assertNotIn(3,r['excluded_pre_person_ids'])
    def test_identical_boxes_distinct_poses_remain_two_people(self):
        p,g=fixture()
        for a,b in zip(p['tracks'][0]['observations'],p['tracks'][1]['observations']):b['bbox']=a['bbox'][:]
        self.assertFalse(_same_person(*p['tracks']));self.assertEqual(select_goalkeeper_role(p,g,60)['seed_track_id'],2)
    def test_only_pre_kick_used(self):
        p,g=fixture();r=select_goalkeeper_role(p,g,60)
        for t in p['tracks']:
            for o in t['observations']:
                if o['frame']>=60:o['keypoints']=[[0,0] for _ in range(17)]
        self.assertEqual(select_goalkeeper_role(p,g,60),r)
    def test_one_sample_insufficient(self):
        p,g=fixture();g={48:g[48]};self.assertIsNone(select_goalkeeper_role(p,g,60)['seed_track_id'])
    def test_no_geometry_abstains(self):
        p,g=fixture();self.assertIsNone(select_goalkeeper_role(p,{},60)['seed_track_id'])
    def test_no_mutation(self):
        p,g=fixture();old=deepcopy((p,g));select_goalkeeper_role(p,g,60);self.assertEqual((p,g),old)
    def test_nan_ankle_abstains(self):
        p,g=fixture();p['tracks']=p['tracks'][:1]
        for o in p['tracks'][0]['observations']:o['keypoints'][15][0]=float('nan')
        self.assertIsNone(select_goalkeeper_role(p,g,60)['seed_track_id'])
    def test_slanted_goal_line_perpendicular_distance(self):
        o=observation(10,300,220);g=[[100,50],[500,50],[500,240],[100,200]]
        self.assertAlmostEqual(line_features(o,g)['foot_distance_goal_height'],0,places=6)
    def test_bad_fps(self):
        p,g=fixture();p['fps']=0
        with self.assertRaises(ValueError):select_goalkeeper_role(p,g,60)
    def test_lock_never_switches_to_excluded_referee_even_same_color(self):
        p,g=fixture();r=select_goalkeeper_role(p,g,60)
        p['tracks'][0]['observations']=[o for o in p['tracks'][0]['observations'] if o['frame']<=60]
        for o in p['tracks'][1]['observations']:
            if o['frame']>60:o.update(observation(o['frame']))
        hist={(o['frame'],t['track_id']):np.array([1.,0.]) for t in p['tracks'] for o in t['observations']}
        lock=associate_locked_goalkeeper(p,r,hist,60)
        self.assertEqual(lock['source_track_ids'],[2]);self.assertNotIn(1,lock['source_track_ids'])
        self.assertTrue(any(x['reason']=='excluded_pre_kick_other_person' for d in lock['decisions'] for x in d['rejected']))
    def test_track_height_collapse_during_dive_is_not_identity_rejection(self):
        p,g=fixture();r=select_goalkeeper_role(p,g,60)
        for o in p['tracks'][0]['observations']:
            if o['frame']>60:o['bbox']=[260,140,340,170]
        hist={(o['frame'],t['track_id']):np.array([1.,0.]) for t in p['tracks'] for o in t['observations']}
        lock=associate_locked_goalkeeper(p,r,hist,60);self.assertEqual(lock['last_observed_frame'],64)
    def test_no_appearance_no_lock(self):
        p,g=fixture();r=select_goalkeeper_role(p,g,60)
        self.assertEqual(associate_locked_goalkeeper(p,r,{},60)['reason'],'missing_pre_kick_appearance')
    def test_disconnected_new_person_rejected(self):
        p,g=fixture();r=select_goalkeeper_role(p,g,60)
        p['tracks'][0]['observations']=[o for o in p['tracks'][0]['observations'] if o['frame']<=60]
        p['tracks'].append(dict(track_id=99,observations=[observation(61,800,200)]))
        hist={(o['frame'],t['track_id']):np.array([1.,0.]) for t in p['tracks'] for o in t['observations']}
        lock=associate_locked_goalkeeper(p,r,hist,60);self.assertNotIn(99,lock['source_track_ids'])
    def test_no_identity_accuracy_claim(self):
        p,g=fixture();r=select_goalkeeper_role(p,g,60)
        self.assertFalse(r['identity_verified']);self.assertFalse(r['calibrated_probability'])
    def test_clockwise_pose_mapping(self):
        box=np.array([[20,30,40,70.]])
        original=np.array([[[25.,35.]]*17]);rotated=original.copy()
        rotated[...,0]=99-original[...,1];rotated[...,1]=original[...,0]
        b,k=unrotate_pose([[30,20,70,40]],rotated,200,100,90)
        np.testing.assert_allclose(b,box);np.testing.assert_allclose(k,original)
    def test_counterclockwise_pose_mapping(self):
        original=np.array([[[25.,35.]]*17]);rotated=original.copy()
        rotated[...,0]=original[...,1];rotated[...,1]=199-original[...,0]
        b,k=unrotate_pose([[30,160,70,180]],rotated,200,100,270)
        np.testing.assert_allclose(b,[[20,30,40,70]]);np.testing.assert_allclose(k,original)
    def test_augmented_same_pose_deduplicated(self):
        o=observation(10);self.assertEqual(len(deduplicate_pose_people([o,deepcopy(o)])),1)
    def test_overlapping_distinct_pose_detections_preserved(self):
        a=observation(10);b=observation(10,300,270);b['bbox']=a['bbox'][:]
        self.assertEqual(len(deduplicate_pose_people([a,b])),2)
    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):unrotate_pose([[0,0,20,30]],np.zeros((1,17,2)),200,100,45)
    def test_supported_duplicate_can_supply_near_kick_anchor(self):
        p,g=fixture();duplicate=deepcopy(p['tracks'][0]);duplicate['track_id']=3
        p['tracks'].append(duplicate)
        p['tracks'][0]['observations']=[o for o in p['tracks'][0]['observations'] if o['frame']<=56]
        role={'seed_track_id':2,'same_person_seed_ids':[2,3],'excluded_pre_person_ids':[1]}
        hist={(o['frame'],t['track_id']):np.array([1.,0.]) for t in p['tracks'] for o in t['observations']}
        lock=associate_locked_goalkeeper(p,role,hist,60)
        self.assertEqual(lock['last_observed_frame'],64);self.assertNotIn(1,lock['source_track_ids'])
    def test_swapped_anatomical_labels_deduplicate_same_person(self):
        o=observation(10);other=deepcopy(o)
        for a,b in [(5,6),(11,12),(15,16)]:other['keypoints'][a],other['keypoints'][b]=other['keypoints'][b],other['keypoints'][a]
        self.assertEqual(len(deduplicate_pose_people([o,other])),1)


if __name__=='__main__':unittest.main()
