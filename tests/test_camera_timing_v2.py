import unittest
import cv2,numpy as np
from src.camera_timing_v2 import (choose_boundaries,classify_view,combine_task_class,goal_camera_hypotheses,
    playback_evidence,pose_action_feature,TASK_CLASSES,kit_hist_distance)


class CameraTimingTests(unittest.TestCase):
    def metrics(self,n=180):return [{'mad':2.,'hist_distance':.02,'near_duplicate':False} for _ in range(n)]

    def test_static_no_cut(self):self.assertEqual(choose_boundaries(self.metrics(),30),[])
    def test_pan_no_cut(self):
        m=self.metrics()
        for v in m:v.update(mad=20.,hist_distance=.12)
        self.assertEqual(choose_boundaries(m,30),[])
    def test_hard_cut_and_suppression(self):
        m=self.metrics();m[60].update(mad=60.,hist_distance=.8);m[62].update(mad=40.,hist_distance=.6)
        self.assertEqual([x['frame'] for x in choose_boundaries(m,30)],[60])
    def test_gradual_transition(self):
        m=self.metrics()
        for j in range(70,74):m[j].update(mad=7.,hist_distance=.36)
        cuts=choose_boundaries(m,30)
        self.assertEqual(len(cuts),1);self.assertEqual(cuts[0]['transition_kind'],'gradual_candidate')
    def test_lone_hist_change_not_soft_cut(self):
        m=self.metrics();m[60].update(mad=7.,hist_distance=.4)
        self.assertEqual(choose_boundaries(m,30),[])
    def test_unknown_not_default_keeper(self):
        result=classify_view([{'width':1920,'height':1080,'people':[]}]*3)
        self.assertEqual(result['view_candidate'],'unknown')
    def test_closeup_not_slow_without_speed(self):
        p={'role':'kicker_candidate','height_fraction':.8,'area_fraction':.25,'bbox':[20,20,300,800]}
        result=classify_view([{'width':1920,'height':1080,'people':[p]}]*3)
        self.assertEqual(result['view_candidate'],'kicker_closeup')
        self.assertEqual(combine_task_class(result['view_candidate'],{'status':'unknown'}),'unknown')
    def test_no_vote_majority(self):
        p={'role':'kicker_candidate','height_fraction':.8,'area_fraction':.25,'bbox':[20,20,300,800]}
        samples=[{'width':1920,'height':1080,'people':[p]}, {'width':1920,'height':1080,'people':[]}]
        self.assertEqual(classify_view(samples)['view_candidate'],'unknown')
    def test_playback_static_abstains(self):
        t=np.linspace(0,2,20);x=np.zeros((20,6))
        self.assertEqual(playback_evidence(t,x,t*2,x)['status'],'unknown')
    def test_playback_relative_stretch(self):
        t=np.linspace(0,2,25);x=np.column_stack([t/2,(t/2)**2,np.sin(t)])
        result=playback_evidence(t,x,t*2,x)
        self.assertEqual(result['status'],'slow_motion_candidate')
        self.assertAlmostEqual(result['relative_time_stretch'],2)
        self.assertFalse(result['reference_speed_verified']);self.assertIsNone(result['absolute_speed_factor'])
    def test_playback_reverse_abstains(self):
        t=np.linspace(0,2,25);x=np.column_stack([t/2,(t/2)**2,np.sin(t)])
        self.assertEqual(playback_evidence(t,x,t,x[::-1])['status'],'unknown')
    def test_missing_pose_abstains(self):self.assertIsNone(pose_action_feature({}))
    def test_four_categories(self):self.assertEqual(len(TASK_CLASSES),4)
    def test_keeper_back_net_view_does_not_prove_slow_motion(self):
        keeper={'role':'keeper_candidate','height_fraction':.3,'area_fraction':.08,'bbox':[350,200,500,450],'face_visible_fraction':.6}
        f={'width':1280,'height':720,'people':[keeper],'foreground_net_cue':{'supported':True}}
        view=classify_view([f]*3)['view_candidate'];self.assertEqual(view,'back_of_goalkeeper')
        self.assertEqual(combine_task_class(view,{'status':'unknown'}),'unknown')
    def test_red_hue_wrap_and_exposure(self):
        a=np.zeros((12,3,3));b=a.copy();a[11,2,2]=1;b[0,1,1]=1
        self.assertLess(kit_hist_distance(a.ravel(),b.ravel()),.55)
    def test_distinct_kit_colours(self):
        a=np.zeros((12,3,3));b=a.copy();a[0,2,2]=1;b[6,2,2]=1
        self.assertGreater(kit_hist_distance(a.ravel(),b.ravel()),.9)
    def test_offcentre_runup_not_offcentre_camera(self):
        goal={'corners':[[500,100],[900,100],[900,300],[500,300]],'bbox':[500,100,900,300]}
        p={'role':'kicker_candidate','height_fraction':.2,'area_fraction':.04,'bbox':[100,400,200,600]}
        frame={'width':1280,'height':720,'people':[p],'goal':goal,'goal_camera':{'median_degrees':4,'min_degrees':1,'max_degrees':8}}
        self.assertEqual(classify_view([frame]*3)['view_candidate'],'back_of_kicker')
    def test_camera_right_left_sign(self):
        world=np.array([[-3.66,0,2.44],[3.66,0,2.44],[3.66,0,0],[-3.66,0,0]],float)
        k=np.array([[1920.,0,960],[0,1920,540],[0,0,1.]])
        for side in (-1,1):
            center=np.array([side*18.,-25.,12.]);forward=np.array([0,0,1.22])-center;forward/=np.linalg.norm(forward)
            right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right)
            rot=np.vstack([right,down,forward]);r=cv2.Rodrigues(rot)[0];t=-rot@center
            image=cv2.projectPoints(world,r,t,k,None)[0].reshape(-1,2)
            result=goal_camera_hypotheses(image,1920,1080)
            self.assertEqual(result['status'],'sensitivity_candidate')
            self.assertGreater(side*result['median_degrees'],10)
            self.assertFalse(result['calibration_verified'])


if __name__=='__main__':unittest.main()
