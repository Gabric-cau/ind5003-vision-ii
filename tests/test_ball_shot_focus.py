import unittest
from copy import deepcopy
from src.ball_shot_focus import focus_shot,ShotFocusConfig,ball_extent


def point(f,x,y=0,w=8,h=8):
    return {'frame':f,'detection_id':str(f),'stabilized_xy':[x,y],
            'x':x,'y':y,'width':w,'height':h}


class ShotFocusTests(unittest.TestCase):
    def test_extent_can_reach_goal_review_band_before_center(self):
        seed=[point(0,0)]
        path=[point(0,0),point(1,60),point(2,85,w=20,h=20)]
        _,d=focus_shot(path,seed,[100,0],0,25,70)
        self.assertEqual(d['reason'],'goal_reference_area_reached')
        self.assertFalse(d['whole_ball_crossing_established'])
        self.assertEqual(d['outcome'],'unknown')

    def test_net_reversal_after_arrival_is_not_a_miss(self):
        path=[point(0,0),point(1,60),point(2,100),point(3,60),point(4,20),point(5,-20)]
        kept,d=focus_shot(path,[path[0]],[100,0],0,25,70)
        self.assertEqual(d['reason'],'goal_reference_area_reached')
        self.assertFalse(d['no_goal_hint'])
        self.assertFalse(d['tail_used_for_outcome'])
        self.assertEqual(d['outcome'],'unknown')

    def test_supported_early_right_angle_is_only_a_miss_hint(self):
        path=[point(0,0),point(1,60),point(2,90),point(3,120),
              point(4,120,30),point(5,120,60),point(6,120,90)]
        _,d=focus_shot(path,[path[0]],[400,0],0,25,70)
        self.assertTrue(d['no_goal_hint'])
        self.assertEqual(d['main_end_frame'],3)
        self.assertEqual(d['outcome'],'unknown')
        self.assertFalse(d['supported_large_turn']['before_physical_goal_line_established'])

    def test_single_bad_jump_does_not_establish_deflection(self):
        path=[point(0,0),point(1,60),point(2,90),point(3,120),point(4,120,30),point(5,180)]
        _,d=focus_shot(path,[path[0]],[400,0],0,25,70)
        self.assertFalse(d['no_goal_hint'])

    def test_late_clothing_cannot_move_an_existing_boundary(self):
        path=[point(0,0),point(1,60),point(2,100)]
        _,before=focus_shot(path,[path[0]],[100,0],0,25,70)
        _,after=focus_shot(path+[point(30,300),point(40,400)],
                           [path[0]],[100,0],0,25,70)
        self.assertEqual(before['main_end_frame'],after['main_end_frame'])
        self.assertEqual(before['focus_end_frame'],after['focus_end_frame'])

    def test_gap_abstains_instead_of_using_late_position(self):
        path=[point(0,0),point(1,60),point(2,90),point(12,120)]
        kept,d=focus_shot(path,[path[0]],[400,0],0,25,70)
        self.assertEqual(d['reason'],'continuity_lost')
        self.assertEqual(kept[-1]['frame'],2)
        self.assertIsNone(d['final_position'])

    def test_original_observations_are_not_modified(self):
        path=[point(0,0),point(1,60),point(2,100)]
        original=deepcopy(path)
        focus_shot(path,[path[0]],[100,0],0,25,70)
        self.assertEqual(path,original)

    def test_extent_is_not_a_whole_ball_goal_measurement(self):
        e=ball_extent(point(0,100,w=30,h=10))
        self.assertEqual(e['half_width_px'],15)
        self.assertFalse(e['whole_ball_crossing_established'])

    def test_gap_cannot_become_goal_area_evidence(self):
        path=[point(0,0),point(1,60),point(2,90),point(20,400)]
        _,d=focus_shot(path,[path[0]],[400,0],0,25,70)
        self.assertEqual(d['reason'],'continuity_lost')
        self.assertIsNone(d['arrival_projection'])

    def test_goal_front_reference_bounds_analysis_before_late_net(self):
        path=[point(0,0),point(1,60),point(2,90),point(3,120),point(4,150),point(5,180)]
        _,d=focus_shot(path,[path[0]],[180,0],0,25,70,
                       goal_polygon=[[100,-30],[240,-30],[240,30],[100,30]])
        self.assertEqual(d['main_end_frame'],2)
        self.assertEqual(d['outcome'],'unknown')


if __name__=='__main__':unittest.main()
