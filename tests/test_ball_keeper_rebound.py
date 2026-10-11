import unittest
from copy import deepcopy
from src.ball_keeper_rebound import associate_keeper_rebound, timed_intersection, ReboundConfig


def detection(f,x,y,confidence=.2):
    return {'frame':f,'x':x,'y':y,'stabilized_xy':[x,y],
            'detection_id':f'{f}:{x}:{y}','confidence':confidence,
            'width':10,'height':10,'sources':['neural']}


class KeeperReboundTests(unittest.TestCase):
    def setUp(self):
        self.prefix=[detection(f,100-20*f,100) for f in range(3)]
        self.pool={str(f):[detection(f,30,135-10*f)] for f in range(4,10)}
        self.regions={str(f):[{'kind':'keeper_candidate','polygon':
            [[20,95],[40,95],[40,115],[20,115]]}] for f in range(3,12)}

    def run_rebound(self,**kw):
        return associate_keeper_rebound(self.prefix,kw.pop('candidates',self.pool),
            kw.pop('keeper_regions',self.regions),30,**kw)

    def test_missing_contact_frame_can_recover_without_inventing_observation(self):
        original=deepcopy(self.prefix)
        path,d=self.run_rebound()
        self.assertTrue(d['turn_confirmed'])
        self.assertEqual(path[:3],original)
        self.assertNotIn(3,[q['frame'] for q in path])
        self.assertAlmostEqual(d['contact_prediction']['frame_float'],3.5)
        self.assertFalse(d['contact_prediction']['is_observation'])

    def test_goal_projection_and_destination_on_person_cannot_unlock_turn(self):
        r={str(f):[{'kind':'goal_projection','polygon':[[0,0],[100,0],[100,200],[0,200]]}]
           for f in range(3,12)}
        path,d=self.run_rebound(keeper_regions=r)
        self.assertFalse(d['turn_confirmed']);self.assertEqual(path,self.prefix)

    def test_high_confidence_late_body_branch_fails_timed_intersection(self):
        pool={str(f):[detection(f,30,90+10*f,.99)] for f in range(4,8)}
        path,d=self.run_rebound(candidates=pool)
        self.assertFalse(d['turn_confirmed']);self.assertEqual(path,self.prefix)

    def test_keeper_body_points_do_not_confirm_separated_ball(self):
        pool={str(f):[detection(f,30,100+3*(f-4),.99)] for f in range(4,7)}
        path,d=self.run_rebound(candidates=pool)
        self.assertFalse(d['turn_confirmed'])

    def test_duplicate_images_are_not_independent_support(self):
        pool={str(f):self.pool[str(f)] for f in (4,5,6)}
        _,d=self.run_rebound(candidates=pool,frame_fingerprints={'4':'a','5':'a','6':'b'})
        self.assertFalse(d['turn_confirmed'])

    def test_single_observation_cannot_reset_direction(self):
        path,d=self.run_rebound(candidates={'4':self.pool['4']})
        self.assertFalse(d['turn_confirmed']);self.assertEqual(path,self.prefix)

    def test_refitting_cannot_reduce_confirmed_support_to_two_points(self):
        pool={str(f):[detection(f,30,135-10*f)] for f in (4,5,6)}
        pool['6'][0].update(y=75.025,stabilized_xy=[30,75.025])
        _,d=self.run_rebound(candidates=pool,config=ReboundConfig(fit_slack_px=.007))
        self.assertFalse(d['turn_confirmed'])

    def test_current_pixels_can_reject_colored_glove_stream(self):
        pool=deepcopy(self.pool)
        for qs in pool.values():qs[0]['rebound_current_white_support']=False
        path,d=self.run_rebound(candidates=pool)
        self.assertFalse(d['turn_confirmed']);self.assertEqual(path,self.prefix)

    def test_repeated_false_straight_stream_does_not_suppress_rebound(self):
        pool=deepcopy(self.pool)
        for f in range(4,10):pool[str(f)].append(detection(f,100-20*f,100,.99))
        path,d=self.run_rebound(candidates=pool)
        self.assertTrue(d['turn_confirmed'])
        self.assertTrue(all(q['x']==30 for q in path[3:]))

    def test_contact_and_direction_do_not_establish_goal_outcome(self):
        path,d=self.run_rebound()
        self.assertTrue(d['turn_confirmed'])
        self.assertEqual(d['outcome'],'unknown')
        self.assertFalse(d['whole_ball_crossing_established'])
        self.assertFalse(d['physical_contact_established'])
        self.assertLessEqual(path[-1]['frame'],d['context_end_frame'])


if __name__=='__main__':unittest.main()
