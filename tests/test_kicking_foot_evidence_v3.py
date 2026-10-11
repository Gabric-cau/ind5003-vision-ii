import unittest
import numpy as np

from src.kicking_foot_evidence_v3 import kicking_foot_evidence_v3


def pose_rows(swap=False):
    rows=[]
    for frame in range(0, 25):
        points=np.tile([140., 130.], (17, 1))
        points[[11,12]]=[[130,150],[150,150]]
        # Left ankle is planted. Right ankle approaches the fixed stationary spot,
        # then passes through it after the departure gate.
        right={10:230,12:215,14:201,17:185,19:170,21:160}.get(frame, 160)
        points[[13,15]]=[[125,180],[140,200]]
        points[[14,16]]=[[155,180],[right,200]]
        if swap:
            points[[13,14]]=points[[14,13]]
            points[[15,16]]=points[[16,15]]
        rows.append(dict(frame=frame,bbox=[100,100,200,200],points=points,
                         confidence=np.full(17,.95)))
    return rows


def call(**kwargs):
    normal=pose_rows(); mirror=pose_rows()
    stationary=kwargs.pop('stationary_ball_observations', [dict(frame=10,detection_id='10:1',xy=[200,200]),
            dict(frame=12,detection_id='12:1',xy=[200,200]),dict(frame=14,detection_id='14:1',xy=[200,200])])
    departure=kwargs.pop('departure_ball_observations', [dict(frame=16,detection_id='16:1',xy=[320,200]),
            dict(frame=17,detection_id='17:2',xy=[380,200])])
    return kicking_foot_evidence_v3(normal,mirror,[14,16],30.,
        stationary_ball_observations=stationary, departure_ball_observations=departure, **kwargs)


class V3Tests(unittest.TestCase):
    def test_paired_stationary_spot_approach_candidate(self):
        result=call()
        self.assertEqual(result['candidate'],'right')
        self.assertEqual(result['status'],'paired_stationary_spot_kinematic_candidate')
        self.assertTrue(result['nearest_side_used_as_label'] is False)

    def test_two_stationary_observations_abstain(self):
        result=call(stationary_ball_observations=[dict(frame=12,detection_id='12:1',xy=[200,200]),
            dict(frame=14,detection_id='14:1',xy=[200,200])])
        self.assertEqual(result['status'],'fewer_than_three_stationary_source_observations')

    def test_departure_is_gate_not_distance_target(self):
        first=call()
        second=call(departure_ball_observations=[dict(frame=16,detection_id='16:1',xy=[500,200]),
            dict(frame=17,detection_id='17:2',xy=[800,200])])
        self.assertEqual(first['candidate'],second['candidate'])
        self.assertEqual(first['views']['normal']['side_metrics'],second['views']['normal']['side_metrics'])

    def test_both_views_must_support_same_side(self):
        result=kicking_foot_evidence_v3(pose_rows(),pose_rows(swap=True),[14,16],30.,
            stationary_ball_observations=[dict(frame=10,detection_id='10:1',xy=[200,200]),dict(frame=12,detection_id='12:1',xy=[200,200]),dict(frame=14,detection_id='14:1',xy=[200,200])],
            departure_ball_observations=[dict(frame=16,detection_id='16:1',xy=[320,200]),dict(frame=17,detection_id='17:2',xy=[380,200])])
        self.assertEqual(result['candidate'],'unknown')

    def test_no_unique_pattern_when_both_ankles_pass(self):
        rows=pose_rows()
        for row in rows:
            if row['frame']>=10:
                row['points'][15,0]=row['points'][16,0]
        result=kicking_foot_evidence_v3(rows,rows,[14,16],30.,
            stationary_ball_observations=[dict(frame=10,detection_id='10:1',xy=[200,200]),dict(frame=12,detection_id='12:1',xy=[200,200]),dict(frame=14,detection_id='14:1',xy=[200,200])],
            departure_ball_observations=[dict(frame=16,detection_id='16:1',xy=[320,200]),dict(frame=17,detection_id='17:2',xy=[380,200])])
        self.assertEqual(result['candidate'],'unknown')


if __name__=='__main__':
    unittest.main()
