import numpy as np
from src.kick_contact import score_contact_candidates, _score_all_tracks


def fixture(moving=True, camera_shift=0):
    balls=[]
    transforms={}
    for frame in range(0,61,6):
        shift=camera_shift*frame
        transforms[frame]=np.array([[1,0,-shift],[0,1,0],[0,0,1]],float)
        if frame<=30:
            x=100
        elif moving:
            x=100+(frame-30)*4
        else:
            continue
        balls.append({"frame":frame,"xy":[x+shift,100],"diameter":6,"confidence":.8})
    poses=[]
    for frame in range(61):
        shift=camera_shift*frame
        x=100+abs(frame-30)*3
        poses.append({"frame":frame,"ankles":[[x+shift,102]],"height":100})
        transforms[frame]=np.array([[1,0,-shift],[0,1,0],[0,0,1]],float)
    return balls,poses,transforms


def test_independent_stationary_then_departure():
    b,p,t=fixture()
    result=score_contact_candidates(b,p,t,30)
    assert result[0]["contact_frame"]==30
    assert result[0]["ball_departure_frames"]
    assert result[0]["score"]>=.75


def test_missing_ball_abstains_at_candidate_level():
    _,p,t=fixture()
    assert score_contact_candidates([],p,t,30)==[]


def test_disappearance_is_not_confirmed_departure():
    b,p,t=fixture(moving=False)
    result=score_contact_candidates(b,p,t,30)
    assert result[0]["ball_departure_frames"] is None
    assert "ball_departure_not_confirmed" in result[0]["warnings"]


def test_camera_translation_compensation():
    b,p,t=fixture(camera_shift=2)
    result=score_contact_candidates(b,p,t,30)
    assert result[0]["contact_frame"]==30
    assert result[0]["ball_departure_frames"]


def test_absolute_frame_offsets_are_preserved():
    b,p,t=fixture()
    for ball in b: ball["frame"]+=90
    for pose in p: pose["frame"]+=90
    t={frame+90:matrix for frame,matrix in t.items()}
    result=score_contact_candidates(b,p,t,30)
    assert result[0]["contact_frame"]==120


def test_no_ankle_encounter_no_candidate():
    b,p,t=fixture()
    for pose in p: pose["ankles"]=[[800,500]]
    assert score_contact_candidates(b,p,t,30)==[]


def test_stationary_object_persisting_after_contact_is_warned():
    b,p,t=fixture()
    for frame in range(36,61,6):
        b.append({"frame":frame,"xy":[100,100],"diameter":6,"confidence":.8})
    result=score_contact_candidates(b,p,t,30)
    assert "stationary_ball_persists_after_candidate" in result[0]["warnings"]


def test_bystander_already_near_false_ball_is_penalized():
    b,p,t=fixture()
    for pose in p: pose["ankles"]=[[100,102]]
    result=score_contact_candidates(b,p,t,30)
    assert not result[0]["ankle_approach_supported"]
    assert result[0]["score"]<.60


def test_approaching_kicker_beats_bystander_identity():
    b,p,t=fixture()
    bystander=[dict(pose,ankles=[[100,100]]) for pose in p]
    result=_score_all_tracks(b,{1:bystander,7:p},t,30,6)
    assert result[0]["contact_associated_track_id"]==7


def test_candidate_window_accounts_for_support_foot_before_strike():
    b,p,t=fixture()
    result=score_contact_candidates(b,p,t,30,sample_stride=2)
    assert result[0]["interval_frames"][1]>=result[0]["contact_frame"]+7
    assert result[0]["interval_kind"]=="heuristic_candidate_window_not_statistical_confidence_interval"


def test_contact_observations_preserve_source_pixels_and_associated_departure():
    b,p,t=fixture(camera_shift=2)
    for index, ball in enumerate(b):
        ball['detection_id']=f'ball-{index}'
        ball['bbox']=[ball['xy'][0]-3,97,ball['xy'][0]+3,103]
    result=score_contact_candidates(b,p,t,30)[0]
    observed={ball['detection_id']:ball for ball in b}
    assert result['stationary_ball_reference_xy']==[100.,100.]
    for observation in result['stationary_ball_observations']+result['departure_ball_observations']:
        assert observation['xy']==observed[observation['detection_id']]['xy']
        assert observation['bbox']==observed[observation['detection_id']]['bbox']
    assert result['stationary_ball_observations'][-1]['xy']==[160.,100.]
    assert [row['frame'] for row in result['departure_ball_observations']]==result['ball_departure_frames']
    assert result['contact_bracket_frames']==[30,36]


def test_same_seed_frame_competitors_are_explicit_and_have_one_primary_observation():
    b,p,t=fixture()
    for index, ball in enumerate(b):ball['detection_id']=f'ball-{index}'
    b.append({**b[2],'detection_id':'other-view','confidence':.9,'xy':[101,100]})
    result=score_contact_candidates(b,p,t,30)[0]
    same_frame=[row for row in result['stationary_ball_observations'] if row['frame']==12]
    assert len(same_frame)==1
    assert same_frame[0]['detection_id']=='other-view'
    assert result['stationary_identity_ambiguous_frames']==[12]
    assert {row['detection_id'] for row in result['stationary_observation_competitors'][0]['observations']}=={'ball-2','other-view'}


def test_seed_only_gate_preserves_unmasked_neural_departure_observations():
    b,p,t=fixture()
    for index, ball in enumerate(b):ball['detection_id']=f'ball-{index}'
    seed_ids={ball['detection_id'] for ball in b if ball['frame']<=30}
    result=score_contact_candidates(b,p,t,30,stationary_seed_detection_ids=seed_ids)[0]
    assert result['ball_departure_frames']==[36,42]
    assert [row['detection_id'] for row in result['departure_ball_observations']]==['ball-6','ball-7']
    assert all(row['detection_id'] in seed_ids for row in result['stationary_ball_observations'])
