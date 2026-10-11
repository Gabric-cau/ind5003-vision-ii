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
