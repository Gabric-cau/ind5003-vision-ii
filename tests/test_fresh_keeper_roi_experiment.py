from copy import deepcopy

from scripts.run_fresh_keeper_roi_experiment import augment_pose, plan_roi, same_pre_person


def observation(frame=8, x=100, confidence=.6):
    return dict(frame=frame, bbox=[x, 100, x+40, 200],
                keypoints=[[x+20, 140]]*17, keypoint_confidences=[confidence]*17)


def test_pre_reference_replacement_requires_same_person_geometry():
    original = observation()
    good, other = observation(confidence=.9), observation(x=140, confidence=.99)
    assert same_pre_person(good, original)
    assert not same_pre_person(other, original)
    pose = dict(tracks=[dict(track_id=2, observations=[original])])
    augmented, replaced = augment_pose(pose, {8: [good, other], 11: [good]},
                                       {'seed_track_id': 2}, 10)
    assert replaced == [8]
    assert augmented['tracks'][0]['observations'][0]['keypoint_confidences'][0] == .9
    assert original['keypoint_confidences'][0] == .6
    assert len(augmented['tracks']) == 1


def test_weaker_roi_observation_does_not_replace_reference():
    pose = dict(tracks=[dict(track_id=2, observations=[observation()])])
    augmented, replaced = augment_pose(pose, {8: [observation(confidence=.5)]},
                                       {'seed_track_id': 2}, 10)
    assert not replaced
    assert augmented == pose


def test_plan_is_bounded_by_source_and_same_shot():
    pose = dict(fps=30, shot_start_frame=0, shot_end_frame=40, frame_width=640,
                frame_height=360, tracks=[dict(track_id=2, observations=[observation(f) for f in [6, 7, 8]])])
    event = dict(kick_frame=10, goal_proposals={'7': {'selected': {'bbox': [40, 30, 600, 210]}}})
    role = dict(status='candidate_supported', seed_track_id=2)
    plan = plan_roi(pose, event, role)
    assert plan['roi_xyxy'] == [0, 0, 640, 290]
    assert plan['start_frame'] == 0
    assert plan['end_frame_exclusive'] == 40
    assert plan_roi(pose, event, {'status': 'abstain'}) is None


def test_post_contact_other_person_keeps_original_exclusion_identity():
    keeper, referee = observation(11), observation(11, x=200)
    pose = dict(tracks=[dict(track_id=2, observations=[keeper]),
                        dict(track_id=1, observations=[referee])])
    extra = {11: [observation(11, x=200, confidence=.9),
                  observation(11, x=300, confidence=.99)]}
    augmented, _ = augment_pose(pose, extra, {'seed_track_id': 2}, 10)
    assert [track['track_id'] for track in augmented['tracks']] == [2, 1]
    assert augmented['tracks'][1]['observations'][0]['original_source_track_id'] == 1
    assert augmented['tracks'][0]['observations'][0] == keeper
