from copy import deepcopy

import numpy as np
import pytest

from src.kicking_foot_strike_window import kicking_foot_strike_window


def fixture():
    samples = []
    for frame in range(4, 21):
        points = np.zeros((17, 2))
        points[[11, 12]] = [[160, 100], [175, 100]]
        points[[13, 14]] = [[155, 150], [180, 150]]
        # Left is nearer but planted; right swings into the ball region.
        points[15] = [151, 197] if frame <= 12 else [151 - 12 * (frame - 12), 180]
        points[16] = [230 - 7 * (frame - 4), 200]
        samples.append(dict(frame=frame, bbox=[100, 0, 260, 200],
                            keypoints=points.tolist(), confidence=[1.] * 17))
    stationary = [dict(frame=frame, xy=[150, 200], detection_id=f'{frame}:ball') for frame in (8, 9, 10)]
    departure = [dict(frame=frame, xy=[100, 180], detection_id=f'{frame}:ball') for frame in (12, 13)]
    return samples, deepcopy(samples), stationary, departure


def run(normal, mirror, stationary, departure):
    return kicking_foot_strike_window(normal, mirror, [10, 12], 30.,
        stationary_ball_observations=stationary, departure_ball_observations=departure)


def test_nearest_planted_foot_and_later_recoil_cannot_choose_label():
    result = run(*fixture())
    assert result['candidate'] == 'right'
    assert result['status'] == 'paired_predeparture_strike_motion_candidate'
    assert result['views']['paired']['target_relevance']['left']['minimum_bracket_ankle_distance_body'] < result['views']['paired']['target_relevance']['right']['minimum_bracket_ankle_distance_body']
    assert result['post_departure_pose_used'] is False
    assert result['shoe_ball_contact_verified'] is False


def test_post_departure_pose_and_departing_ball_coordinates_do_not_affect_prediction():
    args = fixture()
    expected = run(*args)
    for rows in args[:2]:
        for row in rows:
            if row['frame'] > 12:
                row['keypoints'] = (np.asarray(row['keypoints']) * 100).tolist()
    for ball in args[3]:
        ball['xy'] = [10000, 10000]
    assert run(*args) == expected


def test_equal_bilateral_motion_is_unknown():
    args = fixture()
    for rows in args[:2]:
        for row in rows:
            row['keypoints'][15] = [row['keypoints'][16][0] + 3, 200]
    assert run(*args)['candidate'] == 'unknown'


def test_motion_away_from_ball_region_is_unknown():
    args = fixture()
    for ball in args[2]:
        ball['xy'] = [-1000, -1000]
    result = run(*args)
    assert result['candidate'] == 'unknown'
    assert result['status'] == 'motion_side_not_approaching_stationary_ball_region'


def test_mirror_disagreement_and_pose_gap_cannot_be_filled():
    normal, mirror, stationary, departure = fixture()
    for row in mirror:
        row['keypoints'][16][0] += 80
    assert run(normal, mirror, stationary, departure)['candidate'] == 'unknown'
    normal, mirror, stationary, departure = fixture()
    normal = [row for row in normal if row['frame'] not in (7, 8, 9)]
    mirror = [row for row in mirror if row['frame'] not in (7, 8, 9)]
    assert run(normal, mirror, stationary, departure)['candidate'] == 'unknown'


def test_source_translation_and_scale_leave_label_unchanged():
    args = fixture()
    for rows in args[:2]:
        for row in rows:
            row['keypoints'] = (np.asarray(row['keypoints']) * 3 + [1000, 2000]).tolist()
            row['bbox'] = (np.asarray(row['bbox']) * 3 + [1000, 2000, 1000, 2000]).tolist()
    for balls in args[2:]:
        for ball in balls:
            ball['xy'] = (np.asarray(ball['xy']) * 3 + [1000, 2000]).tolist()
    assert run(*args)['candidate'] == 'right'


def test_missing_actual_departure_and_duplicate_ball_frames_rejected():
    normal, mirror, stationary, departure = fixture()
    assert run(normal, mirror, stationary, departure[:1])['candidate'] == 'unknown'
    with pytest.raises(ValueError, match='one selected ball'):
        run(normal, mirror, stationary + stationary[:1], departure)


def test_temporal_channel_swap_stays_unknown_without_anatomical_relabeling():
    args = fixture()
    for rows in args[:2]:
        for row in rows:
            if row['frame'] == 9:
                points = np.asarray(row['keypoints'])
                points[[13, 14, 15, 16]] = points[[14, 13, 16, 15]]
                row['keypoints'] = points.tolist()
    result = run(*args)
    assert result['candidate'] == 'unknown'
    assert result['status'] == 'temporal_anatomical_channel_ambiguity'
    assert result['anatomical_channel_permutation_applied'] is False
