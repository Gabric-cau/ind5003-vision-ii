from copy import deepcopy

import numpy as np
import pytest

from scripts.analyze_shoe_approach import dense_foot
from scripts.run_fresh_shoe_approach_experiment import sha, write
from src.mediapipe_shoe_evidence import COCO_MAP


def shoe_fixture(tmp_path, ambiguity=None):
    records = []
    for frame in range(4, 21):
        coco = np.zeros((17, 2))
        coco[[11, 12]] = [[160, 100], [175, 100]]
        coco[[13, 14]] = [[155, 150], [180, 150]]
        left = [151, 197] if frame <= 12 else [151 - 12 * (frame - 12), 180]
        right = [230 - 7 * (frame - 4), 200]
        points = np.zeros((33, 2))
        points[COCO_MAP] = coco
        points[[29, 31]] = np.asarray(left) + [[-12, 0], [12, 0]]
        points[[30, 32]] = np.asarray(right) + [[-12, 0], [12, 0]]
        for reflected in (False, True):
            records.append(dict(frame=frame, reflected=reflected, unique_decoded_support=True,
                landmarks_source_xy=points.tolist(), landmark_confidence=[1.] * 33,
                anchor_bbox=[100, 0, 260, 200]))
    payload = dict(contact_associated_track_id=1, contact_bracket_frames=[10, 12],
        stationary_ball_observations=[dict(frame=frame, xy=[150, 200], detection_id=f'{frame}:ball')
                                      for frame in (8, 9, 10)],
        departure_ball_observations=[dict(frame=frame, xy=[100, 180], detection_id=f'{frame}:ball')
                                    for frame in (12, 13)], seed_ambiguous_frames=ambiguity or [])
    path = tmp_path / 'contact.json'
    write(path, {'contact': payload})
    inputs = dict(contact=payload, contact_bracket_frames=[10, 12], fps=30.,
                  input_receipts={'contact_export': {'path': str(path), 'sha256': sha(path)}})
    return records, inputs


def test_dense_region_candidate_uses_swing_instead_of_nearest_planted_foot(tmp_path):
    records, inputs = shoe_fixture(tmp_path)
    result = dense_foot(records, inputs)
    assert result['striking_foot'] == result['swing_candidate'] == 'right'
    assert result['joint_region_frames'] == [12]
    assert result['physical_contact_verified'] is False
    assert result['evidence']['post_departure_pose_used'] is False


def test_after_departure_pose_and_departure_coordinates_do_not_change_candidate(tmp_path):
    records, inputs = shoe_fixture(tmp_path)
    expected = dense_foot(records, inputs)
    for row in records:
        if row['frame'] > 12:
            row['landmarks_source_xy'] = (np.asarray(row['landmarks_source_xy']) * 100).tolist()
    payload = deepcopy(inputs['contact'])
    for ball in payload['departure_ball_observations']:
        ball['xy'] = [10000, 10000]
    path = tmp_path / 'new-contact.json'
    write(path, {'contact': payload})
    inputs['input_receipts']['contact_export'] = {'path': str(path), 'sha256': sha(path)}
    assert dense_foot(records, inputs) == expected


@pytest.mark.parametrize('ambiguous_frame', [5, 10, 12])
def test_local_seed_ambiguity_forces_unknown(tmp_path, ambiguous_frame):
    records, inputs = shoe_fixture(tmp_path, ambiguity=[ambiguous_frame])
    result = dense_foot(records, inputs)
    assert result['striking_foot'] == result['swing_candidate'] == 'unknown'
    assert result['joint_region_frames'] == []


def test_repeated_observations_cannot_supply_missing_strike_motion_pairs(tmp_path):
    records, inputs = shoe_fixture(tmp_path)
    for row in records:
        if row['frame'] in (7, 8, 9):
            row['unique_decoded_support'] = False
    assert dense_foot(records, inputs)['striking_foot'] == 'unknown'


def test_changed_contact_bytes_are_rejected(tmp_path):
    records, inputs = shoe_fixture(tmp_path)
    write(tmp_path / 'contact.json', {'contact': inputs['contact'], 'changed': True})
    with pytest.raises(ValueError, match='Contact export bytes changed'):
        dense_foot(records, inputs)


def test_no_decisive_swing_cannot_be_promoted_by_near_shoe_region(tmp_path):
    records, inputs = shoe_fixture(tmp_path)
    for row in records:
        points = np.asarray(row['landmarks_source_xy'])
        points[[29, 31]] = points[[30, 32]] + [3, 0]
        row['landmarks_source_xy'] = points.tolist()
    result = dense_foot(records, inputs)
    assert result['striking_foot'] == 'unknown'
    assert result['joint_region_frames'] == []
