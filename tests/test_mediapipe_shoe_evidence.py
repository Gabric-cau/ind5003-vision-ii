import numpy as np
import pytest

from src.mediapipe_shoe_evidence import coco_samples, restore_landmarks, segment_distance, shoe_ball_distances, source_body_box
from src.pose_refinement import select_anchored_detection


def test_inverse_mirror_restores_anatomical_heel_and_toe_channels():
    points = np.full((33, 2), .5)
    points[29], points[30], points[31], points[32] = [[.1, .2], [.3, .4], [.5, .6], [.7, .8]]
    confidence = np.arange(33) / 33
    xy, conf = restore_landmarks(points, confidence, [100, 200, 200, 400], reflected=True)
    assert np.allclose(xy[29], [169, 280])
    assert np.allclose(xy[31], [129, 360])
    assert conf[29] == confidence[30]
    assert conf[31] == confidence[32]


def test_segment_distance_uses_shoe_segment_and_clamps_outside_endpoints():
    assert segment_distance([2, 1], [0, 0], [4, 0]) == 1
    assert segment_distance([5, 0], [0, 0], [4, 0]) == 1
    assert segment_distance([0, 0], [3, 4], [3, 4]) == 5


def test_shoe_centroid_motion_uses_heel_and_toe_and_their_lower_confidence():
    points = np.zeros((33, 2)); confidence = np.ones(33)
    points[[29, 31]] = [[100, 100], [120, 100]]
    confidence[31] = .2
    row = dict(frame=10, anchor_bbox=[0, 0, 200, 200], landmarks_source_xy=points.tolist(), landmark_confidence=confidence.tolist())
    sample = coco_samples([row], shoe_centroids=True)[0]
    assert sample['keypoints'][15] == [110, 100]
    assert sample['confidence'][15] == .2
    dist = shoe_ball_distances([row], [110, 100], 200)[0]['left']
    assert dist['stationary_point_distance_source_px'] == 0
    assert dist['confident'] is False


def test_invalid_shape_and_degenerate_scale_are_rejected():
    with pytest.raises(ValueError):
        restore_landmarks(np.zeros((32, 2)), np.ones(33), [0, 0, 100, 100])
    with pytest.raises(ValueError):
        shoe_ball_distances([], [0, 0], 0)


def test_torso_region_must_match_source_actor_and_needs_visible_hips():
    points = np.zeros((33, 2)); confidence = np.ones(33)
    points[[0, 11, 12, 23, 24, 25, 26, 27, 28]] = [
        [110, 10], [90, 50], [130, 50], [100, 110], [120, 110],
        [95, 140], [125, 140], [90, 190], [130, 190]]
    anchor = [80, 0, 140, 200]
    box = source_body_box(points, confidence)
    assert select_anchored_detection([dict(bbox=box)], anchor)[0] is not None
    rival = source_body_box(points + [400, 0], confidence)
    assert select_anchored_detection([dict(bbox=rival)], anchor)[0] is None
    confidence[23] = .2
    assert source_body_box(points, confidence) is None


def test_overlapping_rival_cannot_claim_actor_anchor():
    anchor = [0, 0, 100, 200]
    rival = [40, 0, 140, 200]
    assert select_anchored_detection([dict(bbox=rival)], anchor)[0] is not None
    assert select_anchored_detection([dict(bbox=rival)], anchor, rivals=[rival])[0] is None


def test_out_of_crop_foot_is_missing_and_not_clamped_to_the_boundary():
    points = np.full((33, 2), .5)
    points[29] = [-.1, .5]
    points[32] = [1.1, .5]
    xy, confidence = restore_landmarks(points, np.ones(33), [100, 200, 200, 400])
    assert confidence[29] == confidence[32] == 0.
    assert xy[29, 0] == 90 and xy[32, 0] > 200
