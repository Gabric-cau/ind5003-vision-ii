import numpy as np

from scripts import extract_external_ball_geometry as geometry


def test_absent_evidence_is_encoded_with_masks_and_finite_features():
    record = {"observations": [{"goal": {}, "balls": []} for _ in range(64)]}
    features = geometry.geometry_descriptor(record)
    assert features.shape == (48,)
    np.testing.assert_array_equal(features, np.zeros(48))


def test_goal_plane_center_and_evidence_mask_are_explicit():
    goal = {"selected": {"corners": [[100, 100], [300, 100], [300, 200], [100, 200]]}}
    ball = {"bbox": [195, 145, 205, 155], "confidence": .8}
    record = {"observations": [{"goal": goal, "balls": [ball]} for _ in range(64)]}
    features = geometry.geometry_descriptor(record).reshape(4, 12)
    np.testing.assert_allclose(features[:, :2], 1)
    np.testing.assert_allclose(features[:, 2:6], np.tile([.8, .5, .5, 1], (4, 1)))


def test_geometry_features_do_not_depend_on_reference_labels_or_event_id():
    record = {"observations": [{"goal": {}, "balls": []} for _ in range(64)]}
    changed = {**record, "targets": {"ball_direction": "left"}, "event_id": "anything"}
    np.testing.assert_array_equal(geometry.geometry_descriptor(record), geometry.geometry_descriptor(changed))
