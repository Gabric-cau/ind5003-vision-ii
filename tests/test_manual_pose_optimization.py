import pytest

from scripts.optimize_manual_pose import evaluate, match_person, roi_window, yolo_sparse_label


def target():
    xy = [[None, None] for _ in range(17)]
    xy[13], xy[15] = [30, 40], [30, 60]
    mask = [False] * 17
    mask[13] = mask[15] = True
    return {"id": "19-08_f000001_track0", "video_id": "19-08", "joint_xy_px": xy,
            "coordinate_loss_mask": mask, "review_status": ["visible_corrected"] * 17}


def person(index=0, box=None, confidence=.8):
    return {"candidate_id": index, "bbox": box or [10, 10, 50, 70], "confidence": confidence,
            "keypoints": [[30, 40] for _ in range(17)], "keypoint_confidences": [.9] * 17}


def test_missing_people_and_low_confidence_joints_remain_denominator():
    reference = target()
    predictions = {"rows": [{"frame_id": reference["id"], "width": 100, "height": 100, "people": []}]}
    score = evaluate(predictions, [reference])
    assert score["all"]["reference_visible_points"] == 2
    assert score["all"]["within_10px"] == 0
    predictions["rows"][0]["people"] = [person()]
    predictions["rows"][0]["people"][0]["keypoint_confidences"][15] = .1
    score = evaluate(predictions, [reference])
    assert score["all"]["within_10px"] == .5
    assert score["all"]["predicted_visible_points"] == 1


def test_person_matching_does_not_choose_keypoint_error_oracle():
    bad = person(0, confidence=.9)
    bad["keypoints"] = [[90, 90] for _ in range(17)]
    good = person(1, confidence=.8)
    good["keypoints"][15] = [30, 60]
    selected, metadata = match_person(target(), [good, bad])
    assert selected["candidate_id"] == 0
    assert metadata["uses_gt_only_for_evaluation_association"]


def test_sparse_label_has_no_unreviewed_coordinate_or_visibility_targets():
    reference = target()
    label, bbox = yolo_sparse_label(reference, [20, 20, 40, 50], 100, 100, True)
    values = list(map(float, label.split()))
    assert len(values) == 56
    joints = [values[5 + i * 3:8 + i * 3] for i in range(17)]
    assert joints[13] == [.3, .4, 2]
    assert joints[15] == [.3, .6, 2]
    assert joints[0] == [0, 0, 0]
    assert bbox == [20, 20, 40, 62]
    _, val_bbox = yolo_sparse_label(reference, [20, 20, 40, 50], 100, 100, False)
    assert val_bbox == [20, 20, 40, 50]


def test_automatic_roi_depends_only_on_detector_bbox_and_image_bounds():
    assert roi_window([0, 10, 50, 70], 100, 80) == [0, 0, 60, 80]


def test_exact_ten_pixel_threshold_and_nonuniform_image_height_secondary():
    reference = target()
    p = person()
    p["keypoints"][13] = [40, 40]
    p["keypoints"][15] = [40.1, 60]
    predictions = {"rows": [{"frame_id": reference["id"], "width": 100, "height": 720, "people": [p]}]}
    score = evaluate(predictions, [reference])
    assert score["all"]["within_10px"] == .5
    assert score["all"]["within_2pct_image_height"] == 1.
    assert score["all"]["mean_error_px_available_predictions_only"] == pytest.approx(10.05)
