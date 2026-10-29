from copy import deepcopy

import pytest

from scripts.train_keeper_pose_reference_round1 import (
    BULK_ORIGIN, crop_window, evaluate_predictions, paired_changes,
    make_dataset, maximum_trainable_parameter_change, sparse_crop_label,
    trainable_parameter_snapshot, validate_partition, verify_loaded_dataset,
)


def sample(identifier="train", split="train"):
    xy = [[None, None] for _ in range(17)]
    mask = [False] * 17
    status, origin = ["unreviewed"] * 17, [None] * 17
    for joint, point in ((5, [30., 40.]), (6, [50., 40.])):
        xy[joint], mask[joint] = point, True
        status[joint], origin[joint] = "visible_corrected", "original_user_export"
    status[6], origin[6] = "visible_confirmed", BULK_ORIGIN
    return {"id": identifier, "video_id": identifier, "event_id": "event_" + identifier,
            "source_group": "same_venue", "partition_group": "subject_" + identifier,
            "source_sha256": "video_" + identifier, "image_sha256": "image_" + identifier,
            "width": 100, "height": 100, "bbox": [20., 20., 60., 80.],
            "joint_xy_px": xy, "coordinate_loss_mask": mask, "review_status": status,
            "annotation_origin": origin, "split": split}


def person():
    return {"candidate_id": 0, "confidence": .9, "bbox": [20., 20., 60., 80.],
            "keypoints": [[30., 40.] if j == 5 else [50., 40.] for j in range(17)],
            "keypoint_confidences": [.9] * 17}


def prediction(reference, people):
    return {"rows": [{"id": reference["id"], "people": people}]}


def test_fixed_roi_uses_bbox_only_and_clips_to_source_bounds():
    assert crop_window([20, 20, 60, 80], 100, 100) == [10, 5, 70, 95]
    assert crop_window([0, 0, 40, 80], 100, 100) == [0, 0, 50, 100]
    with pytest.raises(ValueError, match="outside"):
        crop_window([-1, 0, 40, 80], 100, 100)


def test_sparse_crop_transforms_only_reviewed_coordinates():
    target = sample()
    label, used, excluded = sparse_crop_label(target, [10, 5, 70, 95])
    values = list(map(float, label.split()))
    assert len(values) == 56
    assert used == [5, 6] and excluded == []
    assert values[5:8] == [0, 0, 0]
    assert values[20:23] == pytest.approx([20 / 60, 35 / 90, 2])
    assert target["joint_xy_px"][5] == [30, 40]


def test_box_point_prompt_is_preserved_if_point_is_in_fixed_roi():
    target = sample()
    target["joint_xy_px"][5] = [65., 40.]
    target["declared_box_point_conflict"] = [j == 5 for j in range(17)]
    _, used, excluded = sparse_crop_label(target, crop_window(target["bbox"], 100, 100))
    assert 5 in used and not excluded
    score = evaluate_predictions(prediction(target, []), [target])
    assert score["by_box_point_prompt"]["flagged"]["denominator"] == 1


def test_outside_crop_point_is_not_fabricated_or_clamped():
    target = sample()
    target["joint_xy_px"][5] = [80., 40.]
    label, used, excluded = sparse_crop_label(target, [10, 5, 70, 95])
    assert used == [6]
    assert excluded == [{"joint_id": 5, "reason": "reference_outside_fixed_crop"}]
    assert list(map(float, label.split()))[20:23] == [0, 0, 0]
    assert target["joint_xy_px"][5] == [80., 40.]


@pytest.mark.parametrize("status", ["unreviewed", "occluded", "left_right_ambiguous"])
def test_nonvisible_joint_cannot_receive_coordinate_loss(status):
    target = sample()
    target["review_status"][5] = status
    with pytest.raises(ValueError, match="visible"):
        sparse_crop_label(target, [10, 5, 70, 95])


@pytest.mark.parametrize("field", ["partition_group", "event_id", "source_sha256", "image_sha256"])
def test_group_or_identical_pixel_leakage_is_rejected(field):
    train, development = sample(), sample("dev", "development_subject_02")
    development[field] = train[field]
    with pytest.raises(ValueError, match="crosses"):
        validate_partition([train, development])


def test_same_venue_is_allowed_without_claiming_unseen_match():
    validate_partition([sample(), sample("dev", "development_subject_02")])


def test_missing_person_and_low_confidence_joint_remain_in_denominator():
    target = sample()
    missing = evaluate_predictions(prediction(target, []), [target])
    assert missing["all"]["denominator"] == 2
    assert missing["all"]["within_10px"] == 0
    p = person()
    p["keypoint_confidences"][6] = .1
    score = evaluate_predictions(prediction(target, [p]), [target])
    assert score["all"]["within_10px_hits"] == 1
    assert score["all"]["available_predictions"] == 1
    assert score["by_stratum"]["original_visible_corrected"]["within_10px"] == 1
    assert score["by_stratum"]["bulk_reference_confirmation"]["within_10px"] == 0


def test_bbox_association_never_selects_by_keypoint_error():
    target, good, bad = sample(), person(), person()
    good["candidate_id"], good["confidence"] = 1, .8
    bad["keypoints"] = [[90., 90.]] * 17
    score = evaluate_predictions(prediction(target, [good, bad]), [target])
    assert score["all"]["within_10px_hits"] == 0


def test_prediction_inventory_must_be_complete_and_unique():
    target = sample()
    with pytest.raises(ValueError, match="inventory"):
        evaluate_predictions({"rows": []}, [target])
    rows = prediction(target, [person()])["rows"]
    with pytest.raises(ValueError, match="inventory"):
        evaluate_predictions({"rows": rows + rows}, [target])


def test_paired_changes_preserve_origin_and_count_corrections_and_regressions():
    target, old, new = sample(), person(), person()
    old["keypoints"][5] = [80., 80.]
    new["keypoints"][6] = [90., 90.]
    before = evaluate_predictions(prediction(target, [old]), [target])
    after = evaluate_predictions(prediction(target, [new]), [target])
    changes = paired_changes(before, after)
    assert changes["all"] == {"corrected": 1, "regressed": 1, "both_correct": 0, "both_incorrect": 0}
    assert changes["by_stratum"]["original_visible_corrected"]["corrected"] == 1
    assert changes["by_stratum"]["bulk_reference_confirmation"]["regressed"] == 1
    modified = deepcopy(after)
    modified["points"][0]["reference_xy"] = [1, 2]
    with pytest.raises(ValueError, match="identity"):
        paired_changes(before, modified)


def test_parameter_delta_detects_active_later_scale_with_dormant_first_scale():
    torch = pytest.importorskip("torch")
    model = torch.nn.ModuleDict({"cv4_0": torch.nn.Linear(1, 1, bias=False),
                                "cv4_2": torch.nn.Linear(1, 1, bias=False)})
    initial = trainable_parameter_snapshot(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model["cv4_2"](torch.ones(1, 1)).sum().backward()
    optimizer.step()
    peak, name = maximum_trainable_parameter_change(model, initial)
    assert peak == pytest.approx(.1)
    assert name == "cv4_2.weight"
    assert torch.equal(model["cv4_0"].weight, initial["cv4_0.weight"])


def test_parameter_delta_rejects_nonfinite_or_changed_trainable_inventory():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    initial = trainable_parameter_snapshot(model)
    assert maximum_trainable_parameter_change(model, initial) == (0., None)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="Non-finite"):
        maximum_trainable_parameter_change(model, initial)
    model.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="inventory"):
        maximum_trainable_parameter_change(model, initial)


def test_dataset_records_uncertain_only_frame_without_creating_negative_label(tmp_path):
    import json
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    visible, uncertain = sample(), sample("uncertain")
    visible["roi_image"] = tmp_path / "roi.png"
    assert cv2.imwrite(str(visible["roi_image"]), numpy.zeros((90, 60, 3), dtype=numpy.uint8))
    uncertain["coordinate_loss_mask"] = [False] * 17
    uncertain["joint_xy_px"] = [[None, None] for _ in range(17)]
    uncertain["review_status"] = ["occluded"] * 17
    output = tmp_path / "dataset"
    make_dataset([visible, uncertain], output)
    evidence = json.loads((output / "provenance.json").read_text())
    assert evidence["input_training_frames"] == 2
    assert evidence["exported_training_images"] == 1
    assert evidence["skipped_training_frames"] == 1
    assert evidence["coordinate_supervision_points"] == 2
    assert evidence["frames"][1]["skipped_reason"] == "no_visible_coordinate_supervision_in_fixed_roi"
    assert len(list((output / "labels/train").glob("*.txt"))) == 1


def test_loader_inventory_checks_both_images_and_visible_coordinates():
    numpy = pytest.importorskip("numpy")

    class Dataset:
        labels = [{"keypoints": numpy.zeros((1, 17, 3))}]

        def __len__(self):
            return 1

    Dataset.labels[0]["keypoints"][0, 5:7, 2] = 2
    expected = {"exported_training_images": 1, "coordinate_supervision_points": 2}
    assert verify_loaded_dataset(Dataset(), expected)["loader_preserved_exported_training_inventory"]
    for changed in ({"exported_training_images": 2}, {"coordinate_supervision_points": 3}):
        with pytest.raises(RuntimeError, match="Loader changed"):
            verify_loaded_dataset(Dataset(), {**expected, **changed})
