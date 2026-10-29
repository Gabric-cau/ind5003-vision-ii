import copy

import pytest

from scripts import prepare_keeper_detector_final as prep
from scripts import train_keeper_detector_final as train


def frame(video, index, status="visible_confirmed"):
    return {"video_id": video, "frame_zero_based": index, "bbox_status": status,
            "bbox_human_reviewed": True,
            "source_json_record": {"identity_resolution_pending": False, "out_of_frame": False}}


def test_selection_keeps_all_training_and_reference_frames_and_never_unknown_negatives():
    rows = [frame("05-04", i) for i in range(8)] + [frame("02-05", i) for i in range(20)]
    rows.append(frame("02-05", 20, "unlocalizable"))
    selected = prep.choose_frames(rows, {("02-05", 7)}, 4)
    assert {("05-04", i) for i in range(8)} <= selected
    assert {("02-05", 0), ("02-05", 6), ("02-05", 7), ("02-05", 13), ("02-05", 19)} <= selected
    assert ("02-05", 20) not in selected
    with pytest.raises(ValueError, match="valid keeper box"):
        prep.choose_frames(rows, {("02-05", 20)})


def sample(video, split, image_hash, event="event"):
    return {"id": video, "video_id": video, "split": split, "event_id": event,
            "subject_group": "subject_02" if video.startswith("02-") else "subject_05",
            "match_id": "fixture", "image_sha256": image_hash, "class_id": 0,
            "class_name": "active_keeper", "bbox": [10, 10, 30, 30],
            "existing_pose_reference": True}


def test_partition_rejects_video_split_hash_and_subject_leakage():
    rows = [sample("05-04", "train", "a"), sample("02-05", "development_subject_02", "b")]
    prep.validate_partition(rows)
    changed = copy.deepcopy(rows)
    changed[1]["split"] = "train"
    with pytest.raises(ValueError, match="frozen partition"):
        prep.validate_partition(changed)
    changed = copy.deepcopy(rows)
    changed[1]["image_sha256"] = "a"
    with pytest.raises(ValueError, match="pixels"):
        prep.validate_partition(changed)
    changed = copy.deepcopy(rows)
    changed[1]["subject_group"] = "subject_05"
    with pytest.raises(ValueError, match="subject or match"):
        prep.validate_partition(changed)


def test_box_target_uses_original_geometry_and_accepts_image_edge():
    assert prep.box_target([0, 0, 100, 50], 100, 50) == [.5, .5, 1., 1.]
    assert prep.box_target([10, 5, 30, 15], 100, 50) == [.2, .2, .2, .2]
    with pytest.raises(ValueError):
        prep.box_target([-1, 5, 30, 15], 100, 50)


def test_inference_descriptor_excludes_reference_and_semantic_labels():
    row = sample("05-04", "train", "abc")
    row.update(image="images/train/example.jpg", width=100, height=50, secret_label="keeper")
    result = train.inference_inputs([row], "/fixture")[0]
    assert set(result) == {"id", "image", "width", "height", "image_sha256"}
    assert result["image"] == "/fixture/images/train/example.jpg"


def test_highest_confidence_is_not_oracle_and_missing_counts_as_failure():
    rows = [sample("02-05", "development_subject_02", "a"),
            sample("02-06", "development_subject_02", "b")]
    predictions = [{"id": "02-05", "detections": [
        {"bbox": [50, 50, 70, 70], "confidence": .9},
        {"bbox": [10, 10, 30, 30], "confidence": .5}]}, {"id": "02-06", "detections": []}]
    score = train.score_predictions(predictions, rows)
    assert score["all"]["frames"] == 2
    assert score["all"]["top1_hits"] == 0
    assert score["all"]["any_candidate_hits"] == 1
    assert score["all"]["any_candidate_recall_iou_0_5"] == .5
    generic = train.score_predictions(predictions, rows, specialized=False)
    assert generic["all"]["top1_hits"] is None
    assert generic["all"]["any_candidate_hits"] == 1


def test_fixed_schedules_match_across_arms_without_repeating_images():
    first = train.schedules_for(["c.jpg", "a.jpg", "b.jpg"], epochs=3)
    assert first == train.schedules_for(["b.jpg", "c.jpg", "a.jpg"], epochs=3)
    assert all(sorted(epoch) == ["a.jpg", "b.jpg", "c.jpg"] for epoch in first)


def test_photometric_transform_has_no_geometry_and_preserves_range():
    torch = pytest.importorskip("torch")
    image = torch.tensor([[[[0., .25, .5, 1.]]]])
    original = image.clone()
    result = train.contrast_brightness(image, 1.1, .02)
    assert result.shape == image.shape and torch.equal(image, original)
    assert float(result.min()) >= 0 and float(result.max()) <= 1
    assert result[0, 0, 0, 2].item() == pytest.approx(.52)
    with pytest.raises(ValueError):
        train.contrast_brightness(image, 2., 0.)


def test_native_horizontal_flip_updates_boxes_for_invariant_keeper_class():
    np = pytest.importorskip("numpy")
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import RandomFlip
    from ultralytics.utils.instance import Instances
    target = {"img": np.zeros((40, 100, 3), np.uint8), "cls": np.array([[0.]]),
              "instances": Instances(np.array([[10., 5., 30., 25.]]),
                  np.empty((0, 0, 2)), bbox_format="xyxy", normalized=False)}
    result = RandomFlip(p=1., direction="horizontal")(target)
    assert result["instances"].bboxes[0].tolist() == pytest.approx([80., 15., 20., 20.])
    assert result["cls"].tolist() == [[0.]]


def test_positive_guard_restores_original_targets_if_geometry_drops_them():
    original = {"cls": [[0.]], "bboxes": [[.5, .5, .1, .2]], "source": "unchanged"}

    def drop(labels):
        labels["source"] = "mutated"
        return {"cls": [], "bboxes": []}

    def fallback(labels):
        assert labels["source"] == "unchanged"
        return labels

    result = train.PositiveKeeperTransforms(drop, fallback)(copy.deepcopy(original))
    assert result["cls"] == [[0.]] and result["keeper_aug_fallback"] is True
    assert result["bboxes"] == original["bboxes"]


def test_optimizer_budget_is_not_naive_samples_divided_by_nominal_batch():
    budget = train.optimizer_budget()
    assert len(budget) == 12 and all(b > a for a, b in zip(budget, budget[1:]))
    assert budget[0] > 835 // 16
    assert budget[-1] > 12 * 835 // 16
