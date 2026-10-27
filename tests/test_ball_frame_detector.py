import pytest
import json

from scripts.train_ball_frame_detector import box_iou, score_frame, summarize, digest, validate_exports


def test_missing_prediction_stays_in_localization_denominator():
    missed = score_frame([0, 0, 10, 10], [])
    found = score_frame([0, 0, 10, 10], [{"bbox": [0, 0, 10, 10], "confidence": .9}])
    metrics = summarize([missed, found])
    assert metrics["annotated_frames"] == 2
    assert metrics["prediction_available_frames"] == 1
    assert metrics["top1_localization_recall_iou_0_5"] == .5


def test_target_matching_does_not_use_oracle_candidate_as_top1():
    row = score_frame([0, 0, 10, 10], [{"bbox": [0, 0, 10, 10], "confidence": .1},
                                      {"bbox": [20, 20, 30, 30], "confidence": .9}])
    assert row["top1_iou"] == 0
    assert row["any_candidate_iou"] == 1


def test_iou_uses_box_overlap_not_center_distance():
    assert box_iou([0, 0, 10, 10], [2, 2, 8, 8]) == pytest.approx(.36)
    assert box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0


def test_empty_evaluation_does_not_report_success():
    with pytest.raises(ValueError):
        summarize([])


def test_training_rejects_label_edits_after_export(tmp_path):
    image, label = tmp_path / "core_01_f000030.jpg", tmp_path / "core_01_f000030.txt"
    image.write_bytes(b"fixture image")
    label.write_text("0 .5 .5 .1 .1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"development_only": True, "independent_ground_truth": False,
        "exports": [{"image": image.name, "label": label.name, "split": "train",
                     "image_sha256": digest(image), "label_sha256": digest(label)}]}))
    items = {"train": [{"image": image, "label": label}]}
    validate_exports(manifest, items)
    label.write_text("0 .1 .1 .1 .1")
    with pytest.raises(ValueError, match="checksum"):
        validate_exports(manifest, items)
