from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_keeper_tiled_proposals as tiled


def person(box, confidence=.8):
    return {"candidate_id": 0, "bbox": list(box), "confidence": confidence,
            "keypoints": [[box[0], box[1]]] * 17, "keypoint_confidences": [.9] * 17}


class Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def result(people):
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([p["bbox"] for p in people]),
                                                conf=Tensor([p["confidence"] for p in people])),
                           keypoints=SimpleNamespace(xy=Tensor([p["keypoints"] for p in people]),
                                                     conf=Tensor([p["keypoint_confidences"] for p in people])))


def test_inference_payload_contains_no_reference_or_roi():
    samples = [{"id": "f", "image": "source.jpg", "width": 1280, "height": 720,
                "bbox": [10, 10, 20, 30], "roi_image": "human.png", "joint_xy_px": [[1, 2]]}]
    assert tiled.inference_frames(samples) == [{key: samples[0][key] for key in ("id", "image", "width", "height")}]


def test_tiling_covers_upper_image_and_uses_only_dimensions():
    windows = tiled.tile_windows(1920, 1080, tiled.TILE_EDGE, tiled.TILE_OVERLAP)
    assert len(windows) == 6
    assert any(x0 <= 912 and x1 >= 966 and y0 <= 258 and y1 >= 353 for x0, y0, x1, y1 in windows)
    assert any(y0 == 0 for _, y0, _, _ in windows)


def test_tiled_coordinates_full_union_and_nms_are_label_free(monkeypatch):
    frame = {"id": "f", "image": "unused", "width": 1280, "height": 720}
    monkeypatch.setattr(tiled, "read_image", lambda _: np.zeros((720, 1280, 3), dtype=np.uint8))
    outputs = [result([person([800, 100, 900, 300], .8)]),
               result([person([480, 100, 580, 300], .9)])]
    calls = []

    class Model:
        def predict(self, images, **kwargs):
            calls.append(kwargs)
            return [outputs.pop(0) for _ in images]

    full = {"rows": [{"id": "f", "people": [person([10, 20, 30, 60], .5)]}],
            "seconds": 1., "reference_labels_used_for_prediction": False}
    prediction = tiled.make_tiled_seeds(Model(), [frame], full, device="cpu", batch=2)
    row = prediction["rows"][0]
    assert len(row["raw_people"]) == 3
    assert len(row["people"]) == 2
    assert row["people"][0]["bbox"] == [800, 100, 900, 300]
    assert row["people"][0]["keypoints"][0] == [800, 100]
    assert row["people"][0]["proposal_crop_xyxy"] == [320, 0, 1280, 720]
    assert [p["candidate_id"] for p in row["people"]] == [0, 1]
    assert calls[0]["imgsz"] == 1280 and calls[0]["conf"] == .1
    assert prediction["reference_labels_used_for_prediction"] is False


def test_small_image_uses_existing_full_without_duplicate_full_call(monkeypatch):
    frame = {"id": "f", "image": "unused", "width": 20, "height": 30}
    monkeypatch.setattr(tiled, "read_image", lambda _: np.zeros((30, 20, 3), dtype=np.uint8))
    full = {"rows": [{"id": "f", "people": []}], "seconds": 0., "reference_labels_used_for_prediction": False}
    prediction = tiled.make_tiled_seeds(object(), [frame], full, device="cpu", batch=1)
    assert prediction["rows"][0]["tile_receipts"] == []
    assert prediction["rows"][0]["people"] == []


def test_box_validation_and_inventory_are_strict():
    with pytest.raises(ValueError, match="outside source"):
        tiled.validate_people([person([0, 0, 20, 31])], 20, 30)
    with pytest.raises(ValueError, match="Non-finite"):
        tiled.validate_people([person([0, 0, float("nan"), 30])], 20, 30)
    with pytest.raises(ValueError, match="inventory"):
        tiled.make_tiled_seeds(object(), [], {"rows": [{"id": "extra", "people": []}]}, device="cpu", batch=1)


def test_nms_keeps_distinct_people_and_stable_ties():
    people = [person([1, 2, 20, 50]), person([1, 2, 20, 50]), person([30, 2, 50, 50])]
    kept = tiled.suppress_duplicates(people)
    assert len(kept) == 2
    assert [p["bbox"] for p in kept] == [people[0]["bbox"], people[2]["bbox"]]


def test_coverage_counts_missing_and_reports_candidate_scope():
    samples = [{"id": "a", "bbox": [0, 0, 20, 40], "split": "development", "event_id": "e"},
               {"id": "b", "bbox": [0, 0, 20, 40], "split": "development", "event_id": "e"}]
    prediction = {"rows": [{"id": "a", "people": [person([0, 0, 20, 40])]}, {"id": "b", "people": []}]}
    metrics = tiled.proposal_coverage(prediction, samples)
    assert metrics["all"]["reference_frames"] == 2
    assert metrics["all"]["candidate_recall_iou_0_3"] == .5
    assert metrics["by_split"]["development"]["covered_iou_0_5"] == 1
    assert metrics["automatic_keeper_role_accuracy"] is None
    assert metrics["reference_used_only_for_evaluation"] is True


def test_coverage_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="inventories"):
        tiled.proposal_coverage({"rows": []}, [{"id": "missing"}])


def test_actual_device_check_uses_predictor_backend():
    model = SimpleNamespace(predictor=SimpleNamespace(device=SimpleNamespace(type="mps")))
    assert tiled.verify_inference_device(model, "mps")
    with pytest.raises(RuntimeError, match="requested device"):
        tiled.verify_inference_device(model, "cpu")
