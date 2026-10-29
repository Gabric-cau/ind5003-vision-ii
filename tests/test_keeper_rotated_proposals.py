from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_keeper_rotated_proposals as rotated


def person(box, confidence=.8):
    return {"candidate_id": 0, "bbox": list(box), "confidence": confidence,
            "keypoints": [[box[0], box[1]]] * 17, "keypoint_confidences": [.9] * 17}


class Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value, dtype=float)

    def tolist(self):
        return self.value


def result(people):
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([p["bbox"] for p in people]),
                                                conf=Tensor([p["confidence"] for p in people])),
                           keypoints=SimpleNamespace(xy=Tensor([p["keypoints"] for p in people]),
                                                     conf=Tensor([p["keypoint_confidences"] for p in people])))


def test_rotation_geometry_preserves_center_and_bounds():
    # Original crop is W=100, H=60. A center point has different inverse
    # coordinates under clockwise and counter-clockwise rotations.
    class Model:
        def predict(self, images, **kwargs):
            return [result([person([10, 20, 30, 40])])]

    # Fake result is in a 90-degree rotated image: inverse maps point and box.
    mapped = rotated.map_rotated_people(result([person([20, 10, 40, 30])]),
                                         [0, 0, 100, 60], 90, 100, 60)
    assert mapped[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    mapped_ccw = rotated.map_rotated_people(result([person([20, 70, 40, 90])]),
                                             [0, 0, 100, 60], 270, 100, 60)
    assert mapped_ccw[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_rotation_clips_keypoints_as_pixel_centers_and_keeps_joint_order():
    mapped = rotated.map_rotated_people(result([person([-5, -4, 110, 70])]),
                                         [10, 20, 110, 80], 90, 120, 100)
    points = np.asarray(mapped[0]["keypoints"])
    assert points[:, 0].min() >= 10 and points[:, 0].max() <= 109
    assert points[:, 1].min() >= 20 and points[:, 1].max() <= 79
    assert len(points) == 17


def test_rotated_seed_requires_dimension_only_frozen_windows():
    frame = {"id": "f", "image": "unused", "width": 1280, "height": 720}
    expected = [list(w) for w in rotated.tile_windows(1280, 720, rotated.TILE_EDGE, rotated.TILE_OVERLAP)
                if w != (0, 0, 1280, 720)]
    frozen = {"rows": [{"id": "f", "raw_people": [],
                         "tile_receipts": [{"crop_xyxy": w} for w in expected]}]}
    class Model:
        def predict(self, images, **kwargs):
            return [result([]) for _ in images]
    original = rotated.read_image
    rotated.read_image = lambda _: np.zeros((720, 1280, 3), dtype=np.uint8)
    try:
        output = rotated.make_rotated_seed_proposals(Model(), [frame], frozen, device="cpu", batch=2)
    finally:
        rotated.read_image = original
    assert len(output["rows"]) == 1
    assert output["rows"][0]["raw_people_by_angle"] == {"0": [], "90": [], "270": []}
    assert output["reference_labels_used_for_prediction"] is False


def test_rotated_roi_selects_highest_confidence_angle_without_reference():
    frame = {"id": "f", "image": "unused", "width": 100, "height": 100}
    seeds = {"rows": [{"id": "f", "people": [person([30, 30, 60, 80], .7)]}]}
    monkey = []
    original = rotated.read_image
    rotated.read_image = lambda _: np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        class Model:
            def predict(self, images, **kwargs):
                monkey.append(kwargs)
                # 90-degree view gets the highest eligible confidence.
                return [result([person([8, 13, 38, 63], .2)]),
                        result([person([13, 8, 63, 38], .9)]), result([])]
        prediction = rotated.predict_rotated_rois(Model(), [frame], seeds, device="cpu", batch=3)
    finally:
        rotated.read_image = original
    receipt = prediction["rows"][0]["raw_people_by_seed"][0]
    assert receipt["selected_angle"] == 90
    assert len(receipt["raw_candidates_by_angle"]["0"]) == 1
    assert prediction["reference_labels_used_for_prediction"] is False


def test_strata_use_reference_bbox_only_and_null_stage_unknown():
    samples = [{"id": "f", "bbox": [0, 0, 30, 10], "stage": None}]
    evaluation = {"points": [{"id": "f", "stratum": "bulk_reference_confirmation",
                               "split": "development", "event_id": "e", "error_px": 1.,
                               "within_10px": True, "within_10pct_box_diagonal": True,
                               "reference_xy": [1, 1], "prediction_xy": [1, 1]}]}
    result = rotated.add_reference_strata(evaluation, samples)
    assert result["points"][0]["reference_bbox_orientation"] == "reference_bbox_horizontal"
    assert result["points"][0]["stage"] == "unknown"
    assert result["by_reference_bbox_orientation"]["reference_bbox_horizontal"]["denominator"] == 1
    assert result["by_stage"]["unknown"]["within_10px"] == 1


def test_control_requires_common_frozen_seed_hash(tmp_path):
    path = tmp_path / "prediction.json"
    path.write_text('{"common_seed_sha256":"wrong","reference_labels_used_for_prediction":false}')
    with pytest.raises(ValueError, match="common seed"):
        rotated.load_control_prediction(path, "expected")
