from copy import deepcopy
from pathlib import Path

import pytest

from scripts.evaluate_shoe_contact_crop import contact_crop_bound
from src.shoe_supervised_model import ShoeConfig


def example():
    inputs = {"source_mapping": {"clip_sha256": "source"}, "reference_labels_read": False,
              "contact_bracket_frames": [49, 51], "fps": 25,
              "inference_interval_frames": [0, 99], "interval_frames": [0, 49]}
    plan = [{"frame": frame, "anchor_bbox": [100, 100, 120, 200],
             "crop_xyxy": [0, 0, 400, 400], "anchor_source": "measured",
             "rival_source_boxes": []} for frame in range(100)]
    plan[0]["anchor_bbox"] = [0, 0, 400, 400]
    camera = {"source_sha256": "source", "segment_ids": {str(f): 2 for f in range(100)},
              "transforms": {str(f): [[1, 0, 0], [0, 1, 0], [0, 0, 1]] for f in range(100)}}
    return inputs, plan, camera


def test_only_automatic_contact_neighborhood_selects_crop():
    inputs, plan, camera = example()
    original = deepcopy((inputs, plan, camera))
    bound, scoped = contact_crop_bound(inputs, plan, camera, video=Path("clip.mp4"),
                                      width=400, height=400, config=ShoeConfig())
    assert bound["inference_interval_frames"] == [39, 57]
    assert bound["fixed_roi"][0] > 0 and bound["fixed_roi"][2] < 400
    assert scoped["interval_frames"] == [39, 49]
    assert all(row["anchor_source"] == "measured" for row in bound["plan"])
    assert (inputs, plan, camera) == original


def test_rejects_contact_camera_boundary_or_wrong_source():
    inputs, plan, camera = example()
    camera["segment_ids"]["50"] = 3
    with pytest.raises(ValueError, match="camera failure"):
        contact_crop_bound(inputs, plan, camera, video=Path("clip.mp4"),
                           width=400, height=400, config=ShoeConfig())
    camera["source_sha256"] = "other"
    with pytest.raises(ValueError, match="source differs"):
        contact_crop_bound(inputs, plan, camera, video=Path("clip.mp4"),
                           width=400, height=400, config=ShoeConfig())
