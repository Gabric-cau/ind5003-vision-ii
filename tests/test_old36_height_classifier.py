import copy

import numpy as np
import pytest

from scripts import train_old36_height_classifier as model
from scripts import train_external_ball_height_augmented as head


def source_manifest():
    return {"labels_present": False, "settings": {"frames": 64, "max_shot_seconds": 20},
            "samples": [{"clip_id": f"col_eng_p{i:02d}_sample", "video_path": "video.mp4",
                "video_sha256": str(i), "pose_path": "pose.json", "start_frame": 60,
                "end_frame_exclusive": 900, "fps": 30, "source_frame_count": 1000,
                "window_method": "first_viable", "window_scan": "adaptive",
                "selected_shot_index": 1, "labels_used": False} for i in range(36)]}


def test_source_manifest_rejects_duplicates_targets_and_non_old_match():
    data = source_manifest()
    model.validate_sources(data)
    for mutation in (lambda d: d["samples"][0].update(height="high"),
                     lambda d: d["samples"][0].update(clip_id="world_cup_2022:match17"),
                     lambda d: d["samples"][0].update(video_sha256="1")):
        changed = copy.deepcopy(data)
        mutation(changed)
        with pytest.raises(ValueError):
            model.validate_sources(changed)


def test_automatic_window_retains_absolute_offset_and_fixed_density():
    data = source_manifest()
    ids = model.window_indices(data["samples"][0], data["settings"])
    assert len(ids) == 64
    assert ids[0] == 60 and ids[-1] == 659
    assert len(set(ids)) == 64
    data["samples"][0]["end_frame_exclusive"] = 130
    assert model.window_indices(data["samples"][0], data["settings"])[-1] == 129


def test_human_target_requires_source_hash_and_human_provenance():
    row = {"clip_id": "col_eng_p01_sample", "video_sha256": "source"}
    ref = {"events": [{"clip_id": row["clip_id"], "source_sha256": "source", "shot_height": "mid",
        "_shot_save_annotation": {"field_provenance": {"shot_height": {
            "status": "human_annotated", "human_verified": True}}}}]}
    assert model.source_targets([row], ref) == ["middle"]
    ref["events"][0]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        model.source_targets([row], ref)


def test_44_training_rows_ignore_validation_pixels_and_export_exactly():
    rng = np.random.default_rng(26)
    features = rng.normal(size=(51, 36))
    indices = list(range(8)) + list(range(15, 51))
    labels = [head.CLASSES[i % 3] for i in range(44)]
    fitted = head.fit_height(features, indices, labels)
    changed = features.copy()
    changed[8:15] = 1e10
    changed_fitted = head.fit_height(changed, indices, labels)
    exported = head.export_height(*fitted)
    for name, value in head.export_height(*changed_fitted).items():
        np.testing.assert_array_equal(exported[name], value)
    transformed = fitted[1].transform(fitted[0].transform(features))
    np.testing.assert_allclose(head.predict_height(features, exported)["probabilities"],
                               fitted[2].predict_proba(transformed), atol=1e-12)
