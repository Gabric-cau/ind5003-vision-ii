from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_external_ball_v2 as experiment
from scripts import train_external_ball_classifier as previous


def manifest():
    return {"settings": deepcopy(previous.PREPROCESS),
            "held_out_world_cup_accessed": False, "labels_used_for_window_selection": False,
            "manual_timeline_used_for_window_selection": False,
            "samples": [{"event_id": str(i), "video_sha256": str(i) * 64,
                         "dataset_id": dataset, "split": split,
                         "filename": f"match_{i + 1:02d}.mp4",
                         "targets": {task: classes[0] for task, classes in previous.CLASSES.items()}}
                        for i, (dataset, split) in enumerate([
                            ("german_cup_2022", "train"), ("coppa_italia_2025", "validation")])]}


def test_heldout_cannot_enter_candidate_experiment():
    data = manifest()
    data["samples"][1]["dataset_id"] = "world_cup_2022"
    with pytest.raises(ValueError, match="Held-out"):
        experiment.validate_frames_manifest(data)


@pytest.mark.parametrize("field", ["held_out_world_cup_accessed", "labels_used_for_window_selection",
                                  "manual_timeline_used_for_window_selection"])
def test_missing_or_true_provenance_is_rejected(field):
    data = manifest()
    del data[field]
    with pytest.raises(ValueError, match="provenance"):
        experiment.validate_frames_manifest(data)
    data[field] = True
    with pytest.raises(ValueError, match="provenance"):
        experiment.validate_frames_manifest(data)


def test_temporal_difference_removes_constant_per_event_appearance():
    rng = np.random.default_rng(13)
    temporal, first, motion = rng.normal(size=(5, 24)), rng.normal(size=(5, 6)), np.zeros((5, 12))
    shift = rng.normal(size=(5, 6))
    original = experiment.descriptor_sets(temporal, first, motion)
    changed = experiment.descriptor_sets(temporal + np.tile(shift, (1, 4)), first + shift, motion)
    np.testing.assert_allclose(original["first_frame_difference"], changed["first_frame_difference"])
    assert not np.allclose(original["appearance_temporal"], changed["appearance_temporal"])


def test_static_frames_have_negligible_motion_and_correct_fixed_dimensions():
    rng = np.random.default_rng(54)
    image = rng.integers(0, 255, size=(144, 256, 3), dtype=np.uint8)
    features = experiment.motion_grid([image] * 8)
    assert features.shape == (4 * 3 * 4 * 4,)
    assert np.max(np.abs(features)) < 1e-4


def test_motion_detects_local_change_and_stays_finite():
    frames = []
    for x in range(12, 100, 12):
        image = np.zeros((144, 256, 3), dtype=np.uint8)
        image[55:75, x:x + 20] = 255
        frames.append(image)
    features = experiment.motion_grid(frames)
    assert np.isfinite(features).all()
    assert np.max(features) > .001


def test_candidate_heads_fit_only_training_features_and_export_exactly():
    rng = np.random.default_rng(45)
    features = rng.normal(size=(15, 32))
    train_rows = list(range(8))
    targets = [{task: classes[i % 3] for task, classes in previous.CLASSES.items()}
               for i in train_rows]
    fitted = previous.fit_heads(features, train_rows, targets)
    altered = features.copy()
    altered[8:] = 1e8
    refit = previous.fit_heads(altered, train_rows, targets)
    expected, actual = previous.export_parameters(*fitted), previous.export_parameters(*refit)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    transformed = fitted[1].transform(fitted[0].transform(features))
    predictions = previous.predict_parameters(features, expected)
    for task in previous.CLASSES:
        np.testing.assert_allclose(predictions[task]["probabilities"],
                                   fitted[2][task].predict_proba(transformed), atol=1e-12)


def test_prediction_rejects_wrong_backbone_before_decoding(tmp_path, monkeypatch):
    backbone = tmp_path / "encoder.pt"
    backbone.write_bytes(b"wrong encoder")
    previous.save_json(tmp_path / "report.json", {"portable_encoder_sha256": "expected"})
    monkeypatch.setattr(previous, "decode_video", lambda *args: pytest.fail("decoded before hash guard"))
    with pytest.raises(ValueError, match="Encoder hash mismatch"):
        experiment.predict(SimpleNamespace(model=tmp_path, backbone=backbone))


def test_prediction_rejects_changed_head_before_decoding(tmp_path, monkeypatch):
    backbone = tmp_path / "encoder.pt"
    backbone.write_bytes(b"encoder")
    (tmp_path / "first_frame_difference_weights.npz").write_bytes(b"changed")
    previous.save_json(tmp_path / "report.json", {
        "portable_encoder_sha256": previous.digest(backbone),
        "candidates": {"first_frame_difference": {"weights_sha256": "expected"}}})
    monkeypatch.setattr(previous, "decode_video", lambda *args: pytest.fail("decoded before hash guard"))
    with pytest.raises(ValueError, match="Classifier hash mismatch"):
        experiment.predict(SimpleNamespace(model=tmp_path, backbone=backbone,
                                           candidate="first_frame_difference"))
