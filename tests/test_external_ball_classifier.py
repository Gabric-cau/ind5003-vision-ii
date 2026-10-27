from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts import train_external_ball_classifier as experiment


def fitting_data():
    rng = np.random.default_rng(194)
    features = rng.normal(size=(12, 24))
    training = [0, 2, 3, 5, 6, 7, 9, 10]
    targets = [
        {task: classes[index % len(classes)] for task, classes in experiment.CLASSES.items()}
        for index in range(len(training))
    ]
    return features, training, targets


def source_samples():
    return [
        {
            "split": split,
            "dataset_id": dataset,
            "event_id": f"event_{index}",
            "video_sha256": str(index) * 64,
            "filename": f"match_{index + 1:02d}.mp4",
            "targets": {task: classes[0] for task, classes in experiment.CLASSES.items()},
        }
        for index, (split, dataset) in enumerate(
            (("train", "german_cup_2022"), ("validation", "coppa_italia_2025"))
        )
    ]


def test_validation_pixels_cannot_change_fitted_preprocessing_or_heads():
    features, training, targets = fitting_data()
    fitted = experiment.fit_heads(features, training, targets)
    changed = features.copy()
    validation = sorted(set(range(len(features))) - set(training))
    changed[validation] = 1e9
    other = experiment.fit_heads(changed, training, targets)
    np.testing.assert_allclose(fitted[0].mean_, features[training].mean(axis=0))
    first_parameters = experiment.export_parameters(*fitted)
    other_parameters = experiment.export_parameters(*other)
    for key in first_parameters:
        np.testing.assert_array_equal(first_parameters[key], other_parameters[key])


def test_exported_npz_predictions_match_sklearn_for_batch_and_single_video(tmp_path):
    features, training, targets = fitting_data()
    scaler, pca, heads = experiment.fit_heads(features, training, targets)
    weights = tmp_path / "weights.npz"
    np.savez_compressed(weights, **experiment.export_parameters(scaler, pca, heads))
    with np.load(weights, allow_pickle=False) as parameters:
        actual = experiment.predict_parameters(features, parameters)
        single = experiment.predict_parameters(features[1], parameters)
    transformed = pca.transform(scaler.transform(features))
    for task, classes in experiment.CLASSES.items():
        expected = heads[task].predict_proba(transformed)
        np.testing.assert_allclose(actual[task]["probabilities"], expected, rtol=1e-12, atol=1e-12)
        assert actual[task]["labels"] == [classes[i] for i in heads[task].predict(transformed)]
        np.testing.assert_allclose(single[task]["probabilities"][0], expected[1], atol=1e-12)
        assert single[task]["labels"] == [actual[task]["labels"][1]]


@pytest.mark.parametrize("split", ["train", "validation", "heldout"])
def test_world_cup_cannot_enter_fitting_even_with_a_relabelled_split(split):
    samples = source_samples()
    samples.append({**samples[0], "event_id": "world_cup_event", "video_sha256": "f" * 64,
                    "dataset_id": "world_cup_2022", "split": split})
    with pytest.raises(ValueError, match="Held-out or unexpected match"):
        experiment.validate_samples(samples)


@pytest.mark.parametrize("field", ["event_id", "video_sha256"])
def test_same_event_or_video_cannot_cross_development_splits(field):
    samples = source_samples()
    samples[1][field] = samples[0][field]
    with pytest.raises(ValueError, match="Duplicate event or video"):
        experiment.validate_samples(samples)


def test_quarantined_event_and_unknown_labels_are_rejected():
    samples = source_samples()
    conflicting = deepcopy(samples)
    conflicting[1]["filename"] = "match_09.mp4"
    with pytest.raises(ValueError, match="Unresolved conflicting event"):
        experiment.validate_samples(conflicting)
    samples[0]["targets"]["ball_direction"] = "unknown"
    with pytest.raises(ValueError, match="Unknown target"):
        experiment.validate_samples(samples)


def test_missing_training_class_fails_instead_of_exporting_wrong_class_order():
    features, training, targets = fitting_data()
    for target in targets:
        target["ball_direction"] = "left"
    with pytest.raises(ValueError, match="does not contain all classes for ball_direction"):
        experiment.fit_heads(features, training, targets)


@pytest.mark.parametrize("flag", ["held_out_world_cup_accessed", "labels_used_for_window_selection",
                                  "manual_timeline_used_for_window_selection"])
def test_invalid_frame_provenance_fails_before_model_loading_or_output_creation(tmp_path, flag):
    pytest.importorskip("torch")
    frames = tmp_path / "frames"
    data = {"samples": source_samples(), "held_out_world_cup_accessed": False,
            "labels_used_for_window_selection": False,
            "manual_timeline_used_for_window_selection": False}
    data[flag] = True
    experiment.save_json(frames / "frames.json", data)
    output = tmp_path / "model"
    with patch.dict(sys.modules, {"torchvision": ModuleType("torchvision")}), \
            patch.object(experiment, "load_encoder") as load_encoder:
        with pytest.raises(ValueError, match="Invalid data provenance"):
            experiment.train(SimpleNamespace(frames=frames, output=output))
        load_encoder.assert_not_called()
    assert not output.exists()


def test_frozen_encoder_keeps_batchnorm_statistics_and_gradients_unchanged(tmp_path):
    torch = pytest.importorskip("torch")

    def tiny_encoder(weights=None):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
        )

    checkpoint = tmp_path / "tiny_backbone.pth"
    torch.save(tiny_encoder().state_dict(), checkpoint)
    torchvision = ModuleType("torchvision")
    models = ModuleType("torchvision.models")
    models.resnet18 = tiny_encoder
    torchvision.models = models
    with patch.dict(sys.modules, {"torchvision": torchvision, "torchvision.models": models}):
        encoder = experiment.load_encoder(checkpoint, threads=1)
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    frames = [np.full((8, 8, 3), index * 8, dtype=np.uint8) for index in range(16)]
    settings = {"spatial_pool": [2, 2], "temporal_bins": 4}
    feature, first, _ = experiment.encode_frames(frames, encoder, settings)
    assert feature.shape == (4 * 4 * 2 * 2,)
    assert first.shape == (4 * 2 * 2,)
    assert all(not module.training for module in encoder.modules())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_sampling_never_includes_the_cut_frame_and_metrics_keep_absent_classes():
    assert experiment.sample_indices(7, 4) == [0, 2, 4, 6]
    assert set(experiment.sample_indices(2, 64)) == {0, 1}
    with pytest.raises(ValueError, match="Empty video"):
        experiment.sample_indices(0, 64)
    result = experiment.metrics(["left", "left"], ["left", "left"],
                                experiment.CLASSES["ball_direction"])
    assert result["accuracy"] == 1
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["confusion_matrix_true_rows"] == [[2, 0, 0], [0, 0, 0], [0, 0, 0]]
