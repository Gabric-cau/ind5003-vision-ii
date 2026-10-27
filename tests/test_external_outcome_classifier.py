from copy import deepcopy

import numpy as np
import pytest

from scripts import train_external_outcome_classifier as outcome


def test_binary_export_and_training_only_preprocessing():
    rng = np.random.default_rng(53)
    features = rng.normal(size=(25, 30))
    training = list(range(8))
    labels = ["goal"] * 6 + ["no_goal"] * 2
    fitted = outcome.fit_head(features, training, labels)
    altered = features.copy()
    altered[8:] = 1e9
    changed = outcome.fit_head(altered, training, labels)
    first, second = outcome.export_head(*fitted), outcome.export_head(*changed)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    x = fitted[1].transform(fitted[0].transform(features))
    np.testing.assert_allclose(outcome.predict_head(features, first)["probabilities"],
                               fitted[2].predict_proba(x), atol=1e-12)


def data():
    samples, clips = [], []
    for i, (group, split) in enumerate([("german_cup_2022", "train"), ("coppa_italia_2025", "validation")]):
        sample = {"filename": f"match_{i + 1:02d}.mp4", "event_id": str(i), "video_sha256": str(i),
                  "dataset_id": group, "split": split, "targets": {
                      "ball_direction": "left", "ball_height_category": "low"}}
        samples.append(sample)
        clips.append({"filename": sample["filename"], "dataset_id": group, "media_check": {"sha256": str(i)},
                      "outcome_conflict": False, "training_goal_label": 1, "official_goal_label": 1,
                      "human_features": {"goal_label": 1, "provenance": "human_user_observation_in_chat"}})
    return samples, {"clips": clips}


def test_world_cup_samples_cannot_enter_goal_fitting():
    samples, metadata = data()
    samples[1]["dataset_id"] = "world_cup_2022"
    with pytest.raises(ValueError, match="Held-out"):
        outcome.external_goal_labels(metadata, samples)


def test_conflicts_and_machine_labels_are_rejected():
    samples, metadata = data()
    changed = deepcopy(metadata)
    changed["clips"][0]["outcome_conflict"] = True
    with pytest.raises(ValueError, match="Conflicting"):
        outcome.external_goal_labels(changed, samples)
    metadata["clips"][0]["human_features"]["provenance"] = "model_prediction"
    with pytest.raises(ValueError, match="disagree"):
        outcome.external_goal_labels(metadata, samples)


def test_unknown_rule_outputs_are_not_treated_as_no_goal():
    rows = [{"sample_id": "a", "goal": "goal"}, {"sample_id": "b", "goal": "no_goal"}]
    result = outcome.score(rows, [None, "no_goal"])
    assert result["prediction_available_events"] == 1
    assert result["strict_accuracy_unknown_as_incorrect"] == .5
    assert result["conditional_accuracy_on_available"] == 1
