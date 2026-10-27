import numpy as np
import pytest

from scripts import train_external_ball_height_augmented as model


def sources():
    samples = [{"core_id": f"core_{i:02d}", "clip_id": str(i), "video_sha256": str(i)}
               for i in range(1, 11)]
    events = [{"clip_id": str(i), "source_sha256": "old" + str(i), "shot_height": "middle",
               "_shot_save_annotation": {"field_provenance": {"shot_height": {
                   "status": "human_annotated", "human_verified": True}}}}
              for i in range(1, 11)]
    return {"samples": samples}, {"events": events}


def test_fixed_conflict_and_unknown_exclusions_keep_height_of_direction_conflict():
    inputs, references = sources()
    references["events"][6]["shot_height"] = "low"
    references["events"][8]["shot_height"] = "unknown"
    admitted, rejected = model.admitted_core_labels(inputs, references)
    assert len(admitted) == 8 and len(rejected) == 2
    assert "core_08" in {row["core_id"] for row in admitted}
    assert not {"core_07", "core_09"} & {row["core_id"] for row in admitted}


def test_machine_generated_height_cannot_be_inherited_as_human_reference():
    inputs, references = sources()
    references["events"][0]["_shot_save_annotation"]["field_provenance"]["shot_height"]["human_verified"] = False
    with pytest.raises(ValueError, match="human provenance"):
        model.admitted_core_labels(inputs, references)


def test_augmented_head_preprocessing_is_training_only_and_export_matches():
    rng = np.random.default_rng(54)
    features = rng.normal(size=(23, 30))
    training = list(range(8)) + list(range(15, 23))
    labels = [model.CLASSES[i % 3] for i in range(16)]
    fitted = model.fit_height(features, training, labels)
    other_features = features.copy()
    other_features[8:15] = 1e9
    changed = model.fit_height(other_features, training, labels)
    expected, actual = model.export_height(*fitted), model.export_height(*changed)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    predictions = model.predict_height(features, expected)
    transformed = fitted[1].transform(fitted[0].transform(features))
    np.testing.assert_allclose(predictions["probabilities"], fitted[2].predict_proba(transformed), atol=1e-12)
