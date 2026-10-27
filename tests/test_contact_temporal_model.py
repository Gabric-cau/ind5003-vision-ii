from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_core_contact_model as trainer
from src import contact_temporal_model as model


def cohort():
    records, references = [], {}
    for index, group in enumerate(group for group, count in model.EXPECTED_GROUPS.items() for _ in range(count)):
        frames = np.arange(0, 150, 2)
        times = frames / 25
        contact = 2. + .08 * index
        record = {"core_id": f"core_{index:02}", "clip_id": f"clip_{index:02}", "match_group": group,
                  "video_sha256": f"{index + 1:064x}", "fps": 25.,
                  "window": {"start_frame": 0, "end_frame_exclusive": 150, "automatic": True},
                  "frame_ids": frames.tolist(), "time_seconds": times.tolist(),
                  "feature_names": ["foot_ball_distance_body", "ball_confidence", "pose_visible", "ball_visible", "seed_available"],
                  "features": np.stack([abs(times - contact), np.exp(-(times - contact) ** 2),
                                        np.ones(len(times)), np.ones(len(times)), np.ones(len(times))], axis=1).tolist(),
                  "provenance": {"human_labels_used": False, "old_contact_used": False,
                                 "model_sha256": {"ball.pt": "a" * 64, "pose.pt": "b" * 64},
                                 "code_sha256": {"extractor.py": "c" * 64}},
                  "settings": {"stride": 2, "ball_imgsz": 1280, "pose_imgsz": 640},
                  "baselines": {"existing_rule": {"time_seconds": contact + .4},
                                "ankle_proximity": {"time_seconds": contact + .2}},
                  "runtime": {"feature_seconds": .02}}
        references[record["core_id"]] = {"core_id": record["core_id"], "clip_id": record["clip_id"],
                                          "video_sha256": record["video_sha256"],
                                          "reference_contact_seconds": contact,
                                          "reference_precision": "coarse_seconds"}
        records.append(record)
    return records, references


def test_four_folds_have_no_match_overlap_and_each_event_is_predicted_once():
    records, references = cohort()
    model.validate_cohort(records, references, strict=True)
    seen = []
    for heldout, training, validation in model.leave_one_match_out(records):
        assert heldout not in {record["match_group"] for record in training}
        assert {record["match_group"] for record in validation} == {heldout}
        assert len(training) + len(validation) == 10
        seen.extend(record["core_id"] for record in validation)
    assert sorted(seen) == sorted(references)


def test_heldout_pixels_and_labels_do_not_change_training_fit():
    records, references = cohort()
    _, training, validation = next(model.leave_one_match_out(records))
    before = model.export_parameters(*model.fit_contact_head(training, references)[:2])
    for record in validation:
        record["features"] = (np.asarray(record["features"]) * 1e6).tolist()
        references[record["core_id"]]["reference_contact_seconds"] += 500
    after = model.export_parameters(*model.fit_contact_head(training, references)[:2])
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_event_class_balancing_and_ambiguous_band_exclusion():
    records, references = cohort()
    _, _, audit = model.fit_contact_head(records, references)
    for record, item in zip(records, audit):
        _, eligible = model.build_temporal_features(record)
        distances = abs(np.asarray(record["time_seconds"]) - references[record["core_id"]]["reference_contact_seconds"])
        assert item["positive_frames"] == np.count_nonzero(eligible & (distances <= .15 + 1e-9))
        assert item["negative_frames"] == np.count_nonzero(eligible & (distances >= .35 - 1e-9))
        assert item["positive_frames"] + item["negative_frames"] < eligible.sum()
        assert item["sample_weight_sum"] == pytest.approx(1)


def test_npz_export_matches_sklearn_without_pickle_for_arbitrary_frames(tmp_path):
    records, references = cohort()
    scaler, head, _ = model.fit_contact_head(records[:4], references)
    features, _ = model.build_temporal_features(records[-1])
    path = tmp_path / "weights.npz"
    np.savez_compressed(path, **model.export_parameters(scaler, head))
    with np.load(path, allow_pickle=False) as parameters:
        actual = model.predict_probabilities(features, parameters)
    np.testing.assert_allclose(actual, head.predict_proba(scaler.transform(features))[:, 1], rtol=1e-12, atol=1e-12)


def test_temporal_context_masks_edges_and_does_not_wrap():
    record = cohort()[0][0]
    x, eligible = model.build_temporal_features(record)
    channels = len(record["feature_names"])
    assert x.shape == (len(record["frame_ids"]), 5 * channels + 5)
    assert not eligible[0] and not eligible[-1]
    np.testing.assert_array_equal(x[0, :channels], 0)
    assert x[0, -5] == 0 and x[0, -3] == 1
    assert eligible[len(eligible) // 2]
    parameters = {"scaler_mean": np.zeros(x.shape[1]), "scaler_scale": np.ones(x.shape[1]),
                  "coef": np.zeros((1, x.shape[1])), "intercept": np.zeros(1)}
    predicted = model.predict_contact(record, parameters)
    assert predicted["frame"] == record["frame_ids"][int(np.flatnonzero(eligible)[0])]


def test_missing_lag_inside_gap_is_masked_and_short_windows_abstain():
    record = cohort()[0][0]
    remove = np.arange(24, 33)
    keep = np.ones(len(record["frame_ids"]), dtype=bool)
    keep[remove] = False
    for key in ("frame_ids", "time_seconds", "features"):
        record[key] = np.asarray(record[key])[keep].tolist()
    x, eligible = model.build_temporal_features(record)
    gap_adjacent = record["frame_ids"].index(46)
    assert not eligible[gap_adjacent]
    short = deepcopy(record)
    for key in ("frame_ids", "time_seconds", "features"):
        short[key] = short[key][:5]
    parameters = {"scaler_mean": np.zeros(x.shape[1]), "scaler_scale": np.ones(x.shape[1]),
                  "coef": np.zeros((1, x.shape[1])), "intercept": np.zeros(1)}
    assert model.predict_contact(short, parameters)["time_seconds"] is None


def test_missing_visual_evidence_abstains_but_seed_allows_a_brief_ball_occlusion():
    record = cohort()[0][0]
    x, _ = model.build_temporal_features(record)
    parameters = {"scaler_mean": np.zeros(x.shape[1]), "scaler_scale": np.ones(x.shape[1]),
                  "coef": np.zeros((1, x.shape[1])), "intercept": np.zeros(1)}
    feature_values = np.asarray(record["features"])
    feature_values[:, record["feature_names"].index("ball_visible")] = 0
    record["features"] = feature_values.tolist()
    assert model.predict_contact(record, parameters)["time_seconds"] is not None
    feature_values[:, record["feature_names"].index("seed_available")] = 0
    record["features"] = feature_values.tolist()
    assert model.predict_contact(record, parameters)["reason"] == "evidence_missing"
    feature_values[:, record["feature_names"].index("ball_visible")] = 1
    feature_values[:, record["feature_names"].index("pose_visible")] = 0
    record["features"] = feature_values.tolist()
    assert model.predict_contact(record, parameters)["time_seconds"] is None


@pytest.mark.parametrize("supported_indices", [[20], [20, 21], [20, 22, 24]])
def test_isolated_or_discontinuous_evidence_cannot_establish_contact(supported_indices):
    record = cohort()[0][0]
    values = np.asarray(record["features"])
    pose_column = record["feature_names"].index("pose_visible")
    values[:, pose_column] = 0
    values[supported_indices, pose_column] = 1
    record["features"] = values.tolist()
    features, eligible = model.build_temporal_features(record)
    assert not eligible.any()
    parameters = {"scaler_mean": np.zeros(features.shape[1]), "scaler_scale": np.ones(features.shape[1]),
                  "coef": np.zeros((1, features.shape[1])), "intercept": np.zeros(1)}
    result = model.predict_contact(record, parameters)
    assert result["time_seconds"] is None and result["eligible_frames"] == 0


def test_three_consecutive_samples_are_supported_and_full_sequences_remain_eligible():
    record = cohort()[0][0]
    _, full = model.build_temporal_features(record)
    times = np.asarray(record["time_seconds"])
    np.testing.assert_array_equal(full, (times >= .4 - 1e-9) & (times <= times[-1] - .4 + 1e-9))
    values = np.asarray(record["features"])
    pose_column = record["feature_names"].index("pose_visible")
    values[:, pose_column] = 0
    values[20:23, pose_column] = 1
    record["features"] = values.tolist()
    _, eligible = model.build_temporal_features(record)
    assert np.flatnonzero(eligible).tolist() == [20, 21, 22]


@pytest.mark.parametrize("field", ["human_labels_used", "old_contact_used"])
def test_annotation_or_old_timing_cannot_enter_feature_provenance(field):
    record = cohort()[0][0]
    record["provenance"][field] = True
    with pytest.raises(ValueError, match="provenance"):
        model.build_temporal_features(record)


@pytest.mark.parametrize("name", ["absolute_time_seconds", "frame_id", "match_id", "relative_progress", "reference_contact_time"])
def test_clock_identifier_and_target_features_are_rejected(name):
    record = cohort()[0][0]
    record["feature_names"][0] = name
    with pytest.raises(ValueError, match="Feature names"):
        model.build_temporal_features(record)


def test_reference_binding_and_external_match_rejected():
    records, references = cohort()
    references[records[0]["core_id"]]["video_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same video"):
        model.validate_cohort(records, references)
    records, references = cohort()
    records[0]["match_group"] = "world_cup_2022"
    with pytest.raises(ValueError, match="Unexpected match"):
        model.validate_cohort(records, references)


@pytest.mark.parametrize("change", ["weights", "settings", "code", "missing"])
def test_same_named_features_from_different_visual_pipeline_are_rejected(change):
    records, references = cohort()
    expected = {"feature_names": records[0]["feature_names"], "lags_seconds": list(model.LAGS_SECONDS),
                "smoothing_seconds": model.SMOOTHING_SECONDS,
                "minimum_support_samples": model.MINIMUM_SUPPORT_SAMPLES, **model.visual_bindings(records[0])}
    altered = records[-1]
    if change == "weights":
        altered["provenance"]["model_sha256"]["ball.pt"] = "d" * 64
    elif change == "settings":
        altered["settings"]["stride"] = 4
    elif change == "code":
        altered["provenance"]["code_sha256"]["extractor.py"] = "e" * 64
    else:
        del altered["provenance"]["model_sha256"]
    with pytest.raises(ValueError, match="differs between events"):
        model.validate_cohort(records, references)
    with pytest.raises(ValueError, match="visual binding differs"):
        model.validate_prediction_bindings(altered, expected)


def test_reference_outside_window_does_not_shift_prediction_or_create_training_positive():
    records, references = cohort()
    scaler, head, _ = model.fit_contact_head(records, references)
    parameters = model.export_parameters(scaler, head)
    before = model.predict_contact(records[0], parameters)
    references[records[0]["core_id"]]["reference_contact_seconds"] = 90.
    assert model.predict_contact(records[0], parameters) == before
    _, _, audit = model.fit_contact_head(records, references)
    assert not audit[0]["used"]
    assert audit[0]["positive_frames"] == 0


def test_comparisons_use_same_events_and_keep_missing_baseline_coverage():
    rows = [{"core_id": f"core_{i}", "reference_contact_seconds": 1., "predictions": {
        "trained_contact": {"time_seconds": 1.1}, "existing_rule": {"time_seconds": None if i else 1.2},
        "ankle_proximity": {"time_seconds": 1.5}, "training_median": {"time_seconds": .5}}} for i in range(2)]
    report = model.compare_predictions(rows)
    assert report["per_method_available_events"]["existing_rule"]["coverage"] == .5
    assert report["per_method_available_events"]["existing_rule"]["within_0.2_seconds"] == 1.
    assert report["per_method_available_events"]["existing_rule"]["within_0.2_seconds_over_all_events"] == .5
    assert report["common_available_core_ids"] == ["core_0"]
    assert all(row["events"] == 1 for row in report["all_methods_on_same_events"].values())
    assert report["pairwise_same_events"]["ankle_proximity"]["trained_contact"]["events"] == 2


def test_full_training_export_and_label_free_prediction_on_synthetic_cohort(tmp_path):
    records, references = cohort()
    features = tmp_path / "features"
    for record in records:
        trainer.save_json(features / f"{record['core_id']}.json", record)
    trainer.save_json(features / "protocol.json", {"should_not_be_a_feature_record": True})
    labels = tmp_path / "labels.json"
    trainer.save_json(labels, {"samples": list(references.values())})
    output = tmp_path / "experiment"
    trainer.train(SimpleNamespace(features_dir=features, labels=labels, output=output))
    report = json.loads((output / "report.json").read_text())
    predictions = json.loads((output / "oof_predictions.json").read_text())["predictions"]
    assert len(report["folds"]) == 4 and len(predictions) == 10
    assert not report["held_out_independent_test"]
    assert len(report["protocol"]["deployment_training_core_ids"]) == 10
    labels.unlink()
    prediction_path = tmp_path / "prediction.json"
    trainer.predict(SimpleNamespace(features=features / "core_00.json", model=output, output=prediction_path))
    actual = json.loads(prediction_path.read_text())
    assert actual["manual_annotation_required"] is False
    assert actual["prediction"]["time_seconds"] is not None
    metadata = trainer.read_json(output / "deployment_metadata.json")
    assert metadata["visual_model_sha256"] == records[0]["provenance"]["model_sha256"]
    assert metadata["visual_settings"] == records[0]["settings"]
    assert metadata["extractor_code_sha256"] == records[0]["provenance"]["code_sha256"]
    assert metadata["minimum_support_samples"] == 3
    records[0]["provenance"]["model_sha256"]["ball.pt"] = "f" * 64
    wrong_features = features / "wrong_backbone.json"
    trainer.save_json(wrong_features, records[0])
    rejected_output = tmp_path / "wrong_prediction.json"
    with pytest.raises(ValueError, match="visual_model_sha256"):
        trainer.predict(SimpleNamespace(features=wrong_features, model=output, output=rejected_output))
    assert not rejected_output.exists()
    with pytest.raises(FileExistsError):
        trainer.train(SimpleNamespace(features_dir=features, labels=features / "protocol.json", output=output))
