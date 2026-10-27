from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_contact_offset_calibration as calibration
from src import contact_temporal_model as temporal
from tests.test_contact_temporal_model import cohort


def test_fitted_offset_is_training_residual_mean_and_ignores_heldout_labels():
    records, references = cohort()
    _, training, validation = next(temporal.leave_one_match_out(records))
    offset, audit = calibration.fit_offset(training, references)
    assert offset == pytest.approx(-.4)
    assert len(audit) == len(training)
    for record in validation:
        references[record["core_id"]]["reference_contact_seconds"] += 20
    changed, changed_audit = calibration.fit_offset(training, references)
    assert changed == offset and changed_audit == audit


def test_abstentions_are_preserved_and_missing_training_rules_are_excluded():
    records, references = cohort()
    records[0]["baselines"]["existing_rule"]["time_seconds"] = None
    offset, audit = calibration.fit_offset(records, references)
    assert len(audit) == 9
    prediction = calibration.calibrate_contact(records[0], offset)
    assert prediction["time_seconds"] is None
    assert prediction["reason"] == "original_rule_abstained"
    for record in records:
        record["baselines"]["existing_rule"]["time_seconds"] = None
    assert calibration.fit_offset(records, references) == (None, [])


@pytest.mark.parametrize("offset", [-50., 50.])
def test_out_of_window_correction_abstains_without_clipping(offset):
    record = cohort()[0][0]
    result = calibration.calibrate_contact(record, offset)
    assert result["time_seconds"] is None
    assert result["reason"] == "calibrated_time_outside_automatic_window"
    assert not result["clipping_applied"]


def test_same_event_metrics_and_full_denominator_preserve_failures():
    rows = [{"core_id": f"core_{i}", "reference_contact_seconds": 1., "predictions": {
        "calibrated_rule": {"time_seconds": 1. if i == 0 else None},
        "existing_rule": {"time_seconds": 1.2 if i == 0 else None},
        "ankle_proximity_control": {"time_seconds": 1.3}}} for i in range(2)]
    scores = calibration.summarize(rows)
    assert scores["main_common_core_ids"] == ["core_0"]
    corrected = scores["per_method_available_events"]["calibrated_rule"]
    assert corrected["within_0.1_seconds"] == 1.
    assert corrected["within_0.1_seconds_over_all_events"] == .5
    assert corrected["coverage"] == .5


def test_four_fold_training_export_and_label_free_inference(tmp_path):
    records, references = cohort()
    features = tmp_path / "features"
    for record in records:
        calibration.save_json(features / f"{record['core_id']}.json", record)
    labels = tmp_path / "labels.json"
    calibration.save_json(labels, {"samples": list(references.values())})
    metadata = {"feature_names": records[0]["feature_names"], **temporal.visual_bindings(records[0]),
                "lags_seconds": list(temporal.LAGS_SECONDS), "smoothing_seconds": temporal.SMOOTHING_SECONDS,
                "minimum_support_samples": temporal.MINIMUM_SUPPORT_SAMPLES}
    metadata_path = tmp_path / "original_metadata.json"
    calibration.save_json(metadata_path, metadata)
    output = tmp_path / "calibration"
    calibration.train(SimpleNamespace(features_dir=features, labels=labels, visual_metadata=metadata_path, output=output))
    report = calibration.read_json(output / "report.json")
    deployment = calibration.read_json(output / "deployment_calibration.json")
    assert len(report["folds"]) == 4
    assert report["events"] == 10
    assert deployment["fitted_offset_seconds"] == pytest.approx(-.4)
    assert deployment["training_datapoints"] == 10
    assert report["experiment_selected_after_prior_development_results"]
    assert not report["neural_weights_updated"] and not report["held_out_independent_test"]
    for fold in report["folds"]:
        assert not set(fold["training_core_ids"]) & set(fold["validation_core_ids"])
        assert all(row["match_group"] != fold["held_out_match"] for row in fold["training_residuals"])
    labels.unlink()
    prediction_path = tmp_path / "prediction.json"
    calibration.predict(SimpleNamespace(features=features / "core_00.json", model=output, output=prediction_path))
    prediction = calibration.read_json(prediction_path)["prediction"]
    assert prediction["time_seconds"] == pytest.approx(2.)
    changed = deepcopy(records[0])
    changed["provenance"]["model_sha256"]["ball.pt"] = "f" * 64
    with pytest.raises(ValueError, match="visual_model_sha256"):
        calibration.calibrate_contact(changed, deployment["fitted_offset_seconds"], deployment)


def test_nonfinite_offset_cannot_create_a_json_prediction():
    with pytest.raises(ValueError, match="finite"):
        calibration.calibrate_contact(cohort()[0][0], np.inf)
