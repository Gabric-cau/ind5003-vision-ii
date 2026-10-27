"""Exploratory constant-offset calibration of unchanged visual contact rules."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_core_contact_model import baseline, digest, read_json, references_from, save_json
from src import contact_temporal_model as temporal


def fit_offset(records, references):
    from sklearn.dummy import DummyRegressor
    temporal.validate_cohort(records, references)
    rows = []
    for record in records:
        raw = baseline(record, "existing_rule")
        if raw["time_seconds"] is None or raw.get("status") == "abstain":
            continue
        reference = temporal.validate_reference(record, references[record["core_id"]])
        rows.append({"core_id": record["core_id"], "match_group": record["match_group"],
                     "raw_contact_seconds": raw["time_seconds"], "reference_contact_seconds": reference,
                     "residual_seconds": reference - raw["time_seconds"],
                     "automatic_window_seconds": [record["window"]["start_frame"] / record["fps"],
                                                   record["window"]["end_frame_exclusive"] / record["fps"]]})
    if not rows:
        return None, rows
    regressor = DummyRegressor(strategy="mean")
    regressor.fit(np.zeros((len(rows), 1)), np.array([row["residual_seconds"] for row in rows]))
    return float(regressor.predict([[0.]])[0]), rows


def calibrate_contact(record, fitted_offset_seconds, metadata=None):
    temporal.validate_record(record)
    if metadata is not None:
        temporal.validate_prediction_bindings(record, metadata)
    if fitted_offset_seconds is not None and (not isinstance(fitted_offset_seconds, (int, float))
            or isinstance(fitted_offset_seconds, bool) or not np.isfinite(fitted_offset_seconds)):
        raise ValueError("Calibration offset must be finite")
    raw = baseline(record, "existing_rule")
    result = {"time_seconds": None, "raw_time_seconds": raw["time_seconds"],
              "original_status": raw.get("status"), "fitted_offset_seconds": fitted_offset_seconds,
              "original_rule": raw, "method": "training_match_mean_contact_residual",
              "neural_weights_updated": False, "clipping_applied": False}
    if raw["time_seconds"] is None or raw.get("status") == "abstain":
        return {**result, "reason": "original_rule_abstained"}
    if fitted_offset_seconds is None:
        return {**result, "reason": "no_available_training_residuals"}
    calibrated = raw["time_seconds"] + fitted_offset_seconds
    start = record["window"]["start_frame"] / record["fps"]
    end = record["window"]["end_frame_exclusive"] / record["fps"]
    if not start <= calibrated < end:
        return {**result, "reason": "calibrated_time_outside_automatic_window"}
    return {**result, "time_seconds": float(calibrated), "frame_estimate": float(calibrated * record["fps"]),
            "reason": "original_rule_plus_fitted_offset"}


def summarize(rows):
    methods = ("calibrated_rule", "existing_rule", "ankle_proximity_control")
    main_common = {row["core_id"] for row in rows if all(row["predictions"][method]["time_seconds"] is not None
                                                       for method in methods[:2])}
    all_common = {row["core_id"] for row in rows if all(row["predictions"][method]["time_seconds"] is not None
                                                      for method in methods)}
    return {"per_method_available_events": {method: temporal.error_metrics(rows, method) for method in methods},
            "main_common_core_ids": sorted(main_common),
            "main_comparison_on_same_events": {method: temporal.error_metrics(rows, method, main_common) for method in methods[:2]},
            "all_method_common_core_ids": sorted(all_common),
            "all_methods_on_same_events": {method: temporal.error_metrics(rows, method, all_common) for method in methods}}


def train(args):
    import sklearn
    if args.output.exists():
        raise FileExistsError("Preserve prior experiments in a separate output directory")
    paths = sorted(args.features_dir.glob("core_*.json"))
    records = [read_json(path) for path in paths]
    references = references_from(read_json(args.labels))
    temporal.validate_cohort(records, references, strict=True)
    original_metadata = read_json(args.visual_metadata)
    for record in records:
        temporal.validate_prediction_bindings(record, original_metadata)
    started = time.perf_counter()
    rows, folds = [], []
    for group, training, validation in temporal.leave_one_match_out(records):
        offset, training_rows = fit_offset(training, references)
        folds.append({"held_out_match": group, "training_core_ids": [row["core_id"] for row in training],
                      "validation_core_ids": [row["core_id"] for row in validation],
                      "fitted_offset_seconds": offset, "training_datapoints": len(training_rows),
                      "training_residuals": training_rows})
        for record in validation:
            reference = temporal.validate_reference(record, references[record["core_id"]])
            rows.append({"core_id": record["core_id"], "clip_id": record["clip_id"],
                         "match_group": group, "video_sha256": record["video_sha256"],
                         "reference_contact_seconds": reference,
                         "reference_precision": references[record["core_id"]].get("reference_precision"),
                         "predictions": {"calibrated_rule": calibrate_contact(record, offset),
                                         "existing_rule": baseline(record, "existing_rule"),
                                         "ankle_proximity_control": baseline(record, "ankle_proximity")},
                         "feature_runtime": record.get("runtime")})
    rows.sort(key=lambda row: row["core_id"])
    offset, training_rows = fit_offset(records, references)
    metadata_fields = ("feature_names", "visual_model_sha256", "visual_settings", "extractor_code_sha256",
                       "lags_seconds", "smoothing_seconds", "minimum_support_samples")
    deployment = {key: original_metadata[key] for key in metadata_fields}
    deployment.update(schema_version=1, model_type="existing_rule_constant_offset_calibration_v1",
                      estimator="sklearn.dummy.DummyRegressor", strategy="mean", learned_parameters=1,
                      fitted_offset_seconds=offset, training_datapoints=len(training_rows),
                      training_core_ids=[row["core_id"] for row in training_rows],
                      training_residuals=training_rows, original_rule_abstention_preserved=True,
                      out_of_window_policy="abstain_without_clipping", neural_weights_updated=False,
                      source_visual_metadata_sha256=digest(args.visual_metadata),
                      deployment_fit_scope="all_available_core_development_references_not_out_of_fold")
    report = {"experiment": "existing_rule_mean_residual_calibration_v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "events": len(rows),
              "estimator": deployment["estimator"], "strategy": "mean", "learned_parameters": 1,
              "folds": folds, "scores": summarize(rows), "deployment_fitted_offset_seconds": offset,
              "evaluation": "leave_one_match_out_development_on_previously_seen_course_events",
              "hypothesis": "A fixed detector-and-ankle rule may have systematic early or late contact timing.",
              "neural_weights_updated": False, "one_parameter_calibration_fitted": True,
              "held_out_independent_test": False, "hyperparameter_search": False,
              "experiment_selected_after_prior_development_results": True,
              "promoted_to_default_pipeline": False, "original_temporal_experiment_modified": False,
              "full_original_pipeline": False, "original_rule_abstention_preserved": True,
              "elapsed_seconds": time.perf_counter() - started,
              "runtime": {"device": "cpu", "sklearn": sklearn.__version__},
              "source_bindings": {"labels_sha256": digest(args.labels),
                                  "visual_metadata_sha256": digest(args.visual_metadata),
                                  "features": [{"file": path.name, "sha256": digest(path)} for path in paths]},
              "code_sha256": {"calibration": digest(__file__), "shared_temporal": digest(temporal.__file__)},
              "limitations": ["This is a one-parameter timing correction, not neural detector training.",
                              "The experiment was proposed after observing the prior temporal model's development results.",
                              "Match-wise fitting prevents same-fold label leakage but does not undo experiment selection on this cohort.",
                              "Human times are coarse references and do not establish frame-exact contact accuracy.",
                              "Original rule abstentions remain unavailable; calibrated times outside the automatic shot also abstain.",
                              "Ankle proximity is an untrained control, not an additional calibrated model.",
                              "The baseline uses sampled visual evidence and is not the complete original processing pipeline."]}
    args.output.mkdir(parents=True)
    save_json(args.output / "deployment_calibration.json", deployment)
    save_json(args.output / "oof_predictions.json", {"schema_version": 1, "predictions": rows})
    save_json(args.output / "report.json", report)
    print(json.dumps(report["scores"], indent=2), flush=True)


def predict(args):
    record, metadata = read_json(args.features), read_json(args.model / "deployment_calibration.json")
    result = calibrate_contact(record, metadata["fitted_offset_seconds"], metadata)
    save_json(args.output, {"core_id": record["core_id"], "clip_id": record["clip_id"],
                           "video_sha256": record["video_sha256"], "prediction": result,
                           "manual_annotation_required": False, "neural_weights_updated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--features-dir", type=Path, required=True)
    fit.add_argument("--labels", type=Path, required=True)
    fit.add_argument("--visual-metadata", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(run=train)
    inference = commands.add_parser("predict")
    inference.add_argument("--features", type=Path, required=True)
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.set_defaults(run=predict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
