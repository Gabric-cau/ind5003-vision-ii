"""Leave-one-match-out contact-time development experiment on ten core clips."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import contact_temporal_model as model


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def references_from(document):
    rows = document.get("samples", document.get("labels", document.get("events"))) if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("Reference JSON requires a samples list")
    references = {}
    for row in rows:
        if row["core_id"] in references:
            raise ValueError("Duplicate reference core ID")
        references[row["core_id"]] = row
    return references


def baseline(record, name):
    result = dict(record.get("baselines", {}).get(name) or {})
    value = result.get("time_seconds")
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value)):
        raise ValueError("Nonfinite baseline contact time")
    result["time_seconds"] = float(value) if value is not None else None
    return result


def train(args):
    import sklearn
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve previous experiments")
    paths = sorted(args.features_dir.glob("core_*.json"))
    records = [read_json(path) for path in paths]
    references = references_from(read_json(args.labels))
    model.validate_cohort(records, references, strict=True)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    rows, folds = [], []
    for group, training, held_out in model.leave_one_match_out(records):
        scaler, head, audit = model.fit_contact_head(training, references)
        parameters = model.export_parameters(scaler, head)
        weights_path = args.output / "fold_weights" / f"{group}.npz"
        weights_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(weights_path, **parameters)
        median = float(np.median([references[row["core_id"]]["reference_contact_seconds"] for row in training]))
        fold = {"held_out_match": group, "training_core_ids": [row["core_id"] for row in training],
                "validation_core_ids": [row["core_id"] for row in held_out], "training_frame_audit": audit,
                "training_reference_median_seconds": median, "weights_file": str(weights_path.relative_to(args.output)),
                "weights_sha256": digest(weights_path)}
        folds.append(fold)
        for record in held_out:
            reference = model.validate_reference(record, references[record["core_id"]])
            prediction_started = time.perf_counter()
            prediction = model.predict_contact(record, parameters)
            window = record["window"]
            rows.append({"core_id": record["core_id"], "clip_id": record["clip_id"],
                         "match_group": group, "video_sha256": record["video_sha256"],
                         "reference_contact_seconds": reference,
                         "reference_precision": references[record["core_id"]].get("reference_precision"),
                         "reference_outside_automatic_window": not window["start_frame"] / record["fps"] <= reference < window["end_frame_exclusive"] / record["fps"],
                         "automatic_window": window, "predictions": {
                             "trained_contact": prediction, "existing_rule": baseline(record, "existing_rule"),
                             "ankle_proximity": baseline(record, "ankle_proximity"),
                             "training_median": {"time_seconds": median, "uses_video": False}},
                         "head_prediction_seconds": time.perf_counter() - prediction_started,
                         "feature_runtime": record.get("runtime")})
        print(f"fold {group}: train {len(training)}, validate {len(held_out)}", flush=True)
    rows.sort(key=lambda row: row["core_id"])
    scaler, head, audit = model.fit_contact_head(records, references)
    final_weights = args.output / "deployment_weights.npz"
    np.savez_compressed(final_weights, **model.export_parameters(scaler, head))
    metadata = {"schema_version": 1, "model_type": "regularized_logistic_temporal_contact_v1",
                "feature_names": records[0]["feature_names"], "lags_seconds": list(model.LAGS_SECONDS),
                "smoothing_seconds": model.SMOOTHING_SECONDS, "edge_policy": "full_lag_context_required",
                "minimum_support_samples": model.MINIMUM_SUPPORT_SAMPLES,
                "maximum_support_gap_median_step_ratio": 1.5,
                "evidence_gate": "current_pose_visible_and_current_ball_visible_or_stationary_seed_available",
                "positive_radius_seconds": model.POSITIVE_RADIUS_SECONDS,
                "negative_radius_seconds": model.NEGATIVE_RADIUS_SECONDS,
                "regularization_C": model.REGULARIZATION_C,
                "sample_weights": "each_event_positive_total_0.5_negative_total_0.5",
                "preprocessing_fit_on_training_only": True, "absolute_time_features": False,
                "deployment_training_core_ids": [row["core_id"] for row in records],
                "weights_sha256": digest(final_weights), "deployment_in_sample_scores_reported": False}
    metadata.update(model.visual_bindings(records[0]))
    save_json(args.output / "deployment_metadata.json", metadata)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": metadata,
              "events": len(records), "folds": folds, "scores": model.compare_predictions(rows),
              "evaluation": "leave_one_match_out_development_on_previously_seen_course_events",
              "held_out_independent_test": False, "hyperparameter_search": False,
              "neural_detector_weights_updated": False, "contact_classifier_weights_trained": True,
              "external_world_cup_accessed": False, "pose_pack_accessed": False,
              "promoted_to_default_pipeline": False, "deployment_training_audit": audit,
              "training_and_evaluation_seconds": time.perf_counter() - started,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "device": "cpu"},
              "source_bindings": {"labels_sha256": digest(args.labels), "features": [
                  {"file": path.name, "sha256": digest(path)} for path in paths]},
              "code_sha256": {"trainer": digest(__file__), "model": digest(model.__file__)},
              "limitations": ["Human contact times are coarse references, not exact frame ground truth.",
                              "The ten original-course events have previously informed project development.",
                              "Same-event frames are correlated; event and match counts, not frame counts, describe sample diversity.",
                              "Automatic windows may omit contact; references never adjust validation windows.",
                              "Frozen detection failures and missing pose evidence limit contact localization.",
                              "Fewer than three consecutive eligible observations abstain; all-event timing hit rates retain unavailable events in the denominator.",
                              "The learned head and ankle baseline share the existing rule's automatic actor and stationary-ball seed when available.",
                              "Camera-pair reliability describes adjacent-frame estimation, not necessarily the full reference prefix.",
                              "OOF models exclude their validation match; deployment weights fit all ten events.",
                              "Probability scores are uncalibrated; no timing accuracy is established on unseen footage."]}
    save_json(args.output / "oof_predictions.json", {"schema_version": 1, "predictions": rows})
    save_json(args.output / "report.json", report)
    print(json.dumps(report["scores"], indent=2), flush=True)


def predict(args):
    record, metadata = read_json(args.features), read_json(args.model / "deployment_metadata.json")
    weights = args.model / "deployment_weights.npz"
    if digest(weights) != metadata["weights_sha256"]:
        raise ValueError("Deployment weights hash mismatch")
    model.validate_prediction_bindings(record, metadata)
    started = time.perf_counter()
    with np.load(weights, allow_pickle=False) as parameters:
        result = model.predict_contact(record, parameters)
    save_json(args.output, {"core_id": record["core_id"], "clip_id": record["clip_id"],
                           "video_sha256": record["video_sha256"], "prediction": result,
                           "manual_annotation_required": False,
                           "head_prediction_seconds": time.perf_counter() - started,
                           "feature_runtime": record.get("runtime"),
                           "scope": "experimental_contact_time_on_automatic_video_features"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--features-dir", type=Path, required=True)
    train_parser.add_argument("--labels", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.set_defaults(run=train)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--features", type=Path, required=True)
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.set_defaults(run=predict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
