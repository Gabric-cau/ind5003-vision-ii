"""Development-only motion ablation on the two admitted external matches.

All candidates are declared together, use the same fixed classifier, and leave
the World Cup match unopened. Dense flow describes visible motion, not a ball
trajectory: player motion and camera motion can still confound these features.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_external_ball_classifier as previous

CANDIDATES = ("appearance_temporal", "first_frame_difference", "motion_grid")
FLOW_SETTINGS = {"width": 256, "height": 144, "grid_rows": 3, "grid_columns": 4,
                 "temporal_bins": 4, "background": "median_dense_translation",
                 "pyr_scale": .5, "levels": 3, "winsize": 15, "iterations": 3,
                 "poly_n": 5, "poly_sigma": 1.2}


def validate_frames_manifest(data):
    previous.validate_samples(data["samples"])
    for field in ("held_out_world_cup_accessed", "labels_used_for_window_selection",
                  "manual_timeline_used_for_window_selection"):
        if data.get(field) is not False:
            raise ValueError(f"Invalid or missing provenance: {field}")
    if data["settings"] != previous.PREPROCESS:
        raise ValueError("Frame preprocessing changed from the audited cache")


def motion_grid(frames):
    if len(frames) < 5:
        raise ValueError("Motion requires at least five ordered frames")
    size = (FLOW_SETTINGS["width"], FLOW_SETTINGS["height"])
    gray = [cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY) for f in frames]
    summaries = []
    for first, second in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(first, second, None, .5, 3, 15, 3, 5, 1.2, 0)
        flow -= np.median(flow.reshape(-1, 2), axis=0)
        flow /= np.array(size, dtype=np.float32)
        cells = []
        for row in np.array_split(flow, FLOW_SETTINGS["grid_rows"], axis=0):
            for cell in np.array_split(row, FLOW_SETTINGS["grid_columns"], axis=1):
                vectors = cell.reshape(-1, 2)
                magnitude = np.linalg.norm(vectors, axis=1)
                cells.extend([*np.mean(vectors, axis=0),
                              float(np.quantile(magnitude, .75)),
                              float(np.quantile(magnitude, .95))])
        summaries.append(cells)
    return np.concatenate([part.mean(axis=0) for part in np.array_split(
        np.asarray(summaries), FLOW_SETTINGS["temporal_bins"])])


def descriptor_sets(temporal, first, motion):
    temporal = np.asarray(temporal, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    motion = np.asarray(motion, dtype=np.float64)
    if temporal.ndim != 2 or first.ndim != 2 or motion.ndim != 2:
        raise ValueError("Expected per-event feature matrices")
    if temporal.shape != (len(first), 4 * first.shape[1]) or len(motion) != len(first):
        raise ValueError("Temporal/first-frame/motion cache dimensions disagree")
    if not all(np.isfinite(x).all() for x in (temporal, first, motion)):
        raise ValueError("Features must be finite")
    return {"appearance_temporal": temporal,
            "first_frame_difference": temporal - np.tile(first, (1, 4)),
            "motion_grid": motion}


def load_frames(folder, row):
    frames = []
    for item in row["frames"]:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Frame path escapes its audited cache")
        if previous.digest(path) != item["sha256"]:
            raise ValueError("Frame checksum mismatch")
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError("Unreadable frame")
        frames.append(frame)
    return frames


def score_predictions(data, predictions):
    scores = {}
    for task, classes in previous.CLASSES.items():
        train_labels = [r["targets"][task] for r in data["samples"] if r["split"] == "train"]
        majority = max(classes, key=train_labels.count)
        scores[task] = {"training_majority_class": majority}
        for split in ("train", "validation"):
            selected = [i for i, r in enumerate(data["samples"]) if r["split"] == split]
            true = [data["samples"][i]["targets"][task] for i in selected]
            scores[task][split] = {
                "model": previous.metrics(true, [predictions[task]["labels"][i]
                                                 for i in selected], classes),
                "majority": previous.metrics(true, [majority] * len(true), classes)}
    return scores


def train(args):
    data = previous.read_json(args.frames / "frames.json")
    validate_frames_manifest(data)
    old = previous.read_json(args.previous_model / "report.json")
    if previous.digest(args.frames / "frames.json") != old["frames_manifest_sha256"]:
        raise ValueError("Prior model cache belongs to a different frame manifest")
    cache = args.previous_model / "feature_cache.npz"
    with np.load(cache, allow_pickle=False) as arrays:
        temporal, first = arrays["temporal"].copy(), arrays["first_frame"].copy()
    if len(temporal) != len(data["samples"]):
        raise ValueError("Prior feature count mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    motions, timings = [], []
    for row in data["samples"]:
        started = time.perf_counter()
        motions.append(motion_grid(load_frames(args.frames, row)))
        timings.append(time.perf_counter() - started)
        print(f"motion {row['filename']}: {timings[-1]:.2f}s", flush=True)
    descriptors = descriptor_sets(temporal, first, motions)
    np.savez_compressed(args.output / "features.npz", **descriptors)
    train_rows = [i for i, row in enumerate(data["samples"]) if row["split"] == "train"]
    train_targets = [data["samples"][i]["targets"] for i in train_rows]
    report = {
        "schema_version": 1, "experiment": "external_ball_motion_ablation_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_test_evaluated": False, "held_out_world_cup_accessed": False,
        "selection_used_validation": False, "development_comparison_only": True,
        "promoted_to_default_pipeline": False, "backbone_trained": False,
        "ball_detector_trained": False, "trajectory_model_trained": False,
        "classifier_trained": True, "training_only_preprocessing": True,
        "candidate_policy": "Three fixed descriptors; fixed C=0.1 and train-only PCA7, no tuning",
        "train_match": "german_cup_2022", "validation_match": "coppa_italia_2025",
        "train_events": len(train_rows), "validation_events": len(data["samples"]) - len(train_rows),
        "frames_manifest_sha256": previous.digest(args.frames / "frames.json"),
        "prior_feature_cache_sha256": previous.digest(cache),
        "portable_encoder_sha256": old["portable_encoder_sha256"],
        "source_annotation_sha256": data["source_annotation_sha256"],
        "script_sha256": previous.digest(__file__), "flow_settings": FLOW_SETTINGS,
        "settings": data["settings"], "motion_feature_seconds": timings,
        "reference_quality": "coarse single-observer categories, not adjudicated ball coordinates",
        "limitations": ["Seven validation events have already been inspected in v1 development",
                        "Motion grid measures all visible motion, not isolated ball motion",
                        "No independent test or full-pipeline accuracy claim",
                        "Training and validation differ in match and camera viewpoint"],
        "candidates": {}, "events": [{k: row[k] for k in
            ("event_id", "filename", "dataset_id", "split", "video_sha256", "targets")}
            for row in data["samples"]]}
    for name, features in descriptors.items():
        started = time.perf_counter()
        fitted = previous.fit_heads(features, train_rows, train_targets)
        weights = previous.export_parameters(*fitted)
        np.savez_compressed(args.output / (name + "_weights.npz"), **weights)
        predictions = previous.predict_parameters(features, weights)
        report["candidates"][name] = {
            "feature_dimensions": features.shape[1], "fit_seconds": time.perf_counter() - started,
            "weights_sha256": previous.digest(args.output / (name + "_weights.npz")),
            "scores": score_predictions(data, predictions), "predictions": predictions}
    previous.save_json(args.output / "report.json", report)
    for name, result in report["candidates"].items():
        print(name, {task: values["validation"]["model"]
                     for task, values in result["scores"].items()}, flush=True)


def predict(args):
    report = previous.read_json(args.model / "report.json")
    if previous.digest(args.backbone) != report["portable_encoder_sha256"]:
        raise ValueError("Encoder hash mismatch")
    if args.candidate not in report["candidates"]:
        raise ValueError("Candidate is absent from this experiment")
    weight_path = args.model / (args.candidate + "_weights.npz")
    if previous.digest(weight_path) != report["candidates"][args.candidate]["weights_sha256"]:
        raise ValueError("Classifier hash mismatch")
    started = time.perf_counter()
    encoded, window = previous.decode_video(args.video, report["settings"])
    frames = [cv2.imdecode(item, cv2.IMREAD_COLOR) for item in encoded]
    if args.candidate == "motion_grid":
        features = motion_grid(frames)
    else:
        encoder = previous.load_encoder(args.backbone, threads=args.threads)
        temporal, first, _ = previous.encode_frames(frames, encoder, report["settings"])
        features = (temporal if args.candidate == "appearance_temporal" else
                    temporal.astype(np.float64) - np.tile(first.astype(np.float64), 4))
    with np.load(weight_path, allow_pickle=False) as parameters:
        predictions = previous.predict_parameters(features, parameters)
    height_model_sha256 = None
    if getattr(args, "height_model", None) is not None:
        if args.candidate != "first_frame_difference":
            raise ValueError("Augmented height model requires the first-frame-difference descriptor")
        from scripts import train_external_ball_height_augmented as height_model
        height_report = previous.read_json(args.height_model / "report.json")
        if (height_report["portable_encoder_sha256"] != report["portable_encoder_sha256"] or
                height_report["settings"] != report["settings"]):
            raise ValueError("Height model encoder/preprocessing mismatch")
        height_weights = args.height_model / "height_weights.npz"
        if previous.digest(height_weights) != height_report["weights_sha256"]:
            raise ValueError("Height classifier hash mismatch")
        with np.load(height_weights, allow_pickle=False) as parameters:
            predictions["ball_height_category"] = height_model.predict_height(features, parameters)
        height_model_sha256 = previous.digest(args.height_model / "report.json")
    previous.save_json(args.output, {"video": str(args.video),
        "video_sha256": previous.digest(args.video), "candidate": args.candidate,
        "predictions": predictions, "window": window,
        "end_to_end_seconds_including_model_load": time.perf_counter() - started,
        "probabilities_calibrated": False, "manual_annotation_required": False,
        "experimental_candidate": True, "held_out_test_accuracy_established": False,
        "height_accuracy_below_training_majority_baseline_on_development": True,
        "model_report_sha256": previous.digest(args.model / "report.json"),
        "height_model_report_sha256": height_model_sha256,
        "feature_dimensions": int(features.size)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--frames", type=Path, required=True)
    fit.add_argument("--previous-model", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.set_defaults(run=train)
    inference = commands.add_parser("predict")
    inference.add_argument("--video", type=Path, required=True)
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--backbone", type=Path, required=True)
    inference.add_argument("--candidate", choices=CANDIDATES, required=True)
    inference.add_argument("--height-model", type=Path)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--threads", type=int, default=2)
    inference.set_defaults(run=predict)
    args = parser.parse_args()
    if getattr(args, "threads", 2) < 1:
        parser.error("threads must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
