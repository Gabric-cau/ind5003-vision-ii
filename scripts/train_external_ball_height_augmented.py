"""Add resolved existing event-level human height labels to the height head.

The old video frame clock is not transferred to the new core clip. The complete
first shot is selected automatically, and old direction labels are never used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_external_ball_classifier as previous
from scripts import train_external_ball_v2 as experiment

CLASSES = ["low", "middle", "high"]
EXCLUDED_CORES = {"core_07": "old middle conflicts with new human low description",
                  "core_09": "old human height unknown for out-of-goal shot"}


def admitted_core_labels(inputs, references):
    by_clip = {row["clip_id"]: row for row in references["events"]}
    admitted, rejected, seen = [], [], set()
    for row in inputs["samples"]:
        core_id = row["core_id"]
        if core_id in seen:
            raise ValueError("Duplicate core event")
        seen.add(core_id)
        if core_id in EXCLUDED_CORES:
            rejected.append({"core_id": core_id, "reason": EXCLUDED_CORES[core_id]})
            continue
        if core_id not in {f"core_{index:02d}" for index in range(1, 11)}:
            raise ValueError("Unexpected core source")
        reference = by_clip[row["clip_id"]]
        provenance = reference.get("_shot_save_annotation", {}).get("field_provenance", {}).get("shot_height", {})
        if provenance.get("status") != "human_annotated" or provenance.get("human_verified") is not True:
            raise ValueError("Core height lacks explicit human provenance")
        height = {"low": "low", "mid": "middle", "middle": "middle", "high": "high"}.get(reference.get("shot_height"))
        if height is None:
            raise ValueError("Unexpected unresolved height in admitted core event")
        admitted.append({**row, "height": height, "old_reference_video_sha256": reference["source_sha256"],
                         "label_inheritance": "same named event, different video cut; qualitative height only",
                         "reference_provenance": provenance})
    if len(admitted) != 8 or len(rejected) != 2:
        raise ValueError("Expected the fixed eight admitted and two excluded core events")
    if len({row["video_sha256"] for row in admitted}) != len(admitted):
        raise ValueError("Duplicate core video")
    return admitted, rejected


def fit_height(features, train_rows, labels):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if set(labels) != set(CLASSES):
        raise ValueError("Training requires all three height classes")
    scaler = StandardScaler().fit(features[train_rows])
    pca = PCA(n_components=min(7, len(train_rows) - 1), svd_solver="full")
    transformed = pca.fit_transform(scaler.transform(features[train_rows]))
    head = LogisticRegression(C=.1, class_weight="balanced", solver="lbfgs", max_iter=2000,
                              random_state=previous.SEED)
    head.fit(transformed, [CLASSES.index(label) for label in labels])
    return scaler, pca, head


def export_height(scaler, pca, head):
    return {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "pca_mean": pca.mean_, "pca_components": pca.components_,
            "coef": head.coef_, "intercept": head.intercept_}


def predict_height(features, weights):
    x = (np.atleast_2d(features) - weights["scaler_mean"]) / weights["scaler_scale"]
    x = (x - weights["pca_mean"]) @ weights["pca_components"].T
    logits = x @ weights["coef"].T + weights["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return {"labels": [CLASSES[i] for i in probability.argmax(axis=1)],
            "probabilities": probability.tolist()}


def train(args):
    data = previous.read_json(args.frames / "frames.json")
    experiment.validate_frames_manifest(data)
    prior = previous.read_json(args.previous_model / "report.json")
    if prior["frames_manifest_sha256"] != previous.digest(args.frames / "frames.json"):
        raise ValueError("Feature-cache manifest mismatch")
    if prior["portable_encoder_sha256"] != previous.digest(args.backbone):
        raise ValueError("Encoder hash mismatch")
    admitted, rejected = admitted_core_labels(previous.read_json(args.core_inputs), previous.read_json(args.references))
    if set(row["video_sha256"] for row in admitted) & set(row["video_sha256"] for row in data["samples"]):
        raise ValueError("A core video duplicates an external development video")
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = previous.load_encoder(args.backbone, threads=args.threads)
    with np.load(args.previous_model / "feature_cache.npz", allow_pickle=False) as cache:
        temporal = cache["temporal"].astype(np.float64)
        first = cache["first_frame"].astype(np.float64)
        features = [*list(temporal - np.tile(first, (1, 4)))]
    for row in admitted:
        video = Path(row["video_path"])
        if previous.digest(video) != row["video_sha256"]:
            raise ValueError("Core video checksum mismatch")
        started = time.perf_counter()
        encoded, window = previous.decode_video(video, data["settings"])
        frames = [cv2.imdecode(frame, cv2.IMREAD_COLOR) for frame in encoded]
        temporal, first, _ = previous.encode_frames(frames, encoder, data["settings"])
        features.append(temporal.astype(np.float64) - np.tile(first.astype(np.float64), 4))
        row["automatic_window"] = window
        row["feature_seconds"] = time.perf_counter() - started
        print(row["core_id"], row["height"], f"{row['feature_seconds']:.2f}s", flush=True)
    features = np.stack(features)
    train_rows = [i for i, row in enumerate(data["samples"]) if row["split"] == "train"]
    train_rows += list(range(len(data["samples"]), len(features)))
    labels = [row["targets"]["ball_height_category"] for row in data["samples"]] + [row["height"] for row in admitted]
    fitted = fit_height(features, train_rows, [labels[i] for i in train_rows])
    weights = export_height(*fitted)
    predictions = predict_height(features, weights)
    np.savez_compressed(args.output / "height_weights.npz", **weights)
    np.savez_compressed(args.output / "features.npz", first_frame_difference=features)
    validation = [i for i, row in enumerate(data["samples"]) if row["split"] == "validation"]
    training_labels = [labels[i] for i in train_rows]
    majority = max(CLASSES, key=training_labels.count)
    report = {"experiment": "external_height_core_augmentation_v2", "classes": CLASSES,
        "train_events": len(train_rows), "validation_events": len(validation),
        "train_match_groups": ["german_cup_2022", *sorted({row["match_group"] for row in admitted})],
        "validation_match_group": "coppa_italia_2025",
        "training_class_counts": {label: training_labels.count(label) for label in CLASSES},
        "height_only": True, "direction_labels_used": False, "old_contact_times_used": False,
        "new_contact_times_used": False, "manual_shot_boundaries_used": False,
        "backbone_trained": False, "classifier_trained": True, "training_only_preprocessing": True,
        "logistic_C": .1, "pca_dimensions": int(fitted[1].n_components_),
        "held_out_test_evaluated": False, "promoted_to_default_pipeline": False,
        "development_comparison_only": True, "settings": data["settings"],
        "admitted_cores": admitted, "excluded_cores": rejected,
        "event_ids": [row["event_id"] for row in data["samples"]] + [row["clip_id"] for row in admitted],
        "reference_labels": labels, "train_indices": train_rows, "validation_indices": validation,
        "predictions": predictions, "scores": {},
        "portable_encoder_sha256": previous.digest(args.backbone),
        "weights_sha256": previous.digest(args.output / "height_weights.npz"),
        "frames_manifest_sha256": previous.digest(args.frames / "frames.json"),
        "core_inputs_sha256": previous.digest(args.core_inputs),
        "old_human_reference_sha256": previous.digest(args.references),
        "script_sha256": previous.digest(__file__),
        "limitations": ["Coarse old event labels are inherited onto different verified video cuts",
                        "Height is ordinal and not calibrated metric height or a 3D trajectory",
                        "Validation match has already been inspected in earlier development",
                        "Automatic first-shot selection can miss action across a camera cut"]}
    for split, indices in (("train", train_rows), ("validation", validation)):
        true = [labels[i] for i in indices]
        report["scores"][split] = {
            "model": previous.metrics(true, [predictions["labels"][i] for i in indices], CLASSES),
            "majority": previous.metrics(true, [majority] * len(indices), CLASSES),
            "training_majority_class": majority}
    previous.save_json(args.output / "report.json", report)
    print(report["scores"], flush=True)


def predict(args):
    report = previous.read_json(args.model / "report.json")
    if previous.digest(args.backbone) != report["portable_encoder_sha256"]:
        raise ValueError("Encoder hash mismatch")
    weights_path = args.model / "height_weights.npz"
    if previous.digest(weights_path) != report["weights_sha256"]:
        raise ValueError("Classifier hash mismatch")
    started = time.perf_counter()
    encoded, window = previous.decode_video(args.video, report["settings"])
    frames = [cv2.imdecode(frame, cv2.IMREAD_COLOR) for frame in encoded]
    encoder = previous.load_encoder(args.backbone, threads=args.threads)
    temporal, first, _ = previous.encode_frames(frames, encoder, report["settings"])
    feature = temporal.astype(np.float64) - np.tile(first.astype(np.float64), 4)
    with np.load(weights_path, allow_pickle=False) as weights:
        prediction = predict_height(feature, weights)
    previous.save_json(args.output, {"video_sha256": previous.digest(args.video),
        "height_prediction": prediction, "window": window, "manual_annotation_required": False,
        "probabilities_calibrated": False, "experimental_candidate": True,
        "end_to_end_seconds_including_model_load": time.perf_counter() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--previous-model", type=Path, required=True)
    p.add_argument("--core-inputs", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.set_defaults(run=train)
    q = commands.add_parser("predict")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--video", type=Path, required=True)
    q.set_defaults(run=predict)
    for command in (p, q):
        command.add_argument("--backbone", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
