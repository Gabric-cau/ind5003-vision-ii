"""Small, isolated video-category experiment; no ball-coordinate supervision.

The ImageNet encoder stays frozen. PCA and two regularized classifiers fit only
the training match. The World Cup match is never read by this experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = {
    "ball_direction": ["left", "center", "right"],
    "ball_height_category": ["low", "middle", "high"],
}
SEED = 20261025
PREPROCESS = {
    "frames": 64, "width": 512, "height": 288,
    "max_shot_seconds": 20.0, "scene_cut_threshold": 30.0,
    "window_policy": "automatic_first_shot",
    "temporal_bins": 4, "spatial_pool": [2, 2],
    "horizontal_flip": False,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_samples(samples):
    expected = {"train": "german_cup_2022", "validation": "coppa_italia_2025"}
    seen = set()
    hashes = set()
    for row in samples:
        if row["split"] not in expected or row["dataset_id"] != expected[row["split"]]:
            raise ValueError("Held-out or unexpected match in training input")
        if row["event_id"] in seen or row["video_sha256"] in hashes:
            raise ValueError("Duplicate event or video in training input")
        seen.add(row["event_id"])
        hashes.add(row["video_sha256"])
        if row.get("filename", Path(row.get("video_path", "")).name) == "match_09.mp4":
            raise ValueError("Unresolved conflicting event cannot enter this experiment")
        for task, classes in CLASSES.items():
            if row["targets"][task] not in classes:
                raise ValueError("Unknown target cannot enter supervised fitting")
    if {s["split"] for s in samples} != {"train", "validation"}:
        raise ValueError("Both development matches are required")


def sample_indices(end_frame, count):
    if end_frame < 1 or count < 1:
        raise ValueError("Empty video window")
    return np.rint(np.linspace(0, end_frame - 1, count)).astype(int).tolist()


def decode_video(video, settings):
    from src.step_pipeline import _scan_camera_shots
    started = time.perf_counter()
    window = _scan_camera_shots(str(video), settings["max_shot_seconds"],
                                settings["scene_cut_threshold"])[0]
    if window["shot_end_sec"] < .75:
        raise ValueError("First shot too short; automatic action selection needs improvement")
    ids = sample_indices(window["shot_end_frame"], settings["frames"])
    needed = set(ids)
    frames = {}
    capture = cv2.VideoCapture(str(video))
    try:
        for index in range(ids[-1] + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video stopped decoding at frame {index}")
            if index in needed:
                frames[index] = cv2.resize(frame, (settings["width"], settings["height"]),
                                           interpolation=cv2.INTER_AREA)
    finally:
        capture.release()
    # Use the same JPEG representation during extraction and standalone inference.
    encoded = [cv2.imencode(".jpg", frames[i], [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
               for i in ids]
    return encoded, {**window, "sample_frame_ids": ids,
                     "selection_uses_labels": False, "selection_human_verified": False,
                     "decode_and_selection_seconds": time.perf_counter() - started}


def extract(args):
    manifest = read_json(args.manifest)
    samples = manifest["samples"]
    validate_samples(samples)
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    records = []
    for index, sample in enumerate(samples):
        path = Path(sample["video_path"])
        if digest(path) != sample["video_sha256"]:
            raise ValueError(f"Video hash mismatch: {path.name}")
        frames, window = decode_video(path, PREPROCESS)
        folder = args.output / "frames" / f"event_{index:02d}"
        folder.mkdir(parents=True)
        files = []
        for j, encoded in enumerate(frames):
            frame_path = folder / f"{j:03d}.jpg"
            frame_path.write_bytes(encoded.tobytes())
            files.append({"path": str(frame_path.relative_to(args.output)),
                          "sha256": digest(frame_path)})
        records.append({"event_id": sample["event_id"], "dataset_id": sample["dataset_id"],
                        "split": sample["split"], "targets": sample["targets"],
                        "video_sha256": sample["video_sha256"], "filename": path.name,
                        "window": window, "frames": files})
        print(f"extracted {path.name}: {window['shot_end_sec']:.2f}s, {len(frames)} frames", flush=True)
    save_json(args.output / "frames.json", {
        "schema_version": 1, "manifest_sha256": digest(args.manifest),
        "settings": PREPROCESS, "samples": records,
        "source_annotation_sha256": manifest["annotation_source"]["sha256"],
        "held_out_world_cup_accessed": False,
        "labels_used_for_window_selection": False,
        "manual_timeline_used_for_window_selection": False,
    })


def load_encoder(checkpoint, device="cpu", threads=2):
    import torch
    torch.set_num_threads(threads)
    if Path(checkpoint).suffix == ".pt":
        encoder = torch.jit.load(str(checkpoint), map_location=device).eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        return encoder
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoder = torch.nn.Sequential(*list(model.children())[:-2]).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def encode_frames(frames, encoder, settings, device="cpu"):
    import torch
    started = time.perf_counter()
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    parts = []
    with torch.inference_mode():
        for batch in tensor.split(8):
            features = encoder(batch.to(device))
            parts.append(torch.nn.functional.adaptive_avg_pool2d(
                features, settings["spatial_pool"]).flatten(1).cpu().numpy())
    spatial = np.concatenate(parts)
    temporal = np.concatenate([part.mean(axis=0) for part in
                               np.array_split(spatial, settings["temporal_bins"])])
    return temporal, spatial[0], time.perf_counter() - started


def fit_heads(features, train_rows, train_targets):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    train_x = features[train_rows]
    scaler = StandardScaler().fit(train_x)
    pca = PCA(n_components=min(7, len(train_rows) - 1), svd_solver="full")
    transformed = pca.fit_transform(scaler.transform(train_x))
    heads = {}
    for task, classes in CLASSES.items():
        labels = np.array([classes.index(t[task]) for t in train_targets])
        if set(labels) != set(range(len(classes))):
            raise ValueError(f"Training split does not contain all classes for {task}")
        classifier = LogisticRegression(C=.1, class_weight="balanced", solver="lbfgs",
                                        max_iter=2000, random_state=SEED)
        classifier.fit(transformed, labels)
        heads[task] = classifier
    return scaler, pca, heads


def metrics(true, predicted, classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    return {"n": len(true), "correct": int(np.sum(np.array(true) == np.array(predicted))),
            "accuracy": float(accuracy_score(true, predicted)),
            "macro_f1": float(f1_score(true, predicted, labels=classes,
                                       average="macro", zero_division=0)),
            "classes": classes, "confusion_matrix_true_rows": confusion_matrix(
                true, predicted, labels=classes).tolist()}


def export_parameters(scaler, pca, heads):
    result = {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
              "pca_mean": pca.mean_, "pca_components": pca.components_}
    for task, head in heads.items():
        result[task + "_coef"] = head.coef_
        result[task + "_intercept"] = head.intercept_
    return result


def predict_parameters(features, parameters):
    x = (np.atleast_2d(features) - parameters["scaler_mean"]) / parameters["scaler_scale"]
    x = (x - parameters["pca_mean"]) @ parameters["pca_components"].T
    predictions = {}
    for task, classes in CLASSES.items():
        logits = x @ parameters[task + "_coef"].T + parameters[task + "_intercept"]
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        predictions[task] = {"labels": [classes[i] for i in probability.argmax(axis=1)],
                             "probabilities": probability.tolist()}
    return predictions


def train(args):
    import sklearn
    import torch
    import torchvision
    data = read_json(args.frames / "frames.json")
    validate_samples(data["samples"])
    if (data["held_out_world_cup_accessed"] or data["labels_used_for_window_selection"]
            or data["manual_timeline_used_for_window_selection"]):
        raise ValueError("Invalid data provenance")
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = load_encoder(args.backbone, threads=args.threads)
    all_features, first_frames, timings = [], [], []
    for row in data["samples"]:
        frames = []
        for record in row["frames"]:
            path = args.frames / record["path"]
            if digest(path) != record["sha256"]:
                raise ValueError("Frame checksum mismatch")
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError("Unreadable frame")
            frames.append(frame)
        feature, first, seconds = encode_frames(frames, encoder, data["settings"])
        all_features.append(feature)
        first_frames.append(first)
        timings.append(seconds)
        print(f"encoded {row['filename']}: {seconds:.2f}s", flush=True)
    x, context = np.stack(all_features), np.stack(first_frames)
    training = [i for i, r in enumerate(data["samples"]) if r["split"] == "train"]
    training_targets = [data["samples"][i]["targets"] for i in training]
    started = time.perf_counter()
    scaler, pca, heads = fit_heads(x, training, training_targets)
    fit_seconds = time.perf_counter() - started
    parameters = export_parameters(scaler, pca, heads)
    np.savez_compressed(args.output / "classifier_weights.npz", **parameters)
    np.savez_compressed(args.output / "feature_cache.npz", temporal=x, first_frame=context)
    # Export the frozen encoder so offline inference does not need torchvision.
    example = torch.zeros(1, 3, data["settings"]["height"], data["settings"]["width"])
    portable_encoder = torch.jit.freeze(torch.jit.trace(encoder, example).eval())
    portable_encoder.save(str(args.output / "encoder.pt"))
    predictions = predict_parameters(x, parameters)
    # Context-only control checks whether a static frame is as predictive as motion.
    static_scaler, static_pca, static_heads = fit_heads(context, training, training_targets)
    static_predictions = predict_parameters(context, export_parameters(static_scaler, static_pca, static_heads))
    report = {"experiment": "frozen_resnet18_temporal_category_head_v1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "backbone_trained": False, "classifier_trained": True,
              "ball_detector_trained": False, "trajectory_model_trained": False,
              "human_verified_accuracy": False, "held_out_test_evaluated": False,
              "promoted_to_default_pipeline": False, "fit_seconds": fit_seconds,
              "train_events": len(training), "validation_events": len(data["samples"]) - len(training),
              "feature_dimensions": int(x.shape[1]), "pca_dimensions": int(pca.n_components_),
              "training_only_preprocessing": True, "selection_used_validation": False,
              "seed": SEED, "logistic_C": .1, "class_weight": "balanced",
              "settings": data["settings"], "classes": CLASSES,
              "backbone_sha256": digest(args.backbone),
              "portable_encoder_sha256": digest(args.output / "encoder.pt"),
              "weights_sha256": digest(args.output / "classifier_weights.npz"),
              "frames_manifest_sha256": digest(args.frames / "frames.json"),
              "source_annotation_sha256": data["source_annotation_sha256"],
              "script_sha256": digest(__file__),
              "runtime": {"device": "cpu", "threads": args.threads,
                          "python": platform.python_version(), "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "sklearn": sklearn.__version__},
              "scores": {}, "predictions": [],
              "limitations": ["Single-observer coarse labels; height definition not yet adjudicated",
                              "Eight independent training events and seven development validation events",
                              "Training and validation differ in match and primary camera viewpoint",
                              "First-shot selection is automatic but not a general penalty event detector",
                              "No independent ball boxes, contact times, coordinates, or 3D trajectory targets",
                              "Probabilities are uncalibrated; context and player appearance may be shortcuts"]}
    for task, classes in CLASSES.items():
        train_labels = [r[task] for r in training_targets]
        majority = max(classes, key=train_labels.count)
        report["scores"][task] = {"training_majority_class": majority}
        for split in ("train", "validation"):
            indices = [i for i, r in enumerate(data["samples"]) if r["split"] == split]
            true = [data["samples"][i]["targets"][task] for i in indices]
            report["scores"][task][split] = {
                "model": metrics(true, [predictions[task]["labels"][i] for i in indices], classes),
                "majority": metrics(true, [majority] * len(indices), classes),
                "first_frame_control": metrics(true, [static_predictions[task]["labels"][i]
                                                        for i in indices], classes),
            }
    for i, row in enumerate(data["samples"]):
        report["predictions"].append({"event_id": row["event_id"], "filename": row["filename"],
            "split": row["split"], "reference": row["targets"], "window": row["window"],
            "feature_seconds": timings[i],
            "prediction": {task: {"label": predictions[task]["labels"][i],
                                  "probabilities_uncalibrated": predictions[task]["probabilities"][i]}
                           for task in CLASSES}})
    save_json(args.output / "report.json", report)
    print(json.dumps(report["scores"], indent=2), flush=True)


def predict(args):
    report = read_json(args.model / "report.json")
    if digest(args.backbone) not in (report["backbone_sha256"], report["portable_encoder_sha256"]):
        raise ValueError("Backbone hash mismatch")
    weights = args.model / "classifier_weights.npz"
    if digest(weights) != report["weights_sha256"]:
        raise ValueError("Classifier hash mismatch")
    started = time.perf_counter()
    encoded, window = decode_video(args.video, report["settings"])
    frames = [cv2.imdecode(frame, cv2.IMREAD_COLOR) for frame in encoded]
    encoder = load_encoder(args.backbone, threads=args.threads)
    feature, _, feature_seconds = encode_frames(frames, encoder, report["settings"])
    with np.load(weights, allow_pickle=False) as parameters:
        predictions = predict_parameters(feature, parameters)
    save_json(args.output, {"video": str(args.video), "video_sha256": digest(args.video),
        "window": window, "predictions": predictions, "probabilities_calibrated": False,
        "end_to_end_seconds_including_model_load": time.perf_counter() - started,
        "feature_seconds": feature_seconds, "manual_annotation_required": False,
        "scope": "experimental category prediction for an event clip starting with the shooting view",
        "final_pipeline_ready": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("extract")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=extract)
    p = commands.add_parser("train")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(run=train)
    p = commands.add_parser("predict")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--backbone", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(run=predict)
    args = parser.parse_args()
    if hasattr(args, "threads") and args.threads < 1:
        parser.error("threads must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
