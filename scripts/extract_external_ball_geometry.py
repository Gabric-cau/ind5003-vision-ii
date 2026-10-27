"""Label-free ball proposals for the admitted external development videos.

Ball detections and classical image goal frames are evidence proposals, not
ground truth or a calibrated 3D trajectory. This script never loads holdouts.
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
from src.goal_geometry import detect_goal_frame

SETTINGS = {"width": 1280, "height": 720, "confidence": .03, "image_size": 1280,
            "classes": [32], "batch": 4, "goal_detector": "existing_conservative_goal_geometry"}


def prepare(args):
    source = previous.read_json(args.manifest)
    previous.validate_samples(source["samples"])
    frames = previous.read_json(args.frames / "frames.json")
    experiment.validate_frames_manifest(frames)
    if frames["manifest_sha256"] != previous.digest(args.manifest):
        raise ValueError("Audited frame cache and source manifest disagree")
    source_rows = {r["event_id"]: r for r in source["samples"]}
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, row in enumerate(frames["samples"]):
        source_row = source_rows[row["event_id"]]
        video = Path(source_row["video_path"])
        if previous.digest(video) != row["video_sha256"]:
            raise ValueError("Source video checksum mismatch")
        ids = row["window"]["sample_frame_ids"]
        capture = cv2.VideoCapture(str(video))
        wanted, decoded = set(ids), {}
        try:
            for index in range(max(ids) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Video ended within the audited automatic window")
                if index in wanted:
                    decoded[index] = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        finally:
            capture.release()
        folder = args.output / "frames" / f"event_{i:02d}"
        folder.mkdir(parents=True)
        files = []
        for j, frame_id in enumerate(ids):
            path = folder / f"{j:03d}.jpg"
            cv2.imwrite(str(path), decoded[frame_id], [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append({"path": str(path.relative_to(args.output)), "sha256": previous.digest(path),
                          "frame_id": frame_id, "time_seconds": frame_id / row["window"]["fps"]})
        records.append({"event_id": row["event_id"], "video_sha256": row["video_sha256"],
                        "dataset_id": row["dataset_id"], "split": row["split"],
                        "filename": row["filename"], "frames": files})
        print(f"prepared {row['filename']}", flush=True)
    previous.save_json(args.output / "manifest.json", {"samples": records,
        "settings": SETTINGS, "labels_used": False, "held_out_world_cup_accessed": False,
        "source_manifest_sha256": previous.digest(args.manifest),
        "audited_frames_manifest_sha256": previous.digest(args.frames / "frames.json")})


def detect(args):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    data = previous.read_json(args.frames / "manifest.json")
    if data["labels_used"] or data["held_out_world_cup_accessed"]:
        raise ValueError("Invalid geometry extraction provenance")
    permitted = {"train": "german_cup_2022", "validation": "coppa_italia_2025"}
    if any(permitted.get(r["split"]) != r["dataset_id"] for r in data["samples"]):
        raise ValueError("Held-out or unexpected source")
    args.output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(args.weights))
    records = []
    for row in data["samples"]:
        started = time.perf_counter()
        images = experiment.load_frames(args.frames, row)
        predictions = []
        for offset in range(0, len(images), 4):
            results = model.predict(images[offset:offset + 4], imgsz=1280, conf=.03,
                                    classes=[32], device=args.device, verbose=False)
            for result in results:
                boxes = result.boxes
                predictions.append([{"bbox": xy.tolist(), "confidence": float(conf)}
                    for xy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())])
        goals = [detect_goal_frame(image) for image in images]
        observations = [{"frame_id": frame["frame_id"], "time_seconds": frame["time_seconds"],
                         "balls": balls, "goal": goal}
                        for frame, balls, goal in zip(row["frames"], predictions, goals)]
        record = {k: v for k, v in row.items() if k != "frames"}
        record.update(observations=observations, elapsed_seconds=time.perf_counter() - started)
        records.append(record)
        previous.save_json(args.output / (row["filename"] + ".json"), record)
        print(row["filename"], f"{record['elapsed_seconds']:.2f}s",
              "ball_frames", sum(bool(p) for p in predictions),
              "goal_frames", sum(bool(g.get("selected")) for g in goals), flush=True)
    previous.save_json(args.output / "detections.json", {"samples": records, "settings": SETTINGS,
        "weights_sha256": previous.digest(args.weights), "script_sha256": previous.digest(__file__),
        "goal_detector_sha256": previous.digest(ROOT / "src" / "goal_geometry.py"),
        "frames_manifest_sha256": previous.digest(args.frames / "manifest.json"),
        "labels_used": False, "held_out_world_cup_accessed": False})


def geometry_descriptor(record):
    observations = record["observations"]
    goals = [o["goal"]["selected"]["corners"] for o in observations if o["goal"].get("selected")]
    goal = np.median(np.asarray(goals), axis=0).astype(np.float32) if goals else None
    transform = (cv2.getPerspectiveTransform(goal, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
                 if goal is not None else None)
    rows = []
    previous_point = None
    for observation in observations:
        balls = observation["balls"]
        # Deterministic continuity is only a proposal; keep uncertainty and gaps.
        candidates = []
        for ball in balls:
            box = np.array(ball["bbox"])
            center = (box[:2] + box[2:]) / 2
            if center[1] < .12 * 720:
                continue
            distance = 0 if previous_point is None else np.linalg.norm((center - previous_point) / [1280, 720])
            candidates.append((ball["confidence"] - .5 * distance, center, ball["confidence"]))
        if not candidates:
            rows.append([0., 0., 0., 0., 0., float(goal is not None)])
            continue
        _, center, confidence = max(candidates, key=lambda item: item[0])
        previous_point = center
        if transform is None:
            normalized = center / [1280, 720]
        else:
            normalized = cv2.perspectiveTransform(center.reshape(1, 1, 2).astype(np.float32), transform)[0, 0]
        normalized = np.clip(normalized, -3, 4)
        rows.append([1., confidence, *normalized, float(0 <= normalized[0] <= 1 and
                                                     0 <= normalized[1] <= 1), float(goal is not None)])
    features = []
    for block in np.array_split(np.asarray(rows), 4):
        available = block[:, 0] > 0
        features.extend([float(available.mean()), float(goal is not None)])
        features.extend(block[available, 1:5].mean(axis=0).tolist() if available.any() else [0.] * 4)
        features.extend(np.quantile(block[available, 2:4], [.1, .5, .9], axis=0).ravel().tolist()
                        if available.any() else [0.] * 6)
    return np.asarray(features, dtype=np.float64)


def fit(args):
    data = previous.read_json(args.manifest)
    previous.validate_samples(data["samples"])
    detections = previous.read_json(args.detections / "detections.json")
    if detections["labels_used"] or detections["held_out_world_cup_accessed"]:
        raise ValueError("Invalid detection provenance")
    rows = {r["event_id"]: r for r in detections["samples"]}
    ordered = []
    for sample in data["samples"]:
        record = rows[sample["event_id"]]
        if record["video_sha256"] != sample["video_sha256"]:
            raise ValueError("Detection/video mismatch")
        ordered.append(record)
    features = np.stack([geometry_descriptor(record) for record in ordered])
    training = [i for i, r in enumerate(data["samples"]) if r["split"] == "train"]
    targets = [data["samples"][i]["targets"] for i in training]
    fitted = previous.fit_heads(features, training, targets)
    weights = previous.export_parameters(*fitted)
    predictions = previous.predict_parameters(features, weights)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "features.npz", geometry=features)
    np.savez_compressed(args.output / "weights.npz", **weights)
    scores = experiment.score_predictions(data, predictions)
    previous.save_json(args.output / "report.json", {"experiment": "external_ball_geometry_control_v2",
        "scores": scores, "predictions": predictions, "train_events": len(training),
        "validation_events": len(ordered) - len(training), "feature_dimensions": features.shape[1],
        "logistic_C": .1, "classifier_trained": True, "backbone_trained": False,
        "ball_detector_trained": False, "trajectory_model_trained": False,
        "held_out_test_evaluated": False, "promoted_to_default_pipeline": False,
        "development_comparison_only": True, "training_only_preprocessing": True,
        "goal_coverage_by_event": [sum(bool(o["goal"].get("selected")) for o in r["observations"])
                                   for r in ordered],
        "ball_coverage_by_event": [sum(bool(o["balls"]) for o in r["observations"]) for r in ordered],
        "limitations": ["Sparse sampled ball boxes can miss the airborne ball",
                        "Image goal homography is not a 3D ball trajectory",
                        "Fallback coordinates use image space when no goal is supported",
                        "Only eight training and seven previously inspected development events"],
        "detections_sha256": previous.digest(args.detections / "detections.json"),
        "manifest_sha256": previous.digest(args.manifest), "script_sha256": previous.digest(__file__)})
    print(scores, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=prepare)
    p = commands.add_parser("detect")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=detect)
    p = commands.add_parser("fit")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=fit)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
