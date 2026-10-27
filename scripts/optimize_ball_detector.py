"""Source-bound development experiments and label-free small-ball inference."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_ball_frame_detector import digest, save_json, score_frame, summarize
from src.ball_specialist import ball_class, predict_images, tile_windows


def load_exports(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for record in manifest["exports"]:
        image = manifest_path.parent / record["image"]
        label = manifest_path.parent / record["label"]
        if digest(image) != record["image_sha256"] or digest(label) != record["label_sha256"]:
            raise ValueError("Export changed after annotation import")
        fields = label.read_text().split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError("Expected exactly one class-0 target")
        values = list(map(float, fields[1:]))
        if not all(0 <= value <= 1 for value in values) or min(values[2:]) <= 0:
            raise ValueError("Invalid normalized target")
        rows.append({**record, "image": image, "label": label, "normalized": values,
                     "video_id": record.get("video_id", image.stem.rsplit("_f", 1)[0]),
                     "split": record.get("split", "test")})
    if not rows:
        raise ValueError("Empty evaluation manifest")
    return rows


def summarize_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    unique = [row for row in rows if row["deduplicated_primary"]]
    return {"all_frames": summarize(rows), "deduplicated_frames": summarize(unique),
            "by_video": {key: {"all_frames": summarize(value),
                         "deduplicated_frames": summarize([r for r in value if r["deduplicated_primary"]])}
                         for key, value in grouped.items()}}


def evaluate_checkpoint(weights, items, output, mode="full", imgsz=1280, device="0", batch=4):
    import cv2
    from ultralytics import YOLO
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    model = YOLO(str(weights))
    rows, seen = [], set()
    for start in range(0, len(items), batch):
        samples = items[start:start + batch]
        images = [cv2.imread(str(sample["image"])) for sample in samples]
        if any(image is None for image in images):
            raise ValueError("Undecodable source image")
        predictions = predict_images(model, images, mode, imgsz, device, batch)
        for sample, image, candidates in zip(samples, images, predictions):
            height, width = image.shape[:2]
            x, y, w, h = sample["normalized"]
            reference = [(x - w / 2) * width, (y - h / 2) * height,
                         (x + w / 2) * width, (y + h / 2) * height]
            key = (sample["video_id"], hashlib.sha256(image.tobytes()).hexdigest())
            primary = sample.get("deduplicated_primary", key not in seen)
            seen.add(key)
            rows.append({"image": sample["image"].name, "video_id": sample["video_id"],
                         "annotation_id": sample.get("annotation_id", sample["image"].stem),
                         "deduplicated_primary": primary, "reference_bbox": reference,
                         "candidates": candidates, **score_frame(reference, candidates)})
    result = {"weights_sha256": digest(weights), "mode": mode, "image_size": imgsz,
              "inference_code_sha256": digest(ROOT / "src/ball_specialist.py"),
              "confidence": .05, "nms_iou": .6, "class_id": ball_class(model),
              "prediction_uses_reference": False, "metrics": summarize_groups(rows), "rows": rows,
              "seconds_including_model_load": time.perf_counter() - started}
    save_json(output, result)
    return {key: value for key, value in result.items() if key != "rows"}


def make_training_dataset(items, output, crop_count):
    import cv2
    import yaml
    if any(row["split"] not in {"train", "validation"} for row in items):
        raise ValueError("Evaluation-only or unknown splits cannot enter model selection")
    rng = random.Random(27)
    records = []
    if {row["video_id"] for row in items if row["split"] == "train"} != {"core_01", "core_02", "core_03"}:
        raise ValueError("Training groups must remain fixed")
    if {row["video_id"] for row in items if row["split"] == "validation"} != {"match_01"}:
        raise ValueError("Validation group must remain fixed")
    for row in items:
        split = "train" if row["split"] == "train" else "val"
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["image"], image_dir / row["image"].name)
        shutil.copy2(row["label"], label_dir / row["label"].name)
        if split != "train":
            continue
        image = cv2.imread(str(row["image"]))
        height, width = image.shape[:2]
        x, y, w, h = row["normalized"]
        for index in range(crop_count):
            fraction = (.5, .75)[index % 2]
            cw, ch = round(width * fraction), round(height * fraction)
            xmin, xmax = max(0, int((x + w / 2) * width) - cw + 2), min(width - cw, int((x - w / 2) * width) - 2)
            ymin, ymax = max(0, int((y + h / 2) * height) - ch + 2), min(height - ch, int((y - h / 2) * height) - 2)
            if xmin > xmax or ymin > ymax:
                continue
            left, top = rng.randint(xmin, xmax), rng.randint(ymin, ymax)
            name = row["image"].stem + f"_crop{index}"
            crop = image[top:top + ch, left:left + cw]
            if not cv2.imwrite(str(image_dir / (name + ".jpg")), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError("Failed to export training crop")
            normalized = [(x * width - left) / cw, (y * height - top) / ch, w * width / cw, h * height / ch]
            (label_dir / (name + ".txt")).write_text("0 " + " ".join(f"{v:.9f}" for v in normalized) + "\n")
            records.append({"source": row["image"].name, "crop": name, "xyxy": [left, top, left + cw, top + ch],
                            "source_sha256": digest(row["image"]), "new_independent_event": False})
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": {0: "ball"}}
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(config))
    save_json(output / "crop_provenance.json", records)
    return path


EXPERIMENTS = [
    {"name": "conservative", "freeze": 20, "lr": .0001, "epochs": 40, "imgsz": 1280, "crops": 0},
    {"name": "adapt_backbone", "freeze": 0, "lr": .00005, "epochs": 40, "imgsz": 1280, "crops": 0},
    {"name": "higher_resolution", "freeze": 10, "lr": .0002, "epochs": 30, "imgsz": 1920, "crops": 0},
    {"name": "mixed_crops", "freeze": 10, "lr": .0002, "epochs": 35, "imgsz": 1280, "crops": 2},
    {"name": "mixed_crops_conservative", "freeze": 20, "lr": .0001, "epochs": 35, "imgsz": 1280, "crops": 2},
]

REFINEMENT_EXPERIMENTS = [
    {"name": "highres_conservative", "freeze": 20, "lr": .0001, "epochs": 60, "imgsz": 1920, "crops": 0},
    {"name": "highres_mixed", "freeze": 10, "lr": .0001, "epochs": 45, "imgsz": 1920, "crops": 2},
    {"name": "highres_repeat", "freeze": 10, "lr": .0002, "epochs": 40, "imgsz": 1920, "crops": 0, "seed": 28},
]


def run_development(args):
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if args.output.exists():
        raise FileExistsError("Use a new experiment directory")
    args.output.mkdir(parents=True)
    items = load_exports(args.manifest)
    source_manifest = json.loads(args.manifest.read_text())
    if not source_manifest.get("development_only") or any(row["split"] not in {"train", "validation"} for row in items):
        raise ValueError("Development optimization requires the training manifest, not evaluation-only references")
    validation = [row for row in items if row["split"] == "validation"]
    configurations = [("full", 1280), ("full", 1920), ("tiles", 1280)]
    experiments = REFINEMENT_EXPERIMENTS if args.round == "refinement" else EXPERIMENTS
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "experiments": experiments,
                "manifest_sha256": digest(args.manifest), "script_sha256": digest(__file__),
                "selection": "top1 IoU >= 0.5 recall on match_01 development validation; unknown counts as miss",
                "inference_configurations": configurations, "test_data_loaded": False,
                "validation_is_independent_test": False, "primary_threshold_not_changed": True,
                "stop_rule": "complete declared comparisons; inspect remaining errors before deciding next round",
                "limitation": "Small development search cannot establish a global optimum or overall project accuracy"}
    save_json(args.output / "protocol.json", protocol)
    scores = []
    def compare(path, name):
        for mode, size in configurations:
            target = args.output / f"{name}_{mode}_{size}.json"
            score = evaluate_checkpoint(path, validation, target, mode, size, args.device, 2)
            scores.append({"name": name, "weights": str(path), "prediction_file": target.name, **score})
            save_json(args.output / "development_scores.json", scores)
            print(json.dumps({"name": name, "mode": mode, "imgsz": size,
                              "hits": score["metrics"]["all_frames"]["top1_iou_0_5_hits"]}), flush=True)
    compare(args.weights, "pretrained")
    compare(args.pilot, "pilot_best")
    if args.inference_only:
        return
    for settings in experiments:
        run = args.output / settings["name"]
        run.mkdir()
        data = make_training_dataset(items, run / "dataset", settings["crops"])
        model = YOLO(str(args.weights))
        model.train(data=str(data), epochs=settings["epochs"], imgsz=settings["imgsz"], batch=2,
                    device=args.device, workers=2, project=str(run), name="fit", exist_ok=False,
                    optimizer="AdamW", lr0=settings["lr"], lrf=.1, weight_decay=.0005,
                    freeze=settings["freeze"], warmup_epochs=2, patience=settings["epochs"],
                    seed=settings.get("seed", 27), deterministic=True, amp=False, cache=False, mosaic=0., mixup=0.,
                    copy_paste=0., scale=.2, translate=.1, fliplr=.5, flipud=0., close_mosaic=0,
                    plots=False, save=True, verbose=False)
        del model
        torch.cuda.empty_cache()
        for checkpoint in ("best", "last"):
            compare(run / "fit" / "weights" / f"{checkpoint}.pt", settings["name"] + "_" + checkpoint)
    best = max(scores, key=lambda s: (s["metrics"]["deduplicated_frames"]["top1_localization_recall_iou_0_5"],
                                     -s["seconds_including_model_load"]))
    save_json(args.output / "selected_candidate.json", {**best, "selection_uses_test": False,
              "promotion": "candidate_only_pending_frozen_module_evaluation_and_video_integration"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--round", choices=("initial", "refinement"), default="initial")
    run_development(parser.parse_args())


if __name__ == "__main__":
    main()
