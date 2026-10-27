"""Small, match-separated ball-box development experiment, not a final test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def score_frame(reference, candidates):
    ordered = sorted(candidates, key=lambda row: row["confidence"], reverse=True)
    overlaps = [box_iou(reference, row["bbox"]) for row in ordered]
    return {"prediction_available": bool(ordered), "candidate_count": len(ordered),
            "top1_iou": overlaps[0] if overlaps else 0.,
            "any_candidate_iou": max(overlaps, default=0.)}


def summarize(rows):
    count = len(rows)
    if not count:
        raise ValueError("Cannot evaluate an empty split")
    return {"annotated_frames": count,
            "prediction_available_frames": sum(r["prediction_available"] for r in rows),
            "top1_iou_0_3_hits": sum(r["top1_iou"] >= .3 for r in rows),
            "top1_iou_0_5_hits": sum(r["top1_iou"] >= .5 for r in rows),
            "top1_localization_recall_iou_0_3": sum(r["top1_iou"] >= .3 for r in rows) / count,
            "top1_localization_recall_iou_0_5": sum(r["top1_iou"] >= .5 for r in rows) / count,
            "any_candidate_recall_iou_0_5": sum(r["any_candidate_iou"] >= .5 for r in rows) / count,
            "scope": "selected positive frames with assisted reference boxes; not event accuracy"}


def dataset_items(yaml_path):
    import yaml
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    items = {}
    for split in ("train", "val"):
        folder = root / config[split]
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError(f"Empty image directory: {folder}")
        items[split] = []
        for image in images:
            parts = list(image.parts)
            parts[parts.index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            values = label.read_text().strip().splitlines()
            if len(values) != 1:
                raise ValueError("This pilot requires one reviewed target per selected frame")
            fields = values[0].split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError("Expected one class-0 ball box")
            normalized = list(map(float, fields[1:]))
            if not all(0 <= v <= 1 for v in normalized) or min(normalized[2:]) <= 0:
                raise ValueError("Invalid normalized box")
            allowed = ("core_01", "core_02", "core_03") if split == "train" else ("match_01",)
            if not any(image.stem.startswith(prefix + "_") for prefix in allowed):
                raise ValueError(f"Unexpected match group in {split}: {image.name}")
            items[split].append({"image": image, "label": label, "normalized": normalized})
    if {digest(r["image"]) for r in items["train"]} & {digest(r["image"]) for r in items["val"]}:
        raise ValueError("Identical images cross the match split")
    return items


def validate_exports(manifest_path, items):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("development_only") or manifest.get("independent_ground_truth"):
        raise ValueError("Unexpected annotation provenance")
    expected = {str((manifest_path.parent / row["image"]).resolve()): row for row in manifest["exports"]}
    actual = {str(row["image"].resolve()) for rows in items.values() for row in rows}
    if len(expected) != len(manifest["exports"]) or set(expected) != actual:
        raise ValueError("Exported image inventory differs from the bound manifest")
    for split, rows in items.items():
        for row in rows:
            reference = expected[str(row["image"].resolve())]
            if reference["split"] != ("validation" if split == "val" else "train"):
                raise ValueError("Manifest split mismatch")
            if digest(row["image"]) != reference["image_sha256"] or digest(row["label"]) != reference["label_sha256"]:
                raise ValueError("Image or label checksum mismatch")


def evaluate(model, items, args, name, class_id):
    import cv2
    import torch
    records = {}
    started = time.perf_counter()
    for split, samples in items.items():
        rows = []
        for offset in range(0, len(samples), args.batch):
            batch = samples[offset:offset + args.batch]
            images = [cv2.imread(str(row["image"])) for row in batch]
            if any(frame is None for frame in images):
                raise ValueError("Failed to read exported source frame")
            results = model.predict(images, imgsz=args.imgsz, conf=.05, iou=.6, max_det=20,
                                    classes=[class_id], device=args.device, verbose=False)
            for sample, frame, result in zip(batch, images, results):
                height, width = frame.shape[:2]
                x, y, w, h = sample["normalized"]
                reference = [(x - w / 2) * width, (y - h / 2) * height,
                             (x + w / 2) * width, (y + h / 2) * height]
                candidates = [{"bbox": box.tolist(), "confidence": float(conf)}
                              for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
                rows.append({"image": sample["image"].name, "reference_bbox": reference,
                             "candidates": candidates, **score_frame(reference, candidates)})
        records[split] = {"metrics": summarize(rows), "rows": rows}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    records["elapsed_seconds"] = time.perf_counter() - started
    save_json(args.output / f"{name}_predictions.json", records)
    return {split: records[split]["metrics"] for split in items}


def run(args):
    import cv2
    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    if args.output.exists():
        raise FileExistsError("Use a new output directory; previous experiments are immutable")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    runtime_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    runtime_config["path"] = str(args.data.parent.resolve())
    runtime_yaml = args.output / "dataset_runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump(runtime_config), encoding="utf-8")
    items = dataset_items(runtime_yaml)
    validate_exports(args.manifest, items)
    protocol = {"schema_version": "ball-frame-detector-development-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "train_videos": ["core_01", "core_02", "core_03"], "validation_videos": ["match_01"],
                "validation_is_final_test": False, "source_labels": "human localization plus automatic bright-spot refinement",
                "weights_sha256": digest(args.weights), "trainer_sha256": digest(__file__),
                "dataset_yaml_sha256": digest(args.data), "annotations_manifest_sha256": digest(args.manifest),
                "runtime_dataset_yaml_sha256": digest(runtime_yaml),
                "epochs": args.epochs, "image_size": args.imgsz, "batch": args.batch, "seed": 27,
                "device": args.device, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "promotion": "experimental only; no deployed config changes",
                "evaluation_confidence": .05, "evaluation_nms_iou": .6,
                "images": {split: [{"image": r["image"].name, "image_sha256": digest(r["image"]),
                                    "label_sha256": digest(r["label"])} for r in rows] for split, rows in items.items()}}
    save_json(args.output / "protocol.json", protocol)
    started = time.perf_counter()
    baseline_model = YOLO(str(args.weights))
    ball_class = next(i for i, name in baseline_model.names.items() if name == "sports ball")
    baseline = evaluate(baseline_model, items, args, "pretrained", ball_class)
    print("Pretrained development metrics", json.dumps(baseline), flush=True)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = YOLO(str(args.weights))
    model.train(data=str(runtime_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=2, project=str(args.output), name="fit", exist_ok=False,
                optimizer="AdamW", lr0=.0005, lrf=.1, weight_decay=.0005,
                freeze=10, warmup_epochs=2, patience=args.epochs, seed=27, deterministic=True,
                amp=False, cache=False, mosaic=0., mixup=0., copy_paste=0.,
                scale=.2, translate=.1, fliplr=.5, flipud=0., close_mosaic=0,
                plots=False, save=True, verbose=False)
    scores = {}
    for checkpoint in ("last", "best"):
        path = args.output / "fit" / "weights" / f"{checkpoint}.pt"
        candidate = YOLO(str(path))
        if candidate.names != {0: "ball"}:
            raise ValueError("Unexpected trained class mapping")
        scores[checkpoint] = {"weights_sha256": digest(path), "scores": evaluate(candidate, items, args, checkpoint, 0)}
        del candidate
    report = {"protocol": protocol, "pretrained": baseline, "trained": scores,
              "elapsed_seconds": time.perf_counter() - started, "detector_weights_updated": True,
              "default_pipeline_changed": False, "independent_test_evaluated": False,
              "limitations": ["Four events from two match groups; repeated frames are correlated.",
                              "Validation is one previously inspected match and selects the best checkpoint.",
                              "Assisted boxes are training candidates, not independent human consensus.",
                              "Only selected clear/blur positive frames are evaluated; occlusions are not negative labels.",
                              "Localization recall is not outcome, trajectory, event or final-position accuracy.",
                              "The new model maps ball to class 0; deployed detector expects COCO class 32."]}
    save_json(args.output / "report.json", report)
    print(json.dumps({"pretrained": baseline, "trained": scores, "seconds": report["elapsed_seconds"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
