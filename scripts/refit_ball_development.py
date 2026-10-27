"""Fixed-epoch refit of the four development events, without evaluation videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_ball_detector import load_exports
from scripts.train_ball_frame_detector import digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    import cv2
    import torch
    import yaml
    from ultralytics import YOLO
    items = load_exports(args.manifest)
    if {row["split"] for row in items} != {"train", "validation"}:
        raise ValueError("Refit requires the original development manifest")
    if {row["video_id"] for row in items} != {"core_01", "core_02", "core_03", "match_01"}:
        raise ValueError("Evaluation groups cannot be added to this refit")
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    dataset = args.output / "dataset"
    images, labels = dataset / "images" / "train", dataset / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for row in items:
        shutil.copy2(row["image"], images / row["image"].name)
        shutil.copy2(row["label"], labels / row["label"].name)
    config = {"path": str(dataset.resolve()), "train": "images/train", "val": "images/train", "names": {0: "ball"}}
    data_path = dataset / "dataset.yaml"
    data_path.write_text(yaml.safe_dump(config))
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": digest(args.manifest), "script_sha256": digest(__file__),
                "initial_weights_sha256": digest(args.weights), "epochs": 30, "imgsz": 1920,
                "freeze": 10, "learning_rate": .0002, "seed": 27, "checkpoint_selection": "fixed_last_epoch",
                "training_videos": ["core_01", "core_02", "core_03", "match_01"], "training_frames": len(items),
                "previous_validation_moved_to_training": True, "evaluation_frames_used_for_training": False,
                "test_results_seen_before_training": True,
                "interpretation": "follow-up development refit after prior evaluation; repeated evaluation is not a fresh test",
                "internal_validation_is_training_fit_only": True, "default_pipeline_changed": False}
    save_json(args.output / "protocol.json", protocol)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model = YOLO(str(args.weights))
    model.train(data=str(data_path), epochs=30, imgsz=1920, batch=2, device=args.device,
                workers=2, project=str(args.output), name="fit", exist_ok=False,
                optimizer="AdamW", lr0=.0002, lrf=.1, weight_decay=.0005,
                freeze=10, warmup_epochs=2, patience=30, seed=27, deterministic=True,
                amp=False, cache=False, mosaic=0., mixup=0., copy_paste=0., scale=.2,
                translate=.1, fliplr=.5, flipud=0., close_mosaic=0, plots=False,
                save=True, verbose=False, val=False)
    weights = args.output / "fit" / "weights" / "last.pt"
    save_json(args.output / "selected_candidate.json", {
        "name": "four_event_fixed_epoch_refit", "weights": str(weights), "weights_sha256": digest(weights),
        "mode": "tiles", "image_size": 1280, "selection_uses_test": False,
        "test_results_seen_before_training": True, "selection_protocol": protocol,
        "promotion": "experimental_development_candidate_only"})


if __name__ == "__main__":
    main()
