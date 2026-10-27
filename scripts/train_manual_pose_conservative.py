"""Predeclared second sparse-head experiment with all BN statistics frozen."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

try:
    from scripts.optimize_manual_pose import (
        MODES, TRAIN_VIDEOS, VALIDATION_VIDEOS, digest, evaluate, load_images,
        make_dataset, predict, save, train_sparse,
    )
except ModuleNotFoundError:
    from optimize_manual_pose import (
        MODES, TRAIN_VIDEOS, VALIDATION_VIDEOS, digest, evaluate, load_images,
        make_dataset, predict, save, train_sparse,
    )


def run(args):
    import cv2
    import torch
    from ultralytics import YOLO
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / "train_manual_pose_conservative.py")
    shutil.copy2(Path(__file__).with_name("optimize_manual_pose.py"), args.output / "optimize_manual_pose.py")
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    samples, manifest = load_images(args.manifest)
    if digest(args.weights) != manifest["model_binding"]["declared_sha256"]:
        raise ValueError("Baseline weight hash mismatch")
    source_protocol = json.loads((args.baseline_run / "protocol.json").read_text())
    if source_protocol["manifest_sha256"] != digest(args.manifest) or source_protocol["weights_sha256"] != digest(args.weights):
        raise ValueError("Previous baseline is not bound to these images and weights")
    prediction_path = args.baseline_run / "baseline_full_1280_predictions.json"
    baseline = json.loads(prediction_path.read_text())
    target_path = args.manifest.parent / "frame_targets.json"
    if digest(target_path) != manifest["artifact_sha256"]["frame_targets.json"]:
        raise ValueError("Manual target hash mismatch")
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
                "manifest_sha256": digest(args.manifest), "weights_sha256": digest(args.weights),
                "baseline_predictions_sha256": digest(prediction_path),
                "train_videos": sorted(TRAIN_VIDEOS), "validation_videos": sorted(VALIDATION_VIDEOS),
                "reason": "first fit improved trained videos but validation regressed; cv4 BN statistics had largest parameter-state changes",
                "change": "freeze every BatchNorm running statistic and reduce LR from1e-4 to2e-5",
                "epochs": 30, "learning_rate": .00002, "warmup_bias_lr": .00002,
                "trainable_parameters": "cv4 only", "all_batchnorm_statistics_frozen": True,
                "box_gain": 0., "cls_gain": 0., "dfl_gain": 0., "kobj_gain": 0.,
                "coordinate_loss": "only manually visible COCO17 joints; unreviewed/occluded v0",
                "inference_configurations": MODES,
                "primary_metric": "within10px, all136visiblepoints; every missingprediction fails",
                "checkpoint_selection": "fixed last epoch; no native sparse-GT AP selection",
                "independent_test": False, "default_pipeline_changed": False}
    save(args.output / "protocol.json", protocol)
    targets = json.loads(target_path.read_text())["frames"]
    data = make_dataset(samples, targets, baseline, args.output / "dataset")
    checkpoint = train_sparse(args.weights, data, args.output, args.device, 30, learning_rate=.00002, freeze_all_bn=True)
    model = YOLO(str(checkpoint))
    comparisons = {}
    for mode, size in MODES:
        name = f"trained_last_{mode}_{size}"
        predictions = predict(model, samples, args.output / f"{name}_predictions.json", mode, size, args.device)
        score = evaluate(predictions, targets)
        save(args.output / f"{name}_evaluation.json", score)
        comparisons[name] = {key: value for key, value in score.items() if key not in {"points", "associations"}}
        print(json.dumps({"name": name, "all": score["all"], "training": score["training_videos"], "validation": score["validation_video"]}), flush=True)
    save(args.output / "comparison.json", comparisons)
    save(args.output / "completion.json", {"checkpoint": str(checkpoint), "sha256": digest(checkpoint),
                                          "default_pipeline_changed": False, "independent_test_evaluated": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    run(parser.parse_args())
