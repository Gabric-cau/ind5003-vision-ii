"""Measure actual MPS training on a small, source-bound training-only subset."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_ball_frame_detector import digest, save_json
from scripts.train_external_ball_round1 import OLD_TRAIN_VIDEOS, TRAIN_MATCH, make_dataset
from src.ball_specialist import predict_images


def training_subset(old_path, new_path, count):
    groups = defaultdict(list)
    for path, is_new in ((old_path, False), (new_path, True)):
        source = json.loads(path.read_text(encoding="utf-8"))
        if is_new and source.get("clean_source_pixels_verified") is not True:
            raise ValueError("Clean source verification is required")
        for row in source["exports"]:
            row = {**row, "video_id": row.get("video_id", Path(row["image"]).stem.rsplit("_f", 1)[0])}
            if is_new:
                if row.get("match_id") != TRAIN_MATCH or row["split"] != "train":
                    continue
            elif row["video_id"] not in OLD_TRAIN_VIDEOS or row["split"] not in {"train", "validation"}:
                raise ValueError("Unexpected original training membership")
            event = row.get("event_id", row["video_id"])
            groups[event].append({**row, "image": path.parent / row["image"],
                                  "label": path.parent / row["label"]})
    rng = random.Random(27)
    for rows in groups.values():
        rows.sort(key=lambda row: str(row["image"]))
        rng.shuffle(rows)
    chosen = []
    while len(chosen) < count:
        added = False
        for event in sorted(groups):
            if groups[event] and len(chosen) < count:
                chosen.append(groups[event].pop())
                added = True
        if not added:
            raise ValueError("Requested more training samples than available")
    for row in chosen:
        if digest(row["image"]) != row["image_sha256"] or digest(row["label"]) != row["label_sha256"]:
            raise ValueError("Source image or annotation changed")
        values = row["label"].read_text().split()
        if len(values) != 5 or values[0] != "0":
            raise ValueError("Expected one ball reference")
        box = list(map(float, values[1:]))
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in box) or min(box[2:]) <= 0:
            raise ValueError("Invalid ball reference")
    return chosen


def verify_checkpoint_inference(weights, image):
    import cv2
    import torch
    from ultralytics import YOLO
    reloaded = YOLO(str(weights))
    if reloaded.names != {0: "ball"}:
        raise RuntimeError("Reloaded checkpoint has unexpected classes")
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError("Training image did not decode")
    records = {}
    for mode in ("full", "tiles"):
        torch.mps.synchronize()
        started = time.perf_counter()
        predictions = predict_images(reloaded, [frame], mode, 1280, "mps", 1)
        torch.mps.synchronize()
        # Ultralytics retains a separate device copy inside the predictor backend.
        if reloaded.predictor.device.type != "mps":
            raise RuntimeError("Reloaded checkpoint inference did not use MPS")
        records[mode] = {"seconds_including_mode_first_call": time.perf_counter() - started,
                         "candidate_count": len(predictions[0]), "completed": True,
                         "actual_device": str(reloaded.predictor.device)}
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, default=ROOT / "outputs/ball_frame_annotations_2026-10-27/manifest.json")
    parser.add_argument("--new-manifest", type=Path, default=ROOT / "outputs/external_ball_frames_2026-10-28/manifest.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs/core_contact_training_2026-10-26/models/yolo11m.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1920)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a fresh probe directory")
    args.output.mkdir(parents=True)

    import cv2
    import psutil
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"sync": False})
    if not torch.backends.mps.is_available():
        raise RuntimeError("This probe requires an available MPS device")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.mps.set_per_process_memory_fraction(.65)
    selected = training_subset(args.old_manifest, args.new_manifest, args.frames)
    data = make_dataset(selected, args.output / "dataset")
    report = {
        "schema": "mps_ball_training_probe/v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "installed_packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "requested": {"frames": args.frames, "epochs": args.epochs, "batch": args.batch,
                      "imgsz": args.imgsz, "freeze": 10, "amp": False, "workers": 0, "nbs": 64},
        "source_weights_sha256": digest(args.weights), "probe_script_sha256": digest(Path(__file__)),
        "selected_images": [row["image"].name for row in selected],
        "heldout_images_or_labels_loaded": False, "accuracy_evaluated": False,
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "mps_memory_fraction": .65, "mps_recommended_memory_bytes": torch.mps.recommended_max_memory(),
        "system_swap_before_bytes": psutil.swap_memory().used,
        "optimizer_steps": 0, "maximum_observed_parameter_delta": 0.,
        "batches": [], "epochs": [], "inference": {}, "completed": False,
    }
    save_json(args.output / "protocol.json", report)
    state = {}

    def train_start(trainer):
        if trainer.device.type != "mps":
            raise RuntimeError(f"Unexpected training device: {trainer.device}")
        name, parameter = next((n, p) for n, p in trainer.model.named_parameters() if p.requires_grad and p.ndim > 1)
        if parameter.device.type != "mps" or parameter.dtype != torch.float32:
            raise RuntimeError("Expected FP32 trainable parameters on MPS")
        initial = parameter.detach().cpu().clone()
        report.update(actual_device=str(trainer.device), tracked_parameter=name,
                      actual_batch=trainer.batch_size,
                      actual_warmup_epochs=min(trainer.args.warmup_epochs, max(trainer.epochs - 1, 0)))

        def step_finished(optimizer, _args, _kwargs):
            torch.mps.synchronize()
            current = parameter.detach().cpu()
            if not torch.isfinite(current).all():
                raise RuntimeError("Non-finite trained parameter")
            report["optimizer_steps"] += 1
            report["maximum_observed_parameter_delta"] = max(
                report["maximum_observed_parameter_delta"], float((current - initial).abs().max()))

        state["step_hook"] = trainer.optimizer.register_step_post_hook(step_finished)

    def epoch_start(trainer):
        torch.mps.synchronize()
        state["epoch_start"] = time.perf_counter()

    def batch_start(trainer):
        torch.mps.synchronize()
        state["batch_start"] = time.perf_counter()

    def batch_end(trainer):
        torch.mps.synchronize()
        seconds = time.perf_counter() - state["batch_start"]
        losses = {key: float(value.detach().cpu()) for key, value in trainer.loss_items.items()}
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError("Non-finite training loss")
        entry = {"epoch": trainer.epoch + 1, "seconds": seconds, "losses": losses,
                 "actual_batch": trainer.batch_size, "accumulate": trainer.accumulate,
                 "oom_retries": trainer._oom_retries,
                 "mps_tensor_bytes": torch.mps.current_allocated_memory(),
                 "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                 "process_rss_bytes": psutil.Process().memory_info().rss}
        report["batches"].append(entry)
        save_json(args.output / "progress.json", report)

    def epoch_end(trainer):
        torch.mps.synchronize()
        entry = {"epoch": trainer.epoch + 1, "training_seconds": time.perf_counter() - state["epoch_start"]}
        report["epochs"].append(entry)
        print(json.dumps({"epoch_done": entry, "optimizer_steps": report["optimizer_steps"],
                          "parameter_delta": report["maximum_observed_parameter_delta"]}), flush=True)

    started = time.perf_counter()
    try:
        model = YOLO(str(args.weights))
        for event, callback in [("on_train_start", train_start), ("on_train_epoch_start", epoch_start),
                                ("on_train_batch_start", batch_start), ("on_train_batch_end", batch_end),
                                ("on_train_epoch_end", epoch_end)]:
            model.add_callback(event, callback)
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="mps",
                    workers=0, project=str(args.output), name="fit", exist_ok=False,
                    optimizer="AdamW", lr0=.0002, lrf=.1, weight_decay=.0005, nbs=64,
                    freeze=10, warmup_epochs=2, patience=30, seed=27, deterministic=True,
                    amp=False, cache=False, mosaic=0., mixup=0., copy_paste=0., scale=.2,
                    translate=.1, fliplr=.5, flipud=0., close_mosaic=0, plots=False,
                    save=True, verbose=False, val=False, compile=False)
        torch.mps.synchronize()
        report["fit_wall_seconds_including_setup_save_and_train_subset_validation"] = time.perf_counter() - started
        state["step_hook"].remove()
        weights = Path(model.trainer.last)
        report["weights_path"] = str(weights)
        report["weights_sha256"] = digest(weights)
        if report["maximum_observed_parameter_delta"] <= 0 or report["optimizer_steps"] < 1:
            raise RuntimeError("No verified non-bias optimizer update")
        if len(report["epochs"]) != args.epochs:
            raise RuntimeError("Requested epochs did not complete")
        if not all(row["actual_batch"] == args.batch and row["oom_retries"] == 0 for row in report["batches"]):
            raise RuntimeError("Requested batch configuration did not run without OOM retry")
        del model
        gc.collect()
        torch.mps.empty_cache()
        report["inference"] = verify_checkpoint_inference(weights, selected[0]["image"])
        report["checkpoint_reloaded_and_inference_passed"] = True
        report["completed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["system_swap_after_bytes"] = psutil.swap_memory().used
        batches = report["batches"]
        if batches:
            report["max_sampled_mps_driver_bytes"] = max(row["mps_driver_bytes"] for row in batches)
            report["max_sampled_mps_tensor_bytes"] = max(row["mps_tensor_bytes"] for row in batches)
            warmed = [row["seconds"] for row in batches if row["epoch"] > 1]
            if warmed:
                report["warm_batch_seconds_median"] = statistics.median(warmed)
                report["warm_batch_seconds_min_max"] = [min(warmed), max(warmed)]
            report["requested_batch_preserved"] = all(row["actual_batch"] == args.batch for row in batches)
            report["oom_retry_observed"] = any(row["oom_retries"] for row in batches)
        if len(report["epochs"]) > 1:
            steady = report["epochs"][-1]["training_seconds"] / math.ceil(args.frames / report["actual_batch"])
            report["estimated_30_epoch_442_frame_training_seconds"] = steady * math.ceil(442 / report["actual_batch"]) * 30
            report["estimate_excludes_final_evaluation_and_is_not_a_full_run_measurement"] = True
        save_json(args.output / "report.json", report)
        print(json.dumps({key: report.get(key) for key in ["completed", "elapsed_seconds", "optimizer_steps",
                          "maximum_observed_parameter_delta", "warm_batch_seconds_median", "max_sampled_mps_driver_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
