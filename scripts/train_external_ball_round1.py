"""Fixed-epoch ball refit with an entire new match reserved for comparison."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_ball_detector import evaluate_checkpoint, load_exports
from scripts.train_ball_frame_detector import digest, save_json, summarize


TRAIN_MATCH = "lev_stu_20240817"
EVALUATION_MATCH = "ars_mci_20230806"
OLD_TRAIN_VIDEOS = {"core_01", "core_02", "core_03", "match_01"}
REGRESSION_VIDEOS = {"core_05", "core_06", "core_09"}


def partition_samples(old, new, regression):
    if {row["video_id"] for row in old} != OLD_TRAIN_VIDEOS:
        raise ValueError("Unexpected original development videos")
    if any(row["split"] not in {"train", "validation"} for row in old):
        raise ValueError("Evaluation-only data cannot enter original training")
    if {row["video_id"] for row in regression} != REGRESSION_VIDEOS:
        raise ValueError("Unexpected regression videos")
    if any(row["split"] != "test" for row in regression):
        raise ValueError("Regression references must remain evaluation-only")
    if {row["match_id"] for row in new} != {TRAIN_MATCH, EVALUATION_MATCH}:
        raise ValueError("New sources must be the two specified complete match groups")
    for row in new:
        expected = "train" if row["match_id"] == TRAIN_MATCH else "validation"
        if row["split"] != expected:
            raise ValueError("An entire new match must stay in one partition")
        if not row["event_id"].startswith(row["match_id"] + "_p"):
            raise ValueError("Event identity is not bound to its source match")
    train = [*old, *(r for r in new if r["match_id"] == TRAIN_MATCH)]
    held = [r for r in new if r["match_id"] == EVALUATION_MATCH]
    if not held or not any(r.get("match_id") == TRAIN_MATCH for r in train):
        raise ValueError("Both new match partitions must contain usable frames")
    if {r["image_sha256"] for r in train} & {r["image_sha256"] for r in held + regression}:
        raise ValueError("Identical source images cross training and evaluation")
    return train, held


def parameter_digests(model):
    return {name: {"shape": list(value.shape),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, value in model.named_parameters()}


class TrainingEvidence:
    def __init__(self, output, batch, device):
        self.output, self.batch, self.device = output, batch, device
        self.steps, self.max_delta = 0, 0.
        self.epochs = []
        self.started = time.perf_counter()

    def start(self, trainer):
        import torch
        expected = "mps" if self.device == "mps" else "cuda" if self.device != "cpu" else "cpu"
        if trainer.device.type != expected or trainer.batch_size != self.batch:
            raise RuntimeError("Training device or batch differs from the frozen protocol")
        self.before = parameter_digests(trainer.model)
        self.name, parameter = next((n, p) for n, p in trainer.model.named_parameters() if p.requires_grad and p.ndim > 1)
        if parameter.dtype != torch.float32 or parameter.device.type != expected:
            raise RuntimeError("Expected an FP32 trainable tensor on the requested device")
        initial = parameter.detach().cpu().clone()

        def step_finished(_optimizer, _args, _kwargs):
            current = parameter.detach().cpu()
            if not torch.isfinite(current).all():
                raise RuntimeError("Non-finite trained parameter")
            self.steps += 1
            self.max_delta = max(self.max_delta, float((current - initial).abs().max()))

        self.handle = trainer.optimizer.register_step_post_hook(step_finished)

    def batch_finished(self, trainer):
        if trainer.batch_size != self.batch or trainer._oom_retries:
            raise RuntimeError("Training batch changed or an OOM retry occurred")
        if not all(math.isfinite(float(value.detach().cpu())) for value in trainer.loss_items.values()):
            raise RuntimeError("Non-finite training loss")

    def saved(self, trainer):
        import torch
        if self.device == "mps":
            torch.mps.synchronize()
        self.epochs.append(trainer.epoch + 1)
        # Keep the preceding complete recovery copy until the new copy is written.
        temporary = self.output / "last_recoverable.pt.tmp"
        checkpoint = self.output / "last_recoverable.pt"
        shutil.copy2(trainer.last, temporary)
        temporary.replace(checkpoint)
        state = {"completed_epochs": self.epochs, "epoch": trainer.epoch + 1,
                 "planned_epochs": trainer.epochs, "actual_batch": trainer.batch_size,
                 "actual_device": str(trainer.device), "optimizer_steps": self.steps,
                 "tracked_parameter": self.name, "maximum_observed_parameter_delta": self.max_delta,
                 "losses": {key: float(value.detach().cpu()) for key, value in trainer.tloss.items()},
                 "elapsed_seconds": time.perf_counter() - self.started,
                 "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                 "checkpoint_contains_optimizer": True,
                 "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        if self.device == "mps":
            state["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        save_json(self.output / "training_progress.json", state)
        print(json.dumps({"saved_epoch": state["epoch"], "optimizer_steps": self.steps,
                          "losses": state["losses"], "elapsed_seconds": state["elapsed_seconds"]}), flush=True)


def add_event_metrics(prediction_path, samples):
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    identities = {row.get("annotation_id", row["image"].stem): row.get("event_id", row["video_id"])
                  for row in samples}
    grouped = defaultdict(list)
    for row in predictions["rows"]:
        row["event_id"] = identities[row["annotation_id"]]
        grouped[row["event_id"]].append(row)
    metrics = {event: {"all_frames": summarize(rows),
                       "deduplicated_frames": summarize([r for r in rows if r["deduplicated_primary"]])}
               for event, rows in grouped.items()}
    predictions["metrics"]["by_event"] = metrics
    predictions["metrics"]["macro_event_localization_recall_iou_0_5"] = sum(
        value["deduplicated_frames"]["top1_localization_recall_iou_0_5"] for value in metrics.values()) / len(metrics)
    save_json(prediction_path, predictions)
    return predictions["metrics"]


def make_dataset(train, destination):
    import yaml
    for kind in ("images", "labels"):
        (destination / kind / "train").mkdir(parents=True)
    copied = []
    seen = set()
    for row in train:
        name = row["image"].name
        if name in seen:
            raise ValueError("Training image filename collision")
        seen.add(name)
        image = destination / "images" / "train" / name
        label = destination / "labels" / "train" / (Path(name).stem + ".txt")
        shutil.copy2(row["image"], image)
        shutil.copy2(row["label"], label)
        copied.append({"image": str(image.relative_to(destination)),
                       "label": str(label.relative_to(destination)),
                       "image_sha256": digest(image), "label_sha256": digest(label),
                       "video_id": row["video_id"], "match_id": row.get("match_id"),
                       "event_id": row.get("event_id"), "source_split": row["split"]})
    # Ultralytics requires a val entry even for a fixed-epoch fit. It sees train only.
    path = destination / "dataset.yaml"
    path.write_text(yaml.safe_dump({"path": str(destination.resolve()),
                                  "train": "images/train", "val": "images/train",
                                  "names": {0: "ball"}}), encoding="utf-8")
    save_json(destination / "manifest.json", {"exports": copied, "validation_is_training_fit_only": True})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--regression-manifest", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--evaluation-batch", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch, args.evaluation_batch) < 1 or args.workers < 0:
        parser.error("Batch sizes must be positive and workers nonnegative")
    args.output = args.output.resolve()
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"sync": False})
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    if args.output.exists():
        raise FileExistsError("Use a new experiment directory")
    old = load_exports(args.old_manifest)
    new = load_exports(args.new_manifest)
    regression = load_exports(args.regression_manifest)
    source = json.loads(args.new_manifest.read_text(encoding="utf-8"))
    if source.get("clean_source_pixels_verified") is not True:
        raise ValueError("Review overlays must be excluded and clean source alignment verified")
    train, held = partition_samples(old, new, regression)
    args.output.mkdir(parents=True)
    data = make_dataset(train, args.output / "dataset")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.65, 0)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(.65)
    versions = {"python": sys.version, "torch": torch.__version__,
                "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
                "cuda": torch.version.cuda,
                "gpu": "Apple Silicon MPS" if args.device == "mps" else
                       torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    protocol = {
        "schema": "external_ball_fixed_round/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs": 30, "image_size": 1920, "batch": args.batch, "freeze": 10,
        "workers": args.workers, "nominal_accumulated_batch": 64, "evaluation_batch": args.evaluation_batch,
        "optimizer": "AdamW", "learning_rate": .0002, "seed": 27,
        "checkpoint_selection": "fixed_last_epoch_before_new_match_evaluation",
        "training_frames": len(train), "training_counts": dict(Counter(r["video_id"] for r in train)),
        "new_match_evaluation_frames": len(held), "new_match_evaluation": EVALUATION_MATCH,
        "regression_frames": len(regression), "evaluation_frames_used_for_training": False,
        "old_match01_remains_training": True, "old_regression_results_previously_seen": True,
        "new_match_used_for_checkpoint_selection": False, "clean_source_pixels_verified": True,
        "manifests": {name: {"path": str(path), "sha256": digest(path)} for name, path in
                      [("old", args.old_manifest), ("new", args.new_manifest), ("regression", args.regression_manifest)]},
        "initial_weights_sha256": digest(args.pretrained), "previous_weights_sha256": digest(args.previous),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
                          ROOT / "scripts/optimize_ball_detector.py", ROOT / "scripts/train_ball_frame_detector.py",
                          ROOT / "src/ball_specialist.py"]},
        "runtime": versions, "device": args.device,
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "prior_remote_round_completion": "unverified_after_epoch_19_connection_loss",
        "evaluation_modes": ["full", "tiles"], "evaluation_image_size": 1280,
        "primary_metric": "top-ranked ball box IoU >= 0.5 on annotated positive frames, missing is a miss",
        "confidence": .05, "nms_iou": .6, "default_pipeline_changed": False,
        "limitations": ["Assisted localization references, not independent consensus.",
                        "Frames from one event are correlated; report per-event counts as well.",
                        "No exhaustive negative-frame annotation; this does not measure false-positive precision.",
                        "New-match comparison concerns this detector only, not all assignment outputs.",
                        "Existing core05/06/09 comparisons are development retests, not fresh final tests."]}
    save_json(args.output / "protocol.json", protocol)
    started = time.perf_counter()
    model = YOLO(str(args.pretrained))
    evidence = TrainingEvidence(args.output, args.batch, args.device)
    model.add_callback("on_train_start", evidence.start)
    model.add_callback("on_train_batch_end", evidence.batch_finished)
    model.add_callback("on_model_save", evidence.saved)
    train_started = time.perf_counter()
    model.train(data=str(data), epochs=30, imgsz=1920, batch=args.batch, device=args.device,
                workers=args.workers, project=str(args.output), name="fit", exist_ok=False,
                optimizer="AdamW", lr0=.0002, lrf=.1, weight_decay=.0005, nbs=64,
                freeze=10, warmup_epochs=2, patience=30, seed=27, deterministic=True,
                amp=False, cache=False, mosaic=0., mixup=0., copy_paste=0., scale=.2,
                translate=.1, fliplr=.5, flipud=0., close_mosaic=0, plots=False,
                save=True, save_period=5, verbose=False, val=False, compile=False)
    training_seconds = time.perf_counter() - train_started
    evidence.handle.remove()
    if evidence.epochs != list(range(1, 31)) or evidence.steps < 1 or evidence.max_delta <= 0:
        raise RuntimeError("Fixed 30-epoch training and actual parameter updates were not verified")
    before, after = evidence.before, parameter_digests(model.trainer.model)
    weights = Path(model.trainer.last)
    candidate = YOLO(str(weights))
    if candidate.names != {0: "ball"}:
        raise ValueError("Expected the trained single ball class")
    shared = [name for name in before if name in after and before[name]["shape"] == after[name]["shape"]]
    changed = [name for name in shared if before[name]["sha256"] != after[name]["sha256"]]
    if not changed:
        raise RuntimeError("No shared model parameters changed during training")
    receipt = {"weights": str(weights), "weights_sha256": digest(weights),
               "selection_uses_test": False, "new_match_evaluation_seen_before_selection": False,
               "test_results_seen_before_training": True, "selection_protocol": "fixed_epoch_30",
               "same_shape_parameter_tensors": len(shared), "changed_parameter_tensors": len(changed),
               "changed_parameter_names": changed, "training_seconds": training_seconds,
               "parameter_comparison": "same FP32 model after head setup and after training",
               "completed_epochs": evidence.epochs, "optimizer_steps": evidence.steps,
               "tracked_parameter": evidence.name, "maximum_observed_parameter_delta": evidence.max_delta,
               "promotion": "experimental_only_pending_full_video_comparison"}
    save_json(args.output / "selected_candidate.json", receipt)
    del model, candidate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.device == "mps":
        torch.mps.empty_cache()
    scores = {}
    for group, samples in [("arsenal_new_match", held), ("core_development_retest", regression)]:
        scores[group] = {}
        for name, path in [("pretrained", args.pretrained), ("previous_refit", args.previous), ("new_round1", weights)]:
            scores[group][name] = {}
            for mode in ("full", "tiles"):
                target = args.output / f"{group}_{name}_{mode}.json"
                score = evaluate_checkpoint(path, samples, target, mode, 1280, args.device, args.evaluation_batch)
                score["metrics"] = add_event_metrics(target, samples)
                scores[group][name][mode] = score
                metric = score["metrics"]["deduplicated_frames"]
                print(json.dumps({"group": group, "model": name, "mode": mode,
                                  "hits": metric["top1_iou_0_5_hits"], "frames": metric["annotated_frames"]}), flush=True)
                save_json(args.output / "evaluation_progress.json", scores)
                gc.collect()
                if args.device == "mps":
                    torch.mps.empty_cache()
    save_json(args.output / "report.json", {"protocol": protocol, "candidate": receipt, "results": scores,
              "elapsed_seconds": time.perf_counter() - started, "training_completed": True,
              "detector_weights_updated": True, "default_pipeline_changed": False,
              "overall_project_accuracy_measured": False})
    print(json.dumps({"completed": True, "training_seconds": training_seconds,
                      "weights_sha256": receipt["weights_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
