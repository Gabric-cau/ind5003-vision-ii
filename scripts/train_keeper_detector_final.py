"""Fixed-budget active-keeper detector comparison with isolated photometric augmentation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.import_manual_pose_joints import save_json, sha256
from scripts.prepare_keeper_detector_final import box_target, validate_partition
from scripts.train_ball_controlled_round2 import ScheduledSampler, sequence_digest
from scripts.train_external_ball_round1 import TrainingEvidence, parameter_digests

EPOCHS = 12
SEED = 29
IMAGE_SIZE = 1280
CONFIDENCE = .10
ARMS = ("control", "photometric")
BASE_AUGMENTATION = {"mosaic": 0., "mixup": 0., "cutmix": 0., "copy_paste": 0.,
    "degrees": 0., "shear": 0., "perspective": 0., "scale": .2, "translate": .1,
    "hsv_h": .015, "hsv_s": .4, "hsv_v": .25, "fliplr": .5, "flipud": 0.,
    "close_mosaic": 0, "augmentations": []}


def schedules_for(names, epochs=EPOCHS):
    names = sorted(names)
    if len(names) != len(set(names)) or not names:
        raise ValueError("Unique nonempty image names required")
    rng = random.Random(SEED)
    schedules = []
    for _ in range(epochs):
        order = names.copy()
        rng.shuffle(order)
        schedules.append(order)
    return schedules


def contrast_brightness(image, contrast, brightness):
    if not .85 <= contrast <= 1.15 or not -.04 <= brightness <= .04:
        raise ValueError("Photometric strength exceeds the frozen range")
    return ((image - .5) * contrast + .5 + brightness).clamp(0, 1)


def optimizer_budget(samples=835, epochs=EPOCHS):
    import numpy as np
    warmup = round(min(2, max(epochs - 1, 0)) * samples)
    steps, last, accumulate, budgets = 0, -1, 16, []
    for epoch in range(epochs):
        for batch in range(samples):
            index = epoch * samples + batch
            if index < warmup:
                accumulate = max(1, int(np.interp(index, [0, warmup], [1, 16]).round()))
            if index - last >= accumulate:
                steps += 1
                last = index
        budgets.append(steps)
    return budgets


class PositiveKeeperTransforms:
    """Keep the known target if a random crop would remove its supervision."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback

    def __call__(self, labels):
        original = copy.deepcopy(labels)
        result = self.primary(labels)
        fallback = len(result["cls"]) == 0
        if fallback:
            result = self.fallback(original)
        if len(result["cls"]) != 1 or len(result["bboxes"]) != 1:
            raise ValueError("Every keeper training frame must retain exactly one target")
        result["keeper_aug_fallback"] = fallback
        return result


def protect_positive_targets(dataset):
    from ultralytics.data.augment import Compose, Format, LetterBox
    fallback = Compose([LetterBox(new_shape=(IMAGE_SIZE, IMAGE_SIZE), scaleup=True),
                        Format(bbox_format="xywh", normalize=True, batch_idx=True)])
    dataset.transforms = PositiveKeeperTransforms(dataset.transforms, fallback)
    return dataset


def preflight_dataset(data_root):
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(SEED, deterministic=True)
    hyp = get_cfg(overrides={**BASE_AUGMENTATION, "imgsz": IMAGE_SIZE})
    dataset = protect_positive_targets(YOLODataset(img_path=str(data_root / "images/train"),
        imgsz=IMAGE_SIZE, batch_size=1, augment=True, hyp=hyp, rect=False, cache=False,
        data={"names": {0: "active_keeper"}}, task="detect"))
    counts, examples = Counter(), []
    for index in range(len(dataset)):
        sample = dataset[index]
        boxes = sample["bboxes"]
        if (not torch.isfinite(boxes).all() or not (boxes[:, 2:] > 0).all()
                or not (boxes >= 0).all() or not (boxes <= 1).all() or sample["cls"].tolist() != [[0.]]):
            raise ValueError("Transformed keeper target is invalid")
        counts["frames"] += 1
        counts["targets"] += len(boxes)
        counts["fallback_frames"] += int(sample["keeper_aug_fallback"])
        if len(examples) < 8:
            examples.append({"image": Path(sample["im_file"]).name,
                "normalized_xywh": boxes.tolist(), "fallback": sample["keeper_aug_fallback"]})
    return {"counts": dict(counts), "examples": examples,
            "native_geometry_transforms_exercised": True, "photometric_targets_unchanged": True}


def scheduled_trainer(schedules, photometric):
    import torch
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.models.yolo.detect import DetectionTrainer

    class Trainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            self.observed_files = defaultdict(list)
            self.photo_rng = random.Random(SEED)
            self.photo_receipts = []
            self.photo_batches = 0
            self.target_counts = defaultdict(Counter)
            super().__init__(*args, **kwargs)

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank != -1 or batch_size != 1 or self.args.workers != 0:
                raise ValueError("The frozen schedule requires one device, batch 1, workers 0")
            dataset = protect_positive_targets(self.build_dataset(dataset_path, mode, batch_size))
            self.frozen_sampler = ScheduledSampler([Path(path).name for path in dataset.im_files], schedules)
            return InfiniteDataLoader(dataset, batch_size=1, num_workers=0, sampler=self.frozen_sampler,
                shuffle=False, collate_fn=dataset.collate_fn, pin_memory=False, drop_last=False,
                generator=torch.Generator().manual_seed(SEED))

        def preprocess_batch(self, batch):
            self.observed_files[self.epoch].extend(Path(p).name for p in batch["im_file"])
            if len(batch["cls"]) != 1 or batch["cls"].tolist() != [[0.]]:
                raise RuntimeError("A positive training frame lost its active-keeper target")
            self.target_counts[self.epoch]["frames"] += len(batch["im_file"])
            self.target_counts[self.epoch]["targets"] += len(batch["cls"])
            self.target_counts[self.epoch]["fallback_frames"] += sum(batch["keeper_aug_fallback"])
            batch = super().preprocess_batch(batch)
            contrast, brightness = self.photo_rng.uniform(.85, 1.15), self.photo_rng.uniform(-.04, .04)
            if photometric:
                batch["img"] = contrast_brightness(batch["img"], contrast, brightness)
                self.photo_batches += 1
            if len(self.photo_receipts) < 16:
                self.photo_receipts.append({"epoch": self.epoch + 1, "image": Path(batch["im_file"][0]).name,
                    "contrast": contrast if photometric else 1.,
                    "brightness": brightness if photometric else 0., "labels_unchanged_by_photometric": True})
            return batch

    return Trainer


class KeeperEvidence(TrainingEvidence):
    def __init__(self, output, schedules, device):
        super().__init__(output, 1, device)
        self.schedules, self.sampling = schedules, []

    def saved(self, trainer):
        observed = trainer.observed_files[trainer.epoch]
        if observed != self.schedules[trainer.epoch]:
            raise RuntimeError("Actual training samples differ from the frozen schedule")
        expected_steps = optimizer_budget(len(observed), EPOCHS)[trainer.epoch]
        if self.steps != expected_steps:
            raise RuntimeError(f"Optimizer steps differ from the fixed budget: {self.steps} != {expected_steps}")
        target_counts = trainer.target_counts[trainer.epoch]
        if target_counts["frames"] != len(observed) or target_counts["targets"] != len(observed):
            raise RuntimeError("Positive target inventory differs from the epoch schedule")
        self.sampling.append({"epoch": trainer.epoch + 1, "samples": len(observed),
            "unique_images": len(set(observed)), "sequence_sha256": sequence_digest(observed),
            "optimizer_steps": self.steps, "target_counts": dict(target_counts)})
        save_json(self.output / "actual_sampling.json", self.sampling)
        save_json(self.output / "photometric_receipt.json", {"batches_augmented": trainer.photo_batches,
            "first_batches": trainer.photo_receipts})
        super().saved(trainer)


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != "active_keeper_detector_final/v1" or not manifest.get("source_pixels_verified")
            or not manifest.get("all_partitions_are_development") or manifest.get("unknown_frames_are_negative_examples")):
        raise ValueError("Expected verified, grouped specialized-keeper references")
    samples = manifest["exports"]
    validate_partition(samples)
    for row in samples:
        for key in ("image", "label"):
            if sha256(path.parent / row[key]) != row[key + "_sha256"]:
                raise ValueError("Exported image or target changed")
        expected = box_target(row["bbox"], row["width"], row["height"])
        tokens = (path.parent / row["label"]).read_text().split()
        if len(tokens) != 5 or tokens[0] != "0" or any(abs(float(a) - b) > 1e-8 for a, b in zip(tokens[1:], expected)):
            raise ValueError("YOLO label does not represent the reviewed keeper box")
    if sum(r["split"] == "train" for r in samples) != 835:
        raise ValueError("Expected all 835 training frames")
    import yaml
    dataset = path.parent / "dataset.yaml"
    definition = yaml.safe_load(dataset.read_text())
    if (Path(definition["path"]).resolve() != path.parent or definition["train"] != "images/train"
            or definition["val"] != "images/train" or definition["names"] != {0: "active_keeper"}):
        raise ValueError("Training YAML may expose development images or change the class")
    return manifest, samples, dataset


def inference_inputs(samples, data_root):
    return [{"id": r["id"], "image": str(Path(data_root) / r["image"]),
             "width": r["width"], "height": r["height"], "image_sha256": r["image_sha256"]} for r in samples]


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def score_predictions(predictions, samples, specialized=True):
    mapping = {row["id"]: row for row in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {r["id"] for r in samples}:
        raise ValueError("Prediction inventory differs from the frozen evaluation frames")
    scored = []
    for sample in samples:
        people = sorted(mapping[sample["id"]]["detections"], key=lambda r: -r["confidence"])
        top = iou(people[0]["bbox"], sample["bbox"]) if people else 0.
        best = max((iou(p["bbox"], sample["bbox"]) for p in people), default=0.)
        scored.append({"id": sample["id"], "video_id": sample["video_id"], "split": sample["split"],
            "event_id": sample["event_id"], "existing_pose_reference": sample["existing_pose_reference"],
            "candidate_count": len(people), "top1_iou": top if specialized else None,
            "top1_hit": top >= .5 if specialized else None, "any_candidate_iou": best,
            "any_candidate_hit": best >= .5})

    def metrics(rows):
        n = len(rows)
        by_clip = defaultdict(list)
        for row in rows:
            by_clip[row["video_id"]].append(row)
        top = sum(r["top1_hit"] for r in rows) if specialized else None
        best = sum(r["any_candidate_hit"] for r in rows)
        macro_top = (sum(sum(r["top1_hit"] for r in part) / len(part) for part in by_clip.values()) / len(by_clip)
                     if specialized and by_clip else None)
        macro_any = (sum(sum(r["any_candidate_hit"] for r in part) / len(part) for part in by_clip.values()) / len(by_clip)
                     if by_clip else None)
        return {"frames": n, "clips": len(by_clip), "top1_hits": top,
            "top1_recall_iou_0_5": top / n if specialized and n else None,
            "macro_clip_top1_recall_iou_0_5": macro_top, "any_candidate_hits": best,
            "any_candidate_recall_iou_0_5": best / n if n else None,
            "macro_clip_any_candidate_recall_iou_0_5": macro_any}

    result = {"all": metrics(scored), "rows": scored,
        "highest_confidence_active_keeper_role_scored": specialized,
        "reference_used_only_for_scoring": True, "independent_test": False,
        "negative_frame_precision_measured": False}
    for field in ("split", "video_id"):
        groups = defaultdict(list)
        for row in scored:
            groups[row[field]].append(row)
        result["by_" + field] = {key: metrics(rows) for key, rows in sorted(groups.items())}
    result["existing_pose_references"] = metrics([r for r in scored if r["existing_pose_reference"]])
    return result


def predict(weights, inputs, device, specialized):
    import cv2
    import torch
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if specialized and model.names != {0: "active_keeper"}:
        raise ValueError("Expected the specialized active-keeper checkpoint")
    if not specialized and model.names.get(0) != "person":
        raise ValueError("Expected the pretrained generic-person detector")
    started = time.perf_counter()
    rows = []
    for index, frame in enumerate(inputs):
        image = cv2.imread(frame["image"])
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError("Cannot read frozen original input")
        result = model.predict(image, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=.7, max_det=100,
                               classes=[0], device=device, verbose=False)[0]
        detections = [{"bbox": box, "confidence": float(conf)} for box, conf in zip(
            result.boxes.xyxy.detach().cpu().tolist(), result.boxes.conf.detach().cpu().tolist())]
        rows.append({"id": frame["id"], "detections": detections})
        if (index + 1) % 50 == 0 or index + 1 == len(inputs):
            print(json.dumps({"evaluation_model": Path(weights).parent.parent.name,
                "frames_completed": index + 1, "frames_total": len(inputs)}), flush=True)
    expected = "mps" if device == "mps" else "cpu" if device == "cpu" else "cuda"
    if model.predictor.device.type != expected:
        raise RuntimeError("Inference device differs from the frozen protocol")
    seconds = time.perf_counter() - started
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return {"rows": rows, "seconds": seconds, "reference_labels_used_for_prediction": False,
            "mode": "full_original_image", "confidence": CONFIDENCE, "image_size": IMAGE_SIZE}


def fit_arm(initial, data, output, schedules, photometric, device):
    import torch
    from ultralytics import YOLO
    output.mkdir()
    model = YOLO(str(initial))
    evidence = KeeperEvidence(output, schedules, device)
    model.add_callback("on_train_start", evidence.start)
    model.add_callback("on_train_batch_end", evidence.batch_finished)
    model.add_callback("on_model_save", evidence.saved)
    started = time.perf_counter()
    model.train(trainer=scheduled_trainer(schedules, photometric), data=str(data), epochs=EPOCHS,
        imgsz=IMAGE_SIZE, batch=1, workers=0, device=device, project=str(output), name="fit",
        exist_ok=False, optimizer="AdamW", lr0=.0001, warmup_bias_lr=.0001, warmup_epochs=2,
        lrf=.1, weight_decay=.0005, nbs=16, freeze=10, patience=EPOCHS, seed=SEED,
        deterministic=True, amp=False, cache=False, plots=False, save=True, save_period=4,
        verbose=False, val=False, compile=False, **BASE_AUGMENTATION)
    evidence.handle.remove()
    if (evidence.epochs != list(range(1, EPOCHS + 1))
            or evidence.steps != optimizer_budget(len(schedules[0]))[-1] or evidence.max_delta <= 0):
        raise RuntimeError("Real fixed-budget training was not verified")
    after = parameter_digests(model.trainer.model)
    changed = [name for name, value in evidence.before.items() if value != after[name]]
    if not changed:
        raise RuntimeError("No parameters changed")
    checkpoint = Path(model.trainer.last)
    if YOLO(str(checkpoint)).names != {0: "active_keeper"}:
        raise ValueError("Saved model class semantics changed")
    receipt = {"initial_weights": str(initial), "initial_sha256": sha256(initial),
        "weights": str(checkpoint), "weights_sha256": sha256(checkpoint),
        "initial_parameter_digests": evidence.before, "completed_epochs": evidence.epochs,
        "optimizer_steps": evidence.steps, "sampling": evidence.sampling,
        "changed_parameters": changed, "tracked_parameter": evidence.name,
        "maximum_observed_parameter_delta": evidence.max_delta,
        "photometric_enabled": photometric, "checkpoint_selection": "fixed_last_epoch_12",
        "training_seconds": time.perf_counter() - started,
        "development_scored_before_checkpoint_freeze": False, "promoted_to_default": False}
    save_json(output / "candidate.json", receipt)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.manifest, args.pretrained = args.manifest.resolve(), args.pretrained.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new final experiment directory")
    import cv2
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False})
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        torch.mps.set_per_process_memory_fraction(.6)
    manifest, samples, data = load_manifest(args.manifest)
    preflight = preflight_dataset(args.manifest.parent)
    if args.preflight_only:
        args.output.mkdir(parents=True)
        save_json(args.output / "preflight.json", preflight)
        print(json.dumps(preflight, indent=2))
        return
    train = [r for r in samples if r["split"] == "train"]
    evaluation = [r for r in samples if r["split"] != "train" or r["existing_pose_reference"]]
    schedules = schedules_for([Path(r["image"]).name for r in train])
    args.output.mkdir(parents=True)
    save_json(args.output / "schedules.json", schedules)
    save_json(args.output / "preflight.json", preflight)
    protocol = {"schema": "active_keeper_final_comparison/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "initial_weights": str(args.pretrained), "initial_weights_sha256": sha256(args.pretrained),
        "dataset_yaml_sha256": sha256(data), "epochs": EPOCHS, "image_size": IMAGE_SIZE,
        "batch": 1, "workers": 0, "device": args.device, "nominal_accumulated_batch": 16,
        "seed": SEED, "optimizer": "AdamW", "learning_rate": .0001, "warmup_bias_lr": .0001,
        "freeze": 10, "base_augmentation": BASE_AUGMENTATION, "photometric_added_only_to_second_arm": {
            "contrast": [.85, 1.15], "brightness": [-.04, .04], "labels_unchanged": True},
        "training_frames": len(train), "evaluation_counts": dict(Counter(r["split"] for r in evaluation)),
        "expected_optimizer_steps_per_epoch": optimizer_budget(),
        "lost_target_augmentation_policy": "fallback to original-frame letterbox; never make false negative",
        "schedule_sha256": sha256(args.output / "schedules.json"),
        "evaluation_frame_ids": [r["id"] for r in evaluation],
        "all_candidates_frozen_before_development_evaluation": True, "independent_test": False,
        "validation_yaml_uses_train_only": True, "human_boxes_used_for_inference": False,
        "checkpoint_selection": "fixed_last_epoch_12", "whole_pipeline_accuracy_measured": False,
        "runtime": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
            ROOT / "scripts/prepare_keeper_detector_final.py", ROOT / "scripts/train_external_ball_round1.py",
            ROOT / "scripts/train_ball_controlled_round2.py"]}}
    save_json(args.output / "protocol.json", protocol)
    candidates = {}
    for arm in ARMS:
        candidates[arm] = fit_arm(args.pretrained, data, args.output / arm, schedules, arm == "photometric", args.device)
        if arm == "photometric":
            first, second = candidates["control"], candidates[arm]
            if (first["initial_parameter_digests"] != second["initial_parameter_digests"]
                    or first["sampling"] != second["sampling"]):
                raise RuntimeError("Control/augmentation initialization, samples, or optimizer budgets differ")
    save_json(args.output / "frozen_candidates.json", candidates)
    inputs = inference_inputs(evaluation, args.manifest.parent)
    results = {}
    for name, weights in [("pretrained_person_candidate_control", args.pretrained),
                          *((name, Path(value["weights"])) for name, value in candidates.items())]:
        specialized = name in ARMS
        prediction = predict(weights, inputs, args.device, specialized)
        score = score_predictions(prediction["rows"], evaluation, specialized)
        save_json(args.output / f"{name}_predictions.json", prediction)
        save_json(args.output / f"{name}_scores.json", score)
        results[name] = {"scores": {k: v for k, v in score.items() if k != "rows"},
                         "seconds": prediction["seconds"], "weights_sha256": sha256(weights)}
        save_json(args.output / "evaluation_progress.json", results)
    report = {"protocol": protocol, "candidates": candidates, "evaluation": results,
        "training_completed": True, "default_pipeline_changed": False,
        "independent_test": False, "full_video_role_tracking_measured": False,
        "limitations": ["Assisted/exposed development labels; no fresh final test.",
            "Only positive active-keeper frames are scored; absent-target false alarms are unmeasured.",
            "Dense frames are correlated; macro-clip results are also reported.",
            "Detector localization is not pose, direction, height, or whole-project accuracy.",
            "Photometric and 2D geometric augmentation do not create true new camera viewpoints."]}
    save_json(args.output / "report.json", report)
    print(json.dumps({name: value["scores"]["by_split"] for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
