"""Frozen, source-matched final ball augmentation comparison on development data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_ball_detector import evaluate_checkpoint, load_exports
from scripts.train_ball_controlled_round2 import make_schedules, sequence_digest
from scripts.train_ball_frame_detector import digest, save_json
from scripts.train_external_ball_round1 import TrainingEvidence, add_event_metrics, parameter_digests, partition_samples
from src.ball_specialist import tile_windows

EPOCHS = 10
SEED = 29
ARMS = ("control_full", "augmented_mixed")
SHARED_AUGMENTATION = {"hsv_h": .015, "hsv_s": .7, "hsv_v": .4,
                       "scale": .2, "translate": .1, "fliplr": .5, "flipud": 0.,
                       "mosaic": 0., "mixup": 0., "copy_paste": 0., "shear": 0., "perspective": 0.}


def contained_grid_crops(normalized, width, height):
    x, y, w, h = normalized
    if min(width, height) <= 0 or not all(0 <= v <= 1 for v in normalized) or min(w, h) <= 0:
        raise ValueError("Invalid image or normalized target")
    bbox = [(x - w / 2) * width, (y - h / 2) * height,
            (x + w / 2) * width, (y + h / 2) * height]
    if bbox[0] < -1e-5 or bbox[1] < -1e-5 or bbox[2] > width + 1e-5 or bbox[3] > height + 1e-5:
        raise ValueError("Original target extends beyond the image")
    variants = []
    for index, window in enumerate(tile_windows(width, height, 960, .25)):
        left, top, right, bottom = window
        if window == (0, 0, width, height):
            continue
        if not (left <= bbox[0] and top <= bbox[1] and right >= bbox[2] and bottom >= bbox[3]):
            continue
        cw, ch = right - left, bottom - top
        target = [(x * width - left) / cw, (y * height - top) / ch, w * width / cw, h * height / ch]
        variants.append({"grid_index": index, "window": list(window), "normalized": target})
    return variants


class EpochScheduleSampler:
    def __init__(self, loaded_names, schedules):
        mapping = {name: i for i, name in enumerate(loaded_names)}
        if len(mapping) != len(loaded_names) or not schedules or not schedules[0]:
            raise ValueError("Unique dataset identities and nonempty schedules required")
        if any(len(s) != len(schedules[0]) or set(s) - mapping.keys() for s in schedules):
            raise ValueError("Each schedule must have the same budget and known images")
        self.indices = [[mapping[n] for n in epoch] for epoch in schedules]
        self.next_epoch = 0

    def __len__(self):
        return len(self.indices[0])

    def __iter__(self):
        if self.next_epoch >= len(self.indices):
            raise RuntimeError("No unplanned training epoch is allowed")
        selected = self.indices[self.next_epoch]
        self.next_epoch += 1
        return iter(selected)


def mixed_schedule(source_schedule, variants, seed=SEED):
    rng = random.Random(seed)
    return [[rng.choice(variants[name]) if variants.get(name) and rng.random() < .5 else name
             for name in epoch] for epoch in source_schedule]


def optimizer_budget(epochs, batches, nominal_batch=16, warmup_epochs=2):
    import numpy as np
    nw = round(min(warmup_epochs, max(epochs - 1, 0)) * batches)
    steps, last, counts = 0, -1, []
    for epoch in range(epochs):
        for i in range(batches):
            ni = epoch * batches + i
            accumulate = max(1, int(np.interp(ni, [0, nw], [1, nominal_batch]).round())) if ni < nw else nominal_batch
            if ni - last >= accumulate:
                steps += 1
                last = ni
        counts.append(steps)
    return counts


class MildContrastBrightness:
    """Photometric-only transform; existing geometry owns all target transforms."""
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.observed = []

    def __call__(self, labels):
        import numpy as np
        alpha, beta = self.rng.uniform(.85, 1.15), self.rng.uniform(-8., 8.)
        image = labels["img"]
        labels["img"] = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        self.observed.append({"file": Path(labels["im_file"]).name, "contrast": alpha, "brightness_255": beta})
        return labels


class PositivePreservingTransforms:
    """Keep a known positive if random geometry removes its only target."""
    def __init__(self, transforms, image_size, photometric=None):
        from ultralytics.data.augment import Compose, LetterBox
        self.geometry = Compose(transforms.transforms[:-1])
        self.format = transforms.transforms[-1]
        self.fallback = LetterBox(new_shape=(image_size, image_size), scaleup=False)
        self.photometric = photometric
        self.observed = []

    def __call__(self, labels):
        if len(labels["instances"]) != 1:
            raise ValueError("Positive-only source must have exactly one ball target")
        original = deepcopy(labels)
        transformed = self.geometry(labels)
        fallback = len(transformed["instances"]) != 1
        if fallback:
            transformed = self.fallback(original)
        if len(transformed["instances"]) != 1:
            raise RuntimeError("Known-positive fallback did not retain its ball")
        if self.photometric is not None:
            transformed = self.photometric(transformed)
        result = self.format(transformed)
        if result["bboxes"].shape[0] != 1:
            raise RuntimeError("Formatting lost a known-positive target")
        self.observed.append({"file": Path(result["im_file"]).name, "geometry_fallback": fallback})
        return result


def make_training_pool(train, output):
    import cv2
    import yaml
    if output.exists():
        raise FileExistsError(output)
    image_dir, label_dir = output / "images/train", output / "labels/train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    records, variants, originals = {}, {}, []
    for sample in train:
        if sample["split"] not in {"train", "validation"}:
            raise ValueError("Evaluation-only frame cannot enter the training pool")
        name = sample["image"].name
        if name in records:
            raise ValueError("Ambiguous source filename")
        image = cv2.imread(str(sample["image"]))
        if image is None:
            raise ValueError("Undecodable training source")
        height, width = image.shape[:2]
        shutil.copy2(sample["image"], image_dir / name)
        shutil.copy2(sample["label"], label_dir / (Path(name).stem + ".txt"))
        provenance = {"source_name": name, "source_sha256": sample["image_sha256"],
                      "source_label_sha256": sample["label_sha256"], "video_id": sample["video_id"],
                      "event_id": sample.get("event_id", sample["video_id"]),
                      "match_id": sample.get("match_id"), "source_split": sample["split"],
                      "independent_new_event": False}
        records[name] = {**provenance, "variant": "original", "image": str((image_dir / name).relative_to(output)),
                         "image_sha256": digest(image_dir / name), "normalized": sample["normalized"],
                         "label": str((label_dir / (Path(name).stem + ".txt")).relative_to(output)),
                         "label_sha256": digest(label_dir / (Path(name).stem + ".txt"))}
        variants[name] = []
        originals.append(name)
        for crop in contained_grid_crops(sample["normalized"], width, height):
            crop_name = f"{Path(name).stem}_grid{crop['grid_index']:02d}.jpg"
            if crop_name in records:
                raise ValueError("Crop identity collision")
            left, top, right, bottom = crop["window"]
            if not cv2.imwrite(str(image_dir / crop_name), image[top:bottom, left:right], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError("Could not write crop")
            (label_dir / (Path(crop_name).stem + ".txt")).write_text(
                "0 " + " ".join(f"{v:.10f}" for v in crop["normalized"]) + "\n")
            records[crop_name] = {**provenance, **crop, "variant": "dimension_grid_crop",
                                  "image": str((image_dir / crop_name).relative_to(output)),
                                  "image_sha256": digest(image_dir / crop_name),
                                  "label": str((label_dir / (Path(crop_name).stem + ".txt")).relative_to(output)),
                                  "label_sha256": digest(label_dir / (Path(crop_name).stem + ".txt"))}
            variants[name].append(crop_name)
    original_list = output / "originals.txt"
    original_list.write_text("".join(str(image_dir.resolve() / n) + "\n" for n in originals))
    for arm in ARMS:
        path = output / f"{arm}.yaml"
        path.write_text(yaml.safe_dump({"path": str(output.resolve()),
            "train": "originals.txt" if arm == ARMS[0] else "images/train",
            "val": "originals.txt", "names": {0: "ball"}}))
    save_json(output / "manifest.json", {"exports": records, "variants": variants,
              "originals": originals, "evaluation_frames_present": False,
              "validation_is_training_fit_only": True})
    return records, variants


def make_preview(records, pool, output):
    import cv2
    import numpy as np
    selected = []
    for source in ("old", "new"):
        candidates = [r for r in records.values() if r["variant"] == "dimension_grid_crop"
                      and (bool(r.get("match_id")) == (source == "new"))]
        distinct = {}
        for candidate in candidates:
            distinct.setdefault(candidate["event_id"], candidate)
        for crop in list(distinct.values())[:3]:
            selected += [records[crop["source_name"]], crop]
    if len(selected) != 12:
        raise ValueError("Expected preview examples from both source pools")
    panels = []
    for row in selected:
        image = cv2.imread(str(pool / row["image"]))
        h, w = image.shape[:2]
        x, y, bw, bh = row["normalized"]
        pt1 = round((x - bw / 2) * w), round((y - bh / 2) * h)
        pt2 = round((x + bw / 2) * w), round((y + bh / 2) * h)
        cv2.rectangle(image, pt1, pt2, (0, 0, 255), max(2, w // 500))
        scale = min(640 / w, 345 / h)
        resized = cv2.resize(image, (round(w * scale), round(h * scale)))
        panel = np.full((380, 640, 3), 245, np.uint8)
        panel[35:35 + resized.shape[0], :resized.shape[1]] = resized
        cv2.putText(panel, row["source_name"][:55], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1)
        cv2.putText(panel, row["variant"], (5, 30), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1)
        panels.append(panel)
    canvas = np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)])
    if not cv2.imwrite(str(output), canvas):
        raise ValueError("Preview save failed")


def training_class(schedules, augmented):
    import torch
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.models.yolo.detect import DetectionTrainer

    class Trainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            self.observed_files = defaultdict(list)
            self.batch_checks = defaultdict(Counter)
            self.photometric = MildContrastBrightness()
            super().__init__(*args, **kwargs)

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank != -1 or batch_size != 1 or self.args.workers != 0:
                raise ValueError("Frozen final comparison requires batch 1 / workers 0 / single device")
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            self.positive_transform = PositivePreservingTransforms(dataset.transforms, self.args.imgsz,
                self.photometric if augmented else None)
            dataset.transforms = self.positive_transform
            names = [Path(p).name for p in dataset.im_files]
            sampler = EpochScheduleSampler(names, schedules)
            return InfiniteDataLoader(dataset, batch_size=1, num_workers=0, sampler=sampler,
                shuffle=False, collate_fn=dataset.collate_fn, pin_memory=False, drop_last=False,
                generator=torch.Generator().manual_seed(SEED))

        def preprocess_batch(self, batch):
            names = [Path(p).name for p in batch["im_file"]]
            self.observed_files[self.epoch].extend(names)
            checks = self.batch_checks[self.epoch]
            checks["batches"] += 1
            checks["remaining_targets"] += int(batch["bboxes"].shape[0])
            checks["empty_after_framework_augmentation"] += int(batch["bboxes"].shape[0] == 0)
            if batch["bboxes"].shape[0] != 1:
                raise RuntimeError("A known-positive frame must never silently become a negative")
            if not torch.isfinite(batch["bboxes"]).all() or not torch.isfinite(batch["img"]).all():
                raise RuntimeError("Non-finite augmented batch")
            if batch["bboxes"].numel() and (batch["bboxes"].min() < 0 or batch["bboxes"].max() > 1):
                raise RuntimeError("Augmented bbox outside normalized bounds")
            return super().preprocess_batch(batch)

    return Trainer


class FinalEvidence(TrainingEvidence):
    def __init__(self, output, schedules, source_schedules, records, old_names, device):
        super().__init__(output, 1, device)
        self.schedules, self.source_schedules = schedules, source_schedules
        self.records, self.old_names, self.sampling = records, set(old_names), []
        self.expected_steps = optimizer_budget(EPOCHS, len(schedules[0]))

    def saved(self, trainer):
        observed = trainer.observed_files[trainer.epoch]
        source_names = [self.records[n]["source_name"] for n in observed]
        if observed != self.schedules[trainer.epoch] or source_names != self.source_schedules[trainer.epoch]:
            raise RuntimeError("Observed model inputs differ from frozen paired source schedules")
        if self.steps != self.expected_steps[trainer.epoch]:
            raise RuntimeError(f"Optimizer steps differ from frozen budget: {self.steps}")
        self.sampling.append({"epoch": trainer.epoch + 1, "samples": len(observed),
            "source_sequence_sha256": sequence_digest(source_names),
            "variant_sequence_sha256": sequence_digest(observed),
            "old": sum(n in self.old_names for n in source_names),
            "new": sum(n not in self.old_names for n in source_names),
            "crops": sum(self.records[n]["variant"] != "original" for n in observed),
            "optimizer_steps": self.steps, "batch_checks": dict(trainer.batch_checks[trainer.epoch]),
            "geometry_fallbacks_cumulative": sum(r["geometry_fallback"] for r in trainer.positive_transform.observed),
            "photometric_calls_cumulative": len(trainer.photometric.observed)})
        save_json(self.output / "actual_sampling.json", self.sampling)
        super().saved(trainer)


def train_arm(initial, pool, output, arm, schedules, source_schedules, records, old_names, device):
    import torch
    from ultralytics import YOLO
    output.mkdir()
    model = YOLO(str(initial))
    if model.names != {0: "ball"}:
        raise ValueError("Expected the existing single-ball checkpoint")
    evidence = FinalEvidence(output, schedules, source_schedules, records, old_names, device)
    model.add_callback("on_train_start", evidence.start)
    model.add_callback("on_train_batch_end", evidence.batch_finished)
    model.add_callback("on_model_save", evidence.saved)
    started = time.perf_counter()
    model.train(trainer=training_class(schedules, arm == ARMS[1]), data=str(pool / f"{arm}.yaml"),
        epochs=EPOCHS, imgsz=1920, batch=1, workers=0, device=device, project=str(output), name="fit",
        exist_ok=False, optimizer="AdamW", lr0=.00005, warmup_bias_lr=.00005,
        lrf=.1, weight_decay=.0005, nbs=16, freeze=10, warmup_epochs=2, patience=EPOCHS,
        seed=SEED, deterministic=True, amp=False, cache=False, close_mosaic=0,
        degrees=3. if arm == ARMS[1] else 0., **SHARED_AUGMENTATION,
        plots=False, save=True, save_period=5, verbose=False, val=False, compile=False)
    evidence.handle.remove()
    if evidence.epochs != list(range(1, EPOCHS + 1)) or evidence.steps != evidence.expected_steps[-1] or evidence.max_delta <= 0:
        raise RuntimeError("Completed epochs and actual updates were not verified")
    after = parameter_digests(model.trainer.model)
    changed = [name for name, before in evidence.before.items() if before != after[name]]
    if not changed:
        raise RuntimeError("No model parameters changed")
    expected_photo = EPOCHS * len(source_schedules[0]) if arm == ARMS[1] else 0
    if len(model.trainer.photometric.observed) != expected_photo:
        raise RuntimeError("Photometric transform calls differ from training budget")
    save_json(output / "photometric_receipt.json", model.trainer.photometric.observed)
    save_json(output / "positive_preservation_receipt.json", model.trainer.positive_transform.observed)
    weights = Path(model.trainer.last)
    receipt = {"initial_weights": str(initial), "initial_sha256": digest(initial),
        "weights": str(weights), "weights_sha256": digest(weights), "completed_epochs": evidence.epochs,
        "optimizer_steps": evidence.steps, "changed_parameter_names": changed,
        "tracked_parameter": evidence.name, "maximum_observed_parameter_delta": evidence.max_delta,
        "training_seconds": time.perf_counter() - started, "sampling": evidence.sampling,
        "checkpoint_selection": "fixed_last_epoch_10", "optimizer_resumed": False,
        "photometric_transform_calls": expected_photo, "prior_development_scores_seen": True,
        "current_round_scores_seen_before_checkpoint_freeze": False}
    save_json(output / "selected_candidate.json", receipt)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return receipt


def sources_from(round1):
    report = json.loads((round1 / "report.json").read_text())
    sources = {}
    for name, source in report["protocol"]["manifests"].items():
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError("Source manifest changed")
        sources[name] = load_exports(path)
    train, ars = partition_samples(sources["old"], sources["new"], sources["regression"])
    if len(train) != 442 or len(sources["old"]) != 122 or len(ars) != 88:
        raise ValueError("Frozen cohort size changed")
    return report, sources, train, ars


def preflight(args):
    import cv2
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.torch_utils import init_seeds
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    pool = args.output / "dataset"
    schedules = json.loads((args.output / "schedules.json").read_text())
    records = json.loads((pool / "manifest.json").read_text())["exports"]
    results = {}
    for arm in ARMS:
        init_seeds(SEED, deterministic=True)
        cfg = get_cfg(overrides={"imgsz": 1920, "degrees": 3. if arm == ARMS[1] else 0.,
                                **SHARED_AUGMENTATION})
        dataset = YOLODataset(img_path=str(pool / ("originals.txt" if arm == ARMS[0] else "images/train")),
            imgsz=1920, augment=True, hyp=cfg, data={"names": {0: "ball"}, "nc": 1},
            batch_size=1, cache=False)
        photo = MildContrastBrightness() if arm == ARMS[1] else None
        positive = PositivePreservingTransforms(dataset.transforms, 1920, photo)
        dataset.transforms = positive
        names = [Path(p).name for p in dataset.im_files]
        loader = InfiniteDataLoader(dataset, batch_size=1, num_workers=0,
            sampler=EpochScheduleSampler(names, [schedules[arm][0]]), shuffle=False,
            collate_fn=dataset.collate_fn, pin_memory=False, drop_last=False,
            generator=torch.Generator().manual_seed(SEED))
        observed, counts = [], Counter()
        for batch in loader:
            observed.extend(Path(p).name for p in batch["im_file"])
            if batch["bboxes"].shape != (1, 4) or not torch.isfinite(batch["bboxes"]).all():
                raise RuntimeError("Preflight lost or corrupted a known-positive target")
            if not ((batch["bboxes"] >= 0) & (batch["bboxes"] <= 1)).all():
                raise RuntimeError("Preflight target outside normalized bounds")
            counts["batches"] += 1
            counts["targets"] += len(batch["bboxes"])
        loader.close()
        if observed != schedules[arm][0] or [records[n]["source_name"] for n in observed] != schedules["source"][0]:
            raise RuntimeError("Preflight source schedule changed")
        results[arm] = {**dict(counts), "geometry_fallbacks": sum(r["geometry_fallback"] for r in positive.observed),
            "photometric_calls": len(photo.observed) if photo else 0,
            "source_sequence_sha256": sequence_digest([records[n]["source_name"] for n in observed]),
            "source_training_split_only": True, "model_loaded": False, "device": "cpu"}
        print(json.dumps({"loader_preflight": arm, **results[arm]}), flush=True)
    save_json(args.output / "loader_preflight.json", results)


def prepare(args):
    report, sources, train, _ = sources_from(args.round1)
    previous_receipt = json.loads((args.round2 / "B_round1_balanced/selected_candidate.json").read_text())
    initial = Path(previous_receipt["weights"])
    if digest(initial) != previous_receipt["weights_sha256"]:
        raise ValueError("Fixed balanced initialization changed")
    if args.output.exists():
        raise FileExistsError("Preserve prior runs; use a new output directory")
    args.output.mkdir(parents=True)
    records, variants = make_training_pool(train, args.output / "dataset")
    names, old_names = [r["image"].name for r in train], {r["image"].name for r in sources["old"]}
    schedule = make_schedules(names, old_names, balanced=True, epochs=EPOCHS, seed=SEED)
    mixed = mixed_schedule(schedule, variants)
    if set(sum(schedule, [])) != set(names):
        raise ValueError("Frozen source deck did not cover every training image")
    schedules = {"source": schedule, ARMS[0]: schedule, ARMS[1]: mixed}
    save_json(args.output / "schedules.json", schedules)
    save_json(args.output / "old_names.json", sorted(old_names))
    make_preview(records, args.output / "dataset", args.output / "crop_examples.jpg")
    protocol = {"schema": "ball_final_augmentation/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "round1": str(args.round1), "round2": str(args.round2), "initial_weights": str(initial),
        "initial_sha256": digest(initial), "epochs_each": EPOCHS, "samples_per_epoch": 442,
        "expected_optimizer_steps_per_epoch": optimizer_budget(EPOCHS, 442),
        "balanced_quota": {"old_source_pool": 221, "new_lev_match": 221},
        "imgsz": 1920, "batch": 1, "workers": 0, "nbs": 16, "lr0": .00005,
        "warmup_bias_lr": .00005, "seed": SEED, "freeze": 10,
        "shared_augmentation": SHARED_AUGMENTATION,
        "intervention": {"type": "combined_augmentation", "crop_probability": .5,
            "grid_edge": 960, "grid_overlap": .25, "complete_bbox_containment_required": True,
            "crop_candidates_dimension_only": True, "positive_window_selection_training_reference_only": True,
            "contrast": [.85, 1.15], "brightness_255": [-8., 8.], "degrees": 3.,
            "known_positive_geometry_drop_policy": "fallback_to_source_variant_letterbox_before_photometric",
            "fake_3d_viewpoints": False, "class_direction_semantics_changed": False},
        "source_manifests": report["protocol"]["manifests"],
        "dataset_manifest_sha256": digest(args.output / "dataset/manifest.json"),
        "schedule_sha256": digest(args.output / "schedules.json"),
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT / "scripts/train_ball_controlled_round2.py", ROOT / "scripts/train_external_ball_round1.py",
            ROOT / "scripts/optimize_ball_detector.py", ROOT / "src/ball_specialist.py"]},
        "selection": "tiles core >=65/91 and ARS >=62/88; then equal-group mean recall; compare current initializer",
        "evaluation": "exposed_development_positive_frame_localization_only",
        "evaluation_before_all_candidates_frozen": False, "independent_test": False,
        "default_pipeline_changed": False, "new_independent_training_events": 0}
    save_json(args.output / "protocol.json", protocol)
    print(json.dumps({"prepared": True, "source_frames": len(names), "pool_images": len(records),
                      "crop_variants": len(records) - len(names), "protocol": str(args.output / "protocol.json")}), flush=True)


def run(args):
    import cv2
    import torch
    from ultralytics.utils import SETTINGS
    SETTINGS.update({"sync": False})
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        torch.mps.set_per_process_memory_fraction(.65)
    protocol = json.loads((args.output / "protocol.json").read_text())
    for relative, expected in protocol["code_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen experiment code changed: {relative}")
    pool = args.output / "dataset"
    if digest(pool / "manifest.json") != protocol["dataset_manifest_sha256"] or digest(args.output / "schedules.json") != protocol["schedule_sha256"]:
        raise ValueError("Frozen dataset or schedule changed")
    initial = Path(protocol["initial_weights"])
    if digest(initial) != protocol["initial_sha256"]:
        raise ValueError("Initial checkpoint changed")
    _, sources, _, ars = sources_from(Path(protocol["round1"]))
    schedules = json.loads((args.output / "schedules.json").read_text())
    records = json.loads((pool / "manifest.json").read_text())["exports"]
    old_names = json.loads((args.output / "old_names.json").read_text())
    for row in records.values():
        if digest(pool / row["image"]) != row["image_sha256"]:
            raise ValueError("Prepared training pixels changed")
        if digest(pool / row["label"]) != row["label_sha256"]:
            raise ValueError("Prepared training targets changed")
    receipts = {}
    started = time.perf_counter()
    for arm in ARMS:
        print(json.dumps({"starting_arm": arm}), flush=True)
        receipts[arm] = train_arm(initial, pool, args.output / arm, arm, schedules[arm],
            schedules["source"], records, old_names, args.device)
        save_json(args.output / "candidates.json", receipts)
    results = {}
    for group, items in (("arsenal_development", ars), ("core_development", sources["regression"])):
        results[group] = {}
        for name, weights in [("initializer", initial)] + [(a, Path(receipts[a]["weights"])) for a in ARMS]:
            results[group][name] = {}
            for mode in ("full", "tiles"):
                target = args.output / f"{group}_{name}_{mode}.json"
                value = evaluate_checkpoint(weights, items, target, mode, 1280, args.device, 1)
                value["metrics"] = add_event_metrics(target, items)
                results[group][name][mode] = value
                save_json(args.output / "evaluation_progress.json", results)
                print(json.dumps({"group": group, "model": name, "mode": mode,
                                  "metrics": value["metrics"]["deduplicated_frames"]}), flush=True)
    def hits(group, arm):
        return results[group][arm]["tiles"]["metrics"]["deduplicated_frames"]["top1_iou_0_5_hits"]
    eligible = [a for a in ARMS if hits("core_development", a) >= 65 and hits("arsenal_development", a) >= 62]
    selected = max(eligible, key=lambda a: hits("core_development", a) / 91 + hits("arsenal_development", a) / 88) if eligible else None
    save_json(args.output / "report.json", {"protocol": protocol, "candidates": receipts, "results": results,
        "eligible_arms": eligible, "selected_candidate": selected, "training_completed": True,
        "elapsed_seconds": time.perf_counter() - started, "default_pipeline_changed": False,
        "independent_test_evaluated": False, "overall_project_accuracy_measured": False,
        "promotion": "candidate_only_pending_full_video_comparison"})
    print(json.dumps({"completed": True, "selected": selected, "eligible": eligible}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round1", type=Path, default=ROOT / "outputs/external_ball_training_mps_2026-10-28")
    parser.add_argument("--round2", type=Path, default=ROOT / "outputs/ball_controlled_round2_mps_2026-10-28")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    args.output, args.round1, args.round2 = args.output.resolve(), args.round1.resolve(), args.round2.resolve()
    {"prepare": prepare, "preflight": preflight, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
