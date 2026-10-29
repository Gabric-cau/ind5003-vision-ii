"""Fixed-budget sampling/initialization comparisons on exposed development data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.optimize_ball_detector import evaluate_checkpoint, load_exports
from scripts.train_ball_frame_detector import digest, save_json
from scripts.train_external_ball_round1 import (
    TrainingEvidence, add_event_metrics, make_dataset, parameter_digests, partition_samples,
)

EPOCHS = 10
ARMS = ("A_round1_natural", "B_round1_balanced", "C_previous_balanced")


def sequence_digest(names):
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()


def make_schedules(names, old_names, *, balanced, epochs=EPOCHS, seed=27):
    names, old_names = sorted(names), set(old_names)
    if len(names) != len(set(names)) or not old_names < set(names) or not old_names:
        raise ValueError("Unique names and two nonempty source groups are required")
    if balanced and len(names) % 2:
        raise ValueError("Equal quotas require an even sample count")
    rng = random.Random(seed)
    groups = [sorted(old_names), sorted(set(names) - old_names)]
    decks = [deque(), deque()]

    def draw(group, count):
        values = []
        for _ in range(count):
            if not decks[group]:
                shuffled = groups[group].copy()
                rng.shuffle(shuffled)
                decks[group].extend(shuffled)
            values.append(decks[group].popleft())
        return values

    schedules = []
    for _ in range(epochs):
        selected = draw(0, len(names) // 2) + draw(1, len(names) // 2) if balanced else names.copy()
        rng.shuffle(selected)
        schedules.append(selected)
    return schedules


class ScheduledSampler:
    def __init__(self, loaded_names, schedules):
        mapping = {name: index for index, name in enumerate(loaded_names)}
        if len(mapping) != len(loaded_names) or not schedules:
            raise ValueError("Dataset names must be unique and schedules nonempty")
        if any(len(s) != len(loaded_names) or set(s) - mapping.keys() for s in schedules):
            raise ValueError("Schedule cannot change epoch size or reference unknown images")
        self.indices = [[mapping[name] for name in schedule] for schedule in schedules]
        self.next_epoch = 0

    def __len__(self):
        return len(self.indices[0])

    def __iter__(self):
        if self.next_epoch >= len(self.indices):
            raise RuntimeError("Sampling requested an unplanned epoch")
        selected = self.indices[self.next_epoch]
        self.next_epoch += 1
        return iter(selected)


def controlled_trainer(schedules):
    import torch
    from ultralytics.data.build import InfiniteDataLoader
    from ultralytics.models.yolo.detect import DetectionTrainer

    class Trainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            self.observed_files = defaultdict(list)
            super().__init__(*args, **kwargs)

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank != -1 or batch_size != 1 or self.args.workers != 0:
                raise ValueError("Controlled loader requires one device, batch 1, workers 0")
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            names = [Path(path).name for path in dataset.im_files]
            self.controlled_sampler = ScheduledSampler(names, schedules)
            generator = torch.Generator().manual_seed(27)
            return InfiniteDataLoader(dataset, batch_size=1, num_workers=0,
                                      sampler=self.controlled_sampler, shuffle=False,
                                      collate_fn=dataset.collate_fn, pin_memory=False,
                                      drop_last=False, generator=generator)

        def preprocess_batch(self, batch):
            self.observed_files[self.epoch].extend(Path(p).name for p in batch["im_file"])
            return super().preprocess_batch(batch)

    return Trainer


class ControlledEvidence(TrainingEvidence):
    def __init__(self, output, schedules, old_names):
        super().__init__(output, 1, "mps")
        self.schedules, self.old_names, self.sampling = schedules, set(old_names), []

    def saved(self, trainer):
        observed = trainer.observed_files[trainer.epoch]
        if observed != self.schedules[trainer.epoch]:
            raise RuntimeError("Actual training order differs from the frozen schedule")
        expected_steps = 49 if trainer.epoch == 0 else 58 + 7 * (trainer.epoch - 1)
        if self.steps != expected_steps:
            raise RuntimeError(f"Unexpected optimizer budget: {self.steps} != {expected_steps}")
        self.sampling.append({"epoch": trainer.epoch + 1, "samples": len(observed),
                              "old": sum(name in self.old_names for name in observed),
                              "new": sum(name not in self.old_names for name in observed),
                              "sequence_sha256": sequence_digest(observed),
                              "optimizer_steps": self.steps})
        save_json(self.output / "actual_sampling.json", self.sampling)
        super().saved(trainer)


def train_arm(initial, data, output, schedules, old_names):
    import torch
    from ultralytics import YOLO
    output.mkdir()
    model = YOLO(str(initial))
    if model.names != {0: "ball"}:
        raise ValueError("Continuation requires the existing single-ball model")
    evidence = ControlledEvidence(output, schedules, old_names)
    model.add_callback("on_train_start", evidence.start)
    model.add_callback("on_train_batch_end", evidence.batch_finished)
    model.add_callback("on_model_save", evidence.saved)
    started = time.perf_counter()
    model.train(trainer=controlled_trainer(schedules), data=str(data), epochs=EPOCHS,
                imgsz=1920, batch=1, workers=0, device="mps", project=str(output), name="fit",
                exist_ok=False, optimizer="AdamW", lr0=.00005, warmup_bias_lr=.00005,
                lrf=.1, weight_decay=.0005, nbs=64, freeze=10, warmup_epochs=2,
                patience=EPOCHS, seed=27, deterministic=True, amp=False, cache=False,
                mosaic=0., mixup=0., copy_paste=0., scale=.2, translate=.1,
                fliplr=.5, flipud=0., close_mosaic=0, plots=False, save=True,
                save_period=5, verbose=False, val=False, compile=False)
    evidence.handle.remove()
    if evidence.epochs != list(range(1, EPOCHS + 1)) or evidence.steps != 114 or evidence.max_delta <= 0:
        raise RuntimeError("Fixed training and actual parameter updates were not verified")
    after = parameter_digests(model.trainer.model)
    changed = [name for name, before in evidence.before.items() if before != after[name]]
    if not changed:
        raise RuntimeError("No parameters changed")
    weights = Path(model.trainer.last)
    receipt = {"initial_weights": str(initial), "initial_sha256": digest(initial),
               "weights": str(weights), "weights_sha256": digest(weights),
               "completed_epochs": evidence.epochs, "optimizer_steps": evidence.steps,
               "changed_parameter_names": changed, "tracked_parameter": evidence.name,
               "maximum_observed_parameter_delta": evidence.max_delta,
               "training_seconds": time.perf_counter() - started,
               "sampling": evidence.sampling, "checkpoint_selection": "fixed_last_epoch_10",
               "optimizer_resumed": False, "prior_development_scores_seen": True,
               "round2_scores_seen_before_checkpoint_freeze": False}
    save_json(output / "selected_candidate.json", receipt)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new experiment directory")
    import cv2
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    SETTINGS.update({"sync": False})
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.mps.set_per_process_memory_fraction(.65)
    baseline_path = args.round1 / "report.json"
    baseline = json.loads(baseline_path.read_text())
    if not baseline["training_completed"] or not baseline["detector_weights_updated"]:
        raise ValueError("Round 1 is incomplete")
    current = Path(baseline["candidate"]["weights"])
    if digest(current) != baseline["candidate"]["weights_sha256"] or digest(args.previous) != baseline["protocol"]["previous_weights_sha256"]:
        raise ValueError("Initial checkpoint identity changed")
    sources = {}
    for name, source in baseline["protocol"]["manifests"].items():
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError("Frozen manifest changed")
        sources[name] = load_exports(path)
    train, ars = partition_samples(sources["old"], sources["new"], sources["regression"])
    if len(train) != 442 or len(sources["old"]) != 122 or len(ars) != 88:
        raise ValueError("Unexpected frozen data counts")
    names = [row["image"].name for row in train]
    old_names = {row["image"].name for row in sources["old"]}
    natural = make_schedules(names, old_names, balanced=False)
    balanced = make_schedules(names, old_names, balanced=True)
    if set(sum(balanced, [])) != set(names):
        raise ValueError("Balanced schedule leaves some training images unseen")
    args.output.mkdir(parents=True)
    data = make_dataset(train, args.output / "dataset")
    save_json(args.output / "schedules.json", {"natural": natural, "balanced": balanced})
    protocol = {"schema": "ball_controlled_development_round2/v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "arms": {ARMS[0]: {"initial": "round1", "sampling": "natural"},
                         ARMS[1]: {"initial": "round1", "sampling": "balanced"},
                         ARMS[2]: {"initial": "previous", "sampling": "balanced"}},
                "epochs_each": EPOCHS, "samples_per_epoch": 442, "expected_optimizer_steps": 114,
                "balanced_quota": {"old_source_pool": 221, "new_lev_match": 221},
                "balances_individual_matches": False, "lr0": .00005, "warmup_bias_lr": .00005,
                "device": "mps", "batch": 1, "workers": 0, "imgsz": 1920,
                "nbs": 64, "freeze": 10, "seed": 27, "optimizer_resumed": False,
                "round1_report_sha256": digest(baseline_path), "round1_weights_sha256": digest(current),
                "previous_weights_sha256": digest(args.previous),
                "schedule_sha256": digest(args.output / "schedules.json"),
                "dataset_manifest_sha256": digest(args.output / "dataset/manifest.json"),
                "source_manifests": baseline["protocol"]["manifests"],
                "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
                    ROOT / "scripts/train_external_ball_round1.py", ROOT / "scripts/optimize_ball_detector.py",
                    ROOT / "scripts/train_ball_frame_detector.py", ROOT / "src/ball_specialist.py"]},
                "runtime": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
                "evaluation": {"scope": "exposed_development_retests", "independent_test": False,
                               "reference_used_in_inference": False, "used_for_gradient_training": False,
                               "all_arms_frozen_before_scoring": True, "modes": ["full", "tiles"],
                               "imgsz": 1280, "batch": 1, "confidence": .05, "nms_iou": .6},
                "selection_rule": "tiles: core >= old 65/91 and ARS >= round1 62/88; then equal-group recall mean",
                "default_pipeline_changed": False,
                "limitations": ["One seed and a short fixed budget; not a global optimum.",
                                "ARS and core are now exposed development data, not fresh independent tests.",
                                "Only selected positive ball frames; not whole-project accuracy or negative precision.",
                                "Old/new source-pool balance is not per-event or per-match balancing.",
                                "All arms use conservative bias warmup; historical round1 used a different warmup."]}
    save_json(args.output / "protocol.json", protocol)
    with tarfile.open(args.output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in protocol["code_sha256"]:
            archive.add(ROOT / path, arcname=path)
    receipts = {}
    started = time.perf_counter()
    for arm in ARMS:
        initial = args.previous if arm == ARMS[2] else current
        schedules = natural if arm == ARMS[0] else balanced
        print(json.dumps({"starting_arm": arm, "epochs": EPOCHS}), flush=True)
        receipts[arm] = train_arm(initial, data, args.output / arm, schedules, old_names)
        save_json(args.output / "candidates.json", receipts)
    if receipts[ARMS[1]]["sampling"] != receipts[ARMS[2]]["sampling"]:
        raise RuntimeError("Balanced arms did not use identical observed schedules")
    results = {}
    for group, rows in (("arsenal_development", ars), ("core_development", sources["regression"])):
        results[group] = {}
        for arm in ARMS:
            results[group][arm] = {}
            for mode in ("full", "tiles"):
                target = args.output / f"{group}_{arm}_{mode}.json"
                value = evaluate_checkpoint(Path(receipts[arm]["weights"]), rows, target, mode, 1280, "mps", 1)
                value["metrics"] = add_event_metrics(target, rows)
                results[group][arm][mode] = value
                save_json(args.output / "evaluation_progress.json", results)
                print(json.dumps({"group": group, "arm": arm, "mode": mode,
                                  "score": value["metrics"]["deduplicated_frames"]}), flush=True)
                gc.collect()
                torch.mps.empty_cache()
    def hits(group, arm):
        return results[group][arm]["tiles"]["metrics"]["deduplicated_frames"]["top1_iou_0_5_hits"]
    eligible = [arm for arm in ARMS if hits("core_development", arm) >= 65 and hits("arsenal_development", arm) >= 62]
    selected = max(eligible, key=lambda arm: (hits("core_development", arm) / 91 + hits("arsenal_development", arm) / 88)) if eligible else None
    save_json(args.output / "report.json", {"protocol": protocol, "candidates": receipts, "results": results,
              "baseline_results": baseline["results"], "eligible_arms": eligible, "selected_candidate": selected,
              "training_completed": True, "elapsed_seconds": time.perf_counter() - started,
              "default_pipeline_changed": False, "independent_test_evaluated": False,
              "promotion": "candidate_only_pending_video_pipeline_comparison"})
    print(json.dumps({"completed": True, "eligible_arms": eligible, "selected": selected}), flush=True)


if __name__ == "__main__":
    main()
