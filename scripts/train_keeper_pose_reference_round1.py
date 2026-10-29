"""Fixed sparse keeper-pose development fit with assisted-label strata."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_manual_pose import box_iou, digest, persons_from_result, save

VISIBLE_STATUSES = {"visible_confirmed", "visible_corrected"}
BULK_ORIGIN = "user_message_bulk_reference_confirmation"
PERSON_CONFIDENCE = .10
KEYPOINT_CONFIDENCE = .15
ROI_MARGIN = .25
EPOCHS = 10
IMAGE_SIZE = 640
LEARNING_RATE = .00002
TRAIN_VIDEOS = {"05-04", "07-08", "10-01", "12-02", "17-08", "19-08"}
SUBJECT_DEVELOPMENT_VIDEOS = {"02-05", "02-06"}
EXTERNAL_DEVELOPMENT_VIDEOS = {f"match_{index}" for index in range(17, 25)}
BASELINE_SHA256 = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"


def crop_window(bbox, width, height):
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        raise ValueError("Finite keeper bbox required")
    x1, y1, x2, y2 = bbox
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("Keeper bbox outside original image")
    dx, dy = ROI_MARGIN * (x2 - x1), ROI_MARGIN * (y2 - y1)
    return [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]


def validate_partition(samples):
    splits = {s["split"] for s in samples}
    if "train" not in splits or len(splits) < 2:
        raise ValueError("Both train and development partitions are required")
    seen = set()
    bindings = {key: {} for key in ("partition_group", "event_id", "source_sha256", "image_sha256")}
    for sample in samples:
        if sample["id"] in seen:
            raise ValueError("Duplicate frame identity")
        seen.add(sample["id"])
        for key, lookup in bindings.items():
            value = sample.get(key)
            if not value:
                raise ValueError(f"Missing grouping/binding: {key}")
            if value in lookup and lookup[value] != sample["split"]:
                raise ValueError(f"{key} crosses train and development")
            lookup[value] = sample["split"]
    for split in sorted(splits):
        if not any(any(s["coordinate_loss_mask"]) for s in samples if s["split"] == split):
            raise ValueError(f"No visible reference points in {split}")


def sparse_crop_label(target, window):
    x0, y0, x3, y3 = window
    width, height = x3 - x0, y3 - y0
    if min(width, height) < 2:
        raise ValueError("Empty keeper crop")
    xy, mask, statuses = (target[key] for key in ("joint_xy_px", "coordinate_loss_mask", "review_status"))
    if not len(xy) == len(mask) == len(statuses) == 17:
        raise ValueError("COCO17 coordinate/mask/status arrays required")
    bx1, by1, bx2, by2 = target["bbox"]
    if not x0 <= bx1 < bx2 <= x3 or not y0 <= by1 < by2 <= y3:
        raise ValueError("Crop must contain the existing keeper bbox")
    values = [(bx1 + bx2 - 2 * x0) / (2 * width),
              (by1 + by2 - 2 * y0) / (2 * height),
              (bx2 - bx1) / width, (by2 - by1) / height]
    used, excluded = [], []
    for joint, (point, use, status) in enumerate(zip(xy, mask, statuses)):
        if use:
            if status not in VISIBLE_STATUSES or point is None or len(point) != 2 or not all(
                    isinstance(v, (float, int)) and math.isfinite(v) for v in point):
                raise ValueError("Coordinate loss requires a visible, finite reviewed point")
            if x0 <= point[0] < x3 and y0 <= point[1] < y3:
                values.extend([(point[0] - x0) / width, (point[1] - y0) / height, 2])
                used.append(joint)
                continue
            excluded.append({"joint_id": joint, "reason": "reference_outside_fixed_crop"})
        values.extend([0., 0., 0.])
    return "0 " + " ".join(f"{v:.9f}" for v in values) + "\n", used, excluded


def point_metrics(points):
    available = [p["error_px"] for p in points if p["error_px"] is not None]
    hits = sum(p["within_10px"] for p in points)
    normalized_hits = sum(p["within_10pct_box_diagonal"] for p in points)
    return {"denominator": len(points), "available_predictions": len(available),
            "within_10px_hits": hits, "within_10px": hits / len(points) if points else None,
            "within_10pct_box_diagonal_hits": normalized_hits,
            "within_10pct_box_diagonal": normalized_hits / len(points) if points else None,
            "mean_error_px_available_only": statistics.mean(available) if available else None,
            "median_error_px_available_only": statistics.median(available) if available else None,
            "missing_predictions_count_as_failures": True}


def label_stratum(status, origin):
    if origin == BULK_ORIGIN:
        return "bulk_reference_confirmation"
    return "original_" + status


def evaluate_predictions(prediction, samples):
    rows = {r["id"]: r for r in prediction["rows"]}
    if len(rows) != len(prediction["rows"]) or set(rows) != {s["id"] for s in samples}:
        raise ValueError("Prediction frame inventory differs from reference inventory")
    points, associations = [], []
    for sample in samples:
        eligible = [(box_iou(sample["bbox"], p["bbox"]), p) for p in rows[sample["id"]]["people"]]
        eligible = [(iou, p) for iou, p in eligible if iou >= .3]
        eligible.sort(key=lambda item: (-item[0], -item[1]["confidence"], item[1]["candidate_id"]))
        person = eligible[0][1] if eligible else None
        associations.append({"id": sample["id"], "matched": person is not None,
                             "iou": eligible[0][0] if eligible else None,
                             "association_uses_reference_bbox_for_evaluation_only": True})
        box = sample["bbox"]
        scale = math.hypot(box[2] - box[0], box[3] - box[1])
        for joint, use in enumerate(sample["coordinate_loss_mask"]):
            if not use:
                continue
            status, origin = sample["review_status"][joint], sample["annotation_origin"][joint]
            if status not in VISIBLE_STATUSES:
                raise ValueError("Non-visible reference cannot enter PCK denominator")
            available = person is not None and person["keypoint_confidences"][joint] >= KEYPOINT_CONFIDENCE
            predicted = person["keypoints"][joint] if available else None
            if predicted is not None and not all(math.isfinite(v) for v in predicted):
                raise ValueError("Non-finite pose prediction")
            error = math.dist(sample["joint_xy_px"][joint], predicted) if available else None
            points.append({"id": sample["id"], "video_id": sample["video_id"],
                           "event_id": sample["event_id"], "source_group": sample["source_group"],
                           "split": sample["split"], "joint_id": joint, "review_status": status,
                           "annotation_origin": origin, "stratum": label_stratum(status, origin),
                           "box_point_prompt": "flagged" if sample.get("declared_box_point_conflict", [False] * 17)[joint] else "not_flagged",
                           "reference_xy": sample["joint_xy_px"][joint], "prediction_xy": predicted,
                           "error_px": error, "within_10px": error is not None and error <= 10,
                           "within_10pct_box_diagonal": error is not None and error <= .1 * scale})
    report = {"all": point_metrics(points), "points": points, "associations": associations,
              "independent_test": False, "automatic_keeper_role_accuracy": None,
              "scope": "assisted-reference pose development; not overall project accuracy"}
    for name in ("split", "stratum", "event_id", "source_group", "box_point_prompt"):
        report["by_" + name] = {key: point_metrics([p for p in points if p[name] == key])
                                for key in sorted({p[name] for p in points})}
    report["by_split_and_stratum"] = {
        split: {stratum: point_metrics([p for p in points if p["split"] == split and p["stratum"] == stratum])
                for stratum in sorted({p["stratum"] for p in points})}
        for split in sorted({s["split"] for s in samples})}
    return report


def paired_changes(before, after):
    left = {(p["id"], p["joint_id"]): p for p in before["points"]}
    right = {(p["id"], p["joint_id"]): p for p in after["points"]}
    if len(left) != len(before["points"]) or len(right) != len(after["points"]) or left.keys() != right.keys():
        raise ValueError("Paired point inventories differ")
    counts = Counter()
    groups = {name: defaultdict(Counter) for name in ("event_id", "split", "stratum")}
    for key, old in left.items():
        new = right[key]
        if any(old[k] != new[k] for k in ("reference_xy", "split", "stratum", "event_id")):
            raise ValueError("Paired reference identity changed")
        a, b = old["within_10px"], new["within_10px"]
        change = "both_correct" if a and b else "regressed" if a else "corrected" if b else "both_incorrect"
        counts[change] += 1
        for name, grouped in groups.items():
            grouped[old[name]][change] += 1
    names = ("corrected", "regressed", "both_correct", "both_incorrect")
    return {"all": {k: counts[k] for k in names},
            **{"by_" + name: {group: {k: values[k] for k in names} for group, values in sorted(grouped.items())}
               for name, grouped in groups.items()}}


def verify_sparse_loss():
    import torch
    from ultralytics.utils.loss import KeypointLoss, v8PoseLoss
    criterion = KeypointLoss(torch.ones(2))
    gradients = []
    for mask in ([True, False], [True, True]):
        prediction = torch.tensor([[[2., 3.], [2., 3.]]], requires_grad=True)
        loss = criterion(prediction, torch.zeros_like(prediction), torch.tensor([mask]), torch.tensor([[100.]]))
        loss.backward()
        gradients.append(prediction.grad.detach().clone())
    if not torch.all(gradients[0][0, 1] == 0) or not torch.any(gradients[1][0, 1] != 0):
        raise RuntimeError("Unknown-coordinate masking failed its non-saturated positive control")
    source = inspect.getsource(v8PoseLoss)
    if "loss[2] *= self.hyp.kobj" not in source or "gt_kpt[..., 2] != 0" not in source:
        raise RuntimeError("Installed sparse pose loss needs review")
    actual = v8PoseLoss.__new__(v8PoseLoss)
    actual.keypoint_loss = criterion
    actual.bce_pose = torch.nn.BCEWithLogitsLoss()
    prediction = torch.tensor([[[[2., 3., .5], [2., 3., .5]]]], requires_grad=True)
    target = torch.tensor([[[0., 0., 2.], [0., 0., 0.]]])
    coordinate, visibility = actual.calculate_keypoints_loss(
        torch.tensor([[True]]), torch.tensor([[0]]), target, torch.tensor([[0]]),
        torch.ones(1, 1), torch.tensor([[[0., 0., 10., 10.]]]), prediction)
    (coordinate + 0. * visibility).backward()
    if not torch.any(prediction.grad[0, 0, 0, :2] != 0) or not torch.all(prediction.grad[0, 0, 1, :2] == 0) or not torch.all(prediction.grad[..., 2] == 0):
        raise RuntimeError("Installed pose objective applies unknown-coordinate or visibility supervision")
    return {"masked_coordinate_gradient_zero": True, "same_coordinate_unmasked_gradient_nonzero": True,
            "visibility_loss_disabled_by_kobj_zero": True,
            "actual_pose_objective_unknown_coordinate_and_visibility_gradients_zero": True,
            "loss_source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def load_samples(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("source_pixels_verified") or not manifest.get("development_only") or manifest.get("independent_ground_truth"):
        raise ValueError("Verified source pixels and assisted development declaration required")
    if manifest.get("full_frame_detection_training_allowed") or manifest.get("unknown_joint_coordinates_synthesized"):
        raise ValueError("Partial keeper references cannot supervise full-frame detection or invented joints")
    samples = []
    for exported in manifest["exports"]:
        sample = dict(exported)
        video = sample["video_id"]
        if video in TRAIN_VIDEOS:
            split, group = "train", sample.get("subject_group", video.split("-")[0])
        elif video in SUBJECT_DEVELOPMENT_VIDEOS:
            split, group = "development_subject_02", "mendeley_subject_02"
        elif video in EXTERNAL_DEVELOPMENT_VIDEOS:
            split, group = "development_external_match", "worldcup_match_17_to_24"
        else:
            raise ValueError(f"Unexpected keeper source: {video}")
        sample.update(split=split, partition_group=group)
        image = manifest_path.parent / sample["image"]
        if digest(image) != sample["image_sha256"]:
            raise ValueError("Keeper source image hash mismatch")
        sample["image"] = image
        # The importer retains original image coordinates; training transforms only the crop.
        if not sample.get("id"):
            sample["id"] = sample.get("frame_id")
        if not sample["id"]:
            raise ValueError("Missing source frame identity")
        window = crop_window(sample["bbox"], sample["width"], sample["height"])
        if sample.get("roi_original_pixels") != window or sample.get("roi_margin_fraction_of_bbox") != ROI_MARGIN:
            raise ValueError("Imported fixed ROI differs from the declared experiment")
        sample["roi_image"] = manifest_path.parent / sample["roi_image"]
        if digest(sample["roi_image"]) != sample["roi_image_sha256"]:
            raise ValueError("Keeper ROI image hash mismatch")
        sparse_crop_label(sample, window)
        if len(sample["annotation_origin"]) != 17:
            raise ValueError("COCO17 annotation origins required")
        samples.append(sample)
    expected = TRAIN_VIDEOS | SUBJECT_DEVELOPMENT_VIDEOS | EXTERNAL_DEVELOPMENT_VIDEOS
    if {s["video_id"] for s in samples} != expected:
        raise ValueError("Fixed 16-video experiment inventory is incomplete")
    validate_partition(samples)
    return samples, manifest


def read_image(sample):
    import cv2
    image = cv2.imread(str(sample["image"]))
    if image is None or image.shape[:2] != (sample["height"], sample["width"]):
        raise ValueError("Source image missing or dimensions changed")
    return image


def predict_full(model, samples, device, batch):
    rows = []
    started = time.perf_counter()
    for start in range(0, len(samples), batch):
        part = samples[start:start + batch]
        results = model.predict([read_image(s) for s in part], imgsz=1280, conf=PERSON_CONFIDENCE,
                                iou=.7, max_det=20, device=device, verbose=False)
        if len(results) != len(part):
            raise RuntimeError("Full-frame inference lost frames")
        rows.extend({"id": sample["id"], "people": persons_from_result(result)}
                    for sample, result in zip(part, results))
    return {"mode": "full", "rows": rows, "seconds": time.perf_counter() - started,
            "reference_labels_used_for_prediction": False}


def predict_rois(model, samples, frozen_full, mode, device, batch):
    if mode not in {"automatic_roi", "oracle_roi"}:
        raise ValueError("Unknown ROI inference mode")
    full = {row["id"]: row["people"] for row in frozen_full["rows"]}
    if set(full) != {s["id"] for s in samples}:
        raise ValueError("Frozen detector frame inventory mismatch")
    rows, started = [], time.perf_counter()
    for sample in samples:
        image = read_image(sample)
        seeds = full[sample["id"]] if mode == "automatic_roi" else [
            {"candidate_id": 0, "bbox": sample["bbox"], "confidence": 1.}]
        windows = [crop_window(p["bbox"], sample["width"], sample["height"]) for p in seeds]
        people, receipts = [], []
        for start in range(0, len(seeds), batch):
            part = seeds[start:start + batch]
            crops = windows[start:start + batch]
            results = model.predict([image[y1:y2, x1:x2] for x1, y1, x2, y2 in crops],
                                    imgsz=IMAGE_SIZE, conf=PERSON_CONFIDENCE, iou=.7,
                                    max_det=10, device=device, verbose=False)
            if len(results) != len(part):
                raise RuntimeError("ROI inference lost crops")
            for seed, window, result in zip(part, crops, results):
                candidates = persons_from_result(result, window[:2])
                eligible = [p for p in candidates if box_iou(p["bbox"], seed["bbox"]) >= .3]
                chosen = max(eligible, key=lambda p: p["confidence"]) if eligible else None
                if chosen is not None:
                    people.append({**chosen, "candidate_id": seed["candidate_id"]})
                receipts.append({"candidate_id": seed["candidate_id"], "crop_xyxy": window,
                                 "seed_bbox": seed["bbox"], "refined_pose_available": chosen is not None})
        rows.append({"id": sample["id"], "people": people, "roi_receipts": receipts})
    return {"mode": mode, "rows": rows, "seconds": time.perf_counter() - started,
            "reference_labels_used_for_prediction": mode == "oracle_roi",
            "oracle_roi_is_diagnostic_only": mode == "oracle_roi",
            "automatic_roi_generator": "frozen_pretrained_full_frame_person_boxes",
            "automatic_keeper_role_assignment_evaluated": False,
            "full_detector_seconds_not_included": frozen_full["seconds"]}


def make_dataset(samples, output):
    import cv2
    import yaml
    image_dir, label_dir = output / "images/train", output / "labels/train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    records = []
    for index, sample in enumerate(s for s in samples if s["split"] == "train"):
        window = crop_window(sample["bbox"], sample["width"], sample["height"])
        label, used, excluded = sparse_crop_label(sample, window)
        record = {"id": sample["id"], "video_id": sample["video_id"], "crop_xyxy": window,
                  "supervised_joints": used, "excluded_joints": excluded,
                  "source_image_sha256": sample["image_sha256"]}
        if used:
            x1, y1, x2, y2 = window
            image = cv2.imread(str(sample["roi_image"]))
            if image is None or image.shape[:2] != (y2 - y1, x2 - x1):
                raise ValueError("Imported ROI dimensions differ from original coordinates")
            name = f"keeper_{index:05d}"
            destination = image_dir / (name + ".png")
            if not cv2.imwrite(str(destination), image):
                raise OSError("Could not write keeper ROI")
            (label_dir / (name + ".txt")).write_text(label, encoding="ascii")
            record["roi_image_sha256"] = digest(destination)
            record["training_image"] = str(destination.relative_to(output))
        else:
            record["skipped_reason"] = "no_visible_coordinate_supervision_in_fixed_roi"
        records.append(record)
    if not any(r["supervised_joints"] for r in records):
        raise ValueError("No training coordinates remain in fixed keeper ROIs")
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/train",
              "names": {0: "person"}, "kpt_shape": [17, 3]}
    data = output / "dataset.yaml"
    data.write_text(yaml.safe_dump(config), encoding="utf-8")
    save(output / "provenance.json", {"frames": records, "roi_margin": ROI_MARGIN,
                                     "input_training_frames": len(records),
                                     "exported_training_images": sum(bool(r["supervised_joints"]) for r in records),
                                     "skipped_training_frames": sum(not r["supervised_joints"] for r in records),
                                     "coordinate_supervision_points": sum(len(r["supervised_joints"]) for r in records),
                                     "native_validation_uses_training_images_only": True,
                                     "native_sparse_pose_map_not_a_selection_metric": True,
                                     "unknown_coordinate_loss_masked": True, "visibility_loss_disabled": True})
    return data


def tensor_digests(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def trainable_parameter_snapshot(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def maximum_trainable_parameter_change(model, initial):
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not initial or current.keys() != initial.keys():
        raise RuntimeError("Trainable parameter inventory changed")
    peak, peak_name = 0., None
    for name, parameter in current.items():
        delta = float((parameter.detach().cpu() - initial[name]).abs().max())
        if not math.isfinite(delta):
            raise RuntimeError(f"Non-finite updated parameter: {name}")
        if delta > peak:
            peak, peak_name = delta, name
    return peak, peak_name


def verify_loaded_dataset(dataset, provenance):
    expected_images = provenance["exported_training_images"]
    expected_points = provenance["coordinate_supervision_points"]
    actual_images = len(dataset)
    actual_points = sum(int((label["keypoints"][..., 2] > 0).sum()) for label in dataset.labels)
    if actual_images != expected_images or actual_points != expected_points:
        raise RuntimeError(f"Loader changed the training inventory: images {actual_images}/{expected_images}, "
                           f"visible coordinates {actual_points}/{expected_points}")
    return {"loaded_training_images": actual_images, "loaded_visible_coordinates": actual_points,
            "loader_preserved_exported_training_inventory": True}


def train_sparse(weights, dataset, output, device, batch, workers):
    import torch
    import ultralytics
    from ultralytics import YOLO
    save(output / "sparse_loss_verification.json", verify_sparse_loss())
    model = YOLO(str(weights))
    head = len(model.model.model) - 1
    prefix = f"model.{head}.cv4."
    if not any(name.startswith(prefix) for name, _ in model.model.named_parameters()):
        raise ValueError("Isolated cv4 keypoint branch unavailable")
    evidence = {"optimizer_steps": 0, "completed_epochs": [], "trainable_scope": prefix,
                "all_batchnorm_statistics_frozen": True, "steps_with_nonzero_supervised_gradient": 0,
                "maximum_absolute_supervised_gradient": 0., "batches_with_nonzero_pose_loss": 0}
    started = time.perf_counter()
    provenance = json.loads((dataset.parent / "provenance.json").read_text(encoding="utf-8"))

    def freeze_and_record(trainer):
        for name, parameter in trainer.model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
        expected_device = "mps" if device == "mps" else "cpu" if device == "cpu" else "cuda"
        if trainer.device.type != expected_device or trainer.batch_size != batch:
            raise RuntimeError("Training device/batch differs from protocol")
        evidence["actual_device"] = str(trainer.device)
        evidence["actual_batch"] = trainer.batch_size
        evidence.update(verify_loaded_dataset(trainer.train_loader.dataset, provenance))
        evidence["before"] = tensor_digests(trainer.model)
        initial = trainable_parameter_snapshot(trainer.model)
        evidence["tracked_trainable_parameter_count"] = len(initial)
        evidence["maximum_delta_parameter"] = None
        evidence["maximum_absolute_parameter_delta"] = 0.

        def record_gradient(_optimizer, _args, _kwargs):
            gradients = [float(p.grad.detach().abs().max().cpu()) for p in trainer.model.parameters()
                         if p.requires_grad and p.grad is not None]
            if not gradients or not all(math.isfinite(value) for value in gradients):
                raise RuntimeError("Missing or non-finite trainable gradients")
            peak = max(gradients)
            evidence["maximum_absolute_supervised_gradient"] = max(evidence["maximum_absolute_supervised_gradient"], peak)
            evidence["steps_with_nonzero_supervised_gradient"] += int(peak > 0)

        def record_step(_optimizer, _args, _kwargs):
            evidence["optimizer_steps"] += 1
            delta, name = maximum_trainable_parameter_change(trainer.model, initial)
            if delta > evidence["maximum_absolute_parameter_delta"]:
                evidence["maximum_absolute_parameter_delta"] = delta
                evidence["maximum_delta_parameter"] = name

        trainer.optimizer.register_step_pre_hook(record_gradient)
        trainer.optimizer.register_step_post_hook(record_step)

    def freeze_statistics(trainer):
        for module in trainer.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def verify_batch(trainer):
        if trainer.batch_size != batch or getattr(trainer, "_oom_retries", 0):
            raise RuntimeError("Unexpected adaptive batch or OOM retry")
        if not all(math.isfinite(float(value.detach().cpu())) for value in trainer.loss_items.values()):
            raise RuntimeError("Non-finite training loss")
        evidence["batches_with_nonzero_pose_loss"] += int(float(trainer.loss_items["pose_loss"].detach().cpu()) > 0)

    def record_epoch(trainer):
        after = tensor_digests(trainer.model)
        changed = [name for name in after if after[name] != evidence["before"][name]]
        bad = [name for name in changed if not name.startswith(prefix) or
               any(part in name for part in ("running_mean", "running_var", "num_batches_tracked"))]
        if bad:
            raise RuntimeError(f"Frozen parameters or BN statistics changed: {bad[:5]}")
        evidence["completed_epochs"].append(trainer.epoch + 1)
        evidence.update(changed_state_names=changed, frozen_state_unchanged=True,
                        losses={name: float(value.detach().cpu()) for name, value in trainer.tloss.items()},
                        elapsed_seconds=time.perf_counter() - started)
        save(output / "training_progress.json", evidence)
        if trainer.epoch + 1 == EPOCHS:
            if not evidence["optimizer_steps"] or not changed or not evidence["steps_with_nonzero_supervised_gradient"] or not evidence["batches_with_nonzero_pose_loss"] or evidence["maximum_absolute_parameter_delta"] <= 0:
                raise RuntimeError("No optimizer update was demonstrated")
            # Retain the actual final FP32 parameters, before EMA/FP16 checkpoint conversion.
            exported_model = deepcopy(trainer.model).cpu().float()
            exported_model.criterion = None
            checkpoint = {"model": exported_model, "ema": None,
                          "epoch": trainer.epoch, "optimizer": None, "train_args": vars(trainer.args),
                          "version": ultralytics.__version__, "date": datetime.now(timezone.utc).isoformat()}
            torch.save(checkpoint, output / "last_fp32.pt")
            save(output / "parameter_update_evidence.json", evidence)
        print(json.dumps({"keeper_epoch": trainer.epoch + 1, "optimizer_steps": evidence["optimizer_steps"],
                          "losses": evidence["losses"]}), flush=True)

    model.add_callback("on_train_start", freeze_and_record)
    model.add_callback("on_train_batch_start", freeze_statistics)
    model.add_callback("on_train_batch_end", verify_batch)
    model.add_callback("on_model_save", record_epoch)
    model.train(data=str(dataset), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=batch, device=device,
                workers=workers, project=str(output), name="fit", exist_ok=False, optimizer="AdamW",
                lr0=LEARNING_RATE, lrf=.1, weight_decay=.0005, freeze=head, warmup_epochs=2,
                warmup_bias_lr=LEARNING_RATE, nbs=16, patience=EPOCHS,
                box=0., cls=0., dfl=0., kobj=0., pose=12.,
                mosaic=0., mixup=0., copy_paste=0., scale=.1, translate=.05,
                fliplr=0., flipud=0., degrees=0., shear=0., perspective=0., close_mosaic=0,
                seed=28, deterministic=True, amp=False, cache=False, plots=False, val=False,
                save=True, verbose=False)
    if evidence["completed_epochs"] != list(range(1, EPOCHS + 1)):
        raise RuntimeError("Training did not complete the fixed epoch budget")
    return output / "last_fp32.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--evaluation-batch", type=int, default=4)
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if min(args.batch, args.evaluation_batch) < 1 or args.workers < 0:
        parser.error("Batch sizes must be positive and workers nonnegative")
    if args.output.exists():
        raise FileExistsError("Use a new keeper experiment directory")
    if digest(args.weights) != BASELINE_SHA256:
        raise ValueError("This experiment requires the frozen original COCO yolo11m-pose checkpoint")
    samples, _manifest = load_samples(args.manifest)
    args.output.mkdir(parents=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.output / ".runtime-cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(args.output / ".runtime-cache/matplotlib")
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device is unavailable")
        torch.mps.set_per_process_memory_fraction(.65)
    protocol = {"schema": "keeper_sparse_reference_round/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": digest(__file__), "manifest_sha256": digest(args.manifest),
                "weights_sha256": digest(args.weights), "torch": torch.__version__,
                "ultralytics": ultralytics.__version__, "epochs": EPOCHS,
                "imgsz": IMAGE_SIZE, "learning_rate": LEARNING_RATE, "roi_margin": ROI_MARGIN,
                "batch": args.batch, "device": args.device, "seed": 28,
                "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
                "train_videos": sorted(TRAIN_VIDEOS), "subject_development_videos": sorted(SUBJECT_DEVELOPMENT_VIDEOS),
                "external_development_videos": sorted(EXTERNAL_DEVELOPMENT_VIDEOS),
                "split_counts": dict(Counter(s["split"] for s in samples)),
                "split_frames_with_coordinate_supervision": {
                    split: sum(any(s["coordinate_loss_mask"]) for s in samples if s["split"] == split)
                    for split in sorted({s["split"] for s in samples})},
                "subject_development_may_share_venue_or_keeper_with_training": True,
                "independent_test": False, "assisted_labels": True,
                "checkpoint_selection": "fixed final epoch raw FP32 weights; no development selection",
                "training_scope": "cv4 only; all BN frozen; box/cls/dfl/kobj gains zero",
                "primary_metric": "original-pixel PCK10 with missing predictions counted as failures",
                "automatic_roi": "frozen pretrained full-frame detections; no manual ROI or point input",
                "oracle_roi": "manual bbox diagnostic only; not deployment accuracy",
                "default_pipeline_changed": False}
    save(args.output / "protocol.json", protocol)
    save(args.output / "sparse_loss_verification.json", verify_sparse_loss())
    model = YOLO(str(args.weights))
    full = predict_full(model, samples, args.device, args.evaluation_batch)
    save(args.output / "pretrained_full_predictions.json", full)
    scores = {"pretrained_full": evaluate_predictions(full, samples)}
    for mode in ("automatic_roi", "oracle_roi"):
        predictions = predict_rois(model, samples, full, mode, args.device, args.evaluation_batch)
        name = "pretrained_" + mode
        save(args.output / (name + "_predictions.json"), predictions)
        scores[name] = evaluate_predictions(predictions, samples)
    for name, score in scores.items():
        save(args.output / (name + "_evaluation.json"), score)
        print(json.dumps({"keeper_evaluation": name, "by_split": score["by_split"]}), flush=True)
    if not args.inference_only:
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
        data = make_dataset(samples, args.output / "dataset")
        checkpoint = train_sparse(args.weights, data, args.output, args.device, args.batch, args.workers)
        model = YOLO(str(checkpoint))
        for mode in ("automatic_roi", "oracle_roi"):
            predictions = predict_rois(model, samples, full, mode, args.device, args.evaluation_batch)
            name = "trained_" + mode
            save(args.output / (name + "_predictions.json"), predictions)
            scores[name] = evaluate_predictions(predictions, samples)
            save(args.output / (name + "_evaluation.json"), scores[name])
            save(args.output / (mode + "_paired_changes.json"), paired_changes(scores["pretrained_" + mode], scores[name]))
            print(json.dumps({"keeper_evaluation": name, "by_split": scores[name]["by_split"]}), flush=True)
    save(args.output / "report.json", {"protocol": protocol,
         "evaluations": {name: {k: v for k, v in score.items() if k not in {"points", "associations"}}
                         for name, score in scores.items()},
         "trained_checkpoint": str(args.output / "last_fp32.pt") if not args.inference_only else None,
         "trained_checkpoint_sha256": digest(args.output / "last_fp32.pt") if not args.inference_only else None,
         "independent_test": False, "overall_project_accuracy": None})


if __name__ == "__main__":
    main()
