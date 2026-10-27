"""Sparse human-joint development comparison and masked keypoint-head fitting."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import statistics
import time

TRAIN_VIDEOS = {"05-04", "10-01", "17-08"}
VALIDATION_VIDEOS = {"19-08"}
MODES = (("full", 1280), ("full", 1920), ("person_roi", 1280))
KEYPOINT_CONFIDENCE = .15
PERSON_CONFIDENCE = .10


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_images(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("development_only") or manifest.get("independent_test"):
        raise ValueError("Only the declared development correction set is accepted")
    images = []
    for exported in manifest["exports"]:
        path = manifest_path.parent / exported["image"]
        if digest(path) != exported["image_sha256"]:
            raise ValueError("Source frame changed")
        images.append({"image": path, "frame_id": exported["frame_id"], "video_id": exported["video_id"],
                       "frame_zero_based": exported["frame_zero_based"]})
    if {row["video_id"] for row in images} != TRAIN_VIDEOS | VALIDATION_VIDEOS:
        raise ValueError("Unexpected fixed video split")
    return images, manifest


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.


def roi_window(bbox, width, height):
    x1, y1, x2, y2 = bbox
    dx, dy = .20 * (x2 - x1), .20 * (y2 - y1)
    return [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]


def persons_from_result(result, offset=(0, 0)):
    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    keypoints = result.keypoints.xy.cpu().tolist()
    confidences = result.keypoints.conf.cpu().tolist()
    if not len(boxes) == len(scores) == len(keypoints) == len(confidences):
        raise ValueError("Unaligned pose outputs")
    people = []
    for index, (box, score, points, confidence) in enumerate(zip(boxes, scores, keypoints, confidences)):
        if len(points) != 17 or len(confidence) != 17:
            raise ValueError("Expected COCO17 pose")
        people.append({"candidate_id": index, "bbox": [box[0] + offset[0], box[1] + offset[1], box[2] + offset[0], box[3] + offset[1]],
                       "confidence": score, "keypoints": [[p[0] + offset[0], p[1] + offset[1]] for p in points],
                       "keypoint_confidences": confidence, "source": "full"})
    return people


def predict(model, samples, output, mode, imgsz, device, batch=4):
    import cv2
    if output.exists():
        raise FileExistsError(output)
    rows = []
    started = time.perf_counter()
    for start in range(0, len(samples), batch):
        part = samples[start:start + batch]
        images = [cv2.imread(str(row["image"])) for row in part]
        if any(image is None for image in images):
            raise ValueError("Could not read source frame")
        results = model.predict(images, imgsz=imgsz, conf=PERSON_CONFIDENCE, iou=.7,
                                max_det=30, device=device, verbose=False)
        for sample, image, result in zip(part, images, results):
            height, width = image.shape[:2]
            original = persons_from_result(result)
            people = original
            crop_records = []
            if mode == "person_roi":
                people = []
                for person in original:
                    x1, y1, x2, y2 = roi_window(person["bbox"], width, height)
                    if min(x2 - x1, y2 - y1) < 16:
                        people.append(person)
                        continue
                    inference = model.predict(image[y1:y2, x1:x2], imgsz=imgsz, conf=PERSON_CONFIDENCE,
                                              iou=.7, max_det=10, device=device, verbose=False)[0]
                    refined = persons_from_result(inference, (x1, y1))
                    eligible = [p for p in refined if box_iou(p["bbox"], person["bbox"]) >= .3]
                    chosen = max(eligible, key=lambda p: p["confidence"]) if eligible else person
                    chosen = {**chosen, "candidate_id": person["candidate_id"],
                              "source": "automatic_person_roi" if eligible else "full_fallback"}
                    people.append(chosen)
                    crop_records.append({"original_candidate_id": person["candidate_id"],
                                         "crop_xyxy": [x1, y1, x2, y2], "all_roi_people": refined,
                                         "selected_source": chosen["source"]})
            elif mode != "full":
                raise ValueError("Unknown pose inference mode")
            rows.append({"frame_id": sample["frame_id"], "video_id": sample["video_id"],
                         "frame_zero_based": sample["frame_zero_based"], "width": width, "height": height,
                         "original_full_people": original, "people": people, "automatic_rois": crop_records})
    report = {"mode": mode, "image_size": imgsz, "reference_coordinates_used_for_prediction": False,
              "ids_are_new_frame_local_candidates": True, "rows": rows,
              "seconds": time.perf_counter() - started}
    save(output, report)
    return report


def match_person(target, people):
    points = [xy for xy, use in zip(target["joint_xy_px"], target["coordinate_loss_mask"]) if use]
    if not points:
        return None, {"status": "no_visible_reference_points"}
    ranked = []
    for person in people:
        x1, y1, x2, y2 = person["bbox"]
        dx, dy = .10 * (x2 - x1), .10 * (y2 - y1)
        covered = sum(x1 - dx <= x <= x2 + dx and y1 - dy <= y <= y2 + dy for x, y in points)
        fraction = covered / len(points)
        if fraction >= .5:
            ranked.append((fraction, person["confidence"], person))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]["candidate_id"]))
    if not ranked:
        return None, {"status": "no_reference_person_box_match", "uses_gt_only_for_evaluation_association": True}
    return ranked[0][2], {"status": "matched", "candidate_id": ranked[0][2]["candidate_id"],
                          "reference_points_inside_expanded_box_fraction": ranked[0][0],
                          "eligible_person_count": len(ranked), "uses_gt_only_for_evaluation_association": True,
                          "selection": "box_coverage_then_detection_confidence; no keypoint-error oracle"}


def metrics(points):
    errors = [p["error_px"] for p in points if p["error_px"] is not None]
    count = len(points)
    return {"reference_visible_points": count, "predicted_visible_points": len(errors),
            "within_10px_hits": sum(p["within_10px"] for p in points),
            "within_10px": sum(p["within_10px"] for p in points) / count if count else None,
            "within_2pct_image_height": sum(p["within_2pct_image_height"] for p in points) / count if count else None,
            "mean_error_px_available_predictions_only": statistics.mean(errors) if errors else None,
            "median_error_px_available_predictions_only": statistics.median(errors) if errors else None,
            "missing_predictions_count_as_pck_failures": True}


def evaluate(predictions, targets):
    lookup = {row["frame_id"]: row for row in predictions["rows"]}
    points, associations = [], []
    if set(lookup) != {row["id"] for row in targets}:
        raise ValueError("Prediction and reference frame inventories differ")
    for target in targets:
        row = lookup[target["id"]]
        person, association = match_person(target, row["people"])
        associations.append({"frame_id": target["id"], **association})
        for joint, use in enumerate(target["coordinate_loss_mask"]):
            if not use:
                continue
            reference = target["joint_xy_px"][joint]
            available = person is not None and person["keypoint_confidences"][joint] >= KEYPOINT_CONFIDENCE
            predicted = person["keypoints"][joint] if available else None
            error = math.dist(reference, predicted) if available else None
            points.append({"frame_id": target["id"], "video_id": target["video_id"], "joint_id": joint,
                           "review_status": target["review_status"][joint], "reference_xy": reference,
                           "prediction_xy": predicted, "error_px": error,
                           "within_10px": error is not None and error <= 10,
                           "within_2pct_image_height": error is not None and error <= .02 * row["height"]})
    return {"all": metrics(points),
            "training_videos": metrics([p for p in points if p["video_id"] in TRAIN_VIDEOS]),
            "validation_video": metrics([p for p in points if p["video_id"] in VALIDATION_VIDEOS]),
            "by_video": {video: metrics([p for p in points if p["video_id"] == video]) for video in sorted(TRAIN_VIDEOS | VALIDATION_VIDEOS)},
            "by_review_status": {status: metrics([p for p in points if p["review_status"] == status])
                                 for status in ("visible_corrected", "visible_confirmed")},
            "associations": associations, "points": points,
            "automatic_shooter_role_accuracy": None,
            "scope": "targeted human correction development set with GT box association; not blind pose or role accuracy"}


def yolo_sparse_label(target, pseudo_box, width, height, expand_for_training):
    x1, y1, x2, y2 = pseudo_box
    points = [point for point, use in zip(target["joint_xy_px"], target["coordinate_loss_mask"]) if use]
    if expand_for_training:
        x1 = max(0., min(x1, min(p[0] for p in points) - 2))
        y1 = max(0., min(y1, min(p[1] for p in points) - 2))
        x2 = min(width, max(x2, max(p[0] for p in points) + 2))
        y2 = min(height, max(y2, max(p[1] for p in points) + 2))
    values = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]
    for xy, use in zip(target["joint_xy_px"], target["coordinate_loss_mask"]):
        values.extend([xy[0] / width, xy[1] / height, 2] if use else [0., 0., 0])
    return "0 " + " ".join(f"{value:.9f}" for value in values) + "\n", [x1, y1, x2, y2]


def make_dataset(samples, targets, baseline, output):
    import yaml
    images = {row["frame_id"]: row["image"] for row in samples}
    predictions = {row["frame_id"]: row for row in baseline["rows"]}
    provenance, skipped = [], []
    for target in targets:
        prediction = predictions[target["id"]]
        person, association = match_person(target, prediction["people"])
        if person is None:
            skipped.append({"frame_id": target["id"], "reason": association["status"]})
            continue
        split = "train" if target["video_id"] in TRAIN_VIDEOS else "val"
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images[target["id"]], image_dir / f"{target['id']}.png")
        label, bbox = yolo_sparse_label(target, person["bbox"], prediction["width"], prediction["height"], split == "train")
        (label_dir / f"{target['id']}.txt").write_text(label, encoding="ascii")
        provenance.append({"frame_id": target["id"], "video_id": target["video_id"], "split": split,
                           "bbox": bbox, "bbox_source": "frozen_detector_pseudo_box",
                           "expanded_to_include_manual_points": split == "train", "manual_person_box": False,
                           "coordinate_supervision_points": sum(target["coordinate_loss_mask"]),
                           "unreviewed_and_occluded_encoded_v0": True, "association": association})
    if {row["video_id"] for row in provenance if row["split"] == "train"} != TRAIN_VIDEOS:
        raise ValueError("Cannot train without frozen detections from every training video")
    if not any(row["split"] == "val" for row in provenance):
        raise ValueError("Missing validation video detections")
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": {0: "person"}, "kpt_shape": [17, 3]}
    (output / "dataset.yaml").write_text(yaml.safe_dump(config))
    save(output / "pseudo_box_provenance.json", {"frames": provenance, "skipped": skipped,
                                                "keypoint_visibility_loss_disabled": True,
                                                "native_pose_map_is_not_the_selection_metric": True})
    return output / "dataset.yaml"


def verify_sparse_loss():
    import torch
    from ultralytics.utils.loss import KeypointLoss, v8PoseLoss
    prediction = torch.tensor([[[2., 3.], [900., 900.]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = KeypointLoss(torch.ones(2))(prediction, target, torch.tensor([[True, False]]), torch.tensor([[100.]]))
    loss.backward()
    if not torch.all(prediction.grad[0, 1] == 0) or not torch.any(prediction.grad[0, 0] != 0):
        raise ValueError("Installed pose loss does not mask unreviewed coordinates")
    source = inspect.getsource(v8PoseLoss)
    if "loss[2] *= self.hyp.kobj" not in source or "gt_kpt[..., 2] != 0" not in source:
        raise ValueError("Installed pose loss must be reviewed before sparse fitting")
    return {"masked_coordinate_gradient_verified": True, "visibility_gain_separate_verified": True,
            "loss_source_sha256": hashlib.sha256(source.encode()).hexdigest(), "loss_source": source}


def train_sparse(weights, data, output, device, epochs, learning_rate=.0001, freeze_all_bn=False):
    import torch
    from ultralytics import YOLO
    loss_check = verify_sparse_loss()
    save(output / "sparse_loss_verification.json", loss_check)
    model = YOLO(str(weights))
    head_index = len(model.model.model) - 1
    prefix = f"model.{head_index}.cv4."
    if not any(name.startswith(prefix) for name, _ in model.model.named_parameters()):
        raise ValueError("Cannot find isolated keypoint head")

    def freeze_parameters(trainer):
        trainable, frozen = [], []
        for name, parameter in trainer.model.named_parameters():
            allowed = name.startswith(prefix)
            parameter.requires_grad_(allowed)
            (trainable if allowed else frozen).append(name)
        save(output / "parameter_freeze.json", {"trainable": trainable, "frozen": frozen,
                                               "trainable_scope": prefix, "frozen_batchnorm_eval": True,
                                               "all_batchnorm_statistics_frozen": freeze_all_bn})

    def freeze_statistics(trainer):
        for name, module in trainer.model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and (freeze_all_bn or not name.startswith(prefix)):
                module.eval()

    model.add_callback("on_train_start", freeze_parameters)
    model.add_callback("on_train_batch_start", freeze_statistics)
    model.train(data=str(data), epochs=epochs, imgsz=1280, batch=2, device=device, workers=2,
                project=str(output), name="fit", exist_ok=False, optimizer="AdamW", lr0=learning_rate, lrf=.1,
                weight_decay=.0005, freeze=head_index, warmup_epochs=2, warmup_bias_lr=learning_rate,
                nbs=16, patience=epochs,
                box=0., cls=0., dfl=0., kobj=0., pose=12.,
                mosaic=0., mixup=0., copy_paste=0., scale=.1, translate=.05,
                fliplr=0., flipud=0., degrees=0., shear=0., perspective=0., close_mosaic=0,
                seed=27, deterministic=True, amp=False, cache=False, plots=False, val=False,
                save=True, verbose=False)
    return output / "fit/weights/last.pt"


def run(args):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    samples, manifest = load_images(args.manifest)
    if digest(args.weights) != manifest["model_binding"]["declared_sha256"]:
        raise ValueError("Pose baseline weight hash differs from reviewed source")
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(__file__),
                "manifest_sha256": digest(args.manifest), "weights_sha256": digest(args.weights),
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "inference_configurations": MODES, "train_videos": sorted(TRAIN_VIDEOS),
                "validation_videos": sorted(VALIDATION_VIDEOS), "epochs": args.epochs,
                "learning_rate": .0001, "warmup_bias_learning_rate": .0001, "nominal_batch_size": 16,
                "primary_metric": "visible human joint within 10 original-image pixels; missing person/joint fails",
                "secondary_metric": "error <= 2 percent image height; not torso-normalized standard PCK",
                "person_confidence": PERSON_CONFIDENCE, "keypoint_confidence": KEYPOINT_CONFIDENCE,
                "association": "GT point containment in 10-percent-expanded predicted person box, then detection confidence",
                "person_role_prediction_evaluated": False, "independent_test": False,
                "checkpoint_selection": "fixed last epoch; native sparse-reference OKS/mAP does not select candidate",
                "parameter_updates": "cv4 keypoint branch only; frozen BN statistics; box/cls/dfl/kobj gains zero",
                "prediction_uses_manual_coordinates": False, "target_source_not_loaded_until_baseline_predictions_saved": True}
    save(args.output / "protocol.json", protocol)
    model = YOLO(str(args.weights))
    predictions = {}
    for mode, size in MODES:
        name = f"baseline_{mode}_{size}"
        predictions[name] = predict(model, samples, args.output / f"{name}_predictions.json", mode, size, args.device)
    target_path = args.manifest.parent / "frame_targets.json"
    if digest(target_path) != manifest["artifact_sha256"]["frame_targets.json"]:
        raise ValueError("Manual targets changed")
    targets = json.loads(target_path.read_text(encoding="utf-8"))["frames"]
    scores = {}
    for name, prediction in predictions.items():
        score = evaluate(prediction, targets)
        save(args.output / f"{name}_evaluation.json", score)
        scores[name] = {key: value for key, value in score.items() if key not in {"points", "associations"}}
        print(json.dumps({"name": name, "all": score["all"], "validation": score["validation_video"]}), flush=True)
    save(args.output / "comparison.json", scores)
    if args.inference_only:
        return
    data = make_dataset(samples, targets, predictions["baseline_full_1280"], args.output / "dataset")
    del model
    torch.cuda.empty_cache()
    checkpoint = train_sparse(args.weights, data, args.output, args.device, args.epochs)
    trained = YOLO(str(checkpoint))
    for mode, size in MODES:
        name = f"trained_last_{mode}_{size}"
        prediction = predict(trained, samples, args.output / f"{name}_predictions.json", mode, size, args.device)
        score = evaluate(prediction, targets)
        save(args.output / f"{name}_evaluation.json", score)
        scores[name] = {key: value for key, value in score.items() if key not in {"points", "associations"}}
        print(json.dumps({"name": name, "all": score["all"], "validation": score["validation_video"]}), flush=True)
    save(args.output / "comparison.json", scores)
    save(args.output / "completion.json", {"checkpoint": str(checkpoint), "sha256": digest(checkpoint),
                                          "trained_parameters": "cv4 only", "default_pipeline_changed": False,
                                          "independent_test_evaluated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("Epoch count must be positive")
    run(args)


if __name__ == "__main__":
    main()
