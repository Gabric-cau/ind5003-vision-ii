"""Evaluate label-free rotated person proposals and three-angle keeper pose inference."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import os
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_keeper_tiled_proposals import (
    MAX_PEOPLE_PER_VIEW,
    PERSON_NMS_IOU,
    SEED_IMAGE_SIZE,
    TILE_EDGE,
    TILE_OVERLAP,
    inference_frames,
    proposal_coverage,
    suppress_duplicates,
    validate_people,
    verify_inference_device,
)
from scripts.optimize_manual_pose import box_iou, digest, persons_from_result, save
from scripts.train_keeper_pose_reference_round1 import (
    IMAGE_SIZE,
    KEYPOINT_CONFIDENCE,
    PERSON_CONFIDENCE,
    ROI_MARGIN,
    crop_window,
    evaluate_predictions,
    load_samples,
    paired_changes,
    point_metrics,
    read_image,
)
from src.ball_specialist import tile_windows
from src.goalkeeper_role_association import unrotate_pose

ANGLES = (0, 90, 270)


def rotate_crop(crop, angle):
    import cv2
    if angle == 0:
        return crop
    if angle == 90:
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    if angle == 270:
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Only 0, 90 and 270 degree rotations are supported")


def map_rotated_people(result, window, angle, source_width, source_height):
    """Map a pose result from a rotated crop back to source pixel coordinates."""
    people = persons_from_result(result)
    if not people:
        return []
    x0, y0, x1, y1 = map(int, window)
    crop_width, crop_height = x1 - x0, y1 - y0
    boxes = np.asarray([person["bbox"] for person in people], dtype=float)
    points = np.asarray([person["keypoints"] for person in people], dtype=float)
    boxes, points = unrotate_pose(boxes, points, crop_width, crop_height, angle)
    mapped = []
    for person, box, keypoints in zip(people, boxes, points):
        # Boxes are edges, while keypoints are pixel centers. Keep this
        # distinction at the crop boundary before adding the source offset.
        box = np.asarray(box, dtype=float)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, crop_width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, crop_height)
        keypoints = np.asarray(keypoints, dtype=float)
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, max(crop_width - 1, 0))
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, max(crop_height - 1, 0))
        box[[0, 2]] += x0
        box[[1, 3]] += y0
        keypoints[:, 0] += x0
        keypoints[:, 1] += y0
        mapped.append({**person, "bbox": box.tolist(), "keypoints": keypoints.tolist(),
                       "proposal_angle": angle})
    validate_people(mapped, source_width, source_height)
    return mapped


def _validate_windows(row, width, height):
    expected = [list(window) for window in tile_windows(width, height, TILE_EDGE, TILE_OVERLAP)
                if window != (0, 0, width, height)]
    observed = [list(receipt["crop_xyxy"]) for receipt in row.get("tile_receipts", [])]
    if observed != expected:
        raise ValueError("Frozen tile windows differ from the dimension-only proposal protocol")
    return [tuple(window) for window in expected]


def make_rotated_seed_proposals(model, frames, frozen, *, device, batch, angles=ANGLES):
    """Generate common person seeds; no reference boxes or joints enter this function."""
    if batch < 1 or tuple(angles) != ANGLES:
        raise ValueError("The frozen rotation protocol requires batch >= 1 and angles 0/90/270")
    frozen_rows = {row["id"]: row for row in frozen["rows"]}
    if set(frozen_rows) != {frame["id"] for frame in frames}:
        raise ValueError("Frozen tiled seed inventory differs from source inventory")
    rows, started = [], time.perf_counter()
    for frame_index, frame in enumerate(frames):
        image = read_image(frame)
        frozen_row = frozen_rows[frame["id"]]
        windows = _validate_windows(frozen_row, frame["width"], frame["height"])
        per_angle = {str(angle): [] for angle in ANGLES}
        for person in frozen_row.get("raw_people", []):
            per_angle["0"].append({**person, "proposal_angle": 0,
                                    "proposal_source": person.get("proposal_source", "cached_full")})
        requests = []
        for window in windows:
            x0, y0, x1, y1 = window
            crop = image[y0:y1, x0:x1]
            for angle in (90, 270):
                requests.append((window, angle, rotate_crop(crop, angle)))
        for start in range(0, len(requests), batch):
            part = requests[start:start + batch]
            predictions = model.predict([request[2] for request in part], imgsz=SEED_IMAGE_SIZE,
                                        conf=PERSON_CONFIDENCE, iou=PERSON_NMS_IOU,
                                        max_det=MAX_PEOPLE_PER_VIEW, device=device, verbose=False)
            if len(predictions) != len(part):
                raise RuntimeError("Rotated tiled detector lost input views")
            for (window, angle, _), prediction in zip(part, predictions):
                people = map_rotated_people(prediction, window, angle, frame["width"], frame["height"])
                for person in people:
                    person.update(proposal_source="grid_tile_rotated",
                                  proposal_crop_xyxy=list(window))
                per_angle[str(angle)].extend(people)
        raw_all = [person for angle in ANGLES for person in per_angle[str(angle)]]
        unrotated = suppress_duplicates(per_angle["0"])
        rotated = suppress_duplicates(raw_all)
        rows.append({"id": frame["id"], "raw_people_by_angle": per_angle,
                     "raw_people": raw_all, "unrotated_people": unrotated,
                     "rotated_people": rotated, "tile_windows": [list(window) for window in windows],
                     "selected_angle_by_nms": [person.get("proposal_angle") for person in rotated]})
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(frames):
            print(json.dumps({"rotated_seed_frames_completed": frame_index + 1,
                              "rotated_seed_frames_planned": len(frames),
                              "seconds": time.perf_counter() - started}), flush=True)
    return {"mode": "frozen_full_plus_grid_tiled_person_proposals_with_rotations", "rows": rows,
            "angles": list(ANGLES), "seconds": time.perf_counter() - started,
            "reference_labels_used_for_prediction": False,
            "automatic_keeper_role_assignment_evaluated": False}


def seed_prediction(seeds, mode):
    if mode not in {"unrotated", "rotated"}:
        raise ValueError("Unknown seed mode")
    key = "unrotated_people" if mode == "unrotated" else "rotated_people"
    return {"mode": mode, "rows": [{"id": row["id"], "people": row[key],
                                    "reference_labels_used_for_prediction": False}
                                   for row in seeds["rows"]],
            "reference_labels_used_for_prediction": False}


def _best_by_confidence(people, seed_box):
    eligible = [person for person in people if box_iou(seed_box, person["bbox"]) >= .3]
    return max(eligible, key=lambda person: (person["confidence"], -person.get("candidate_id", 0))) if eligible else None


def predict_rotated_rois(model, frames, seeds, *, device, batch, angles=ANGLES):
    """Run three-angle pose inference from common detector-generated seeds."""
    if tuple(angles) != ANGLES:
        raise ValueError("The frozen rotation protocol requires angles 0/90/270")
    seed_by_id = {row["id"]: row for row in seeds["rows"]}
    if set(seed_by_id) != {frame["id"] for frame in frames}:
        raise ValueError("Seed and source inventories differ")
    rows, started = [], time.perf_counter()
    for frame_index, frame in enumerate(frames):
        image = read_image(frame)
        seed_people = seed_by_id[frame["id"]]["people"]
        requests = []
        for seed_index, seed in enumerate(seed_people):
            window = crop_window(seed["bbox"], frame["width"], frame["height"])
            crop = image[window[1]:window[3], window[0]:window[2]]
            for angle in ANGLES:
                requests.append((seed_index, seed, window, angle, rotate_crop(crop, angle)))
        raw_by_seed = defaultdict(lambda: {str(angle): [] for angle in ANGLES})
        for start in range(0, len(requests), batch):
            part = requests[start:start + batch]
            predictions = model.predict([request[4] for request in part], imgsz=IMAGE_SIZE,
                                        conf=PERSON_CONFIDENCE, iou=.7, max_det=10,
                                        device=device, verbose=False)
            if len(predictions) != len(part):
                raise RuntimeError("Rotated ROI inference lost input views")
            for (seed_index, seed, window, angle, _), prediction in zip(part, predictions):
                raw_by_seed[seed_index][str(angle)] = map_rotated_people(
                    prediction, window, angle, frame["width"], frame["height"])
        selected, receipts = [], []
        for seed_index, seed in enumerate(seed_people):
            by_angle = raw_by_seed[seed_index]
            best = {angle: _best_by_confidence(by_angle[str(angle)], seed["bbox"])
                    for angle in ANGLES}
            chosen_angle, chosen = max(
                ((angle, person) for angle, person in best.items() if person is not None),
                key=lambda pair: (pair[1]["confidence"], -pair[0]), default=(None, None))
            if chosen is not None:
                selected.append({**chosen, "seed_candidate_id": seed.get("candidate_id"),
                                 "selected_angle": chosen_angle})
            receipts.append({"seed_candidate_id": seed.get("candidate_id"),
                             "seed_bbox": seed["bbox"],
                             "raw_candidates_by_angle": by_angle,
                             "selected_angle": chosen_angle,
                             "selected_confidence": chosen["confidence"] if chosen else None})
        rows.append({"id": frame["id"], "people": suppress_duplicates(selected),
                     "raw_people_by_seed": receipts, "seed_count": len(seed_people),
                     "angles": list(ANGLES)})
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(frames):
            print(json.dumps({"rotated_roi_frames_completed": frame_index + 1,
                              "rotated_roi_frames_planned": len(frames),
                              "seconds": time.perf_counter() - started}), flush=True)
    return {"mode": "three_angle_automatic_seed_roi_pose", "angles": list(ANGLES), "rows": rows,
            "seconds": time.perf_counter() - started,
            "reference_labels_used_for_prediction": False,
            "automatic_keeper_role_assignment_evaluated": False}


def add_reference_strata(evaluation, samples):
    sample_by_id = {sample["id"]: sample for sample in samples}
    points = evaluation["points"]
    for point in points:
        sample = sample_by_id[point["id"]]
        bbox = sample["bbox"]
        point["reference_bbox_orientation"] = (
            "reference_bbox_horizontal" if bbox[2] - bbox[0] > bbox[3] - bbox[1]
            else "reference_bbox_nonhorizontal")
        point["stage"] = sample.get("stage") if sample.get("stage") is not None else "unknown"
    for field in ("reference_bbox_orientation", "stage"):
        groups = defaultdict(list)
        for point in points:
            groups[point[field]].append(point)
        evaluation["by_" + field] = {key: point_metrics(value) for key, value in sorted(groups.items())}
    paired = defaultdict(list)
    for point in points:
        paired[(point["stratum"], point["reference_bbox_orientation"])].append(point)
    evaluation["by_stratum_and_orientation"] = {
        f"{stratum}|{orientation}": point_metrics(value)
        for (stratum, orientation), value in sorted(paired.items())}
    evaluation["strata_definition"] = {
        "reference_bbox_horizontal": "reference bbox width > height, evaluated from supplied reference only",
        "reference_bbox_nonhorizontal": "reference bbox width <= height, evaluated from supplied reference only",
        "stage": "manifest stage value; null mapped to unknown",
    }
    return evaluation


def load_control_prediction(path, seed_hash):
    if not path.is_file():
        raise FileNotFoundError(path)
    prediction = json.loads(path.read_text())
    if prediction.get("common_seed_sha256") != seed_hash:
        raise ValueError("No-rotation control does not use the frozen common seed")
    if prediction.get("reference_labels_used_for_prediction") is not False:
        raise ValueError("No-rotation control must declare label-free inference")
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiled-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new rotated proposal output directory")
    if args.batch < 1:
        parser.error("--batch must be positive")
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    SETTINGS.update({"sync": False})
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(.65)
    samples, _manifest = load_samples(args.manifest)
    frames = inference_frames(samples)
    tiled_protocol_path = args.tiled_dir / "protocol.json"
    tiled_seed_path = args.tiled_dir / "frozen_tiled_seeds.json"
    tiled_protocol = json.loads(tiled_protocol_path.read_text())
    tiled_seeds = json.loads(tiled_seed_path.read_text())
    if tiled_protocol["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Frozen tiled protocol manifest differs")
    if tiled_protocol["weights_sha256"].get("pretrained") != digest(args.pretrained):
        raise ValueError("Frozen tiled protocol pretrained weights differ")
    if tiled_seeds.get("source_manifest_sha256") != digest(args.manifest):
        raise ValueError("Frozen tiled seeds manifest differs")
    if tiled_seeds.get("weights_sha256") != digest(args.pretrained):
        raise ValueError("Frozen tiled seeds pretrained weights differ")
    if tiled_seeds.get("reference_labels_used_for_prediction") is not False:
        raise ValueError("Frozen tiled seeds must declare label-free inference")
    seed_hash = digest(tiled_seed_path)
    weights = {"pretrained": args.pretrained, "trained": args.trained}
    args.output.mkdir(parents=True)
    protocol = {
        "schema": "keeper_rotated_proposals/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest(args.manifest), "tiled_protocol_sha256": digest(tiled_protocol_path),
        "tiled_seed_sha256": seed_hash, "weights_sha256": {name: digest(path) for name, path in weights.items()},
        "device": args.device, "batch": args.batch,
        "runtime": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "proposal_protocol": {"angles": list(ANGLES), "tile_edge": TILE_EDGE, "tile_overlap": TILE_OVERLAP,
                              "seed_image_size": SEED_IMAGE_SIZE, "person_confidence": PERSON_CONFIDENCE,
                              "person_nms_iou": PERSON_NMS_IOU, "maximum_people_per_view": MAX_PEOPLE_PER_VIEW,
                              "zero_angle_reused_from_frozen_raw": True, "tile_windows_dimension_only": True},
        "roi_protocol": {"angles": list(ANGLES), "imgsz": IMAGE_SIZE, "roi_margin": ROI_MARGIN,
                          "keypoint_confidence": KEYPOINT_CONFIDENCE, "association_iou": .3,
                          "same_common_seed_for_both_checkpoints": True,
                          "angle_selection": "highest eligible model person confidence; no reference input"},
        "reference_labels_used_for_prediction": False, "new_training": False,
        "automatic_keeper_role_accuracy": None, "independent_test": False,
        "default_pipeline_changed": False,
        "strata": {"horizontal": "reference bbox width > height at evaluation time only",
                    "stage": "manifest stage, null mapped to unknown"},
        "limitations": ["Exposed assisted-reference development data, not an independent final test.",
                        "Candidate recall and pose PCK do not measure keeper identity accuracy.",
                        "Rotation comparisons use one frozen common detector-seed run and one seed.",
                        "Timing is a single-run measurement, not complete-video latency."],
    }
    save(args.output / "protocol.json", protocol)
    started = time.perf_counter()
    model = YOLO(str(args.pretrained))
    if model.task != "pose":
        raise ValueError("Pose weights required for rotated seed inference")
    rotated_seeds = make_rotated_seed_proposals(model, frames, tiled_seeds, device=args.device, batch=args.batch)
    rotated_seeds.update(weights_sha256=protocol["weights_sha256"]["pretrained"],
                         source_manifest_sha256=protocol["manifest_sha256"],
                         seconds_including_model_load=time.perf_counter() - started)
    if any(row["raw_people_by_angle"]["90"] or row["raw_people_by_angle"]["270"] for row in rotated_seeds["rows"]):
        rotated_seeds["actual_device"] = verify_inference_device(model, args.device)
    save(args.output / "frozen_rotated_seeds.json", rotated_seeds)
    rotated_seed_hash = digest(args.output / "frozen_rotated_seeds.json")
    del model
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    seed_modes = {mode: seed_prediction(rotated_seeds, mode) for mode in ("unrotated", "rotated")}
    seed_modes["unrotated"]["source"] = "frozen_tiled_proposals_zero_degree"
    seed_modes["rotated"]["source"] = "frozen_tiled_proposals_zero_degree_plus_rotated_tiles"
    for mode, prediction in seed_modes.items():
        save(args.output / f"{mode}_seed_proposals.json", prediction)
    predictions, timings = {}, {}
    for name, path in weights.items():
        if digest(path) != protocol["weights_sha256"][name]:
            raise ValueError("Checkpoint changed after protocol freeze")
        model_started = time.perf_counter()
        model = YOLO(str(path))
        if model.task != "pose":
            raise ValueError("Pose weights required for rotated ROI inference")
        model_load_seconds = time.perf_counter() - model_started
        for mode, seeds in seed_modes.items():
            prediction = predict_rotated_rois(model, frames, seeds, device=args.device, batch=args.batch)
            prediction.update(weights_sha256=protocol["weights_sha256"][name], common_seed_sha256=rotated_seed_hash,
                              seed_mode=mode, model_load_seconds=model_load_seconds,
                              # The same loaded model is reused for both modes. This is
                              # an equivalent per-mode total, not an additive wall time.
                              seconds_including_model_load=model_load_seconds + prediction["seconds"])
            if any(row["raw_people_by_seed"] for row in prediction["rows"]):
                prediction["actual_device"] = verify_inference_device(model, args.device)
            predictions[(name, mode)] = prediction
            timings[f"{name}_{mode}"] = {
                "model_load_seconds": model_load_seconds,
                "seconds_including_model_load_equivalent": prediction["seconds_including_model_load"],
                "roi_prediction_seconds": prediction["seconds"],
                "model_load_shared_across_modes": True,
            }
            save(args.output / f"{name}_{mode}_rotated_roi_predictions.json", prediction)
        del model
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    # Existing no-rotation outputs are the explicit after/control comparison and
    # use the same frozen tiled seeds; defer all reference scoring until now.
    controls = {}
    for name in weights:
        control_path = args.tiled_dir / f"{name}_tiled_seed_roi_predictions.json"
        controls[name] = load_control_prediction(control_path, seed_hash)
    evaluations = {}
    for (name, mode), prediction in predictions.items():
        evaluation = add_reference_strata(evaluate_predictions(prediction, samples), samples)
        evaluation.update(weights_sha256=protocol["weights_sha256"][name], common_seed_sha256=rotated_seed_hash,
                          prediction_sha256=digest(args.output / f"{name}_{mode}_rotated_roi_predictions.json"),
                          seed_mode=mode)
        evaluations[(name, mode)] = evaluation
        save(args.output / f"{name}_{mode}_rotated_roi_evaluation.json", evaluation)
    for name, control in controls.items():
        evaluation = add_reference_strata(evaluate_predictions(control, samples), samples)
        evaluation.update(weights_sha256=protocol["weights_sha256"][name], common_seed_sha256=seed_hash,
                          seed_mode="no_rotation_control", control_prediction_sha256=digest(
                              args.tiled_dir / f"{name}_tiled_seed_roi_predictions.json"))
        evaluations[(name, "no_rotation_control")] = evaluation
        save(args.output / f"{name}_no_rotation_control_evaluation.json", evaluation)
    coverage = {mode: proposal_coverage(seed_prediction(rotated_seeds, mode), samples)
                for mode in ("unrotated", "rotated")}
    coverage["by_angle"] = {
        str(angle): proposal_coverage({"rows": [{"id": row["id"], "people": row["raw_people_by_angle"][str(angle)]}
                                                for row in rotated_seeds["rows"]]}, samples)
        for angle in ANGLES}
    save(args.output / "proposal_coverage.json", coverage)
    paired = {mode: paired_changes(evaluations[("pretrained", mode)], evaluations[("trained", mode)])
              for mode in ("no_rotation_control", "unrotated", "rotated")}
    save(args.output / "paired_changes.json", paired)
    report = {
        "protocol": protocol, "rotated_seed_sha256": rotated_seed_hash,
        "proposal_coverage": {key: {inner: value for inner, value in data.items() if inner != "rows"}
                               for key, data in coverage.items()},
        "pose_evaluation": {f"{name}_{mode}": {key: value for key, value in evaluation.items()
                                                  if key not in {"points", "associations"}}
                            for (name, mode), evaluation in evaluations.items()},
        "paired_changes": paired, "timing": timings,
        "elapsed_seconds": time.perf_counter() - started, "completed": True,
        "new_training": False, "default_pipeline_changed": False,
        "automatic_keeper_role_accuracy": None, "independent_test": False,
    }
    save(args.output / "report.json", report)
    print(json.dumps({"completed": True, "proposal_coverage": report["proposal_coverage"],
                      "evaluation_keys": sorted(report["pose_evaluation"])}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
