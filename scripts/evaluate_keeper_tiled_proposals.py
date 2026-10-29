"""Compare two pose models using frozen, label-free full-frame/tiled person seeds."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gc
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_manual_pose import box_iou, digest, persons_from_result, save
from scripts.train_keeper_pose_reference_round1 import (
    IMAGE_SIZE, KEYPOINT_CONFIDENCE, PERSON_CONFIDENCE, ROI_MARGIN,
    evaluate_predictions, load_samples, paired_changes, predict_rois, read_image,
)
from src.ball_specialist import tile_windows

TILE_EDGE = 960
TILE_OVERLAP = .25
SEED_IMAGE_SIZE = 1280
PERSON_NMS_IOU = .7
MAX_PEOPLE_PER_VIEW = 20


def inference_frames(samples):
    """Pass only image identity and dimensions to neural inference."""
    return [{key: sample[key] for key in ("id", "image", "width", "height")}
            for sample in samples]


def validate_people(people, width, height):
    for person in people:
        box, confidence = person["bbox"], person["confidence"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Non-finite person box")
        if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
            raise ValueError("Person box outside source frame")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid person confidence")


def suppress_duplicates(people, threshold=PERSON_NMS_IOU):
    """Deterministic confidence NMS in original image coordinates."""
    order = sorted(range(len(people)), key=lambda index: (-people[index]["confidence"], index))
    kept = []
    for index in order:
        if any(box_iou(people[index]["bbox"], people[other]["bbox"]) > threshold for other in kept):
            continue
        kept.append(index)
    return [{**people[index], "candidate_id": candidate_id}
            for candidate_id, index in enumerate(kept)]


def verify_inference_device(model, requested):
    device = getattr(getattr(model, "predictor", None), "device", None)
    expected = "mps" if requested == "mps" else "cpu" if requested == "cpu" else "cuda"
    if getattr(device, "type", None) != expected:
        raise RuntimeError("Pose inference did not use the requested device")
    return str(device)


def make_tiled_seeds(model, frames, frozen_full, *, device, batch):
    if batch < 1:
        raise ValueError("Positive inference batch required")
    full = {row["id"]: row["people"] for row in frozen_full["rows"]}
    if (len(full) != len(frozen_full["rows"]) or len(frames) != len({s["id"] for s in frames})
            or set(full) != {s["id"] for s in frames}):
        raise ValueError("Frozen full-frame inventory differs from inference inventory")
    if frozen_full.get("reference_labels_used_for_prediction") is not False:
        raise ValueError("Full-frame seed cache must declare label-free inference")
    rows, started = [], time.perf_counter()
    for frame_index, frame in enumerate(frames):
        image = read_image(frame)
        width, height = frame["width"], frame["height"]
        windows = [window for window in tile_windows(width, height, TILE_EDGE, TILE_OVERLAP)
                   if window != (0, 0, width, height)]
        raw = [{**person, "proposal_source": "cached_full", "proposal_crop_xyxy": [0, 0, width, height]}
               for person in full[frame["id"]]]
        receipts = []
        for start in range(0, len(windows), batch):
            part = windows[start:start + batch]
            results = model.predict([image[y0:y1, x0:x1] for x0, y0, x1, y1 in part],
                                    imgsz=SEED_IMAGE_SIZE, conf=PERSON_CONFIDENCE,
                                    iou=PERSON_NMS_IOU, max_det=MAX_PEOPLE_PER_VIEW,
                                    device=device, verbose=False)
            if len(results) != len(part):
                raise RuntimeError("Tiled detector lost input views")
            for window, result in zip(part, results):
                people = persons_from_result(result, window[:2])
                validate_people(people, width, height)
                raw.extend({**person, "proposal_source": "grid_tile", "proposal_crop_xyxy": list(window)}
                           for person in people)
                receipts.append({"crop_xyxy": list(window), "candidate_count": len(people)})
        validate_people(raw, width, height)
        rows.append({"id": frame["id"], "people": suppress_duplicates(raw),
                     "raw_people": raw, "tile_receipts": receipts,
                     "full_candidate_count": len(full[frame["id"]])})
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(frames):
            print(json.dumps({"seed_frames_completed": frame_index + 1, "seed_frames_planned": len(frames),
                              "seconds": time.perf_counter() - started}), flush=True)
    return {"mode": "frozen_full_plus_grid_tiled_person_proposals", "rows": rows,
            "seconds": time.perf_counter() - started,
            "reference_labels_used_for_prediction": False,
            "automatic_keeper_role_assignment_evaluated": False,
            "cached_full_seconds_not_included": frozen_full["seconds"]}


def proposal_coverage(prediction, samples):
    predictions = {row["id"]: row for row in prediction["rows"]}
    if len(predictions) != len(prediction["rows"]) or set(predictions) != {s["id"] for s in samples}:
        raise ValueError("Proposal/reference frame inventories differ")
    rows = []
    for sample in samples:
        people = predictions[sample["id"]]["people"]
        best_iou = max((box_iou(sample["bbox"], person["bbox"]) for person in people), default=0.)
        rows.append({"id": sample["id"], "split": sample["split"], "event_id": sample["event_id"],
                     "candidate_count": len(people), "maximum_reference_box_iou": best_iou,
                     "covered_iou_0_3": best_iou >= .3, "covered_iou_0_5": best_iou >= .5})

    def metrics(group):
        count = len(group)
        return {"reference_frames": count, "frames_with_any_person": sum(r["candidate_count"] > 0 for r in group),
                "covered_iou_0_3": sum(r["covered_iou_0_3"] for r in group),
                "covered_iou_0_5": sum(r["covered_iou_0_5"] for r in group),
                "candidate_recall_iou_0_3": sum(r["covered_iou_0_3"] for r in group) / count if count else None,
                "candidate_recall_iou_0_5": sum(r["covered_iou_0_5"] for r in group) / count if count else None}

    report = {"all": metrics(rows), "rows": rows, "reference_used_only_for_evaluation": True,
              "automatic_keeper_role_accuracy": None,
              "scope": "any-person proposal coverage of assisted keeper boxes; not keeper identification accuracy"}
    for field in ("split", "event_id"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        report["by_" + field] = {key: metrics(value) for key, value in sorted(groups.items())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a new tiled proposal output directory")
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
    baseline_protocol_path = args.baseline_dir / "protocol.json"
    full_path = args.baseline_dir / "pretrained_full_predictions.json"
    baseline_protocol = json.loads(baseline_protocol_path.read_text())
    if (baseline_protocol["weights_sha256"] != digest(args.pretrained)
            or baseline_protocol["manifest_sha256"] != digest(args.manifest)):
        raise ValueError("Original seed weights or source manifest differ from the frozen baseline")
    full = json.loads(full_path.read_text())
    weights = {"pretrained": args.pretrained, "trained": args.trained}
    args.output.mkdir(parents=True)
    protocol = {"schema": "keeper_tiled_proposals/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": digest(args.manifest), "baseline_protocol_sha256": digest(baseline_protocol_path),
                "cached_full_predictions_sha256": digest(full_path),
                "weights_sha256": {name: digest(path) for name, path in weights.items()},
                "device": args.device, "batch": args.batch,
                "runtime": {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
                "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
                "seed_policy": {"weights": "frozen_original_pretrained_pose", "include_cached_full": True,
                                "tile_edge": TILE_EDGE, "overlap": TILE_OVERLAP, "imgsz": SEED_IMAGE_SIZE,
                                "person_confidence": PERSON_CONFIDENCE, "nms_iou": PERSON_NMS_IOU,
                                "maximum_people_per_view": MAX_PEOPLE_PER_VIEW, "global_candidate_cap": None,
                                "tile_positions_use_image_dimensions_only": True, "rotations": [0]},
                "roi_pose": {"imgsz": IMAGE_SIZE, "margin": ROI_MARGIN, "keypoint_confidence": KEYPOINT_CONFIDENCE,
                             "same_frozen_seeds_for_both_models": True, "reference_labels_used_in_inference": False},
                "reference_association_iou": .3, "primary_point_metric": "original-pixel PCK10; missing counts as failure",
                "independent_test": False, "automatic_keeper_role_accuracy": None, "new_training": False,
                "checkpoint_or_policy_selection_uses_scores": False, "default_pipeline_changed": False,
                "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
                    Path(__file__), ROOT / "scripts/train_keeper_pose_reference_round1.py",
                    ROOT / "scripts/optimize_manual_pose.py", ROOT / "src/ball_specialist.py")},
                "limitations": ["Exposed assisted-reference development data, not an independent final test.",
                                "Any-person candidate coverage and associated pose PCK do not measure keeper identity.",
                                "References do not exhaustively label other people; candidate precision is unmeasured.",
                                "Timings are single-run measurements, not complete-video latency or repeat benchmarks."]}
    save(args.output / "protocol.json", protocol)
    started = time.perf_counter()
    seed_started = time.perf_counter()
    model = YOLO(str(args.pretrained))
    if model.task != "pose":
        raise ValueError("Pose weights required for seed detection")
    seeds = make_tiled_seeds(model, frames, full, device=args.device, batch=args.batch)
    if any(row["tile_receipts"] for row in seeds["rows"]):
        seeds["actual_device"] = verify_inference_device(model, args.device)
    seeds["seconds_including_model_load"] = time.perf_counter() - seed_started
    seeds["weights_sha256"] = protocol["weights_sha256"]["pretrained"]
    seeds["source_manifest_sha256"] = protocol["manifest_sha256"]
    seed_path = args.output / "frozen_tiled_seeds.json"
    save(seed_path, seeds)
    seed_hash = digest(seed_path)
    del model
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    prediction_paths, pose_times = {}, {}
    for name, path in weights.items():
        if digest(path) != protocol["weights_sha256"][name] or digest(seed_path) != seed_hash:
            raise ValueError("Frozen model or common seed cache changed")
        stage_started = time.perf_counter()
        model = YOLO(str(path))
        if model.task != "pose":
            raise ValueError("Pose weights required for ROI comparison")
        prediction = predict_rois(model, frames, seeds, "automatic_roi", args.device, args.batch)
        if any(row["roi_receipts"] for row in prediction["rows"]):
            prediction["actual_device"] = verify_inference_device(model, args.device)
        prediction.update(weights_sha256=protocol["weights_sha256"][name], common_seed_sha256=seed_hash,
                          automatic_roi_generator=seeds["mode"], source_manifest_sha256=protocol["manifest_sha256"],
                          seconds_including_model_load=time.perf_counter() - stage_started)
        pose_times[name] = {"seconds_including_model_load": prediction["seconds_including_model_load"],
                            "roi_prediction_seconds": prediction["seconds"]}
        prediction_paths[name] = args.output / f"{name}_tiled_seed_roi_predictions.json"
        save(prediction_paths[name], prediction)
        del model
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    # Both prediction files are fixed before reference boxes or points are scored.
    evaluations = {}
    for name, path in prediction_paths.items():
        evaluations[name] = evaluate_predictions(json.loads(path.read_text()), samples)
        evaluations[name].update(weights_sha256=protocol["weights_sha256"][name], common_seed_sha256=seed_hash,
                                 predictions_sha256=digest(path), source_manifest_sha256=protocol["manifest_sha256"])
        save(args.output / f"{name}_tiled_seed_roi_evaluation.json", evaluations[name])
    coverage = {"full": proposal_coverage(full, samples), "full_plus_tiles": proposal_coverage(seeds, samples)}
    save(args.output / "proposal_coverage.json", coverage)
    paired = paired_changes(evaluations["pretrained"], evaluations["trained"])
    save(args.output / "paired_changes.json", paired)
    report = {"protocol": protocol, "common_seed_sha256": seed_hash,
              "prediction_sha256": {name: digest(path) for name, path in prediction_paths.items()},
              "proposal_coverage": {mode: {key: value for key, value in data.items() if key != "rows"}
                                    for mode, data in coverage.items()},
              "pose_evaluation": {name: {key: value for key, value in data.items() if key not in {"points", "associations"}}
                                  for name, data in evaluations.items()},
              "paired_changes": paired, "elapsed_seconds": time.perf_counter() - started,
              "timing": {"seed_generation_seconds_including_model_load": seeds["seconds_including_model_load"],
                         "historical_cached_full_seconds_not_rerun": full["seconds"],
                         "roi_pose_by_model": pose_times},
              "completed": True, "new_training": False, "default_pipeline_changed": False,
              "automatic_keeper_role_accuracy": None, "independent_test": False}
    save(args.output / "report.json", report)
    print(json.dumps({"completed": True, "proposal_coverage": report["proposal_coverage"],
                      "pose_by_split": {name: data["by_split"] for name, data in evaluations.items()}}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
