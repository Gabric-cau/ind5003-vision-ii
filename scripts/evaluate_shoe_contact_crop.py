"""Source-only local-crop ablation for a frozen shoe head; never load labels."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import time

import cv2
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.predict_video_shoes import load_bound_model
from scripts.run_fresh_shoe_approach_experiment import infer_event, read, sha, write
from src.pose_refinement import crop_bounds
from src.shoe_supervised_model import ShoeConfig, extract_evidence, predict_event


def contact_crop_bound(inputs, plan, camera, *, video, width, height, config):
    source = inputs["source_mapping"]
    if camera["source_sha256"] != source["clip_sha256"]:
        raise ValueError("Camera source differs from the measured actor")
    if inputs.get("reference_labels_read") is not False:
        raise ValueError("Source poses must be independent of reference labels")
    bracket = inputs["contact_bracket_frames"]
    fps = float(inputs["fps"])
    start = max(inputs["inference_interval_frames"][0],
                math.ceil(bracket[0] - config.foot_before_seconds * fps))
    end = min(inputs["inference_interval_frames"][1],
              math.floor(bracket[1] + config.foot_after_seconds * fps))
    if not start <= bracket[0] < bracket[1] <= end:
        raise ValueError("Automatic contact lies outside its source pose window")
    tasks = {row["frame"]: row for row in plan}
    if len(tasks) != len(plan) or any(frame not in tasks for frame in range(start, end + 1)):
        raise ValueError("Measured ROI plan must cover each local frame exactly once")
    segment_ids = [camera["segment_ids"].get(str(frame)) for frame in range(start, end + 1)]
    if None in segment_ids or len(set(segment_ids)) != 1:
        raise ValueError("Local contact window crosses a camera failure")
    boxes = [tasks[frame]["anchor_bbox"] for frame in range(start, end + 1)
             if tasks[frame]["anchor_bbox"] is not None]
    if len(boxes) < 5:
        raise ValueError("Insufficient measured actor observations for local crop")
    bounds = [crop_bounds(box, width, height, margin=.45) for box in boxes]
    fixed = [min(box[0] for box in bounds), min(box[1] for box in bounds),
             max(box[2] for box in bounds), max(box[3] for box in bounds)]
    scoped = deepcopy(inputs)
    scoped.update(inference_interval_frames=[start, end],
                  interval_frames=[start, min(end, inputs["interval_frames"][1])],
                  original_camera_motion={"transforms": {
                      str(frame): camera["transforms"][str(frame)] for frame in range(start, end + 1)}})
    bound = deepcopy(scoped)
    bound.update(video=video, width=width, height=height, fixed_roi=fixed,
                 plan=[{**tasks[frame], "crop_xyxy": fixed} for frame in range(start, end + 1)])
    return bound, scoped


def run(args):
    if args.output.exists():
        raise FileExistsError("Choose a fresh output to preserve the original evidence")
    started = time.perf_counter()
    model, metadata = load_bound_model(args.model, args.pose33_model)
    evidence_dir = args.analysis_dir / "trained_shoes"
    inputs = read(evidence_dir / "inputs.json")
    plan = read(evidence_dir / "roi_plan.json")
    records = read(evidence_dir / "poses33.json")
    if sha(evidence_dir / "poses33.json") != inputs["poses33_sha256"]:
        raise ValueError("Original cached poses changed")
    source = inputs["source_mapping"]
    filename = source["clip_filename"]
    if Path(filename).name != filename:
        raise ValueError("Expected a source basename")
    video = args.analysis_dir / "source" / filename
    if sha(video) != source["clip_sha256"]:
        raise ValueError("Source video changed")
    cached_plan = {(row["frame"], row["reflected"]): row for row in records}
    for task in plan:
        for reflected in (False, True):
            cached = cached_plan.get((task["frame"], reflected))
            if cached is None or any(task[key] != cached[key] for key in (
                    "crop_xyxy", "anchor_bbox", "anchor_source", "rival_source_boxes")):
                raise ValueError("ROI plan differs from the source-bound pose cache")
    cap = cv2.VideoCapture(str(video))
    try:
        if (not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - inputs["fps"]) > .01
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != source["frame_count"]):
            raise ValueError("Decoder clock differs from source evidence")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    camera = read(args.camera_json)
    bound, scoped = contact_crop_bound(inputs, plan, camera, video=video, width=width,
                                      height=height, config=ShoeConfig(**model["config"]))
    original_prediction = read(evidence_dir / "prediction.json")
    if original_prediction["striking_foot"] in ("left", "right"):
        raise ValueError("This ablation only addresses unavailable striking-foot evidence")
    cv2.setNumThreads(args.threads)
    args.output.mkdir(parents=True)
    write(args.output / "roi_plan.json", bound["plan"])
    with threadpool_limits(limits=args.threads):
        fresh, hashes, quality = infer_event(bound, args.pose33_model)
        for key, digest in hashes.items():
            if inputs["decoded_frame_hashes"].get(key) != digest:
                raise ValueError("Local crop decoded pixels differ from original pose evidence")
        write(args.output / "poses33.json", fresh)
        scoped.update(decoded_frame_hashes=hashes, poses33_sha256=sha(args.output / "poses33.json"),
                      new_neural_inference=True, extraction_scope="automatic_contact_local_crop_ablation",
                      inference_distribution_changed=True, matched_training_protocol=False)
        write(args.output / "inputs.json", scoped)
        features = extract_evidence(scoped, fresh)
        candidate = predict_event(model, features, inputs["fps"])
    after = deepcopy(original_prediction)
    after.update(striking_foot=candidate["striking_foot"],
                 right_score_uncalibrated=candidate["right_score_uncalibrated"])
    report = {"schema_version": "shoe-contact-local-crop-ablation-v1", "source_sha256": sha(video),
              "original_inputs_sha256": sha(evidence_dir / "inputs.json"),
              "original_plan_sha256": sha(evidence_dir / "roi_plan.json"),
              "camera_sha256": sha(args.camera_json), "model_sha256": metadata["model_sha256"],
              "pose_model_sha256": sha(args.pose33_model),
              "original_crop_xyxy": plan[0]["crop_xyxy"], "local_crop_xyxy": bound["fixed_roi"],
              "local_interval_frames": bound["inference_interval_frames"],
              "automatic_contact_bracket_frames": inputs["contact_bracket_frames"],
              "original": original_prediction, "foot_only_candidate": after,
              "local_raw_prediction": candidate, "local_quality": quality,
              "supported_side_frames": features["supported_side_frames"],
              "foot_vector_available": features["foot_vector"] is not None,
              "landings_changed": after["landings"] != original_prediction["landings"],
              "human_labels_read": False, "weights_updated": False,
              "inference_distribution_changed": True, "matched_training_protocol": False,
              "evaluation_scope": "unvalidated source-only crop ablation with frozen head, not accuracy",
              "seconds": time.perf_counter() - started,
              "code_sha256": {name: sha(ROOT / name) for name in (
                  "scripts/evaluate_shoe_contact_crop.py", "scripts/run_fresh_shoe_approach_experiment.py",
                  "src/shoe_supervised_model.py", "src/pose_refinement.py")}}
    write(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis-dir", "model", "pose33-model", "camera-json", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    report = run(args)
    print({key: report[key] for key in ("local_crop_xyxy", "local_interval_frames",
                                      "foot_vector_available", "supported_side_frames", "seconds")})
    print(report["foot_only_candidate"]["striking_foot"])


if __name__ == "__main__":
    main()
