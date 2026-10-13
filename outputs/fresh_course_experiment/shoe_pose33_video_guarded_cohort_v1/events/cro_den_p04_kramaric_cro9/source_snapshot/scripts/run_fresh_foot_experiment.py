"""Run paired high-resolution cropped poses around fresh contact evidence.

Inputs use clip-absolute frames and source pixels. No reviewer labels, player
names or camera-side priors are used to infer a striking foot.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_results import sha256
from src.kicking_foot_evidence import kicking_foot_evidence
from src.kicking_foot_redetection import undo_mirror
from src.pose_refinement import _detections, anchor_at_frame, crop_bounds, select_anchored_detection
from src.runtime import inference_provenance, resolve_device


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_pose_records(path: Path, clip_ids: list[str] | None) -> list[tuple[dict, Path]]:
    if path.is_dir():
        paths = [path / f"{clip_id}.json" for clip_id in clip_ids] if clip_ids else sorted(path.glob("*.json"))
        rows = [(read(item), item) for item in paths]
        rows = [(row, item) for row, item in rows if isinstance(row, dict) and "pose_frames" in row]
    else:
        data = read(path)
        rows = [(row, path) for row in (data if isinstance(data, list) else [data])]
        rows = [(row, item) for row, item in rows if clip_ids is None or row.get("clip_id") in clip_ids]
    if not rows or len({row["clip_id"] for row, _ in rows}) != len(rows):
        raise ValueError("Fresh pose input must contain distinct rich pose records")
    if clip_ids is not None and set(clip_ids) != {row["clip_id"] for row, _ in rows}:
        raise ValueError("Some requested fresh pose records are unavailable")
    return rows


def pose_scene_rejections(pose: dict) -> list[str]:
    selection = pose.get("shot_selection", {})
    attempts = [attempt for attempt in selection.get("attempts", [])
                if attempt.get("shot_index") == selection.get("selected_shot_index")]
    reasons = []
    if selection.get("found_qualified_shot") is not True:
        reasons.append("upstream_did_not_find_qualified_shot")
    if len(attempts) != 1 or attempts[0].get("qualifies") is not True:
        reasons.append("selected_shot_not_uniquely_qualified")
    if attempts and attempts[0].get("scene_rejections") != []:
        reasons.append("selected_shot_has_scene_rejections_or_missing_scene_check")
    if pose.get("selected_track_qualified") is not True:
        reasons.append("selected_actor_not_qualified")
    return reasons


def contact_payload(contact: dict | None) -> dict:
    if not contact:
        return {}
    return contact.get("contact", contact.get("selected_candidate", contact)) or {}


def selected_ball_observations(contact: dict | None) -> list[dict] | None:
    payload = contact_payload(contact)
    observations = payload.get("ball_observations", (contact or {}).get("ball_observations"))
    if observations is None:
        stationary = payload.get("stationary_ball_observations")
        departure = payload.get("departure_ball_observations")
        if any(value is not None and not isinstance(value, list) for value in (stationary, departure)):
            raise ValueError("Stationary/departure evidence must preserve lists of actual detections")
        if stationary is not None or departure is not None:
            observations = (stationary or []) + (departure or [])
    if observations is None:
        return None
    if not isinstance(observations, list):
        raise ValueError("Selected ball observations must be a list of actual source-pixel detections")
    return observations


def freeze_code(output_dir: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        target = output_dir / "source_snapshot" / name
        if target.exists():
            if sha256(target) != expected:
                raise FileExistsError("Saved source snapshot does not match the current experiment code")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        if sha256(target) != expected:
            raise ValueError("Experiment source snapshot failed its integrity check")


def validate_pose_model(model) -> dict:
    task = getattr(model, "task", None)
    if task != "pose":
        raise ValueError(f"Foot refinement requires task=pose; loaded model task={task!r}")
    network = getattr(model, "model", None)
    configuration = getattr(network, "yaml", {}) or {}
    shape = configuration.get("kpt_shape", getattr(network, "kpt_shape", None))
    if shape is None or list(shape) != [17, 3]:
        raise ValueError(f"Foot refinement requires COCO-17 xy/confidence keypoints [17, 3]; model has {shape!r}")
    return {"expected_task": "pose", "actual_task": task, "keypoint_shape": list(shape),
            "model_loaded_and_checked": True}


def summarize_plan(pose: dict, timing: dict, tasks: list[dict], rejected: list[dict],
                   balls: list[dict] | None) -> dict:
    frame_ids = sorted({task["frame"] for task in tasks})
    first, departure = timing["contact_bracket_frames"]
    ball_ids = sorted({row["frame"] for row in balls or []})
    matched = [frame for frame in ball_ids if first - 1 <= frame <= departure and frame in frame_ids]
    warnings = []
    if not balls:
        warnings.append("no_actual_source_pixel_ball_observations_for_distance_evidence")
    elif len(matched) < 3:
        warnings.append("fewer_than_three_planned_contact_frame_matched_ball_samples")
    ambiguity = [frame for frame in timing.get("seed_ambiguous_frames", [])
                 if first - 5 <= frame <= departure]
    if ambiguity:
        warnings.append("local_ball_identity_ambiguity_will_keep_candidate_unknown")
    return {"source_pixel_dimensions": [pose["frame_width"], pose["frame_height"]],
            "source_fps": pose["fps"], "contact_bracket_frames": [first, departure],
            "contact_bracket_clip_seconds": [first / pose["fps"], departure / pose["fps"]],
            "timing_source": timing["timing_source"], "actor_track_id": timing["actor_track_id_candidate"],
            "actor_reselected_by_ball_evidence": timing.get("actor_reselected_by_ball_evidence", False),
            "planned_frame_ids": frame_ids, "planned_source_frames": len(frame_ids),
            "planned_normal_pose_crops": sum(not task["reflected"] for task in tasks),
            "planned_mirrored_pose_crops": sum(task["reflected"] for task in tasks),
            "actual_ball_observation_count": len(balls or []),
            "planned_contact_ball_frame_matches": matched,
            "local_seed_identity_ambiguous_frames": ambiguity,
            "source_rejection_counts": dict(Counter(row["reason"] for row in rejected)),
            "warnings": warnings, "new_neural_inference": False}


def timing_and_actor(pose: dict, contact: dict | None, allow_pose_timing: bool) -> dict:
    """The accepted contact JSON contract is deliberately explicit."""
    evidence = contact_payload(contact)
    explicit_bracket = "contact_bracket_frames" in evidence
    bracket = evidence.get("contact_bracket_frames")
    if not explicit_bracket and bracket is None and evidence.get("departure_frame") is not None:
        first = evidence.get("last_stationary_frame", evidence.get("contact_frame"))
        if first is not None:
            bracket = [first, evidence["departure_frame"]]
    if not explicit_bracket and bracket is None and evidence.get("ball_departure_frames"):
        first = evidence.get("contact_frame")
        if first is not None:
            bracket = [first, min(evidence["ball_departure_frames"])]
    supported = bracket is not None
    if bracket is None and allow_pose_timing and pose.get("kick_frame") is not None:
        frame = int(pose["kick_frame"])
        bracket = [max(pose["shot_start_frame"], frame - 2), min(pose["shot_end_frame"] - 1, frame + 2)]
    actor = evidence.get("contact_associated_track_id", evidence.get("actor_track_id", pose.get("selected_track_id")))
    if bracket is not None:
        if (len(bracket) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in bracket)
                or not pose["shot_start_frame"] <= bracket[0] <= bracket[1] < pose["shot_end_frame"]):
            raise ValueError("Contact bracket must lie inside the selected fresh shot")
    return {"contact_bracket_frames": bracket, "actor_track_id_candidate": actor,
            "explicit_contact_bracket_supplied": explicit_bracket,
            "original_pose_selected_track_id": pose.get("selected_track_id"),
            "actor_reselected_by_ball_evidence": evidence.get("actor_reselected_by_ball_evidence", False),
            "actor_field_support": evidence.get("actor_field_support"),
            "actor_goal_line_excluded": evidence.get("actor_goal_line_excluded"),
            "seed_ambiguous_frames": evidence.get("seed_ambiguous_frames", []),
            "ball_departure_evidence_available": supported,
            "timing_source": "independent_ball_contact_candidate" if supported else "pose_motion_proposal" if bracket else "unavailable",
            "contact_observed_and_verified": False, "actor_identity_verified": False}


def restore_crop_detections(prediction, bounds, reflected: bool, frame: int) -> list[dict]:
    origin = np.asarray(bounds[:2], float)
    width = bounds[2] - bounds[0]
    output = []
    for item in _detections(prediction):
        points, confidence = np.asarray(item["keypoints"]), np.asarray(item["keypoint_confidences"])
        box = np.asarray(item["bbox"])
        if reflected:
            points, confidence = undo_mirror(points, confidence, width)
            box = np.asarray([width - 1 - box[2], box[1], width - 1 - box[0], box[3]])
        valid = (confidence > 0) & np.isfinite(points).all(axis=1) & np.any(abs(points) > 1e-9, axis=1)
        points = points.copy()
        points[valid] += origin
        points[~valid], confidence[~valid] = 0, 0
        output.append({"frame": frame, "bbox": (box + np.tile(origin, 2)).tolist(),
                       "keypoints": points.tolist(), "keypoint_confidences": confidence.tolist(),
                       "box_confidence": item["box_confidence"]})
    return output


def prepare_tasks(video: Path, pose: dict, timing: dict) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    scene_rejections = pose_scene_rejections(pose)
    if scene_rejections:
        return [], [], [{"reason": reason} for reason in scene_rejections]
    actor_id = timing["actor_track_id_candidate"]
    excluded = (pose.get("goal_line_role_exclusions") or {}).get("excluded_track_ids", [])
    if actor_id in excluded or timing.get("actor_goal_line_excluded") is True:
        return [], [], [{"reason": "contact_actor_is_excluded_goal_line_occupant"}]
    if actor_id != pose.get("selected_track_id"):
        support = timing.get("actor_field_support")
        if (timing.get("actor_reselected_by_ball_evidence") is not True
                or timing.get("actor_goal_line_excluded") is not False
                or support is None or not np.isfinite(support) or support < .20):
            return [], [], [{"reason": "reselected_contact_actor_has_insufficient_field_role_evidence"}]
    actor = next((track for track in pose["tracks"] if track["track_id"] == timing["actor_track_id_candidate"]), None)
    if actor is None:
        return [], [], [{"reason": "contact_actor_not_present_in_fresh_pose_tracks"}]
    observations = {item["frame"]: item for item in actor["observations"]}
    first, departure = timing["contact_bracket_frames"]
    fps = pose["fps"]
    start = max(pose["shot_start_frame"], first - round(.30 * fps))
    end = min(pose["shot_end_frame"] - 1, departure + round(.30 * fps))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode fresh clip: {video}")
    if abs(capture.get(cv2.CAP_PROP_FPS) - fps) > .01:
        raise ValueError("Fresh pose FPS and actual video FPS disagree")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    images, tasks, rejected = [], [], []
    for frame in range(start, end + 1):
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Fresh clip ended before planned frame {frame}")
        if image.shape[1] != pose["frame_width"] or image.shape[0] != pose["frame_height"]:
            raise ValueError("Fresh pose pixels and source image dimensions disagree")
        anchor, source = anchor_at_frame(observations, frame, max_gap=3)
        if anchor is None:
            rejected.append({"frame": frame, "reason": source})
            continue
        bounds = crop_bounds(anchor, image.shape[1], image.shape[0], margin=.45)
        crop = image[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        for reflected in (False, True):
            images.append(np.ascontiguousarray(crop[:, ::-1]) if reflected else crop.copy())
            tasks.append({"frame": frame, "crop_xyxy": list(bounds), "anchor_bbox": anchor,
                          "anchor_source": source, "reflected": reflected})
    capture.release()
    return images, tasks, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-input", type=Path, required=True, help="Fresh pose JSON directory or rich-record list")
    parser.add_argument("--clips-dir", type=Path, required=True, help="Fresh MP4 folder with source_manifest.json")
    parser.add_argument("--contact-json-dir", type=Path, help="Fresh ball contacts/ or event_windows/ JSON folder; frame IDs and xy are clip-absolute source pixels")
    parser.add_argument("--model", type=Path, help="COCO-17 YOLO pose .pt model; required for inference, optional for --plan-only")
    parser.add_argument("--output-dir", type=Path, required=True, help="New evidence folder; changed inputs/code require a new folder")
    parser.add_argument("--clips", nargs="+", help="Explicit pilot clip IDs; omit only for the complete source cohort")
    parser.add_argument("--device", default="cuda:0", help="Actual inference device; unused in --plan-only")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size for each normal/mirrored person crop")
    parser.add_argument("--batch-size", type=int, default=8, help="Combined normal/mirrored crops per model batch")
    parser.add_argument("--allow-pose-timing", action="store_true", help="Motion diagnosis only; absent ball departure keeps the foot output unknown")
    parser.add_argument("--plan-only", action="store_true", help="Validate source and report plans without neural inference")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.model is None and not args.plan_only:
        parser.error("--model is required for actual pose inference")
    if args.batch_size < 1 or args.imgsz < 32:
        raise ValueError("Invalid positive inference settings")
    records = load_pose_records(args.pose_input, args.clips)
    for pose, _ in records:
        if (pose.get("new_neural_inference") is not True
                or pose.get("reference_labels_used_for_prediction") is not False):
            raise ValueError("Foot experiment requires fresh source-only pose inference records")
    manifest_path = args.clips_dir / "source_manifest.json"
    manifest_records = read(manifest_path)["clips"]
    manifest = {record["clip_id"]: record for record in manifest_records}
    if len(manifest) != len(manifest_records):
        raise ValueError("Fresh source manifest has duplicate clip IDs")
    if args.clips is None and {pose["clip_id"] for pose, _ in records} != set(manifest):
        raise ValueError("Use --clips for an explicit pilot; complete inference needs every source event")
    model_hash = sha256(args.model) if args.model is not None else None
    device = "not_resolved_plan_only" if args.plan_only else resolve_device(args.device)
    script_hash = sha256(Path(__file__))
    module_hash = sha256(ROOT / "src/kicking_foot_evidence.py")
    manifest_hash = sha256(manifest_path)
    code_files = ["scripts/run_fresh_foot_experiment.py", "src/kicking_foot_evidence.py",
                  "src/pose_refinement.py", "src/kicking_foot_redetection.py", "src/runtime.py"]
    code_hashes = {name: sha256(ROOT / name) for name in code_files}
    settings = {"imgsz": args.imgsz, "batch_size": args.batch_size, "crop_margin": .45,
                "confidence": .15, "device": device, "allow_pose_timing": args.allow_pose_timing}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze_code(args.output_dir, code_hashes)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "settings": settings, "model_sha256": model_hash, "script_sha256": script_hash,
                "module_sha256": module_hash, "source_manifest_sha256": manifest_hash,
                "code_sha256": code_hashes, "cohort": [pose["clip_id"] for pose, _ in records],
                "plan_only": args.plan_only, "reference_labels_read": False,
                "human_verified": False, "held_out_test": False, "status": "started"}
    model_check = {"expected_task": "pose", "expected_keypoint_shape": [17, 3],
                   "model_loaded_and_checked": False,
                   "status": "not_loaded_plan_only" if args.plan_only else "awaiting_first_inference_batch"}
    protocol["model_task_validation"] = model_check
    if not args.plan_only:
        protocol["runtime"] = inference_provenance(args.device, device, [args.model])
    save(args.output_dir / "protocol.json", protocol)
    model = None
    outputs = []
    for pose, pose_path in records:
        clip_id = pose["clip_id"]
        source = manifest[clip_id]
        video = args.clips_dir / source["clip_filename"]
        if sha256(video) != source["clip_sha256"]:
            raise ValueError(f"Fresh source-video hash mismatch: {clip_id}")
        if pose.get("inference_identity", {}).get("source_sha256") != source["clip_sha256"]:
            raise ValueError("Pose cache and fresh clip are from different source bytes")
        contact_path = args.contact_json_dir / f"{clip_id}.json" if args.contact_json_dir else None
        contact = read(contact_path) if contact_path and contact_path.is_file() else None
        if contact and contact.get("clip_id") not in (None, clip_id):
            raise ValueError("Contact evidence has a different clip identity")
        if contact and contact.get("source_sha256") != source["clip_sha256"]:
            raise ValueError("Contact evidence is from different clip bytes")
        timing = timing_and_actor(pose, contact, args.allow_pose_timing)
        identity = {"pose_sha256": sha256(pose_path), "source_sha256": source["clip_sha256"],
                    "contact_sha256": sha256(contact_path) if contact else None, "model_sha256": model_hash,
                    "script_sha256": script_hash, "module_sha256": module_hash,
                    "source_manifest_sha256": manifest_hash, "code_sha256": code_hashes, "settings": settings}
        target = args.output_dir / "results" / f"{clip_id}.json"
        if target.exists():
            previous = read(target)
            if not args.resume or previous.get("inference_identity") != identity:
                raise FileExistsError("Use a new output directory for changed inference or inputs")
            outputs.append(previous)
            continue
        result = {"clip_id": clip_id, "timing": timing, "inference_identity": identity,
                  "kicking_foot_candidate": "unknown", "human_verified": False, "accuracy_validated": False,
                  "held_out_test": False, "reference_labels_used_for_prediction": False,
                  "source_mapping": source, "new_neural_inference": False}
        scene_rejections = pose_scene_rejections(pose)
        result["upstream_scene_rejections"] = scene_rejections
        if scene_rejections:
            result["status"] = "unqualified_upstream_shot_or_actor"
        elif timing["contact_bracket_frames"] is None:
            result["status"] = "awaiting_independent_ball_departure_evidence"
        else:
            images, tasks, rejected = prepare_tasks(video, pose, timing)
            result.update(planned_pose_crops=len(tasks), source_rejections=rejected)
            balls = selected_ball_observations(contact)
            result["plan"] = summarize_plan(pose, timing, tasks, rejected, balls)
            if args.plan_only:
                result.update(status="plan_only_no_inference" if tasks else "plan_only_rejected_source_crops",
                              planned_frame_ids=result["plan"]["planned_frame_ids"], model_task_validation=model_check)
            elif not tasks:
                result["status"] = "no_identity_supported_source_crop"
            else:
                if model is None:
                    from ultralytics import YOLO
                    model = YOLO(str(args.model))
                    model_check = validate_pose_model(model)
                    protocol["model_task_validation"] = model_check
                    save(args.output_dir / "protocol.json", protocol)
                result["model_task_validation"] = model_check
                started = time.monotonic()
                samples = {False: [], True: []}
                raw = []
                for offset in range(0, len(images), args.batch_size):
                    predictions = model.predict(images[offset:offset + args.batch_size], imgsz=args.imgsz,
                                                conf=.15, device=device, verbose=False)
                    for prediction, task in zip(predictions, tasks[offset:offset + args.batch_size]):
                        detections = restore_crop_detections(prediction, task["crop_xyxy"], task["reflected"], task["frame"])
                        selected, reason = select_anchored_detection(detections, task["anchor_bbox"])
                        record = {**task, "selection_reason": reason, "detection_count": len(detections), "selected": selected}
                        raw.append(record)
                        if selected:
                            samples[task["reflected"]].append({"frame": task["frame"], "bbox": selected["bbox"],
                                "keypoints": selected["keypoints"], "confidence": selected["keypoint_confidences"]})
                save(args.output_dir / "poses" / f"{clip_id}.json", raw)
                evidence = kicking_foot_evidence(samples[False], samples[True], timing["contact_bracket_frames"],
                                                pose["fps"], ball_observations=balls)
                result.update(status=evidence["status"], temporal_evidence=evidence, new_neural_inference=True,
                              inference_elapsed_sec=round(time.monotonic() - started, 3),
                              normal_pose_frames=len(samples[False]), mirrored_pose_frames=len(samples[True]))
                if timing["ball_departure_evidence_available"]:
                    ambiguous = [frame for frame in timing["seed_ambiguous_frames"]
                                 if timing["contact_bracket_frames"][0] - 5 <= frame <= timing["contact_bracket_frames"][1]]
                    result["local_seed_identity_ambiguous_frames"] = ambiguous
                    if ambiguous:
                        result["status"] = "ambiguous_contact_ball_identity"
                    else:
                        result["kicking_foot_candidate"] = evidence["candidate"]
                else:
                    result["status"] = "pose_timed_motion_diagnostic_not_contact_prediction"
        save(target, result)
        outputs.append(result)
        print(json.dumps({"clip_id": clip_id, "status": result["status"], "candidate": result["kicking_foot_candidate"],
                          "new_neural_inference": result["new_neural_inference"],
                          "contact_bracket": timing["contact_bracket_frames"],
                          "actor_track_id": timing["actor_track_id_candidate"],
                          "pose_crops": result.get("planned_pose_crops", 0),
                          "plan_warnings": result.get("plan", {}).get("warnings", []),
                          "upstream_scene_rejections": scene_rejections}), flush=True)
    protocol["status"] = "completed"
    save(args.output_dir / "protocol.json", protocol)
    save(args.output_dir / "summary.json", {"events": len(outputs),
         "new_neural_inference_events": sum(row["new_neural_inference"] for row in outputs),
         "candidate_counts": dict(Counter(row["kicking_foot_candidate"] for row in outputs)),
         "status_counts": dict(Counter(row["status"] for row in outputs)),
         "held_out_test": False, "accuracy_validated": False})


if __name__ == "__main__":
    main()
