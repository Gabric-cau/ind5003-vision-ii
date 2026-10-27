"""Infer fresh keeper-region poses from automatic goal geometry and contact."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.predict_video_keeper import hybrid_direction, predict_analysis, supported_rule
from scripts.run_fresh_keeper_roi_experiment import infer_roi
from src.keeper_contact_recovery import (automatic_goal_geometry, bounded_contact,
                                         guard_recovered_prediction, recover_keeper)
from src.keeper_supervised_model import predict_keeper
from src.runtime import file_sha256, resolve_device
from src.step_pipeline import _build_tracks


def plan_goal_roi(pose, geometry, frame):
    boxes = [row["selected"]["bbox"] for current, row in geometry.items()
             if frame-round(.5*pose["fps"]) <= current < frame and row.get("selected")]
    if len(boxes) < 3:
        return None
    array = np.asarray(boxes, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
        raise ValueError("Invalid automatic goal boxes")
    center = np.median(array, axis=0)
    width, height = center[2]-center[0], center[3]-center[1]
    if width <= 20 or height <= 10:
        return None
    coherent = np.max(np.abs(array-center) / np.array([width, height, width, height]), axis=1) < .25
    if np.count_nonzero(coherent) < 3 or np.mean(coherent) < .6:
        return None
    x0, y0 = array[coherent, :2].min(axis=0)-[.6*height, .5*height]
    x1, y1 = array[coherent, 2:].max(axis=0)+[.6*height, 1.2*height]
    roi = [max(0, int(x0)), max(0, int(y0)), min(pose["frame_width"], int(np.ceil(x1))),
           min(pose["frame_height"], int(np.ceil(y1)))]
    return {
        "roi_xyxy": roi, "start_frame": max(pose["shot_start_frame"], frame-round(1.5*pose["fps"])),
        "end_frame_exclusive": min(pose["shot_end_frame"], frame+round(1.2*pose["fps"])+1),
        "rotations": [0, 90, 270], "seed_track_id": None,
        "selection": "coherent automatic pre-contact goal geometry; no person seed or human box",
        "source_pixels_are_not_upscaled_evidence": True,
        "goal_geometry_samples": len(boxes), "coherent_goal_samples": int(coherent.sum()),
    }


def add_new_tracks(pose, detections):
    tracks = _build_tracks([{"frame": frame, "detections": rows} for frame, rows in sorted(detections.items())])
    merged = deepcopy(pose)
    old_ids = [int(track["track_id"]) for track in merged.get("tracks", [])]
    offset = max(old_ids, default=-1)+1
    additions = []
    for local_id, source in sorted(tracks.items()):
        track = deepcopy(source)
        track["track_id"] = local_id+offset
        track["inference_track_scope"] = "automatic_goal_roi_pose_recovery"
        track["inference_local_track_id"] = local_id
        for observation in track["observations"]:
            observation["source_inference_track_id"] = local_id
        additions.append(track)
    merged["tracks"] = merged.get("tracks", []) + additions
    return merged, additions


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def fullshot_contact_inputs(folder, pose, source):
    """Use an independently scanned source-bound shot without relabeling old poses."""
    paths = {name: folder/(name+".json") for name in ("prediction", "features", "evidence")}
    artifacts = {name: json.loads(path.read_text()) for name, path in paths.items()}
    contact, features, evidence = (artifacts[name] for name in ("prediction", "features", "evidence"))
    if any(item.get("video_sha256") != source["clip_sha256"] for item in artifacts.values()):
        raise ValueError("Fullshot contact artifacts/source hash mismatch")
    if contact.get("manual_annotations_required") is not False:
        raise ValueError("Fullshot contact must be automatic")
    if (contact.get("model_bindings") or {}).get("mode") != "matched_training_protocol":
        raise ValueError("Keeper ROI requires a matching fullshot training protocol")
    diagnostics = (features.get("diagnostics") or {}).get("fullshot_protocol") or {}
    if (diagnostics.get("selection") != "first_automatic_camera_shot" or
            diagnostics.get("contact_time_or_labels_used_to_select_window") is not False or
            evidence.get("fullshot_protocol") != diagnostics):
        raise ValueError("Fullshot window is not source-only automatic camera segmentation")
    window = contact.get("automatic_window") or {}
    if window.get("automatic") is not True or features.get("window") != window:
        raise ValueError("Fullshot prediction/features windows disagree")
    start, end = window.get("start_frame"), window.get("end_frame_exclusive")
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) or
            not 0 <= start < end <= source["frame_count"] or
            diagnostics.get("analysis_end_frame_exclusive") != end):
        raise ValueError("Fullshot contact window outside source video")
    if not np.isclose(features["fps"], source["fps"], rtol=1e-5):
        raise ValueError("Fullshot features/source clock mismatch")
    updated = deepcopy(pose)
    updated.update(shot_start_frame=start, shot_end_frame=end)
    adjustment = {
        "method": "source_bound_fullshot_automatic_camera_window",
        "original_pose_shot_start_frame": pose.get("shot_start_frame"),
        "original_pose_shot_end_frame": pose.get("shot_end_frame"),
        "analysis_shot_start_frame": start, "analysis_shot_end_frame": end,
        "original_pose_observations_preserved": True,
        "original_pose_hash_reinterpreted_as_new_inference": False,
        "new_neural_roi_inference_required": True,
    }
    bindings = {"fullshot_"+name+"_sha256": file_sha256(path) for name, path in paths.items()}
    return contact, updated, adjustment, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path)
    parser.add_argument("--reuse-roi-cache", type=Path,
                        help="Replay source-bound cached neural detections without using the GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--events", nargs="+")
    parser.add_argument("--contact-predictions", type=Path,
                        help="Optional source-bound fullshot predictions root containing event subdirectories")
    parser.add_argument("--allow-cross-view-aliases", action="store_true",
                        help="Optional source-evidenced cross-view aliases in the recovery guard only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new keeper ROI output directory")
    if args.reuse_roi_cache is None and args.pose_model is None:
        parser.error("--pose-model is required unless --reuse-roi-cache is supplied")
    cv2.setNumThreads(2)
    model, cached_protocol = None, None
    if args.reuse_roi_cache:
        cached_protocol = json.loads((args.reuse_roi_cache/"protocol.json").read_text())
        pose_model_hash = cached_protocol["pose_model_sha256"]
        device = "cpu"
    else:
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(2)
        device = resolve_device(args.device)
        model = YOLO(str(args.pose_model))
        if model.task != "pose":
            raise ValueError("Expected pose weights")
        pose_model_hash = file_sha256(args.pose_model)
    head = json.loads(args.model.read_text())
    inputs = sorted((args.batch / "events").glob("*/attempt_001/pipeline"))
    if args.events:
        inputs = [path for path in inputs if path.parents[1].name in args.events]
    if not inputs:
        raise ValueError("No keeper pipeline inputs matched the batch and event selection")
    args.output.mkdir(parents=True)
    protocol = {
        "schema_version": "automatic-goal-roi-pose-recovery-v1", "imgsz": args.imgsz,
        "pose_model_sha256": pose_model_hash, "keeper_head_sha256": file_sha256(args.model),
        "code_sha256": {name: file_sha256(ROOT/name) for name in (
            "scripts/recover_keeper_roi_pose.py", "scripts/run_fresh_keeper_roi_experiment.py",
            "src/keeper_contact_recovery.py", "src/goalkeeper_role_association.py",
            "src/goalkeeper_pipeline.py", "src/goalkeeper_pose_direction.py",
            "src/goalkeeper_direction_evidence.py", "src/keeper_view_aliases.py", "src/step_pipeline.py")},
        "reference_labels_read_during_inference": False, "new_neural_inference": args.reuse_roi_cache is None,
        "new_training": False, "manual_roi_or_contact_used": False,
        "scope": "previously examined World Cup development regression; not final unseen-match testing",
        "device": device,
        "contact_predictions_root": str(args.contact_predictions) if args.contact_predictions else None,
        "reused_roi_protocol_sha256": file_sha256(args.reuse_roi_cache/"protocol.json") if args.reuse_roi_cache else None,
        "cross_view_aliases_requested": args.allow_cross_view_aliases,
    }
    save(args.output/"protocol.json", protocol)
    rows = []
    for directory in inputs:
        started = time.perf_counter()
        event = directory.parents[1].name
        before = predict_analysis(directory, args.model)
        manifest = json.loads((directory/"source/source_manifest.json").read_text())
        source = manifest["clips"][0]
        video = directory/"source"/source["clip_filename"]
        pose_path = directory/"pose"/(source["clip_id"]+".json")
        pose = json.loads(pose_path.read_text())
        contact_path = directory/"trained_contact/prediction.json"
        window_adjustment, additional_bindings = None, {}
        if args.contact_predictions:
            contact_path = args.contact_predictions/event/"prediction.json"
            contact, pose, window_adjustment, additional_bindings = fullshot_contact_inputs(
                args.contact_predictions/event, pose, source)
        else:
            contact = json.loads(contact_path.read_text())
        # Independent ROI inference may recover a shot rejected by the kicker branch.
        if pose.get("shot_start_frame") is None:
            pose["shot_start_frame"] = contact["automatic_window"]["start_frame"]
            pose["shot_end_frame"] = contact["automatic_window"]["end_frame_exclusive"]
        frame, reason = bounded_contact(pose, contact, source)
        result = {"event": event, "video_sha256": source["clip_sha256"], "before": before["prediction"],
                  "after": before["prediction"], "recovery_applied": False, "contact_frame": frame,
                  "source_bindings": {"source_manifest_sha256": file_sha256(directory/"source/source_manifest.json"),
                      "source_pose_sha256": file_sha256(pose_path), "contact_sha256": file_sha256(contact_path)},
                  "recovery_reason": reason, "reference_labels_read_during_inference": False}
        result["source_bindings"].update(additional_bindings)
        if window_adjustment is not None:
            result["analysis_window_adjustment"] = window_adjustment
        if frame is not None:
            geometry = automatic_goal_geometry(video, pose, frame)
            plan = plan_goal_roi(pose, geometry, frame)
            result["plan"] = plan
            if plan is None:
                result["recovery_reason"] = "no_coherent_automatic_goal_roi"
            else:
                if args.reuse_roi_cache:
                    cached_path = args.reuse_roi_cache/"pose"/(event+".json")
                    cached = json.loads(cached_path.read_text())
                    if (cached["plan"] != plan or cached["source_bindings"] != result["source_bindings"] or
                            cached["source_video_sha256"] != source["clip_sha256"] or
                            cached["pose_model_sha256"] != pose_model_hash or cached_protocol["imgsz"] != args.imgsz):
                        raise ValueError("Cached ROI detections no longer match source, contact, pose or inference plan")
                    detections = {int(frame): items for frame, items in cached["detections"].items()}
                    receipts = cached["receipts"]
                    result["source_bindings"]["reused_roi_cache_sha256"] = file_sha256(cached_path)
                else:
                    detections, receipts = infer_roi(model, video, plan, device=device, imgsz=args.imgsz)
                merged, additions = add_new_tracks(pose, detections)
                save(args.output/"pose"/(event+".json"), {
                    "plan": plan, "new_tracks": additions, "detections": detections,
                    "receipts": receipts, "geometry": geometry, "source_bindings": result["source_bindings"],
                    "pose_model_sha256": protocol["pose_model_sha256"], "source_video_sha256": source["clip_sha256"],
                })
                keeper = recover_keeper(merged, contact, source, video, geometry,
                                        allow_cross_view_aliases=args.allow_cross_view_aliases)
                unguarded = predict_keeper(keeper, head)
                raw = guard_recovered_prediction(keeper, unguarded)
                recovered_rule = supported_rule(keeper) if raw["actual_dive_evidence_supported"] else "unknown"
                direction, method = hybrid_direction(before["existing_rule_direction"],
                    recovered_rule if recovered_rule != "unknown" else raw["direction"])
                if before["existing_rule_direction"] == "unknown" and recovered_rule != "unknown":
                    method = "observed_lateral_body_direction"
                result["keeper"] = keeper
                result["learned_prediction"] = raw
                result["unguarded_learned_prediction"] = unguarded
                result["recovery_reason"] = keeper.get("reason")
                result["recovery_applied"] = direction != "unknown" or raw["height"] != "unknown"
                result["after"] = {**before["prediction"], "direction": direction,
                    "dive_direction": {"left": "image_left", "center": "center", "right": "image_right", "unknown": "unknown"}[direction],
                    "direction_method": "automatic_goal_roi_"+method,
                    "height_model_candidate": raw["height"] if raw["height"] != "unknown" else before["prediction"]["height_model_candidate"]}
                result["new_pose_tracks"] = len(additions)
                result["new_pose_observations"] = sum(len(track["observations"]) for track in additions)
        result["runtime_seconds"] = time.perf_counter()-started
        save(args.output/(event+".json"), result)
        summary = {key: value for key, value in result.items() if key not in ("keeper", "learned_prediction", "source_bindings")}
        summary.update(prediction_sha256=file_sha256(args.output/(event+".json")),
                       head_reasons=(result.get("learned_prediction") or {}).get("reasons"),
                       role_reason=((result.get("keeper") or {}).get("role") or {}).get("reason"))
        rows.append(summary)
        save(args.output/"summary.json", {"protocol": protocol, "rows": rows, "events": len(rows),
             "reference_labels_read_during_inference": False})
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
