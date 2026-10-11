"""Run fresh pose extraction on hashed course context clips, without old caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.final_results import sha256
from src.runtime import inference_provenance, resolve_device
from src.step_pipeline import process_video_steps


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-shot-sec", type=float, default=30.0)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from ultralytics import YOLO
    source = json.loads((args.clips_dir / "source_manifest.json").read_text(encoding="utf-8"))
    records = {r["clip_id"]: r for r in source["clips"]}
    clip_ids = args.clips or list(records)
    if len(clip_ids) != len(set(clip_ids)) or any(cid not in records for cid in clip_ids):
        raise ValueError("Requested clip IDs must be distinct members of the source manifest")
    if args.batch_size < 1 or args.imgsz < 32 or args.max_shot_sec <= 0:
        raise ValueError("Invalid positive inference settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model_hash = sha256(args.model)
    script_hash = sha256(Path(__file__))
    algorithm_hash = sha256(ROOT / "src/step_pipeline.py")
    module_hashes = {name: sha256(ROOT / "src" / name) for name in (
        "step_pipeline.py", "kicker_analysis.py", "field_masks.py", "goal_geometry.py", "runtime.py",
        "camera_timing_v2.py")}
    settings = {"imgsz": args.imgsz, "batch_size": args.batch_size,
                "max_shot_sec": args.max_shot_sec, "confidence": 0.10,
                "keypoint_confidence": 0.15, "roi_mode": "auto",
                "shot_selection": "first_viable", "device": device,
                "use_field_mask": True, "require_goal_context": True, "exclude_goal_line_actors": True,
                "shot_scan_method": "adaptive"}
    settings["event_slice_policy"] = "selected_shot_must_overlap_manual_course_slice_contact_remains_automatic"
    protocol = inference_provenance(args.device, device, [args.model])
    protocol.update({"settings": settings, "source_manifest_sha256": sha256(args.clips_dir / "source_manifest.json"),
                     "script_sha256": script_hash, "algorithm_sha256": algorithm_hash,
                     "module_sha256": module_hashes,
                     "cohort_clip_ids": clip_ids, "old_caches_used": False,
                     "reference_labels_used_for_prediction": False, "human_verified": False,
                     "held_out_test": False, "complete_raw_to_all_features": False})
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    model = YOLO(str(args.model))
    if model.task != "pose" or list(model.model.yaml.get("kpt_shape", [])) != [17, 3]:
        raise ValueError("Course pose extraction requires a COCO 17-keypoint pose model")
    results = []
    for index, clip_id in enumerate(clip_ids):
        video = args.clips_dir / records[clip_id]["clip_filename"]
        if sha256(video) != records[clip_id]["clip_sha256"]:
            raise ValueError(f"Source clip hash mismatch: {clip_id}")
        path = args.output_dir / f"{clip_id}.json"
        identity = {"source_sha256": records[clip_id]["clip_sha256"], "model_sha256": model_hash,
                    "algorithm_sha256": algorithm_hash, "script_sha256": script_hash,
                    "module_sha256": module_hashes,
                    "settings": settings}
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if not args.resume or previous.get("inference_identity") != identity:
                raise FileExistsError(f"Existing experiment is not safely reusable: {path}")
            results.append(previous)
            print(f"Reused matching fresh run {index + 1}/{len(clip_ids)}: {clip_id}", flush=True)
            continue
        started = time.monotonic()
        result = process_video_steps(str(video), model, clip_id=clip_id,
                    batch_size=args.batch_size, max_shot_sec=args.max_shot_sec,
                    pose_conf=0.10, keypoint_conf=0.15, roi_mode="auto",
                    predict_kwargs={"imgsz": args.imgsz, "conf": 0.10, "device": device},
                    shot_selection="first_viable", use_field_mask=True, require_goal_context=True,
                    shot_scan_method="adaptive",
                    event_slice_interval=(records[clip_id]["original_slice_start_frame_in_clip"],
                                          records[clip_id]["original_slice_end_frame_in_clip"]))
        result.update({"inference_identity": identity, "source_mapping": records[clip_id],
                       "match_key": records[clip_id]["match_id"], "event_id": records[clip_id]["event_id"],
                       "kicker": records[clip_id]["kicker"], "team": records[clip_id]["team"],
                       "inference_elapsed_sec": round(time.monotonic() - started, 3),
                       "new_neural_inference": True, "human_verified": False,
                       "accuracy_validated": False, "reference_labels_used_for_prediction": False})
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        results.append(result)
        summary = {"clip_id": clip_id, "status": result["status"],
                   "step_count": result["step_count"], "kicking_foot": result["kicking_foot"],
                   "kick_frame": result["kick_frame"], "selected_track_id": result["selected_track_id"],
                   "selected_shot_index": result["shot_selection"]["selected_shot_index"],
                   "review_required": result["review_required"], "elapsed_sec": result["inference_elapsed_sec"]}
        print(json.dumps({"completed": index + 1, "total": len(clip_ids), **summary}), flush=True)
        (args.output_dir / "progress.json").write_text(json.dumps({"completed": index + 1,
                        "total": len(clip_ids), "last_clip": clip_id}, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "step_diagnostics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
