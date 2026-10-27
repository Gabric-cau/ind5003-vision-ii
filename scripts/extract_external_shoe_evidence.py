"""Extract source-only contact and paired shoe evidence from external event videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_shoe_evidence import read, write, sha, build_bound


def shoe_stage(args):
    from scripts.run_fresh_shoe_approach_experiment import infer_event
    from src.shoe_supervised_model import extract_evidence
    sample = read(args.event_dir / "source.json")
    contact = read(args.event_dir / "contact/prediction.json")
    evidence = read(args.event_dir / "contact/evidence.json")
    if sha(sample["video_path"]) != sample["video_sha256"]:
        raise ValueError("External video changed")
    source = {"clip_id": sample["clip_id"], "clip_sha256": sample["video_sha256"],
        "clip_filename": Path(sample["video_path"]).name, "frame_count": sample["frame_count"], "fps": sample["fps"]}
    mapping = {"mapping_kind": "original_video_no_reencoding", "source_start_frame": 0,
        "source_end_frame_exclusive": contact["automatic_window"]["end_frame_exclusive"],
        "shot_sha256": sample["video_sha256"], "fps": sample["fps"],
        "width": sample["width"], "height": sample["height"]}
    # lossless_shot is the shared helper's legacy mapping-container name; this
    # run directly decodes the original video and has no intermediate encoding.
    bundle = {"source_mapping": source, "video_path": sample["video_path"],
              "old_selected_actor": None, "lossless_shot": mapping}
    bound, reason = build_bound(bundle, contact, evidence)
    directory = args.event_dir / "shoes"
    if directory.exists():
        raise FileExistsError("External shoe evidence already exists")
    directory.mkdir()
    if bound is None:
        write(directory / "summary.json", {"status": "unavailable", "reason": reason})
        return
    failed = [pair[1] for pair in evidence.get("camera", {}).get("failed_pairs", [])]
    if failed:
        first_failed = min(failed)
        bound["original_camera_motion"]["transforms"] = {
            k: value if int(k) < first_failed else None
            for k, value in bound["original_camera_motion"]["transforms"].items()}
    started = time.perf_counter()
    records, hashes, quality = infer_event(bound, args.pose33_model)
    write(directory / "poses33.json", records)
    inputs = {k: bound[k] for k in ("source_mapping", "fps", "scene_interval_frames", "interval_frames",
        "inference_interval_frames", "contact_bracket_frames", "actor_track_id", "original_pose_selected_track_id",
        "original_camera_motion", "original_phase_result", "contact", "dense_actor_association")}
    inputs.update(model_sha256=sha(args.pose33_model), reference_labels_read=False,
        new_neural_inference=True, human_verified=False, decoded_frame_hashes=hashes,
        poses33_sha256=sha(directory / "poses33.json"), coordinate_space="original_source_image_pixels",
        extraction_scope="complete_automatic_first_shot_no_human_timing",
        clock_mapping=mapping, contact_prediction_sha256=sha(args.event_dir / "contact/prediction.json"))
    write(directory / "inputs.json", inputs)
    features = extract_evidence(inputs, records)
    write(directory / "summary.json", {"status": "fresh_external_pose_candidate", **quality,
        "supported_side_frames": features["supported_side_frames"],
        "foot_features_available": features["foot_vector"] is not None,
        "runtime_seconds": time.perf_counter() - started,
        "inputs_sha256": sha(directory / "inputs.json"), "poses33_sha256": sha(directory / "poses33.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path)
    parser.add_argument("--pose33-model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-for-videos", type=float, default=0.,
                        help="Maximum seconds to wait for an in-progress source upload, per event")
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(args.runtime_root))
    cv2.setNumThreads(2)
    if args.event_dir:
        shoe_stage(args)
        return
    if args.output.exists():
        raise FileExistsError("Choose a new external extraction directory")
    document = read(args.inputs)
    if document.get("manual_annotations_included") is not False:
        raise ValueError("Expected a label-free external video manifest")
    allowed = {"clip_id", "match_group", "video_path", "video_sha256", "fps", "frame_count", "width", "height"}
    if any(set(s) != allowed or s["match_group"] == "world_cup_2022" or s["clip_id"] == "match_09"
           for s in document["samples"]):
        raise ValueError("Input includes annotations, held-out or quarantined events")
    config = read(args.config)
    for key in ("contact_model", "pose_model", "ball_model", "pose33_model", "shoe_python"):
        if not Path(config[key]).is_absolute():
            config[key] = str(args.config.parent / config[key])
    args.output.mkdir(parents=True)
    rows = []
    for sample in document["samples"]:
        directory = args.output / sample["clip_id"]
        directory.mkdir()
        write(directory / "source.json", sample)
        started = time.perf_counter()
        try:
            wait_started = time.perf_counter()
            while not Path(sample["video_path"]).exists() and time.perf_counter() - wait_started < args.wait_for_videos:
                time.sleep(2.)
            upload_wait_seconds = time.perf_counter() - wait_started
            started = time.perf_counter()
            if sha(sample["video_path"]) != sample["video_sha256"]:
                raise ValueError("External video checksum mismatch")
            command = [sys.executable, str(args.runtime_root / "scripts/predict_video_contact.py"),
                "--video", sample["video_path"], "--model", config["contact_model"],
                "--ball-model", config["ball_model"], "--pose-model", config["pose_model"],
                "--output", str(directory / "contact"), "--device", args.device, "--threads", "2"]
            with (directory / "contact.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = [config["shoe_python"], str(Path(__file__).resolve()), "--runtime-root", str(args.runtime_root),
                "--event-dir", str(directory), "--pose33-model", config["pose33_model"]]
            with (directory / "shoes.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = read(directory / "shoes/summary.json")
        except Exception as error:
            result = {"status": "failed", "reason": str(error)}
        row = {"clip_id": sample["clip_id"], "total_runtime_seconds": time.perf_counter() - started,
               "upload_wait_seconds_excluded": upload_wait_seconds, **result}
        rows.append(row)
        write(args.output / "summary.json", {"manual_annotations_used": False, "world_cup_holdout_accessed": False, "rows": rows})
        print(json.dumps({k: row.get(k) for k in ("clip_id", "status", "foot_features_available", "actual_inference_calls", "total_runtime_seconds", "reason")}), flush=True)


if __name__ == "__main__":
    main()
