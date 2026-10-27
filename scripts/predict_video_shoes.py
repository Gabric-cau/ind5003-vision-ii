"""Run the trained shoe heads from a video and completed automatic stage artifacts."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fresh_shoe_approach_experiment import bind_event, decode_interval, infer_event, read, sha, write
from src.shoe_supervised_model import extract_evidence, predict_event
from src.shoe_video_recovery import bind_recovered_event
from src.shoe_camera_segments import recover_missing_foot

BOUND_INPUT_KEYS = ("source_mapping", "fps", "scene_interval_frames", "interval_frames",
    "inference_interval_frames", "contact_bracket_frames", "actor_track_id",
    "original_pose_selected_track_id", "original_camera_motion", "original_phase_result", "contact")


def bound_pose_cache(directory, bound, pose_model, manifest_hash):
    """Reuse neural evidence only after exact current binding and pixel checks."""
    receipt = {"requested": directory is not None, "used": False}
    if directory is None:
        return None, {**receipt, "reason": "not_requested"}
    directory = Path(directory)
    paths = {name: directory / name for name in ("inputs.json", "poses33.json", "roi_plan.json", "prediction.json")}
    if not all(path.is_file() for path in paths.values()):
        return None, {**receipt, "reason": "cache_artifact_missing"}
    cached, prediction, plan = (read(paths[name]) for name in ("inputs.json", "prediction.json", "roi_plan.json"))
    if (cached.get("poses33_sha256") != sha(paths["poses33.json"])
            or cached.get("model_sha256") != sha(pose_model)
            or cached.get("source_manifest_sha256") != manifest_hash
            or cached.get("reference_labels_read") is not False
            or prediction.get("source_code_sha256", {}).get("scripts/run_fresh_shoe_approach_experiment.py")
                != sha(ROOT / "scripts/run_fresh_shoe_approach_experiment.py")):
        return None, {**receipt, "reason": "cache_provenance_mismatch"}
    differing = [key for key in BOUND_INPUT_KEYS if cached.get(key) != bound[key]]
    if differing or plan != bound["plan"]:
        return None, {**receipt, "reason": "current_binding_differs", "differing_fields": differing,
                      "roi_plan_differs": plan != bound["plan"]}
    hashes = {}
    for frame, image, _ in decode_interval(bound):
        hashes[str(frame)] = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if hashes != cached["decoded_frame_hashes"]:
        return None, {**receipt, "reason": "decoded_pixels_differ"}
    records = read(paths["poses33.json"])
    quality_keys = ("decoded_interval_frames", "distinct_decoded_pixel_frames", "repeated_decoded_pixel_frames",
        "accepted_roi_pose_calls", "unique_pixel_accepted_roi_pose_calls", "roi_identity_status_counts",
        "original_actor_missing_frames")
    quality = {key: prediction[key] for key in quality_keys}
    quality.update(actual_inference_calls=0,
        cached_original_inference_calls=prediction.get("cached_original_inference_calls", prediction["actual_inference_calls"]))
    receipt.update(used=True, reason="exact_source_window_actor_roi_camera_and_pixels_match",
                   cache_directory=str(directory), artifact_sha256={name: sha(path) for name, path in paths.items()})
    return (records, hashes, quality), receipt


def load_bound_model(model_dir, pose33_path):
    metadata = read(model_dir / "deployment_metadata.json")
    model_path = model_dir / "deployment.joblib"
    if sha(model_path) != metadata["model_sha256"]:
        raise ValueError("Supervised shoe model changed after its metadata was written")
    contract = metadata["evidence_contract"]
    if sha(ROOT / "src/shoe_supervised_model.py") != contract["feature_code_sha256"]:
        raise ValueError("Shoe feature implementation differs from the trained model")
    if sha(pose33_path) not in contract["pose_model_sha256"]:
        raise ValueError("Pose checkpoint differs from the supervised training evidence")
    if version("scikit-learn") != metadata["dependencies"]["scikit-learn"]:
        raise ValueError("Use the scikit-learn version recorded in deployment_metadata.json")
    model = joblib.load(model_path)
    if model.get("evidence_contract") != contract:
        raise ValueError("Model and metadata evidence contracts disagree")
    return model, metadata


def run(args):
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; existing predictions are preserved")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    started = time.perf_counter()
    model, metadata = load_bound_model(args.model, args.pose33_model)
    source_dir = args.analysis_dir / "source"
    manifest_path = source_dir / "source_manifest.json"
    manifest = read(manifest_path)
    sources = manifest.get("clips", [])
    if len(sources) != 1 or manifest.get("manual_annotations_used") is not False:
        raise ValueError("Expected one unannotated-video source manifest")
    source = sources[0]
    filename = source.get("clip_filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Source clip filename must be a plain basename")
    bindings = SimpleNamespace(clips_dir=source_dir, pose_dir=args.analysis_dir / "pose",
                              ball_dir=args.analysis_dir / "ball", phase_dir=args.analysis_dir / "phase")
    bound, provenance = bind_event(bindings, source, sha(manifest_path))
    if bound is None:
        recovered, recovery = bind_recovered_event(args.analysis_dir, source, sha(manifest_path), args.threads)
        provenance["automatic_contact_recovery"] = recovery
        if recovered is not None:
            bound = recovered
            provenance["original_unavailable_reason"] = provenance.pop("unavailable_reason", None)
            provenance["selected_binding"] = "measured_trained_contact_actor_fallback"
    args.output.mkdir(parents=True)
    write(args.output / "input_receipts.json", provenance)
    result = {"schema_version": "unannotated-video-supervised-shoes-v1", "clip_id": source["clip_id"],
        "source_sha256": source["clip_sha256"], "manual_annotations_used": False,
        "binding_method": provenance.get("selected_binding", "original_independent_contact_and_phase"),
        "labels_read_at_inference": False, "supervised_model_sha256": metadata["model_sha256"],
        "pose_model_sha256": sha(args.pose33_model), "fps": source["fps"],
        "coordinate_space": "original_source_image_pixels", "frame_clock": "clip_local_zero_based",
        "full_approach_visible": False, "count_complete": False, "total_run_up_steps": None,
        "human_verified": False, "accuracy_validated_on_this_video": False,
        "interpretation": "Observed automatic approach-window landing candidates and anatomical striking-foot candidate",
        "limitations": ["The approach window may omit earlier steps.",
                        "A pose-derived landing score is not proof of physical shoe-ground contact.",
                        "Failure to bind actor, contact or approach leaves the output unavailable.",
                        "Scores are uncalibrated; new-video accuracy requires separate evaluation."]}
    if bound is None:
        result.update(status="unavailable", reason=provenance["unavailable_reason"],
                      landings=None, candidate_count=None, striking_foot="unknown",
                      right_score_uncalibrated=None, actual_inference_calls=0, automatic_window=None)
    else:
        cv2.setNumThreads(args.threads)
        write(args.output / "roi_plan.json", bound["plan"])
        with threadpool_limits(limits=args.threads):
            cached, cache_receipt = bound_pose_cache(getattr(args, "pose_cache", None), bound,
                                                    args.pose33_model, sha(manifest_path))
            records, hashes, quality = cached if cached is not None else infer_event(bound, args.pose33_model)
            write(args.output / "poses33.json", records)
            inputs = {key: bound[key] for key in BOUND_INPUT_KEYS}
            inputs.update(source_manifest_sha256=sha(manifest_path), input_receipts=provenance,
                model_sha256=sha(args.pose33_model), reference_labels_read=False, new_neural_inference=cached is None,
                pose_cache=cache_receipt,
                human_verified=False, decoded_frame_hashes=hashes,
                poses33_sha256=sha(args.output / "poses33.json"), coordinate_space="original_source_image_pixels")
            write(args.output / "inputs.json", inputs)
            evidence = extract_evidence(inputs, records)
            prediction = predict_event(model, evidence, bound["fps"])
            camera_recovery, measured_camera = {"used": False, "reason": "optional_recovery_disabled"}, None
            if getattr(args, "recover_camera_segments", False):
                prediction, camera_recovery, measured_camera = recover_missing_foot(
                    inputs, records, model, prediction, video=bound["video"],
                    pose=read(args.analysis_dir / "pose" / (source["clip_id"] + ".json")))
            if measured_camera is not None:
                write(args.output / "camera_segments.json", measured_camera)
                camera_recovery["camera_sha256"] = sha(args.output / "camera_segments.json")
            write(args.output / "camera_segment_recovery.json", camera_recovery)
        result.update(status="automatic_candidates", **prediction, **quality,
                      automatic_window={"start_frame": bound["interval_frames"][0],
                                        "end_frame": bound["interval_frames"][1]},
                      supported_side_frames=evidence["supported_side_frames"],
                      foot_camera_recovery=camera_recovery,
                      pose_cache=cache_receipt,
                      inputs_sha256=sha(args.output / "inputs.json"),
                      poses33_sha256=sha(args.output / "poses33.json"))
        if prediction["landings"] is None and prediction["striking_foot"] == "unknown":
            result.update(status="unavailable", reason="no_eligible_paired_pose_evidence")
    result["runtime_seconds_including_model_load_and_pose_inference"] = time.perf_counter() - started
    result["prerequisite_stage_runtime_included"] = False
    result["source_code_sha256"] = {p: sha(ROOT / p) for p in (
        "scripts/predict_video_shoes.py", "scripts/run_fresh_shoe_approach_experiment.py", "src/shoe_supervised_model.py",
        "src/shoe_video_recovery.py", "src/shoe_camera_segments.py",
        "scripts/recover_shoe_evidence.py", "src/kick_contact.py")}
    write(args.output / "prediction.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Completed analyze_penalty_video directory with source, pose, ball and phase")
    parser.add_argument("--model", type=Path, required=True, help="Supervised deployment model directory")
    parser.add_argument("--pose33-model", type=Path, required=True, help="Bound MediaPipe heavy task checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--pose-cache", type=Path,
                        help="Optional prior shoe output; reuse only after exact source, ROI, camera and pixel checks")
    parser.add_argument("--recover-camera-segments", action="store_true",
                        help="Experimental foot-only camera recovery; requires a valid automatic contact window")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output / "prediction.json"), "status": result["status"],
                      "candidate_count": result["candidate_count"], "striking_foot": result["striking_foot"]}))


if __name__ == "__main__":
    main()
