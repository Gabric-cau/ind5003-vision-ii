"""Contact extraction and inference on the complete first automatic shot, up to 60s."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contact_fullshot import (fullshot_settings, scan_fullshot, stamp_protocol,
                                  reusable_cache, validate_fullshot_prediction)
from src.contact_video_features import extract_contact_record, SETTINGS
from src.contact_roi_refinement import refine_contact_record, REFINEMENT_SETTINGS
from src.contact_temporal_model import predict_contact
from src.contact_actor_support import RECOVERY_SETTINGS, recover_actor_support
from src.contact_causal_actor import CAUSAL_SETTINGS, recover_causal_actor, predict_causal_contact
from src.contact_candidate_window import WINDOW_SETTINGS, bind_candidate_window, predict_candidate_contact
from src.runtime import file_sha256, resolve_device


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--inputs", type=Path)
    parser.add_argument("--event-id", default="new_video")
    parser.add_argument("--cores", nargs="+")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cache-model", type=Path, help="Original metadata binding an unchanged shorter-window cache")
    parser.add_argument("--single-output", action="store_true", help="Write the three result files directly into --output")
    parser.add_argument("--allow-window-transfer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.single_output and (args.video is None or args.cores):
        parser.error("--single-output requires --video and does not accept --cores")
    if args.output.exists():
        raise FileExistsError("Use a new fullshot output directory")
    import cv2
    import numpy as np
    metadata = read_json(args.model / "deployment_metadata.json")
    weights = args.model / "deployment_weights.npz"
    if file_sha256(weights) != metadata["weights_sha256"]:
        raise ValueError("Contact model hash mismatch")
    if {p.name: file_sha256(p) for p in (args.ball_model, args.pose_model)} != metadata["visual_model_sha256"]:
        raise ValueError("Visual models differ from contact training")
    for name, expected in metadata["extractor_code_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Feature code differs from training: {name}")
    settings = fullshot_settings(metadata["visual_settings"])
    actor_recovery = settings.get("actor_support_recovery_version") is not None
    causal_recovery = settings.get("causal_actor_recovery_version") is not None
    bounded_recovery = settings.get("recovered_candidate_window_version") is not None
    if actor_recovery and any(settings.get(k) != v for k, v in RECOVERY_SETTINGS.items()):
        raise ValueError("Unsupported actor support protocol")
    if causal_recovery and (not actor_recovery or any(settings.get(k) != v for k, v in CAUSAL_SETTINGS.items())):
        raise ValueError("Unsupported causal actor protocol")
    if bounded_recovery and (not causal_recovery or any(settings.get(k) != v for k, v in WINDOW_SETTINGS.items())):
        raise ValueError("Unsupported candidate window protocol")
    cache_metadata = read_json(args.cache_model / "deployment_metadata.json") if args.cache_model else metadata
    wrapper_keys = set(RECOVERY_SETTINGS) | set(CAUSAL_SETTINGS) | set(WINDOW_SETTINGS)
    cache_visual = {k: v for k, v in fullshot_settings(cache_metadata["visual_settings"]).items() if k not in wrapper_keys}
    target_visual = {k: v for k, v in settings.items() if k not in wrapper_keys}
    if cache_visual != target_visual or cache_metadata["visual_model_sha256"] != metadata["visual_model_sha256"]:
        raise ValueError("Cache protocol changes more than the permitted window cap")
    for name, expected in cache_metadata["extractor_code_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("Cache extractor code differs")
    refine = bool(settings.get("pose_gap_roi"))
    expected_keys = (set(SETTINGS) | (set(REFINEMENT_SETTINGS) if refine else set())
                     | (set(RECOVERY_SETTINGS) if actor_recovery else set()) | (set(CAUSAL_SETTINGS) if causal_recovery else set())
                     | (set(WINDOW_SETTINGS) if bounded_recovery else set()))
    if set(settings) != expected_keys or (refine and any(settings[k] != v for k, v in REFINEMENT_SETTINGS.items())):
        raise ValueError("Unsupported visual settings")
    if args.inputs:
        samples = read_json(args.inputs)["samples"]
    else:
        cap = cv2.VideoCapture(str(args.video))
        try:
            if not cap.isOpened():
                raise ValueError("Cannot open source video")
            samples = [{"core_id": args.event_id, "clip_id": args.video.stem, "match_group": "unassigned",
                        "video_path": str(args.video), "video_sha256": file_sha256(args.video),
                        "fps": float(cap.get(cv2.CAP_PROP_FPS)), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}]
        finally:
            cap.release()
    if args.cores:
        if set(args.cores) - {s["core_id"] for s in samples}:
            raise ValueError("Unknown requested event IDs")
        samples = [s for s in samples if s["core_id"] in args.cores]
    if len({s["core_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate event IDs")
    cv2.setNumThreads(args.threads)
    device = resolve_device(args.device)
    ball = pose = None
    args.output.mkdir(parents=True)
    for sample in samples:
        started = time.perf_counter()
        if any("reference" in k or "contact_seconds" in k for k in sample):
            raise ValueError("Annotations cannot enter fullshot inference")
        video = args.video_root / (sample["core_id"] + ".mp4") if args.video_root else Path(sample["video_path"])
        if file_sha256(video) != sample["video_sha256"]:
            raise ValueError("Fullshot source hash mismatch")
        shot, frame_ids = scan_fullshot(video, settings)
        if shot["fps"] != sample["fps"]:
            raise ValueError("Fullshot source FPS mismatch")
        cache_source, record, evidence = None, None, None
        if args.cache_dir:
            cache = args.cache_dir / (sample["core_id"] + ".json")
            cache_evidence = args.cache_dir / "evidence" / cache.name
            if cache.exists() and cache_evidence.exists():
                candidate, visual = read_json(cache), read_json(cache_evidence)
                if reusable_cache(candidate, visual, sample, cache_metadata, shot, frame_ids):
                    record, evidence = candidate, visual
                    cache_source = {"features_sha256": file_sha256(cache), "evidence_sha256": file_sha256(cache_evidence),
                                    "sampled_frames_unchanged": True}
        if record is None:
            if ball is None:
                import torch
                from ultralytics import YOLO
                torch.set_num_threads(args.threads)
                ball, pose = YOLO(str(args.ball_model)), YOLO(str(args.pose_model))
                if ball.task != "detect" or ball.names[32] != "sports ball" or pose.task != "pose":
                    raise ValueError("Require COCO ball and pose checkpoints")
            print("extracting " + sample["core_id"], flush=True)
            record, evidence = extract_contact_record(sample, ball, pose, [args.ball_model, args.pose_model],
                video_root=args.video_root, device=device, settings={k: settings[k] for k in SETTINGS})
            if refine:
                record, evidence = refine_contact_record(record, evidence, video, pose, device=device)
        record, evidence = stamp_protocol(record, evidence, shot, cache_source=cache_source)
        if actor_recovery and not record["settings"].get("actor_support_recovery_version"):
            record, evidence = recover_actor_support(record, evidence, sample["height"])
        if causal_recovery and not record["settings"].get("causal_actor_recovery_version"):
            record, evidence = recover_causal_actor(record, evidence, sample["height"])
        if bounded_recovery and not record["settings"].get("recovered_candidate_window_version"):
            record, evidence = bind_candidate_window(record, evidence)
        bindings = validate_fullshot_prediction(record, metadata, allow_window_transfer=args.allow_window_transfer)
        with np.load(weights, allow_pickle=False) as parameters:
            predict = predict_candidate_contact if bounded_recovery else predict_causal_contact if causal_recovery else predict_contact
            prediction = predict(record, parameters)
        output = {"video_sha256": sample["video_sha256"], "prediction": prediction,
                  "selected_method": "temporal_classifier", "temporal_prediction": prediction,
                  "calibrated_prediction": None, "manual_annotations_required": False,
                  "automatic_window": record["window"], "baselines": record["baselines"],
                  "diagnostics": record["diagnostics"], "feature_runtime": record["runtime"], "device": device,
                  "end_to_end_seconds_including_model_load": time.perf_counter() - started,
                  "model_bindings": bindings, "contact_model_metadata_sha256": file_sha256(args.model / "deployment_metadata.json"),
                  "scope": "first automatic camera shot up to 60 seconds; later shots are not searched",
                  "promoted_to_default_pipeline": False}
        folder = args.output if args.single_output else args.output / sample["core_id"]
        for name, data in (("prediction.json", output), ("features.json", record), ("evidence.json", evidence)):
            write_json(folder / name, data)
        if not args.single_output:
            write_json(args.output / "features" / (sample["core_id"] + ".json"), record)
            write_json(args.output / "features" / "evidence" / (sample["core_id"] + ".json"), evidence)
        print(json.dumps({"event": sample["core_id"], "prediction": prediction, "window": record["window"],
                          "cache_reused": cache_source is not None, "bindings": bindings,
                          "seconds": output["end_to_end_seconds_including_model_load"]}), flush=True)


if __name__ == "__main__":
    main()
