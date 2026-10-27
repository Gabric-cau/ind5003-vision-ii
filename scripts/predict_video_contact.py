"""Offline video-to-contact inference with no manual labels or time inputs."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime import file_sha256, resolve_device
from src.contact_video_features import extract_contact_record, SETTINGS
from src.contact_temporal_model import (predict_contact, LAGS_SECONDS, SMOOTHING_SECONDS,
                                        validate_prediction_bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Optional fitted rule-offset model directory")
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Prediction output already exists")
    if args.threads < 1:
        parser.error("threads must be positive")
    started = time.perf_counter()
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    metadata = json.loads((args.model / "deployment_metadata.json").read_text())
    weights = args.model / "deployment_weights.npz"
    if file_sha256(weights) != metadata["weights_sha256"]:
        raise ValueError("Contact model hash mismatch")
    visual = {p.name: file_sha256(p) for p in (args.ball_model, args.pose_model)}
    if visual != metadata["visual_model_sha256"]:
        raise ValueError("Visual model differs from training")
    settings = metadata["visual_settings"]
    refine = bool(settings.get("pose_gap_roi"))
    from src.contact_roi_refinement import REFINEMENT_SETTINGS, refine_contact_record
    expected_keys = set(SETTINGS) | (set(REFINEMENT_SETTINGS) if refine else set())
    if set(settings) != expected_keys or (refine and any(settings[k] != v for k, v in REFINEMENT_SETTINGS.items())):
        raise ValueError("Unsupported visual feature settings schema")
    if metadata["lags_seconds"] != list(LAGS_SECONDS) or metadata["smoothing_seconds"] != SMOOTHING_SECONDS:
        raise ValueError("Temporal settings differ from training")
    for name, expected in metadata["extractor_code_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError("Feature code differs from training")
    capture = cv2.VideoCapture(str(args.video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open input video")
        sample = {"core_id": "new_video", "clip_id": args.video.stem, "match_group": "unassigned",
                  "video_path": str(args.video), "video_sha256": file_sha256(args.video),
                  "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                  "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
    finally:
        capture.release()
    device = resolve_device(args.device)
    ball, pose = YOLO(str(args.ball_model)), YOLO(str(args.pose_model))
    record, evidence = extract_contact_record(sample, ball, pose,
        [args.ball_model, args.pose_model], device=device,
        settings={k: settings[k] for k in SETTINGS})
    if refine:
        record, evidence = refine_contact_record(record, evidence, args.video, pose, device=device)
    if record["feature_names"] != metadata["feature_names"]:
        raise ValueError("Feature schema differs from training")
    validate_prediction_bindings(record, metadata)
    with np.load(weights, allow_pickle=False) as parameters:
        prediction = predict_contact(record, parameters)
    temporal_prediction = prediction
    calibrated_prediction = None
    if args.calibration:
        from scripts.train_contact_offset_calibration import calibrate_contact
        calibration = json.loads((args.calibration / "deployment_calibration.json").read_text())
        if calibration["source_visual_metadata_sha256"] != file_sha256(args.model / "deployment_metadata.json"):
            raise ValueError("Calibration and temporal model use different visual metadata")
        calibrated_prediction = calibrate_contact(record, calibration["fitted_offset_seconds"], calibration)
        prediction = calibrated_prediction
    output = {"video_sha256": sample["video_sha256"], "prediction": prediction,
              "selected_method": "fitted_rule_offset" if args.calibration else "temporal_classifier",
              "temporal_prediction": temporal_prediction, "calibrated_prediction": calibrated_prediction,
              "manual_annotations_required": False, "automatic_window": record["window"],
              "baselines": record["baselines"], "diagnostics": record["diagnostics"],
              "feature_runtime": record["runtime"], "device": device,
              "end_to_end_seconds_including_model_load": time.perf_counter() - started,
              "scope": "experimental contact localization on a penalty clip starting with its shooting view",
              "promoted_to_default_pipeline": False}
    args.output.mkdir(parents=True)
    for name, data in (("prediction.json", output), ("features.json", record), ("evidence.json", evidence)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
