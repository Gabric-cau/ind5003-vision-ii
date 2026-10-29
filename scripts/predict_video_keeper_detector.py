"""Produce offline active-keeper boxes and optional pose without manual labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.active_keeper_detector import run_video, validate_model, validate_paths
from src.runtime import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.stride, args.threads) < 1:
        parser.error("stride and threads must be positive")
    video, weights, output, pose_weights = validate_paths(args.video, args.weights, args.output, args.pose_model)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime-cache" / "ultralytics"))
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"sync": False})
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    device = resolve_device(args.device)
    detector = YOLO(str(weights))
    validate_model(detector)
    pose = YOLO(str(pose_weights)) if pose_weights is not None else None
    if pose is not None:
        validate_model(pose, pose=True)
    result = run_video(video, output, detector, weights=weights, device=device, stride=args.stride,
                       pose_model=pose, pose_weights=pose_weights)
    print(json.dumps({"output": str(output / "prediction.json"), "overlay": str(output / "overlay.mp4"),
                      "sampled_frames": result["sampled_frame_count"],
                      "frames_with_detection": result["frames_with_detection"],
                      "seconds": result["end_to_end_seconds"], "accuracy_measured": False}), flush=True)


if __name__ == "__main__":
    main()
