"""Extract label-free ball/pose temporal evidence with frozen local YOLO weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contact_video_features import extract_contact_record, SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", nargs="+")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pose-imgsz", type=int, default=SETTINGS["pose_imgsz"])
    parser.add_argument("--pose-conf", type=float, default=SETTINGS["pose_conf"])
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.pose_imgsz < 32 or args.pose_imgsz % 32 or not 0 < args.pose_conf <= 1:
        parser.error("pose image size must be a positive multiple of 32 and confidence in (0, 1]")
    settings = {**SETTINGS, "pose_imgsz": args.pose_imgsz, "pose_conf": args.pose_conf}
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    data = json.loads(args.inputs.read_text())
    samples = data["samples"]
    if args.cores:
        unknown = set(args.cores) - {s["core_id"] for s in samples}
        if unknown:
            raise ValueError(f"Unknown event IDs: {unknown}")
        samples = [s for s in samples if s["core_id"] in args.cores]
    args.output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        if (args.output / (sample["core_id"] + ".json")).exists():
            raise FileExistsError("Existing features cannot be overwritten")
    ball, pose = YOLO(str(args.ball_model)), YOLO(str(args.pose_model))
    if ball.task != "detect" or ball.names[32] != "sports ball" or pose.task != "pose":
        raise ValueError("Require COCO ball detector and 17-keypoint pose checkpoint")
    for sample in samples:
        print("extracting " + sample["core_id"], flush=True)
        record, evidence = extract_contact_record(sample, ball, pose,
            [args.ball_model, args.pose_model], video_root=args.video_root, device=args.device,
            settings=settings)
        for folder, content in ((args.output, record), (args.output / "evidence", evidence)):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (sample["core_id"] + ".json")).write_text(
                json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"core": sample["core_id"], "window": record["window"],
            "runtime": record["runtime"], "baselines": record["baselines"]}), flush=True)


if __name__ == "__main__":
    main()
