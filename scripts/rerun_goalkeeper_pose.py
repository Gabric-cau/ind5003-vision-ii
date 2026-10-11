"""Targeted high-resolution pose fallback for clips missing a goalkeeper track."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "processed" / "penalty_goalkeeper_features.csv")
    parser.add_argument("--kick-frames", type=Path, default=PROJECT_ROOT / "data" / "annotations" / "penalty_kick_frames.csv")
    parser.add_argument("--step-diagnostics", type=Path, default=PROJECT_ROOT / "outputs" / "step_diagnostics.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "goalkeeper_pose_fallback.json")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "weights" / "yolo11n_pose.pt"),
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--all", action="store_true", help="Run the goal-band pass for all 39 clips.")
    parser.add_argument("--allow-unverified-annotations", action="store_true")
    args = parser.parse_args(argv)

    with args.features.open("r", encoding="utf-8", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    missing = {
        row["clip_id"]
        for row in feature_rows
        if args.all or row["status"] == "review_required"
    }
    with args.kick_frames.open("r", encoding="utf-8", newline="") as handle:
        kick_rows = list(csv.DictReader(handle))
    from src.annotation_provenance import require_verified
    require_verified([row for row in kick_rows if row["clip_id"] in missing], "Kick frames", args.allow_unverified_annotations)
    kicks = {row["clip_id"]: int(row["kick_frame"]) for row in kick_rows}
    step = json.loads(args.step_diagnostics.read_text(encoding="utf-8"))
    sources = {row["clip_id"]: row for row in step}

    from ultralytics import YOLO
    from src.goalkeeper_pipeline import infer_goalkeeper_pose_window

    model = YOLO(args.model)
    outputs = []
    for index, clip_id in enumerate(sorted(missing), 1):
        source = sources[clip_id]
        print(f"[{index}/{len(missing)}] {clip_id}", flush=True)
        outputs.append(
            infer_goalkeeper_pose_window(
                source["video_path"], model, kicks[clip_id], clip_id=clip_id, imgsz=args.imgsz
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} fallback pose records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
