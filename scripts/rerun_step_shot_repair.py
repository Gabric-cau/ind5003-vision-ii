"""Re-run failed step clips without masks, preserving the historical exports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def render_qa(result, destination):
    import cv2
    import numpy as np

    start, end = result["shot_start_frame"], result["shot_end_frame"]
    selected = next((t for t in result["tracks"] if t["track_id"] == result["selected_track_id"]), {})
    observations = {o["frame"]: o for o in selected.get("observations", [])}
    targets = sorted(set([start, end - 1, *result.get("step_frames", []),
                          *[f for f in [result.get("runup_start_frame"), result.get("kick_frame")]
                            if f is not None]]))
    capture = cv2.VideoCapture(result["video_path"])
    panels = []
    for frame_id in targets:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode QA frame {frame_id}")
        scale = 480 / frame.shape[1]
        frame = cv2.resize(frame, (480, 270))
        item = observations.get(frame_id)
        if item:
            box = [round(v * scale) for v in item["bbox"]]
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
            for index in [11, 12, 15, 16]:
                if item["keypoint_confidences"][index] >= .15:
                    point = tuple(round(v * scale) for v in item["keypoints"][index])
                    cv2.circle(frame, point, 3, (0, 0, 255), -1)
        labels = []
        if frame_id == result.get("kick_frame"):
            labels.append("KICK ESTIMATE")
        if frame_id == result.get("runup_start_frame"):
            labels.append("RUN-UP START")
        if frame_id in result.get("step_frames", []):
            labels.append("STEP CANDIDATE")
        frame = cv2.copyMakeBorder(frame, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(frame, f"f{frame_id} {frame_id/result['fps']:.2f}s {' / '.join(labels)}",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        panels.append(frame)
    capture.release()
    while len(panels) % 2:
        panels.append(np.zeros_like(panels[0]))
    sheet = np.vstack([np.hstack(panels[i:i+2]) for i in range(0, len(panels), 2)])
    ok, encoded = cv2.imencode(".jpg", sheet)
    if not ok:
        raise ValueError("QA image encoding failed")
    destination.write_bytes(encoded.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/shot_repair_2026-10-09")
    parser.add_argument("--clip-id", action="append", help="Otherwise re-run the historical failed clips")
    parser.add_argument("--all", action="store_true", help="Regression-run all canonical rows, still without replacing main exports")
    parser.add_argument("--model", type=Path, default=ROOT / "models/weights/yolo11n_pose.pt")
    parser.add_argument("--reuse", action="store_true", help="Use this run's existing per-clip diagnostics")
    args = parser.parse_args()
    if args.all and args.clip_id:
        parser.error("Use either --all or --clip-id, not both")
    from ultralytics import YOLO
    import torch
    from src.step_pipeline import process_video_steps

    torch.set_num_threads(min(4, torch.get_num_threads()))
    with (ROOT / "data/processed/penalty_step_counts.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline = list(csv.DictReader(handle))
    targets = [r for r in baseline if (args.all or (r["clip_id"] in args.clip_id if args.clip_id else r["status"] == "failed"))]
    if args.clip_id and set(args.clip_id) != {r["clip_id"] for r in targets}:
        parser.error("Unknown clip ID")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    comparisons = []
    for previous in targets:
        clip_id = previous["clip_id"]
        destination = args.output_dir / f"{clip_id}.json"
        started = time.perf_counter()
        if args.reuse and destination.exists():
            result = json.loads(destination.read_text(encoding="utf8"))
        else:
            if model is None:
                model = YOLO(str(args.model))
            print(f"START {clip_id}", flush=True)
            result = process_video_steps(
                str(ROOT / "data/raw_clips" / f"{clip_id}.mp4"), model,
                clip_id=clip_id, use_field_mask=False, shot_selection="first_viable",
                batch_size=8, predict_kwargs={"imgsz": 640, "conf": .10, "device": "cpu"},
            )
            result["run_metadata"] = {"model": str(args.model), "device": "cpu",
                                      "field_mask": False, "human_verified": False,
                                      "step_pipeline_sha256": hashlib.sha256((ROOT / "src/step_pipeline.py").read_bytes()).hexdigest(),
                                      "elapsed_sec": round(time.perf_counter() - started, 2)}
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
        render_qa(result, args.output_dir / f"{clip_id}_qa.jpg")
        summary = {k: result.get(k) for k in ["clip_id", "status", "shot_start_frame", "shot_end_frame",
                   "runup_start_frame", "kick_frame", "step_count", "kicking_foot", "review_required", "qa_warnings"]}
        summary.update({"old_status": previous["status"], "old_step_count": previous["step_count"],
                        "old_shot_end_frame": previous["shot_end_frame"],
                        "human_verified": False, "mask_used": False})
        comparisons.append(summary)
        (args.output_dir / "comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf8")
        print(json.dumps(summary), flush=True)
    return 0 if all(r["status"] == "ok" for r in comparisons) else 1


if __name__ == "__main__":
    raise SystemExit(main())
