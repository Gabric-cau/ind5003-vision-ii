"""Render contact sheets for visual QA of selected ball trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_frame(video_path: str, frame_id: int):
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def panel(row, item, width=400, height=225):
    frame_id = int(item["frame"])
    frame = read_frame(row["video_path"], frame_id)
    if frame is None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    original_height, original_width = frame.shape[:2]
    scale_x, scale_y = width / original_width, height / original_height
    frame = cv2.resize(frame, (width, height))
    x, y = int(round(float(item["x"]) * scale_x)), int(round(float(item["y"]) * scale_y))
    cv2.circle(frame, (x, y), 10, (0, 255, 255), 2)
    for anchor in row.get("pose_anchors", []):
        ax, ay = int(round(float(anchor[0]) * scale_x)), int(round(float(anchor[1]) * scale_y))
        cv2.drawMarker(frame, (ax, ay), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.putText(frame, f"f{frame_id} conf={float(item['confidence']):.2f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "outputs" / "ball_outcome_diagnostics.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ball_qa")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--rows-per-sheet", type=int, default=7)
    args = parser.parse_args(argv)
    rows = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    if args.review_only:
        rows = [row for row in rows if row["ball_track_status"] == "review_required"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for page_start in range(0, len(rows), args.rows_per_sheet):
        page_rows = rows[page_start: page_start + args.rows_per_sheet]
        rendered = []
        for row in page_rows:
            trajectory = row.get("trajectory") or []
            if not trajectory:
                continue
            indices = sorted(set([0, len(trajectory) // 2, len(trajectory) - 1]))
            panels = [panel(row, trajectory[index]) for index in indices]
            while len(panels) < 3:
                panels.append(panels[-1].copy())
            strip = np.hstack(panels[:3])
            header = np.zeros((34, strip.shape[1], 3), dtype=np.uint8)
            label = f"{row['clip_id']} | {row['ball_track_status']} | outcome={row['outcome']}"
            cv2.putText(header, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            rendered.append(np.vstack([header, strip]))
        if not rendered:
            continue
        sheet = np.vstack(rendered)
        page = page_start // args.rows_per_sheet + 1
        suffix = "review" if args.review_only else "all"
        output = args.output_dir / f"ball_qa_{suffix}_{page:02d}.jpg"
        cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
