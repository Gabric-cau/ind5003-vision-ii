"""Render compact goalkeeper QA contact sheets from diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def nearest_sample(candidate, frame, fps):
    samples = candidate["pre_trajectory"] + candidate["post_trajectory"]
    if not samples:
        return None
    selected = min(samples, key=lambda row: abs(row[0] * fps - frame))
    return selected if abs(selected[0]*fps-frame) <= max(2, round(.12*fps)) else None


def render_tile(frame, sample, bounds, label):
    x0, y0, x1, y1 = bounds
    crop = frame[y0:y1, x0:x1].copy()
    if sample is not None:
        x, y = int(sample[1] - x0), int(sample[2] - y0)
        cv2.circle(crop, (x, y), 8, (0, 255, 255), 2)
    else:
        label += " | NO OBS"
    crop = cv2.resize(crop, (256, 160))
    cv2.rectangle(crop, (0, 0), (256, 22), (0, 0, 0), -1)
    cv2.putText(crop, label, (7, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def render_row(result):
    fps = float(result["fps"])
    kick = int(result["kick_frame"])
    candidate = result["selected_candidate"]
    if candidate is None:
        row = np.zeros((160, 1024, 3), dtype=np.uint8)
        title = f"{result['clip_id']} | NO QUALIFIED GOALKEEPER TRACK | review=True"
        cv2.putText(row, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1, cv2.LINE_AA)
        return row
    samples = candidate["pre_trajectory"] + candidate["post_trajectory"]
    xs = np.asarray([row[1] for row in samples])
    ys = np.asarray([row[2] for row in samples])
    scales = np.asarray([row[3] for row in samples])
    scale = max(50.0, float(np.median(scales)))
    centre_x = float((xs.min() + xs.max()) / 2)
    centre_y = float((ys.min() + ys.max()) / 2)
    crop_w = max(280.0, float(xs.max() - xs.min()) + 3.2 * scale)
    crop_h = max(190.0, float(ys.max() - ys.min()) + 2.4 * scale)

    capture = cv2.VideoCapture(result["video_path"])
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0 = int(np.clip(centre_x - crop_w / 2, 0, max(0, width - crop_w)))
    y0 = int(np.clip(centre_y - crop_h / 2, 0, max(0, height - crop_h)))
    x1 = min(width, int(x0 + crop_w))
    y1 = min(height, int(y0 + crop_h))
    offsets = [(-1.0, "kick-1.0s"), (0.0, "kick"), (0.4, "kick+0.4s"), (0.8, "kick+0.8s")]
    tiles = []
    try:
        for offset, label in offsets:
            frame_id = max(0, int(round(kick + offset * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                frame = np.zeros((height or 720, width or 1280, 3), dtype=np.uint8)
            sample = nearest_sample(candidate, frame_id, fps)
            tiles.append(render_tile(frame, sample, (x0, y0, x1, y1), label))
    finally:
        capture.release()
    row = np.hstack(tiles)
    title = (
        f"{result['clip_id']} | track {result['goalkeeper_track_id']} | "
        f"{result['gk_prekick_movement_level']} | {result['dive_direction']} / {result['dive_height']} | "
        f"review={result['review_required']}"
    )
    cv2.rectangle(row, (0, 0), (1024, 28), (22, 22, 22), -1)
    cv2.putText(row, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, default=PROJECT_ROOT / "outputs" / "goalkeeper_diagnostics.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "goalkeeper_qa")
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    args = parser.parse_args(argv)
    results = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [render_row(result) for result in results]
    for page, start in enumerate(range(0, len(rows), args.rows_per_sheet), 1):
        sheet = np.vstack(rows[start : start + args.rows_per_sheet])
        output = args.output_dir / f"goalkeeper_qa_{page:02d}.jpg"
        cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
