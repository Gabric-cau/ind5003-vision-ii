"""Render candidate identity sheets for goalkeeper track review."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def tile(frame, candidate):
    h, w = frame.shape[:2]
    x = float(candidate["baseline_x"])
    y = float(candidate["baseline_y"])
    scale = max(60.0, float(candidate["body_scale_px"]))
    crop_w, crop_h = 2.5 * scale, 2.2 * scale
    x0 = int(np.clip(x - crop_w / 2, 0, max(0, w - crop_w)))
    y0 = int(np.clip(y - crop_h / 2, 0, max(0, h - crop_h)))
    x1, y1 = min(w, int(x0 + crop_w)), min(h, int(y0 + crop_h))
    crop = cv2.resize(frame[y0:y1, x0:x1], (190, 170))
    px = int((x - x0) * 190 / max(1, x1 - x0))
    py = int((y - y0) * 170 / max(1, y1 - y0))
    cv2.circle(crop, (px, py), 8, (0, 255, 255), 2)
    text = f"id={candidate['track_id']} pre={candidate['pre_coverage']:.2f} y={candidate['upper_field_prior']:.2f}"
    cv2.rectangle(crop, (0, 0), (190, 22), (0, 0, 0), -1)
    cv2.putText(crop, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def main():
    results = json.loads((ROOT / "outputs" / "goalkeeper_diagnostics.json").read_text(encoding="utf-8"))
    out = ROOT / "outputs" / "goalkeeper_qa" / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    for result in results:
        fps = float(result["fps"])
        frame_id = max(0, int(result["kick_frame"] - 0.35 * fps))
        capture = cv2.VideoCapture(result["video_path"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            continue
        candidates = [row for row in result["track_candidates"] if row["pre_count"] >= 2]
        candidates.sort(key=lambda row: (-row["upper_field_prior"], -row["pre_coverage"]))
        candidates = candidates[:8]
        if not candidates:
            continue
        row = np.hstack([tile(frame, candidate) for candidate in candidates])
        output = out / f"{result['clip_id']}.jpg"
        cv2.imwrite(str(output), row, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
