"""Render source pixels and pose channels around fresh contact candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def render(result: dict, clips_dir: Path, output_dir: Path) -> None:
    mapping = result["source_mapping"]
    capture = cv2.VideoCapture(str(clips_dir / mapping["clip_filename"]))
    if not capture.isOpened():
        raise ValueError("Fresh source clip is unavailable")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    kick = result.get("kick_frame")
    if kick is None:
        frames = np.linspace(0, count - 1, 8).astype(int).tolist()
    else:
        frames = [kick - 90, kick - 60, kick - 30, kick - 8,
                  kick - 3, kick, kick + 3, kick + 10]
    frames = sorted({max(0, min(count - 1, int(frame))) for frame in frames})
    poses = {int(row["frame"]): row for row in result.get("pose_frames", [])}
    panels = []
    for frame_id in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode evidence frame {frame_id}")
        row = poses.get(frame_id, {})
        for detection in row.get("detections", []):
            selected = detection.get("track_id") == result.get("selected_track_id")
            color = (30, 210, 30) if selected else (220, 120, 30)
            box = np.asarray(detection["bbox"], dtype=int)
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
            cv2.putText(frame, f"track {detection.get('track_id')}",
                        (int(box[0]), max(15, int(box[1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            points = np.asarray(detection.get("keypoints", []))
            confidence = np.asarray(detection.get("keypoint_confidences", []))
            if points.shape != (17, 2) or confidence.shape != (17,):
                continue
            for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12)):
                if min(confidence[a], confidence[b]) >= .15:
                    cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 2)
            for joint, label in ((15, "L"), (16, "R")):
                if confidence[joint] >= .15:
                    cv2.putText(frame, label, tuple(points[joint].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        panel = np.full((390, 640, 3), 250, np.uint8)
        panel[:360] = cv2.resize(frame, (640, 360))
        source_sec = (mapping["source_start_frame"] + frame_id) / mapping["fps"]
        cv2.putText(panel, f"clip {frame_id / fps:.3f}s ({frame_id}) | source {source_sec:.3f}s",
                    (8, 381), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1)
        panels.append(panel)
    capture.release()
    sheet = np.full((75 + 390 * ((len(panels) + 1) // 2), 1280, 3), 255, np.uint8)
    cv2.putText(sheet, result["clip_id"], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1)
    cv2.putText(sheet, f"Candidate contact={kick}, steps={result.get('step_count')}; pose channels unverified",
                (8, 55), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        sheet[75 + row * 390:75 + (row + 1) * 390, column * 640:(column + 1) * 640] = panel
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_dir / f"{result['clip_id']}.jpg"), sheet):
        raise ValueError("Evidence image export failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in sorted(args.pose_dir.glob("*_p*.json")):
        render(json.loads(path.read_text(encoding="utf-8")), args.clips_dir, args.output_dir)


if __name__ == "__main__":
    main()
