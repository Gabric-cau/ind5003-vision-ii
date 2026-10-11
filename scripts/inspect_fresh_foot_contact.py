"""Render actual source pixels for an unverified fresh ball/actor contact.

The output is diagnostic media. Pose channel names are model predictions,
and neither the displayed contact nor the kicking foot is human validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fresh_foot_experiment import contact_payload, selected_ball_observations, timing_and_actor
from src.final_results import sha256
from src.pose_refinement import anchor_at_frame, crop_bounds


def contain(image: np.ndarray, width: int, height: int) -> np.ndarray:
    factor = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * factor), round(image.shape[0] * factor)))
    panel = np.full((height, width, 3), 24, np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--contact-json", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pose = json.loads(args.pose_json.read_text())
    record = json.loads(args.contact_json.read_text())
    contact = contact_payload(record)
    timing = timing_and_actor(pose, record, False)
    first, departure = timing["contact_bracket_frames"]
    actor_id = timing["actor_track_id_candidate"]
    manifest = json.loads((args.clips_dir / "source_manifest.json").read_text())
    source = next(row for row in manifest["clips"] if row["clip_id"] == pose["clip_id"])
    video = args.clips_dir / source["clip_filename"]
    if sha256(video) != source["clip_sha256"] or record["source_sha256"] != source["clip_sha256"]:
        raise ValueError("Contact/source video identity mismatch")
    if record.get("pose_sha256") != sha256(args.pose_json):
        raise ValueError("Contact/source pose identity mismatch")
    track = next(row for row in pose["tracks"] if row["track_id"] == actor_id)
    poses = {row["frame"]: row for row in track["observations"]}
    balls = {row["frame"]: row for row in selected_ball_observations(record) or []}
    frames = sorted({first - 8, first - 4, first - 2, first - 1, first, first + 1, departure, departure + 2})
    frames = [frame for frame in frames if pose["shot_start_frame"] <= frame < pose["shot_end_frame"]]
    capture = cv2.VideoCapture(str(video))
    tiles, metadata = [], []
    for fid in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, original = capture.read()
        if not ok:
            raise ValueError("Failed to read the requested actual source frame")
        annotated = original.copy()
        anchor, origin = anchor_at_frame(poses, fid, max_gap=3)
        if anchor is None:
            raise ValueError("No observed/interpolated actor crop for the requested contact frame")
        bounds = crop_bounds(anchor, original.shape[1], original.shape[0], margin=.35)
        observation = poses.get(fid)
        ankles = []
        if observation:
            for index, name, color in ((15, "pose L", (70, 90, 245)), (16, "pose R", (245, 180, 50))):
                xy = observation["keypoints"][index]
                confidence = observation["keypoint_confidences"][index]
                if confidence > .15:
                    point = tuple(np.rint(xy).astype(int))
                    cv2.circle(annotated, point, 4, color, 1)
                    cv2.putText(annotated, name, (point[0] + 4, point[1] - 3), 0, .3, color, 1, cv2.LINE_AA)
                ankles.append({"pose_channel": name, "xy": xy, "confidence": confidence})
        ball = balls.get(fid)
        if ball:
            point = tuple(np.rint(ball["xy"]).astype(int))
            cv2.circle(annotated, point, 8, (60, 230, 80), 2)
        scene = contain(annotated, 480, 270)
        crop = contain(annotated[bounds[1]:bounds[3], bounds[0]:bounds[2]], 320, 270)
        tile = np.hstack((scene, crop))
        source_time = (source["source_start_frame"] + fid) / source["fps"]
        cv2.putText(tile, f"f{fid} source {source_time:.3f}s actor {actor_id}", (8, 20), 0, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, "AI diagnostic; contact and pose L/R unverified", (8, 260), 0, .42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        metadata.append({"clip_frame": fid, "source_recording_frame": source["source_start_frame"] + fid,
                         "source_recording_seconds": source_time, "crop_xyxy": bounds,
                         "anchor_source": origin, "pose_ankles": ankles, "selected_ball_detection": ball})
    capture.release()
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[-1]))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{pose['clip_id']}_contact_source.jpg"
    if not cv2.imwrite(str(path), sheet):
        raise ValueError("Could not write diagnostic source sheet")
    evidence = {"clip_id": pose["clip_id"], "contact_bracket_frames": [first, departure],
                "actor_track_id_candidate": actor_id, "source_sha256": source["clip_sha256"],
                "pose_sha256": sha256(args.pose_json), "contact_sha256": sha256(args.contact_json),
                "source_sheet_sha256": sha256(path), "script_sha256": sha256(Path(__file__)),
                "human_verified": False, "accuracy_validated": False,
                "actual_source_pixels_decoded": True, "frames": metadata}
    (args.output_dir / f"{pose['clip_id']}_contact_source.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"sheet": str(path), "source_frames": len(metadata), "actor": actor_id,
                      "contact_bracket": [first, departure], "human_verified": False}))


if __name__ == "__main__":
    main()
