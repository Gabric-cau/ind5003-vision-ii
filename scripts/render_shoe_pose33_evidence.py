"""Render actual source-frame and paired 33-point channel evidence for review."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_results import sha256
from src.fresh_results import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoe-dir", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a new evidence directory")
    args.output_dir.mkdir(parents=True)
    records = []
    for row in read_json(args.shoe_dir / "summary.json")["rows"]:
        if row["shoe_region_candidate"] == "unknown":
            continue
        event = args.shoe_dir / "events" / row["clip_id"]
        result, protocol = read_json(event / "result.json"), read_json(event / "protocol.json")
        if sha256(event / "result.json") != row["result_sha256"]:
            raise ValueError("Evidence result differs from cohort summary")
        source = result["source_mapping"]
        video = args.clips_dir / source["clip_filename"]
        if sha256(video) != source["clip_sha256"]:
            raise ValueError("Evidence source video differs from inference")
        poses = read_json(event / "poses33.json")
        views = [{p["frame"]: p for p in poses if p["reflected"] is reflected} for reflected in (False, True)]
        accepted = [p["frame"] for p in result["evidence"]["shoe_centroid_motion"]["accepted_paired_frames"]]
        joint = result["joint_near_shoe_and_paired_pose_frames"]
        # Preserve the preparation/strike sequence and every near-region frame.
        frames = sorted(set(accepted[::max(1, len(accepted) // 6)] + accepted[-1:] + joint))
        target = np.asarray(result["stationary_ball_reference"]["xy"])
        points = np.concatenate([np.asarray(views[0][frame]["landmarks_source_xy"])[[23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
                                 for frame in frames] + [target[None]])
        left, top = np.floor(points.min(axis=0) - 18).astype(int)
        right, bottom = np.ceil(points.max(axis=0) + 18).astype(int)
        cap = cv2.VideoCapture(str(video))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
        tile_w, tile_h, header = 260, 330, 92
        sheet = np.full((3 * tile_h + header, len(frames) * tile_w, 3), 245, np.uint8)
        cv2.putText(sheet, f"{row['clip_id']} | shoe-region candidate: {row['shoe_region_candidate']} (unverified)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Top: source pixels. Middle/bottom: independent normal/mirror model, mapped to source.",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Red: model left. Blue: model right. Yellow: fixed last-stationary-ball point. Gray: confidence < 0.5.",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, .46, (20, 20, 20), 1, cv2.LINE_AA)
        factor = min((tile_w - 12) / (right - left), (tile_h - 36) / (bottom - top))
        for column, frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - frame - 1) > .5:
                raise ValueError("Evidence source frame cannot be decoded with the bound clock")
            for view_index in range(3):
                crop = image[top:bottom, left:right].copy()
                if view_index:
                    pose = views[view_index - 1][frame]
                    pts = np.asarray(pose["landmarks_source_xy"]) - [left, top]
                    conf = np.asarray(pose["landmark_confidence"])
                    for indices, color in (([23, 25, 27, 29, 31, 27], (45, 45, 230)),
                                           ([24, 26, 28, 30, 32, 28], (230, 105, 30))):
                        for a, b in zip(indices, indices[1:]):
                            line_color = color if min(conf[[a, b]]) >= .5 else (140, 140, 140)
                            cv2.line(crop, tuple(np.rint(pts[a]).astype(int)), tuple(np.rint(pts[b]).astype(int)), line_color, 1)
                        for index in indices[:-1]:
                            cv2.circle(crop, tuple(np.rint(pts[index]).astype(int)), 2,
                                       color if conf[index] >= .5 else (140, 140, 140), -1)
                    cv2.circle(crop, tuple(np.rint(target - [left, top]).astype(int)), 4, (0, 235, 235), 1)
                resized = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                x = column * tile_w + (tile_w - resized.shape[1]) // 2
                y = header + view_index * tile_h + 28
                sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                label = ("source", "normal", "mirror inverse")[view_index]
                extra = " [joint near]" if frame in joint else ""
                cv2.putText(sheet, f"{label} f{frame}{extra}", (column * tile_w + 8, y - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1, cv2.LINE_AA)
        cap.release()
        destination = args.output_dir / f"{row['clip_id']}.jpg"
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise ValueError("Evidence image could not be written")
        records.append(dict(clip_id=row["clip_id"], candidate=row["shoe_region_candidate"],
            image=destination.name, image_sha256=sha256(destination), source_frames=frames,
            source_full_video_frames=[source["source_start_frame"] + frame for frame in frames],
            source_video_sha256=source["clip_sha256"], result_sha256=sha256(event / "result.json"),
            protocol_sha256=sha256(event / "protocol.json"), poses33_sha256=sha256(event / "poses33.json"),
            model_sha256=protocol["model_sha256"], crop_xyxy=[int(v) for v in (left, top, right, bottom)],
            source_frame_pixels_resampled=True, human_verified=False, anatomical_side_verified=False,
            shoe_ball_contact_verified=False))
    write_json(args.output_dir / "manifest.json", dict(records=records,
        render_script_sha256=sha256(Path(__file__)),
        scope="Source frame and anatomical-channel review; no verified shoe contact or human labels"))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
