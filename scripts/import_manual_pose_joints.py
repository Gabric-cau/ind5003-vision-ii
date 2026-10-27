"""Import sparse human pose corrections without inventing unreviewed joints."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

import cv2

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
PACK_ROOT = DOWNLOADS / "IND5003_\u8d44\u6599\u5ba1\u9605_20261025/vision2_pose_pack_v2"
DEFAULT_JSON = DOWNLOADS / "vision2_\u4eba\u5de5\u5173\u8282\u6807\u6ce8.json"
DEFAULT_CSV = DOWNLOADS / "vision2_\u4eba\u5de5\u5173\u8282\u5750\u6807.csv"
DEFAULT_REPORT = DOWNLOADS / "Vision2\u89c6\u9891\u4e0e\u59ff\u6001\u6570\u636e\u5206\u5de5\u6210\u679c\u62a5\u544a.pdf"
CSV_FIELDS = ["video_id", "track_id", "frame", "timestamp_s", "joint_id", "joint_name", "status",
              "x_px", "y_px", "human_reviewed", "reviewer", "updated_at", "frame_note", "source_sha256"]
JOINT_NAMES = ["\u9f3b", "\u5de6\u773c", "\u53f3\u773c", "\u5de6\u8033", "\u53f3\u8033",
               "\u5de6\u80a9", "\u53f3\u80a9", "\u5de6\u8098", "\u53f3\u8098", "\u5de6\u8155", "\u53f3\u8155",
               "\u5de6\u9acb", "\u53f3\u9acb", "\u5de6\u819d", "\u53f3\u819d", "\u5de6\u8e1d", "\u53f3\u8e1d"]
STATUSES = {"visible_corrected", "visible_confirmed", "occluded"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid nonnegative integer {name}: {value!r}")
    return value


def finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid finite number {name}: {value!r}")
    return float(value)


def key(row):
    return row["video_id"], row["track_id"], row["frame"], row["joint_id"]


def load_sources(json_path, csv_path):
    document = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if document.get("schema") != "vision2-manual-joints-v1":
        raise ValueError("Unexpected manual joint schema")
    if document.get("coordinate_system") != "original image pixels; top-left origin, x right, y down":
        raise ValueError("Manual coordinate system is not original image pixels")
    if document.get("frame_numbering") != "zero-based":
        raise ValueError("Manual frame numbering must be zero-based")
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError("Unexpected manual CSV headers")
        csv_rows = list(reader)
    records, seen = [], set()
    for index, row in enumerate(document["annotations"]):
        if row["video_id"] not in document["clips"] or not re.fullmatch(r"[A-Za-z0-9_-]+", row["video_id"]):
            raise ValueError("Unknown or unsafe video identifier")
        for field in ("track_id", "frame", "joint_id"):
            integer(row[field], field)
        if row["joint_id"] >= 17 or row["joint_name"] != JOINT_NAMES[row["joint_id"]]:
            raise ValueError("Joint name does not match COCO17 index")
        if row["status"] not in STATUSES or row["human_reviewed"] is not True:
            raise ValueError("All imported rows must be explicitly human reviewed")
        if row["status"] == "occluded":
            if row["x"] is not None or row["y"] is not None:
                raise ValueError("Occluded joints must preserve unknown coordinates")
        else:
            finite(row["x"], "x")
            finite(row["y"], "y")
        finite(row["timestamp_s"], "timestamp_s")
        if key(row) in seen:
            raise ValueError("Duplicate manual joint record")
        seen.add(key(row))
        records.append({**row, "source_annotation_index_zero_based": index, "source_json_record": row})
    if not records:
        raise ValueError("No manual joint records")
    csv_lookup = {}
    for line, row in enumerate(csv_rows, 2):
        parsed = {**row, "track_id": int(row["track_id"]), "frame": int(row["frame"]), "joint_id": int(row["joint_id"])}
        if key(parsed) in csv_lookup:
            raise ValueError("Duplicate manual CSV joint record")
        csv_lookup[key(parsed)] = (line, row)
    if set(csv_lookup) != seen:
        raise ValueError("JSON and CSV record inventories differ")
    for row in records:
        line, csv_row = csv_lookup[key(row)]
        for field in ("joint_name", "status", "updated_at"):
            if row[field] != csv_row[field]:
                raise ValueError(f"JSON and CSV disagree: {field}")
        for field, csv_field in (("x", "x_px"), ("y", "y_px"), ("timestamp_s", "timestamp_s")):
            value = float(csv_row[csv_field]) if csv_row[csv_field] else None
            if value != row[field]:
                raise ValueError(f"JSON and CSV disagree: {field}")
        clip = document["clips"][row["video_id"]]
        if csv_row["human_reviewed"] != "true" or csv_row["reviewer"] != document["reviewer"]:
            raise ValueError("JSON and CSV review metadata disagree")
        if csv_row["source_sha256"] != clip["source_sha256"] or row["track_id"] != clip["track_id"]:
            raise ValueError("JSON and CSV source/track binding disagree")
        row["source_csv_line_one_based"] = line
        row["source_csv_record"] = csv_row
    return document, records


def verify_model_binding(clips, model_weights=None, remote_host=None, remote_model=None):
    expected = {row["model_sha256"] for row in clips.values()}
    if len(expected) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(expected))):
        raise ValueError("Expected one declared model hash")
    expected_hash = next(iter(expected))
    if model_weights and remote_host:
        raise ValueError("Choose local or remote model verification")
    if bool(remote_host) != bool(remote_model):
        raise ValueError("Remote model verification requires host and path")
    result = {"declared_sha256": expected_hash, "file_verified": False,
              "status": "declared_hash_only", "same_weights_does_not_establish_same_inference": True}
    if model_weights:
        actual = sha256(model_weights)
        result.update(path=str(Path(model_weights).resolve()), method="local_sha256")
    elif remote_host:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", remote_host) or not str(remote_model).startswith("/"):
            raise ValueError("Unsafe remote model binding arguments")
        command = "sha256sum " + shlex.quote(str(remote_model))
        completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote_host, command],
                                   check=True, capture_output=True, text=True, timeout=30)
        fields = completed.stdout.strip().split(maxsplit=1)
        if len(fields) != 2 or fields[1] != str(remote_model) or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise ValueError("Unexpected remote sha256sum response")
        actual = fields[0]
        result.update(host=remote_host, path=str(remote_model), method="ssh_sha256sum_read_only")
    else:
        return result
    if actual != expected_hash:
        raise ValueError("Reviewed pose model weights do not match available weights")
    result.update(file_verified=True, status="model_file_hash_verified", verified_sha256=actual)
    return result


def bind_videos(document, records, video_root, timestamps_root, pose_json_root=None):
    videos = {}
    for clip, reference in document["clips"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
            raise ValueError("Unsafe clip identifier")
        path = (Path(video_root) / f"{clip}.mp4").resolve(strict=True)
        actual = sha256(path)
        if actual != reference["source_sha256"]:
            raise ValueError(f"Source video hash mismatch: {clip}")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {clip}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        times_path = (Path(timestamps_root) / f"{clip}.csv").resolve(strict=True)
        with times_path.open(encoding="utf-8-sig", newline="") as stream:
            time_rows = list(csv.DictReader(stream))
        times = {}
        for time_row in time_rows:
            frame = int(time_row["frame"])
            if time_row["video_id"] != clip or frame in times:
                raise ValueError("Invalid frame timestamp inventory")
            times[frame] = finite(float(time_row["time_s"]), "PTS seconds")
        if set(times) != set(range(count)):
            raise ValueError(f"PTS inventory differs from video frame count: {clip}")
        if any(times[i] >= times[i + 1] for i in range(count - 1)):
            raise ValueError("Frame PTS must increase")
        pose_binding = {"declared_sha256": reference["pose_json_sha256"], "file_verified": False,
                        "status": "exact_inference_hash_provided_file_not_located",
                        "track_id_join_requires_this_exact_output": True}
        if not re.fullmatch(r"[0-9a-f]{64}", reference["pose_json_sha256"]):
            raise ValueError("Invalid declared pose JSON hash")
        pose_path = Path(pose_json_root) / f"{clip}.json" if pose_json_root else None
        if pose_path is not None and pose_path.is_file():
            pose_hash = sha256(pose_path)
            if pose_hash != reference["pose_json_sha256"]:
                raise ValueError(f"Exact reviewed pose JSON mismatch: {clip}")
            pose_binding.update(path=str(pose_path.resolve()), file_verified=True,
                                verified_sha256=pose_hash, status="exact_inference_file_hash_verified")
        for row in records:
            if row["video_id"] != clip:
                continue
            if row["frame"] not in times:
                raise ValueError("Manual frame outside video")
            if abs(row["timestamp_s"] - times[row["frame"]]) > 1e-6:
                raise ValueError("Manual timestamp differs from frame PTS")
            if row["x"] is not None and not (0 <= row["x"] < width and 0 <= row["y"] < height):
                raise ValueError("Manual joint outside original image")
        videos[clip] = {"path": str(path), "sha256": actual, "width": width, "height": height,
                        "reported_frame_count": count, "timestamp_csv": str(times_path),
                        "timestamp_csv_sha256": sha256(times_path), "time_basis": "source_frame_pts",
                        "reviewed_track_id": reference["track_id"], "pose_output_binding": pose_binding}
    if set(videos) != {row["video_id"] for row in records}:
        raise ValueError("Declared clips and annotation clips differ")
    return videos


def sparse_frames(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["video_id"], row["track_id"], row["frame"])].append(row)
    frames = []
    for (clip, track, frame), annotations in sorted(grouped.items()):
        target = {"id": f"{clip}_f{frame:06d}_track{track}", "video_id": clip,
                  "reviewed_track_id": track, "frame_zero_based": frame,
                  "timestamp_s": annotations[0]["timestamp_s"],
                  "joint_xy_px": [[None, None] for _ in range(17)],
                  "coordinate_loss_mask": [False] * 17,
                  "visibility_target": [None] * 17,
                  "visibility_loss_mask": [False] * 17,
                  "review_status": ["unreviewed"] * 17,
                  "source_annotation_indices": [],
                  "person_role": "shooter_as_reported", "split": "development",
                  "complete_person_pose_truth": False}
        for row in annotations:
            joint = row["joint_id"]
            visible = row["status"] != "occluded"
            target["joint_xy_px"][joint] = [row["x"], row["y"]]
            target["coordinate_loss_mask"][joint] = visible
            target["visibility_target"][joint] = visible
            target["visibility_loss_mask"][joint] = True
            target["review_status"][joint] = row["status"]
            target["source_annotation_indices"].append(row["source_annotation_index_zero_based"])
        frames.append(target)
    return frames


def export_frames(videos, frames, staging):
    lookup = {(row["video_id"], row["frame_zero_based"]): row for row in frames}
    exports = []
    for clip, video in videos.items():
        capture = cv2.VideoCapture(video["path"])
        decoded = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape[:2] != (video["height"], video["width"]):
                    raise ValueError("Decoded video size changed")
                row = lookup.get((clip, decoded))
                if row:
                    relative = f"images/{row['id']}.png"
                    path = staging / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), image):
                        raise ValueError("Could not export reviewed source frame")
                    row["image"] = relative
                    row["decoded_pixels_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
                    exports.append({"frame_id": row["id"], "image": relative, "image_sha256": sha256(path),
                                    "decoded_pixels_sha256": row["decoded_pixels_sha256"],
                                    "video_id": clip, "frame_zero_based": decoded, "split": "development"})
                decoded += 1
        finally:
            capture.release()
        if decoded != video["reported_frame_count"] or sha256(video["path"]) != video["sha256"]:
            raise ValueError("Video changed or failed complete decoding")
        video["decoded_frame_count"] = decoded
    if len(exports) != len(frames):
        raise ValueError("Missing reviewed source frame exports")
    return exports


def prepare(json_path, csv_path, output, *, video_root=PACK_ROOT / "videos",
            timestamps_root=PACK_ROOT / "timestamps", pose_json_root=None, report_path=None,
            model_weights=None, remote_model_host=None, remote_model_path=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    source_paths = {"manual_json": Path(json_path).resolve(strict=True),
                    "manual_csv": Path(csv_path).resolve(strict=True)}
    if report_path:
        source_paths["handoff_report"] = Path(report_path).resolve(strict=True)
    source_files = {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_paths.items()}
    document, records = load_sources(json_path, csv_path)
    model_binding = verify_model_binding(document["clips"], model_weights, remote_model_host, remote_model_path)
    videos = bind_videos(document, records, video_root, timestamps_root, pose_json_root)
    frames = sparse_frames(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        exports = export_frames(videos, frames, staging)
        common = {"schema": "source_bound_sparse_manual_pose/v1", "development_only": True,
                  "independent_test": False, "independent_consensus": False,
                  "human_reviewed_as_declared": True, "reviewer_identity_independently_verified": False,
                  "selection": "targeted previously observed shooter errors and adjacent transitions"}
        summary = {**common, "videos": len(videos), "reviewed_frames": len(frames),
                   "joint_records": len(records), "status_counts": dict(Counter(row["status"] for row in records)),
                   "coordinate_targets": sum(sum(row["coordinate_loss_mask"]) for row in frames),
                   "visibility_targets": sum(sum(row["visibility_loss_mask"]) for row in frames),
                   "unreviewed_joint_slots": sum(row["review_status"].count("unreviewed") for row in frames),
                   "by_video": {clip: {"reviewed_frames": sum(row["video_id"] == clip for row in frames),
                                       "joint_records": sum(row["video_id"] == clip for row in records),
                                       "statuses": dict(Counter(row["status"] for row in records if row["video_id"] == clip))}
                                for clip in videos},
                   "source_hashes_verified": True, "timestamps_match_source_pts": True,
                   "json_csv_agreement": True, "model_file_hash_verified": model_binding["file_verified"],
                   "exact_pose_output_files_verified": sum(video["pose_output_binding"]["file_verified"] for video in videos.values())}
        targets = {**common, "joint_names_coco17": JOINT_NAMES, "frames": frames,
                   "mask_policy": {"unreviewed": "neither coordinate nor visibility loss",
                                   "occluded": "visibility supervision only; coordinates remain unknown",
                                   "visible_confirmed": "human confirmed original prediction; retain separate provenance",
                                   "visible_corrected": "human-positioned original-image coordinate",
                                   "person_detection_negative_labels": False,
                                   "plain_full_pose_yolo_export": False}}
        manifest = {**common, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_files": source_files, "script_sha256": sha256(__file__),
                    "source_metadata": {name: value for name, value in document.items() if name != "annotations"},
                    "model_binding": model_binding, "videos": videos, "exports": exports,
                    "scope_limits": ["47 reviewed frames are not 139 independent examples.",
                                     "No keeper joint, foot-contact, step, landing or 3D truth is supplied here.",
                                     "Declared track IDs may only join the exact reviewed output hash.",
                                     "Current predictions must be produced without access to these target coordinates."]}
        save_json(staging / "annotations.json", {**common, "annotations": records})
        save_json(staging / "frame_targets.json", targets)
        save_json(staging / "summary.json", summary)
        manifest["artifact_sha256"] = {name: sha256(staging / name) for name in ("annotations.json", "frame_targets.json", "summary.json")}
        if any(sha256(source_paths[name]) != value["sha256"] for name, value in source_files.items()):
            raise ValueError("Manual source changed during import")
        save_json(staging / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(f"Output directory appeared during import: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--video-root", type=Path, default=PACK_ROOT / "videos")
    parser.add_argument("--timestamps-root", type=Path, default=PACK_ROOT / "timestamps")
    parser.add_argument("--pose-json-root", type=Path)
    parser.add_argument("--model-weights", type=Path)
    parser.add_argument("--remote-model-host")
    parser.add_argument("--remote-model-path")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/manual_pose_joints_2026-10-27")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    print(json.dumps(prepare(args.source_json, args.source_csv, args.output_dir, video_root=args.video_root,
                             timestamps_root=args.timestamps_root, pose_json_root=args.pose_json_root,
                             report_path=args.report, model_weights=args.model_weights,
                             remote_model_host=args.remote_model_host, remote_model_path=args.remote_model_path),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
