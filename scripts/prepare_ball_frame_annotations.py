"""Import reviewed ball-frame candidates without changing the source workbook."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import cv2
import openpyxl


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
DEFAULT_SOURCE = DOWNLOADS / "\u7403\u8def\u9010\u5e27\u8865\u6807_\u603b\u8868.xlsx"
DEFAULT_CORE_ROOT = DOWNLOADS / "IND5003_\u8d44\u6599\u5ba1\u9605_20261026/extracted/\u5c0f\u76ae_10\u6bb5\u6838\u5fc3\u7f5a\u7403_\u4eba\u5de5\u8865\u6807\u4ea4\u4ed8/playback"
DEFAULT_MATCH_ROOT = DOWNLOADS / "IND5003_\u8d44\u6599\u5ba1\u9605_20261025/three_matches_release/extracted/Penalty_3_Matches_Export_20261025/videos"
FRAME_SHEET = "\u7403\u8def\u9010\u5e27"
SUMMARY_SHEET = "\u6700\u7ec8\u4f4d\u7f6e\u6c47\u603b"
NOTES_SHEET = "\u6807\u6ce8\u8bf4\u660e"
FRAME_HEADERS = [
    "\u5c40\u90e8\u653e\u5927\u56fe", "\u5b8c\u6574\u753b\u9762\u6838\u5bf9\u56fe\uff08\u53f3\u4e0a\u89d2\u4e3a3\u500d\u653e\u5927\uff09",
    "\u89c6\u9891\u7f16\u53f7", "\u65f6\u95f4(\u79d2)", "\u5e27\u53f7", "\u7403\u5fc3x(px)", "\u7403\u5fc3y(px)",
    "\u7403\u6846x1", "\u7403\u6846y1", "\u7403\u6846x2", "\u7403\u6846y2", "\u80fd\u5426\u770b\u6e05",
    "\u5b9a\u4f4d\u65b9\u5f0f", "\u9636\u6bb5", "\u5907\u6ce8",
]
SPLITS = {"core_01": "train", "core_02": "train", "core_03": "train", "match_01": "validation"}
STATUS_MAP = {
    "\u6e05\u6670": "clear", "\u53ef\u89c1\uff08\u8fd0\u52a8\u6a21\u7cca\uff09": "blur",
    "\u90e8\u5206\u53ef\u89c1": "partial", "\u770b\u4e0d\u6e05": "unknown",
    "\u753b\u9762\u5916\uff08\u955c\u5934\u5207\u8d70\uff09": "shot_cut",
    "\u753b\u9762\u5916": "out_of_view",
}
SCHEMA = "ball_frame_annotations/v1"
PROVENANCE = "human_reviewed_auto_refined_candidates_not_independent_ground_truth"
TIME_TOLERANCE = 0.00051


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def row_source(sheet, row, values, workbook_hash):
    serialized = json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"sheet": sheet, "row_1_based": row, "workbook_sha256": workbook_hash,
            "row_values_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "raw_values": list(values)}


def numeric(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number: {value!r}")
    return float(value)


def frame_integer(value, name):
    number = numeric(value, name)
    if not number.is_integer() or number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(number)


def read_workbook(source, workbook_hash):
    workbook = openpyxl.load_workbook(source, read_only=True, data_only=False)
    try:
        if set(workbook.sheetnames) != {FRAME_SHEET, SUMMARY_SHEET, NOTES_SHEET}:
            raise ValueError("Unexpected workbook sheets")
        sheets = {}
        for sheet in workbook:
            sheets[sheet.title] = [row_source(sheet.title, index, values, workbook_hash)
                                   for index, values in enumerate(sheet.values, 1)]
        if sheets[FRAME_SHEET][0]["raw_values"] != FRAME_HEADERS:
            raise ValueError("Unexpected ball-frame worksheet headers")
        return sheets
    finally:
        workbook.close()


def video_sources(core_root, match_root):
    # The allowlist prevents discovery or decoding of held-out core videos.
    return {clip: (Path(match_root) if clip == "match_01" else Path(core_root)).resolve() / f"{clip}.mp4"
            for clip in SPLITS}


def inspect_videos(paths, summary_rows):
    descriptions = {}
    contacts = []
    for source in summary_rows[1:]:
        values = source["raw_values"]
        clip = values[0]
        if clip not in paths or clip in descriptions:
            raise ValueError(f"Unexpected or duplicate summary video: {clip}")
        match = re.search(r"(\d+)[\u00d7x](\d+)\s+(\d+(?:\.\d+)?)fps", str(values[1]))
        if not match:
            raise ValueError(f"Missing declared dimensions/fps in summary: {clip}")
        descriptions[clip] = tuple(float(x) for x in match.groups())
        contact = re.match(r"(\d+)\u2192(\d+)\u4e4b\u95f4", str(values[2]))
        if not contact:
            raise ValueError(f"Contact must remain an interval: {clip}")
        lower, upper = (int(x) for x in contact.groups())
        if upper != lower + 1:
            raise ValueError(f"Invalid contact interval: {clip}")
        contacts.append({"video_id": clip, "frame_interval_zero_based": [lower, upper],
                         "semantics": "transition_between_observed_frames",
                         "is_single_frame_ground_truth": False, "source": source})
    if set(descriptions) != set(paths):
        raise ValueError("Summary must contain all four allowlisted videos")
    videos = {}
    for clip, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        expected_width, expected_height, expected_fps = descriptions[clip]
        if (width, height) != (expected_width, expected_height) or not math.isclose(fps, expected_fps, abs_tol=1e-6):
            raise ValueError(f"Video dimensions/fps do not match workbook summary: {clip}")
        if frame_count <= 0 or fps <= 0:
            raise ValueError(f"Invalid video frame count/fps: {clip}")
        videos[clip] = {"path": str(path), "sha256": sha256(path), "width": width, "height": height,
                        "fps": fps, "reported_frame_count": frame_count, "split": SPLITS[clip]}
    for contact in contacts:
        video = videos[contact["video_id"]]
        if contact["frame_interval_zero_based"][1] >= video["reported_frame_count"]:
            raise ValueError("Contact interval outside video")
        contact["time_interval_seconds"] = [frame / video["fps"] for frame in contact["frame_interval_zero_based"]]
    return videos, contacts


def normalize_rows(source_rows, videos):
    rows, seen = [], set()
    for source in source_rows[1:]:
        values = source["raw_values"]
        clip = values[2]
        if clip not in videos:
            raise ValueError(f"Video is not allowlisted: {clip}")
        video = videos[clip]
        frame = frame_integer(values[4], "frame")
        if (clip, frame) in seen:
            raise ValueError(f"Duplicate video/frame: {clip}/{frame}")
        seen.add((clip, frame))
        if frame >= video["reported_frame_count"]:
            raise ValueError(f"Frame outside video: {clip}/{frame}")
        seconds = numeric(values[3], "time")
        if abs(seconds - frame / video["fps"]) > TIME_TOLERANCE:
            raise ValueError(f"Frame/time mismatch: {clip}/{frame}")
        if values[11] not in STATUS_MAP:
            raise ValueError(f"Unsupported visibility: {values[11]!r}")
        status = STATUS_MAP[values[11]]
        note = str(values[14] or "")
        status_basis = "source_visibility"
        if status == "unknown" and "\u51fa\u753b\u9762" in note:
            status, status_basis = "out_of_view", "explicit_source_note"
        if values[13] == "\u955c\u5934\u5207\u6362":
            status, status_basis = "shot_cut", "source_phase"
        interpolated = "\u63d2\u503c" in note
        truncated = "\u534a\u622a\u51fa\u753b" in note
        center_values, bbox_values = values[5:7], values[7:11]
        if any(v is None for v in center_values) != all(v is None for v in center_values):
            raise ValueError(f"Incomplete center: {clip}/{frame}")
        if any(v is None for v in bbox_values) != all(v is None for v in bbox_values):
            raise ValueError(f"Incomplete bounding box: {clip}/{frame}")
        center = None if all(v is None for v in center_values) else [numeric(v, "center") for v in center_values]
        bbox = None if all(v is None for v in bbox_values) else [numeric(v, "bbox") for v in bbox_values]
        width, height = video["width"], video["height"]
        if center is not None and not (0 <= center[0] < width and 0 <= center[1] < height):
            raise ValueError(f"Center outside image: {clip}/{frame}")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid bounding box geometry: {clip}/{frame}")
            if center is None or not (x1 <= center[0] <= x2 and y1 <= center[1] <= y2):
                raise ValueError(f"Center outside bounding box: {clip}/{frame}")
        if status in {"unknown", "out_of_view", "shot_cut"} and (center is not None or bbox is not None):
            raise ValueError(f"Unlocalized status has geometry: {clip}/{frame}")
        reasons = []
        if status not in {"clear", "blur"}:
            reasons.append(f"visibility_{status}")
        if bbox is None:
            reasons.append("no_source_bounding_box")
        if interpolated:
            reasons.append("interpolated_position")
        if truncated:
            reasons.append("truncated_at_picture_boundary")
        rows.append({"id": f"{clip}_f{frame:06d}", "video_id": clip, "frame_zero_based": frame,
                     "time_seconds": seconds, "exact_frame_time_seconds": frame / video["fps"],
                     "visibility": status, "visibility_basis": status_basis, "center_xy_px": center,
                     "bbox_xyxy_px": bbox, "interpolated": interpolated, "truncated": truncated,
                     "geometry_kind": "bbox_and_center" if bbox else "center_only" if center else "unlocalized",
                     "method_raw": values[12], "phase_raw": values[13], "note_raw": values[14],
                     "split": SPLITS[clip], "detection_train_eligible": not reasons,
                     "detection_exclusion_reasons": reasons, "negative_sample_eligible": False,
                     "partial_visibility_candidate": status == "partial", "provenance": PROVENANCE,
                     "independent_ground_truth": False, "source": source})
    if {row["video_id"] for row in rows} != set(videos):
        raise ValueError("Frame sheet must contain all four allowlisted videos")
    return rows


def decode_and_export(videos, rows, staging, output, export_yolo):
    eligible = {(row["video_id"], row["frame_zero_based"]): row for row in rows if row["detection_train_eligible"]}
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
                    raise ValueError(f"Decoded image size changed: {clip}/{decoded}")
                row = eligible.get((clip, decoded))
                if export_yolo and row is not None:
                    directory = "val" if row["split"] == "validation" else "train"
                    image_relative = f"yolo/images/{directory}/{row['id']}.jpg"
                    label_relative = f"yolo/labels/{directory}/{row['id']}.txt"
                    image_path, label_path = staging / image_relative, staging / label_relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise ValueError(f"Cannot write extracted frame: {image_path}")
                    x1, y1, x2, y2 = row["bbox_xyxy_px"]
                    width, height = video["width"], video["height"]
                    target = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                              (x2 - x1) / width, (y2 - y1) / height]
                    label_path.write_text("0 " + " ".join(f"{value:.9f}" for value in target) + "\n", encoding="ascii")
                    exports.append({"annotation_id": row["id"], "image": image_relative, "label": label_relative,
                                    "image_sha256": sha256(image_path), "label_sha256": sha256(label_path),
                                    "width": width, "height": height, "split": row["split"]})
                decoded += 1
        finally:
            capture.release()
        if decoded != video["reported_frame_count"]:
            raise ValueError(f"Decoded frame count mismatch: {clip}: {decoded} != {video['reported_frame_count']}")
        video["decoded_frame_count"] = decoded
        if sha256(video["path"]) != video["sha256"]:
            raise ValueError(f"Source video changed during import: {clip}")
    if export_yolo:
        if len(exports) != len(eligible):
            raise ValueError("Not every eligible annotation was exported")
        dataset = ("# Development only; human-reviewed candidates, not independent ground truth.\n"
                   "# Grouped by entire source video; no unseen-data performance claim.\n"
                   f"path: {json.dumps(str(output / 'yolo'), ensure_ascii=False)}\n"
                   "train: images/train\nval: images/val\nnames:\n  0: ball\n")
        (staging / "yolo/dataset.yaml").write_text(dataset, encoding="utf-8")
    return exports


def prepare(source, output, *, core_root=DEFAULT_CORE_ROOT, match_root=DEFAULT_MATCH_ROOT, export_yolo=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    workbook_hash = sha256(source)
    sheets = read_workbook(source, workbook_hash)
    videos, contacts = inspect_videos(video_sources(core_root, match_root), sheets[SUMMARY_SHEET])
    rows = normalize_rows(sheets[FRAME_SHEET], videos)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        exports = decode_and_export(videos, rows, staging, output, export_yolo)
        if sha256(source) != workbook_hash:
            raise ValueError("Source workbook changed during import")
        common = {"schema": SCHEMA, "provenance": PROVENANCE, "independent_ground_truth": False,
                  "development_only": True, "unseen_data_claim": False}
        summary = {**common, "rows": len(rows), "by_video": dict(Counter(row["video_id"] for row in rows)),
                   "visibility_counts": dict(Counter(row["visibility"] for row in rows)),
                   "raw_visibility_counts": dict(Counter(row["source"]["raw_values"][11] for row in rows)),
                   "geometry_counts": dict(Counter(row["geometry_kind"] for row in rows)),
                   "detection_train_eligible": sum(row["detection_train_eligible"] for row in rows),
                   "eligible_by_split": dict(Counter(row["split"] for row in rows if row["detection_train_eligible"])),
                   "partial_visibility_rows": sum(row["partial_visibility_candidate"] for row in rows),
                   "partial_with_bbox": sum(row["partial_visibility_candidate"] and row["bbox_xyxy_px"] is not None for row in rows),
                   "interpolated_rows": [row["id"] for row in rows if row["interpolated"]],
                   "truncated_rows": [row["id"] for row in rows if row["truncated"]],
                   "negative_samples": 0, "yolo_exported_images": len(exports),
                   "validation": {"duplicate_frames": 0, "geometry_errors": 0, "time_errors": 0,
                                  "source_workbook_unchanged": True, "all_video_frame_counts_decoded": True}}
        annotations = {**common, "frame_headers": FRAME_HEADERS, "annotations": rows,
                       "contact_intervals": contacts, "source_summary_rows": sheets[SUMMARY_SHEET],
                       "source_instruction_rows": sheets[NOTES_SHEET]}
        manifest = {**common, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_workbook": {"path": str(source), "sha256": workbook_hash},
                    "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
                    "source_video_roots": {"core_playback": str(Path(core_root).resolve()), "match_videos": str(Path(match_root).resolve())},
                    "videos": videos, "split_policy": {"unit": "entire_source_video", "assignments": SPLITS,
                                                          "frame_random_split": False, "development_only": True},
                    "policy": {"eligible_visibility": ["clear", "blur"], "source_bbox_required": True,
                               "interpolation_excluded": True, "truncation_excluded": True,
                               "partial_visibility_excluded": True, "unlocalized_is_negative": False,
                               "contact_phase_is_single_frame_truth": False, "time_tolerance_seconds": TIME_TOLERANCE,
                               "row_hash_encoding": "UTF-8 JSON array; ensure_ascii=False; separators=(',', ':')",
                               "embedded_workbook_images": "retained in unchanged source workbook; not copied to JSON"},
                    "unopened_holdout_video_ids": ["core_05", "core_06", "core_09"],
                    "exports": exports, "dataset_yaml": "yolo/dataset.yaml" if export_yolo else None}
        write_json(staging / "annotations.json", annotations)
        write_json(staging / "summary.json", summary)
        manifest["artifact_sha256"] = {name: sha256(staging / name) for name in ("annotations.json", "summary.json")}
        write_json(staging / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(f"Output directory appeared during import: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-xlsx", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ball_frame_annotations_2026-10-27")
    parser.add_argument("--core-playback-root", type=Path, default=DEFAULT_CORE_ROOT)
    parser.add_argument("--match-videos-root", type=Path, default=DEFAULT_MATCH_ROOT)
    parser.add_argument("--export-yolo", action="store_true")
    args = parser.parse_args()
    summary = prepare(args.source_xlsx, args.output_dir, core_root=args.core_playback_root,
                      match_root=args.match_videos_root, export_yolo=args.export_yolo)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
