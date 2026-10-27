"""Bind the three delivered ball-reference videos for evaluation only."""
from __future__ import annotations

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

try:
    from scripts.prepare_ball_frame_annotations import (
        DEFAULT_CORE_ROOT, DOWNLOADS, FRAME_HEADERS, FRAME_SHEET, NOTES_SHEET,
        PROVENANCE, ROOT, STATUS_MAP, SUMMARY_SHEET, TIME_TOLERANCE,
        frame_integer, numeric, read_workbook, sha256, write_json,
    )
except ModuleNotFoundError:
    from prepare_ball_frame_annotations import (
        DEFAULT_CORE_ROOT, DOWNLOADS, FRAME_HEADERS, FRAME_SHEET, NOTES_SHEET,
        PROVENANCE, ROOT, STATUS_MAP, SUMMARY_SHEET, TIME_TOLERANCE,
        frame_integer, numeric, read_workbook, sha256, write_json,
    )


VIDEO_IDS = ("core_05", "core_06", "core_09")
DEFAULT_SOURCE = DOWNLOADS / "\u6d4b\u8bd5\u96c6_\u7403\u8def\u9010\u5e27_core05_06_09.xlsx"
SCHEMA = "ball_frame_evaluation/v1"


def inspect_videos(core_root, summary_rows):
    descriptions, contacts = {}, []
    for source in summary_rows[1:]:
        values = source["raw_values"]
        clip = values[0]
        if clip not in VIDEO_IDS or clip in descriptions:
            raise ValueError(f"Unexpected or duplicate summary video: {clip}")
        description = re.search(r"(\d+)[\u00d7x](\d+)\s+(\d+(?:\.\d+)?)fps", str(values[1]))
        contact = re.match(r"(\d+)\u2192(\d+)\u4e4b\u95f4", str(values[2]))
        if not description or not contact:
            raise ValueError(f"Missing video description or contact interval: {clip}")
        descriptions[clip] = tuple(float(x) for x in description.groups())
        lower, upper = map(int, contact.groups())
        if upper != lower + 1:
            raise ValueError(f"Contact reference is not adjacent frames: {clip}")
        contacts.append({"video_id": clip, "frame_interval_zero_based": [lower, upper],
                         "is_single_frame_ground_truth": False, "source": source})
    if set(descriptions) != set(VIDEO_IDS):
        raise ValueError("Missing evaluation video summary")
    videos = {}
    for clip in VIDEO_IDS:
        path = (Path(core_root) / f"{clip}.mp4").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        expected_width, expected_height, expected_fps = descriptions[clip]
        if (width, height) != (expected_width, expected_height) or not math.isclose(fps, expected_fps, abs_tol=1e-6):
            raise ValueError(f"Video dimensions/fps do not match reference: {clip}")
        if count <= 0 or fps <= 0:
            raise ValueError(f"Invalid video: {clip}")
        videos[clip] = {"path": str(path), "sha256": sha256(path), "width": width,
                        "height": height, "fps": fps, "reported_frame_count": count, "split": "test"}
    for contact in contacts:
        video = videos[contact["video_id"]]
        if not 0 <= contact["frame_interval_zero_based"][0] < contact["frame_interval_zero_based"][1] < video["reported_frame_count"]:
            raise ValueError("Contact interval outside video")
        contact["time_interval_seconds"] = [frame / video["fps"] for frame in contact["frame_interval_zero_based"]]
    return videos, contacts


def normalize_rows(source_rows, videos):
    rows, seen = [], set()
    for source in source_rows[1:]:
        values = source["raw_values"]
        clip = values[2]
        if clip not in VIDEO_IDS or clip not in videos:
            raise ValueError(f"Video is not in evaluation allowlist: {clip}")
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
        if status == "unknown" and any(text in note for text in ("\u51fa\u753b\u9762", "\u4ece\u753b\u9762\u5de6\u8fb9\u7f18\u51fa\u753b")):
            status, status_basis = "out_of_view", "explicit_source_note"
        if values[13] == "\u955c\u5934\u5207\u6362":
            status, status_basis = "shot_cut", "source_phase"
        interpolated = "\u63d2\u503c" in note
        truncated = "\u534a\u622a" in note and "\u753b" in note
        declared_repeat = "\u4e0e\u4e0a\u4e00\u5e27\u753b\u9762\u91cd\u590d" in note
        quality_flags = []
        if declared_repeat and "\u540c\u4e0a" in note:
            predecessor = next((row for row in reversed(rows)
                                if row["video_id"] == clip and row["frame_zero_based"] == frame - 1), None)
            if predecessor and predecessor["visibility"] not in {"clear", "blur"}:
                quality_flags.append("copied_previous_frame_has_lower_visibility")
                status = predecessor["visibility"]
                status_basis = "source_note_same_as_previous_frame"
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
        identity_uncertain = clip == "core_09" and frame >= 109
        rows.append({"id": f"{clip}_f{frame:06d}", "video_id": clip, "frame_zero_based": frame,
                     "time_seconds": seconds, "exact_frame_time_seconds": frame / video["fps"],
                     "visibility": status, "visibility_basis": status_basis,
                     "center_xy_px": center, "bbox_xyxy_px": bbox,
                     "interpolated": interpolated, "truncated": truncated,
                     "geometry_kind": "bbox_and_center" if bbox else "center_only" if center else "unlocalized",
                     "method_raw": values[12], "phase_raw": values[13], "note_raw": values[14],
                     "split": "test", "detection_train_eligible": False,
                     "detection_evaluation_eligible": not reasons,
                     "detection_exclusion_reasons": reasons,
                     "negative_sample_eligible": False,
                     "declared_repeat_of_previous_frame": declared_repeat,
                     "quality_flags": quality_flags,
                     "identity_uncertain_after_reentry": identity_uncertain,
                     "continuous_shot_identity_reference": not identity_uncertain,
                     "provenance": PROVENANCE, "independent_ground_truth": False, "source": source})
    if {row["video_id"] for row in rows} != set(VIDEO_IDS):
        raise ValueError("Frame sheet must contain all three evaluation videos")
    return rows


def assign_duplicate_groups(rows):
    # Both actual identical pixels and the annotator's explicit repeat evidence are retained.
    parents = {row["id"]: row["id"] for row in rows}
    by_frame = {(row["video_id"], row["frame_zero_based"]): row for row in rows}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    def join(first, second):
        a, b = root(first), root(second)
        if a != b:
            parents[max(a, b)] = min(a, b)

    signatures = {}
    for row in rows:
        key = (row["video_id"], row["decoded_pixels_sha256"])
        if key in signatures:
            join(row["id"], signatures[key])
        else:
            signatures[key] = row["id"]
        previous = by_frame.get((row["video_id"], row["frame_zero_based"] - 1))
        if row["declared_repeat_of_previous_frame"] and previous:
            join(row["id"], previous["id"])
    selected = {}
    for row in sorted(rows, key=lambda value: (value["video_id"], value["frame_zero_based"])):
        group = root(row["id"])
        row["duplicate_group_id"] = group
        eligible = row["detection_evaluation_eligible"]
        row["deduplicated_primary"] = eligible and group not in selected
        if eligible:
            selected.setdefault(group, row["id"])
    for row in rows:
        row["deduplicated_representative_id"] = selected.get(row["duplicate_group_id"])


def decode_and_export(videos, rows, staging):
    lookup = {(row["video_id"], row["frame_zero_based"]): row for row in rows}
    exports = []
    for clip, video in videos.items():
        capture = cv2.VideoCapture(video["path"])
        decoded = 0
        previous_hash = None
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape[:2] != (video["height"], video["width"]):
                    raise ValueError(f"Decoded image size changed: {clip}/{decoded}")
                pixels_hash = hashlib.sha256(image.tobytes()).hexdigest()
                row = lookup.get((clip, decoded))
                if row:
                    row["decoded_pixels_sha256"] = pixels_hash
                    row["previous_decoded_pixels_sha256"] = previous_hash
                    row["pixels_identical_to_previous_frame"] = pixels_hash == previous_hash
                    if row["detection_evaluation_eligible"]:
                        image_relative = f"yolo/images/test/{row['id']}.jpg"
                        label_relative = f"yolo/labels/test/{row['id']}.txt"
                        image_path, label_path = staging / image_relative, staging / label_relative
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        label_path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise ValueError(f"Cannot write evaluation frame: {image_path}")
                        x1, y1, x2, y2 = row["bbox_xyxy_px"]
                        width, height = video["width"], video["height"]
                        target = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                                  (x2 - x1) / width, (y2 - y1) / height]
                        label_path.write_text("0 " + " ".join(f"{value:.9f}" for value in target) + "\n", encoding="ascii")
                        exports.append({"annotation_id": row["id"], "image": image_relative,
                                        "label": label_relative, "image_sha256": sha256(image_path),
                                        "label_sha256": sha256(label_path), "width": width, "height": height,
                                        "split": "test", "normalized": target})
                previous_hash = pixels_hash
                decoded += 1
        finally:
            capture.release()
        if decoded != video["reported_frame_count"]:
            raise ValueError(f"Decoded frame count mismatch: {clip}")
        if sha256(video["path"]) != video["sha256"]:
            raise ValueError(f"Source video changed during import: {clip}")
        video["decoded_frame_count"] = decoded
    if any("decoded_pixels_sha256" not in row for row in rows):
        raise ValueError("Some annotated frames were not decoded")
    assign_duplicate_groups(rows)
    by_id = {row["id"]: row for row in rows}
    for exported in exports:
        row = by_id[exported["annotation_id"]]
        for key in ("video_id", "frame_zero_based", "decoded_pixels_sha256", "duplicate_group_id",
                    "deduplicated_primary", "declared_repeat_of_previous_frame",
                    "identity_uncertain_after_reentry"):
            exported[key] = row[key]
    return exports


def prepare(source, output, *, core_root=DEFAULT_CORE_ROOT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    workbook_hash = sha256(source)
    sheets = read_workbook(source, workbook_hash)
    videos, contacts = inspect_videos(core_root, sheets[SUMMARY_SHEET])
    rows = normalize_rows(sheets[FRAME_SHEET], videos)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        exports = decode_and_export(videos, rows, staging)
        if sha256(source) != workbook_hash:
            raise ValueError("Source workbook changed during import")
        common = {"schema": SCHEMA, "provenance": PROVENANCE,
                  "evaluation_only": True, "training_permitted": False,
                  "independent_ground_truth": False, "whole_pipeline_unseen_data_claim": False,
                  "module_holdout_scope": "not used by the new ball-frame detector pilot; older pipeline exposure exists"}
        summary = {**common, "rows": len(rows),
                   "by_video": dict(Counter(row["video_id"] for row in rows)),
                   "geometry_counts": dict(Counter(row["geometry_kind"] for row in rows)),
                   "evaluation_frames": len(exports),
                   "evaluation_frames_by_video": dict(Counter(row["video_id"] for row in exports)),
                   "deduplicated_evaluation_frames": sum(row["deduplicated_primary"] for row in exports),
                   "deduplicated_frames_by_video": dict(Counter(row["video_id"] for row in exports if row["deduplicated_primary"])),
                   "declared_repeat_rows": sum(row["declared_repeat_of_previous_frame"] for row in rows),
                   "exact_repeat_rows": sum(row["pixels_identical_to_previous_frame"] for row in rows),
                   "interpolated_rows": [row["id"] for row in rows if row["interpolated"]],
                   "truncated_rows": [row["id"] for row in rows if row["truncated"]],
                   "quality_flagged_rows": [row["id"] for row in rows if row["quality_flags"]],
                   "identity_uncertain_evaluation_frames": sum(row["identity_uncertain_after_reentry"] for row in exports),
                   "negative_samples": 0}
        annotations = {**common, "frame_headers": FRAME_HEADERS, "annotations": rows,
                       "contact_intervals": contacts, "source_summary_rows": sheets[SUMMARY_SHEET],
                       "source_instruction_rows": sheets[NOTES_SHEET]}
        manifest = {**common, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_workbook": {"path": str(source), "sha256": workbook_hash},
                    "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
                    "videos": videos, "exports": exports,
                    "policy": {"split": "test", "eligible_visibility": ["clear", "blur"],
                               "source_bbox_required": True, "interpolation_excluded": True,
                               "truncation_excluded": True, "partial_visibility_excluded": True,
                               "unlocalized_is_negative": False, "time_tolerance_seconds": TIME_TOLERANCE,
                               "primary_metric": "one earliest eligible frame per duplicate group",
                               "duplicate_group_evidence": ["identical decoded pixels within video", "source-declared adjacent repeat"],
                               "reentry_identity": "ball detection only; no continuous shot identity claim from core09 frame109",
                               "contact_phase_is_single_frame_truth": False,
                               "embedded_workbook_images": "retained in unchanged source workbook"}}
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ball_frame_evaluation_2026-10-27")
    parser.add_argument("--core-playback-root", type=Path, default=DEFAULT_CORE_ROOT)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    print(json.dumps(prepare(args.source_xlsx, args.output_dir, core_root=args.core_playback_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
