"""Import sparse keeper references from an immutable review ZIP into target ROIs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from scripts.import_manual_pose_joints import JOINT_NAMES, finite, integer, save_json, sha256


DOWNLOADS = Path.home() / "Downloads"
DEFAULT_ZIP = DOWNLOADS / "IND5003_\u8d44\u6599\u5ba1\u9605_20261028/\u95e8\u5c06\u539f\u70b9\u786e\u8ba4\u66f4\u65b0_2026-10-28.zip"
DEFAULT_PACK = DOWNLOADS / "IND5003_\u8d44\u6599\u5ba1\u9605_20261025/vision2_pose_pack_v2"
DEFAULT_MATCHES = DOWNLOADS / "IND5003_\u8d44\u6599\u5ba1\u9605_20261025/three_matches_release/extracted/Penalty_3_Matches_Export_20261025"
DEFAULT_OUTPUT = ROOT / "outputs/keeper_reference_import_2026-10-28"
VISIBLE = {"visible_confirmed", "visible_corrected"}
UNLOCATED = {"occluded", "left_right_ambiguous"}
ORIGINS = {"original_user_export", "user_message_bulk_reference_confirmation"}
TRAIN_CLIPS = {"05-04", "07-08", "10-01", "12-02", "17-08", "19-08"}
VALIDATION_CLIPS = {"02-05", "02-06"}
TEST_CLIPS = {f"match_{i}" for i in range(17, 25)}
EMPTY_REFERENCE_FILE = "\u539f\u53c2\u8003\u4e3a\u7a7a\u7684\u9636\u6bb5\u5173\u8282.csv"
CONFLICT_FILE = "\u539f\u70b9\u4e0e\u4eba\u5de5\u6846\u4e00\u81f4\u6027\u63d0\u793a.csv"


def optional_bool(value):
    if value in (None, ""):
        return None
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def csv_records(content):
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def read_package(path):
    """Read data members only; never extract or execute package code."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP member names")
        for name in names:
            normalized = name.replace("\\", "/")
            member = PurePosixPath(normalized)
            if member.is_absolute() or ".." in member.parts or re.match(r"^[A-Za-z]:", normalized):
                raise ValueError(f"Unsafe ZIP member: {name}")
        clips = {}
        member_hashes = {}
        for name in names:
            if name.startswith("human_reviewed/") and name.endswith(".json"):
                data = archive.read(name)
                document = json.loads(data)
                clip = document["video_id"]
                if not re.fullmatch(r"(?:\d{2}-\d{2}|match_\d{2})", clip) or name != f"human_reviewed/{clip}.json" or clip in clips:
                    raise ValueError("Unsafe or repeated clip identity")
                if document.get("frame_numbering") != "zero-based original source":
                    raise ValueError("Unexpected source-frame numbering")
                if document.get("coordinate_system") != "original pixels; x right/y down; anatomical left/right":
                    raise ValueError("Unexpected coordinate system")
                clips[clip] = document
                member_hashes[name] = hashlib.sha256(data).hexdigest()
        if not clips:
            raise ValueError("No reviewed clips")
        required = ["human_boxes.csv", "human_joints.csv", EMPTY_REFERENCE_FILE, CONFLICT_FILE,
                    "current_status.json", "validation.json"]
        members = {name: archive.read(name) for name in required}
        member_hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in members.items()})
        return {"clips": clips, "boxes_csv": csv_records(members["human_boxes.csv"]),
                "joints_csv": csv_records(members["human_joints.csv"]),
                "empty_references": csv_records(members[EMPTY_REFERENCE_FILE]),
                "declared_conflicts": csv_records(members[CONFLICT_FILE]),
                "declared_status": json.loads(members["current_status.json"]),
                "declared_validation": json.loads(members["validation.json"]), "member_sha256": member_hashes}


def identity(clip, pose_inventory, match_inventory):
    if clip in pose_inventory:
        metadata = pose_inventory[clip]
        subject = clip.split("-")[0]
        return {"event_id": f"mendeley_br_x9bsxnpx_v2_{clip}", "match_id": "mendeley_br_x9bsxnpx_v2",
                "subject_group": f"mendeley_subject_{subject}", "source_group": "mendeley_br_x9bsxnpx_v2",
                "source_url": metadata["source_url"], "identity_basis": "dataset_trial_filename",
                "same_subject_trials_must_share_split": True,
                "split": "validation" if clip in VALIDATION_CLIPS else "train"}
    metadata = match_inventory.get(clip)
    if metadata is None:
        raise ValueError(f"Missing original event identity: {clip}")
    return {"event_id": metadata["original_event_id"], "match_id": metadata["dataset_id"],
            "subject_group": None, "source_group": metadata["dataset_id"], "source_url": metadata["source_url"],
            "identity_basis": "original_release_event_mapping", "original_match_id": metadata["original_match_id"],
            "source_start_frame_zero_based": metadata["source_start_frame_zero_based"],
            "same_event_views_must_share_split": True, "split": "test"}


def normalize_package(package):
    boxes_by_key, joints_by_key = {}, {}
    for row in package["boxes_csv"]:
        key = row["video_id"], int(row["frame"])
        if key in boxes_by_key:
            raise ValueError("Duplicate box CSV reference")
        boxes_by_key[key] = row
    for row in package["joints_csv"]:
        key = row["video_id"], int(row["frame"]), int(row["joint_id"])
        if key in joints_by_key:
            raise ValueError("Duplicate joint CSV reference")
        joints_by_key[key] = row
    declared_conflicts = {(row["video_id"], int(row["frame"]), int(row["joint_id"])) for row in package["declared_conflicts"]}
    frames, seen_boxes, seen_joints = [], set(), set()
    for clip, document in sorted(package["clips"].items()):
        width, height = document["source_dimensions"]
        if not re.fullmatch(r"[0-9a-f]{64}", document["source_sha256"]):
            raise ValueError("Invalid source hash")
        for source_index, raw in enumerate(document["frames"]):
            frame = integer(raw["frame"], "frame")
            key = clip, frame
            if key in seen_boxes or raw["video_id"] != clip or key not in boxes_by_key:
                raise ValueError("JSON/CSV box inventories disagree")
            seen_boxes.add(key)
            csv_box = boxes_by_key[key]
            for field in ("keeper_id", "bbox_status"):
                if csv_box[field] != raw[field]:
                    raise ValueError(f"JSON/CSV box disagree: {field}")
            for field in ("bbox_human_reviewed", "bbox_changed_by_human", "occluded", "out_of_frame", "proposal_wrong_person",
                          "identity_resolution_pending", "proposal_was_interpolated"):
                if optional_bool(csv_box[field]) != raw[field]:
                    raise ValueError(f"JSON/CSV box disagree: {field}")
            if raw["bbox_human_reviewed"] is not True:
                raise ValueError("Unreviewed box")
            for field in ("timestamp_s", "review_timestamp_s"):
                if abs(finite(raw[field], field) - float(csv_box[field])) > 1e-9:
                    raise ValueError("JSON/CSV timestamp disagreement")
            bbox = raw["bbox"]
            if bbox is not None:
                if len(bbox) != 4:
                    raise ValueError("Invalid bbox size")
                bbox = [finite(x, "bbox") for x in bbox]
                x1, y1, x2, y2 = bbox
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Box outside source image: {key}")
                if bbox != [float(csv_box[field]) for field in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]:
                    raise ValueError("JSON/CSV box coordinates disagree")
            elif any(csv_box[field] for field in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
                raise ValueError("Missing box acquired CSV coordinates")
            if (raw["bbox_status"] in VISIBLE) != (bbox is not None):
                raise ValueError("Box visibility and localization disagree")
            normalized_joints = []
            for joint_key, joint in raw["human_joints"].items():
                joint_id = integer(joint["joint_id"], "joint_id")
                jkey = clip, frame, joint_id
                if joint_id >= 17 or str(joint_id) != joint_key or joint["joint_name"] != JOINT_NAMES[joint_id]:
                    raise ValueError("Invalid COCO17 joint identity")
                if jkey in seen_joints or jkey not in joints_by_key:
                    raise ValueError("JSON/CSV joint inventories disagree")
                seen_joints.add(jkey)
                csv_joint = joints_by_key[jkey]
                if joint["video_id"] != clip or joint["frame"] != frame or joint["track_id"] != raw["keeper_id"]:
                    raise ValueError("Joint source/keeper identity mismatch")
                if joint["human_reviewed"] is not True or optional_bool(csv_joint["human_reviewed"]) is not True:
                    raise ValueError("Unreviewed joint")
                for field in ("status", "joint_name", "reference_source", "updated_at", "track_id", "reviewer", "import_group"):
                    if csv_joint[field] != joint[field]:
                        raise ValueError(f"JSON/CSV joint disagree: {field}")
                for field in ("x", "y", "timestamp_s"):
                    csv_value = float(csv_joint[field]) if csv_joint[field] else None
                    if csv_value != joint[field]:
                        raise ValueError(f"JSON/CSV joint disagree: {field}")
                if abs(joint["timestamp_s"] - raw["timestamp_s"]) > 1e-9:
                    raise ValueError("Joint/frame timestamp disagreement")
                status = joint["status"]
                if status in VISIBLE:
                    x, y = finite(joint["x"], "x"), finite(joint["y"], "y")
                    if not (0 <= x < width and 0 <= y < height):
                        raise ValueError("Joint outside image")
                    distance = max(bbox[0] - x, x - bbox[2], bbox[1] - y, y - bbox[3], 0) if bbox else None
                elif status in UNLOCATED:
                    if joint["x"] is not None or joint["y"] is not None:
                        raise ValueError("Unknown joint coordinates must stay empty")
                    distance = None
                else:
                    raise ValueError(f"Unknown joint status: {status}")
                origin = csv_joint["annotation_origin"]
                if origin not in ORIGINS or joint.get("annotation_origin", "original_user_export") != origin:
                    raise ValueError("Inconsistent joint annotation origin")
                independent = optional_bool(csv_joint["independent_blinded_annotation"])
                if origin == "user_message_bulk_reference_confirmation":
                    if independent is not False or joint.get("independent_blinded_annotation") is not False:
                        raise ValueError("Bulk acceptance cannot become independent annotation")
                    if joint["reference_source_video_sha256"] != document["source_sha256"] or joint["reference_proposal_sha256"] != document["proposal_sha256"]:
                        raise ValueError("Bulk reference source binding mismatch")
                conflict = jkey in declared_conflicts
                normalized_joints.append({**joint, "annotation_origin": origin, "independent_blinded_annotation": independent,
                                          "declared_box_point_conflict": conflict, "distance_outside_bbox_px": distance,
                                          "coordinate_loss_eligible": status in VISIBLE,
                                          "visibility_loss_eligible": status in VISIBLE | {"occluded"},
                                          "source_csv_record": csv_joint, "source_json_record": joint})
            frames.append({"annotation_id": f"{clip}_f{frame:06d}_{raw['keeper_id']}", "video_id": clip,
                           "frame_zero_based": frame, "timestamp_s": raw["timestamp_s"], "review_timestamp_s": raw["review_timestamp_s"],
                           "keeper_id": raw["keeper_id"], "bbox_original_pixels": bbox, "bbox_status": raw["bbox_status"],
                           "stage": raw["stage"], "joints": normalized_joints, "bbox_human_reviewed": True,
                           "derived_review_flag_csv_stale": optional_bool(csv_box["all_required_stage_joints_reviewed"]) != raw["all_required_stage_joints_reviewed"],
                           "proposal_was_interpolated": raw["proposal_was_interpolated"], "split": "unassigned_grouped",
                           "full_frame_person_annotation_exhaustive": False, "unmarked_people_are_negatives": False,
                           "source_member": f"human_reviewed/{clip}.json", "source_annotation_index_zero_based": source_index,
                           "source_json_record": raw, "source_csv_record": csv_box})
    if seen_boxes != set(boxes_by_key) or seen_joints != set(joints_by_key):
        raise ValueError("JSON/CSV annotation inventories disagree")
    if not declared_conflicts <= seen_joints:
        raise ValueError("Conflict has no matching joint")
    for row in package["empty_references"]:
        key = row["video_id"], int(row["frame"]), int(row["joint_id"])
        if row["x"] or row["y"] or key in seen_joints:
            raise ValueError("Empty reference was filled or overlaps a reviewed joint")
    return frames


def roi_bounds(bbox, width, height, margin):
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]


def sparse_roi_targets(frame, bounds):
    x1, y1, x2, y2 = bounds
    target = {"joint_xy_roi_pixels": [[None, None] for _ in range(17)],
              "joint_xy_original_pixels": [[None, None] for _ in range(17)],
              "coordinate_loss_mask": [False] * 17, "visibility_loss_mask": [False] * 17,
              "visibility_target": [None] * 17, "joint_status": ["unreviewed"] * 17,
              "annotation_origin": [None] * 17, "reference_source": [None] * 17,
              "independent_blinded_annotation": [None] * 17,
              "declared_box_point_conflict": [False] * 17}
    for joint in frame["joints"]:
        index, status = joint["joint_id"], joint["status"]
        target["joint_status"][index] = status
        for key in ("annotation_origin", "reference_source", "independent_blinded_annotation", "declared_box_point_conflict"):
            target[key][index] = joint[key]
        if status in VISIBLE:
            x, y = joint["x"], joint["y"]
            target["joint_xy_original_pixels"][index] = [x, y]
            target["joint_xy_roi_pixels"][index] = [x - x1, y - y1]
            inside = x1 <= x < x2 and y1 <= y < y2
            target["coordinate_loss_mask"][index] = joint["coordinate_loss_eligible"] and inside
        target["visibility_loss_mask"][index] = joint["visibility_loss_eligible"]
        if status in VISIBLE | {"occluded"}:
            target["visibility_target"][index] = status in VISIBLE
    return target


def import_update(zip_path, output, pose_pack=DEFAULT_PACK, matches_root=DEFAULT_MATCHES, roi_margin=0.25):
    zip_path, output, pose_pack, matches_root = map(lambda p: Path(p).resolve(), (zip_path, output, pose_pack, matches_root))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing import: {output}")
    if not 0 <= roi_margin <= 1:
        raise ValueError("ROI margin must be between 0 and 1")
    package_hash = sha256(zip_path)
    package = read_package(zip_path)
    frames = normalize_package(package)
    pose_inventory_path, matches_inventory_path = pose_pack / "inventory.json", matches_root / "metadata/annotations.json"
    pose_inventory = json.loads(pose_inventory_path.read_text())
    match_inventory = {Path(row["filename"]).stem: row for row in json.loads(matches_inventory_path.read_text())["clips"]}
    metadata_hashes = {str(path): sha256(path) for path in (pose_inventory_path, matches_inventory_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    sources, exports, hashes = {}, [], {}
    try:
        (staging / "roi_images").mkdir()
        (staging / "source_frames").mkdir()
        for clip, document in sorted(package["clips"].items()):
            source = (pose_pack if clip in pose_inventory else matches_root) / "videos" / f"{clip}.mp4"
            actual_hash = sha256(source)
            if actual_hash != document["source_sha256"]:
                raise ValueError(f"Source video SHA mismatch: {clip}")
            hashes[source] = actual_hash
            group = identity(clip, pose_inventory, match_inventory)
            selected = {row["frame_zero_based"]: row for row in frames if row["video_id"] == clip}
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open source video: {clip}")
                width, height = [int(capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
                reported_count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
                if [width, height] != document["source_dimensions"]:
                    raise ValueError(f"Source dimensions differ: {clip}")
                index, seen = 0, set()
                previous_pts = None
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if not math.isfinite(pts) or (previous_pts is not None and pts <= previous_pts):
                        raise ValueError(f"Nonincreasing decoded PTS: {clip}/{index}")
                    previous_pts = pts
                    if index in selected:
                        row = selected[index]
                        if abs(row["timestamp_s"] - pts) > 1e-6:
                            raise ValueError(f"Source frame PTS mismatch: {clip}/{index}")
                        row.update(group)
                        row.update(source_video_sha256=actual_hash, source_video_path=str(source),
                                   decoded_pts_seconds=pts, source_width=width, source_height=height)
                        seen.add(index)
                        if row["joints"] and row["bbox_status"] in VISIBLE:
                            bounds = roi_bounds(row["bbox_original_pixels"], width, height, roi_margin)
                            x1, y1, x2, y2 = bounds
                            crop = image[y1:y2, x1:x2]
                            image_path = staging / "roi_images" / f"{row['annotation_id']}.png"
                            if not cv2.imwrite(str(image_path), crop):
                                raise ValueError("Cannot export source ROI")
                            original_path = staging / "source_frames" / f"{row['annotation_id']}.jpg"
                            if not cv2.imwrite(str(original_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                                raise ValueError("Cannot export original source frame")
                            targets = sparse_roi_targets(row, bounds)
                            exports.append({**{key: row[key] for key in ("annotation_id", "video_id", "event_id", "match_id", "subject_group", "source_group", "keeper_id", "split", "frame_zero_based", "timestamp_s", "stage")},
                                            "id": row["annotation_id"], "image": str(original_path.relative_to(staging)),
                                            "image_sha256": sha256(original_path), "width": width, "height": height,
                                            "roi_image": str(image_path.relative_to(staging)), "roi_image_sha256": sha256(image_path),
                                            "roi_width": x2 - x1, "roi_height": y2 - y1, "roi_original_pixels": bounds,
                                            "roi_margin_fraction_of_bbox": roi_margin, "bbox_original_pixels": row["bbox_original_pixels"],
                                            "bbox": row["bbox_original_pixels"], "source_sha256": actual_hash,
                                            "source_video_sha256": actual_hash, "source_video_path": str(source),
                                            "source_width": width, "source_height": height, "decoded_pts_seconds": pts,
                                            "training_task": "single_target_roi_sparse_pose", "unmarked_people_are_negatives": False,
                                            "joint_xy_px": targets["joint_xy_original_pixels"],
                                            "review_status": targets["joint_status"], **targets})
                    index += 1
                if index != reported_count or seen != set(selected):
                    raise ValueError(f"Source decode/frame inventory mismatch: {clip}")
                sources[clip] = {"path": str(source), "sha256": actual_hash, "width": width, "height": height,
                                 "fps": fps, "decoded_frames": index, "source_frame_pts_verified": True,
                                 "reviewed_box_frames": len(selected), **group}
            finally:
                capture.release()
            print(f"Imported {clip}: {len(selected)} boxes, {sum(bool(row['joints']) for row in selected.values())} sparse ROI frames", flush=True)
        if sha256(zip_path) != package_hash or any(sha256(path) != digest for path, digest in hashes.items()) or any(sha256(path) != digest for path, digest in metadata_hashes.items()):
            raise ValueError("Source changed during import")
        all_joints = [joint for row in frames for joint in row["joints"]]
        summary = {"clips": len(sources), "box_records": len(frames), "box_status_counts": dict(Counter(row["bbox_status"] for row in frames)),
                   "joint_records": len(all_joints), "joint_status_counts": dict(Counter(row["status"] for row in all_joints)),
                   "joint_origin_counts": dict(Counter(row["annotation_origin"] for row in all_joints)),
                   "empty_references_preserved": len(package["empty_references"]), "declared_conflicts_preserved": len(package["declared_conflicts"]),
                   "roi_exports": len(exports), "coordinate_supervision_points": sum(sum(row["coordinate_loss_mask"]) for row in exports),
                   "stale_derived_csv_review_flags_preserved": sum(row["derived_review_flag_csv_stale"] for row in frames),
                   "visibility_supervision_points": sum(sum(row["visibility_loss_mask"]) for row in exports),
                   "source_zip_and_videos_unchanged": True}
        if summary["box_records"] != package["declared_status"]["box_records"] or summary["joint_records"] != package["declared_status"]["human_joint_records"]:
            raise ValueError("Imported inventory differs from package declaration")
        save_json(staging / "annotations.json", {"frames": frames, "empty_references": package["empty_references"],
                                                 "declared_conflicts": package["declared_conflicts"],
                                                 "source_member_sha256": package["member_sha256"]})
        save_json(staging / "summary.json", summary)
        save_json(staging / "manifest.json", {"schema": "keeper_reference_roi/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                              "source_zip": str(zip_path), "source_zip_sha256": package_hash,
                                              "source_metadata_sha256": metadata_hashes, "sources": sources,
                                              "source_pixels_verified": True, "independent_ground_truth": False, "development_only": True,
                                              "training_task": "single_target_roi_sparse_pose", "full_frame_detection_training_allowed": False,
                                              "unknown_joint_coordinates_synthesized": False, "checkpoint_selection_policy": "must_be_frozen_by_training_protocol",
                                              "split_policy": {"train": sorted(TRAIN_CLIPS), "validation": sorted(VALIDATION_CLIPS),
                                                               "test": sorted(TEST_CLIPS), "all_partitions_are_development": True,
                                                               "same_shooter_02_trials_kept_together": True,
                                                               "world_cup_final_source_group_kept_together": True},
                                              "limitations": ["Human reviewed model/assistant proposals are not independent blinded truth.",
                                                              "Only the target keeper is annotated; other people must not become full-frame negatives.",
                                                              "Sparse ROI pose losses must use the provided masks; unreviewed joints are not zeros.",
                                                              "Bounding-box and joint discrepancies retain original values and conflict flags; only points outside the fixed ROI lose coordinate supervision.",
                                                              "ROI pose accuracy with supplied keeper boxes is not end-to-end person detection accuracy."],
                                              "summary": summary, "exports": exports})
        staging.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pose-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--matches-root", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--roi-margin", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(import_update(args.zip_path, args.output, args.pose_pack, args.matches_root, args.roi_margin), indent=2))


if __name__ == "__main__":
    main()
