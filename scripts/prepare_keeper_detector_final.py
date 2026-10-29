"""Export reviewed defending-keeper boxes without treating unknown frames as negatives."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from scripts.import_keeper_reference_update import (
    DEFAULT_MATCHES, DEFAULT_PACK, DEFAULT_ZIP, TEST_CLIPS, TRAIN_CLIPS,
    VALIDATION_CLIPS, VISIBLE, identity, normalize_package, read_package,
)
from scripts.import_manual_pose_joints import save_json, sha256


def split_for(video_id):
    if video_id in TRAIN_CLIPS:
        return "train"
    if video_id in VALIDATION_CLIPS:
        return "development_subject_02"
    if video_id in TEST_CLIPS:
        return "development_external_match"
    raise ValueError(f"Unexpected keeper source: {video_id}")


def box_target(box, width, height):
    if width <= 0 or height <= 0 or len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError("Invalid image dimensions or box")
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("Keeper box outside original image")
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def choose_frames(frames, pose_reference_keys, per_development_clip=32):
    if per_development_clip < 2:
        raise ValueError("At least two temporal development samples are required")
    grouped = defaultdict(list)
    for row in frames:
        split_for(row["video_id"])
        if row["bbox_status"] in VISIBLE:
            raw = row["source_json_record"]
            if not row["bbox_human_reviewed"] or raw["identity_resolution_pending"] or raw["out_of_frame"]:
                raise ValueError("A localized box has unresolved target identity")
            grouped[row["video_id"]].append(row)
    valid_keys = {(row["video_id"], row["frame_zero_based"]) for rows in grouped.values() for row in rows}
    if not set(pose_reference_keys) <= valid_keys:
        raise ValueError("Pose reference no longer has a valid keeper box")
    selected = set(pose_reference_keys)
    for video_id, rows in grouped.items():
        rows.sort(key=lambda row: row["frame_zero_based"])
        if split_for(video_id) == "train":
            selected.update((video_id, row["frame_zero_based"]) for row in rows)
        else:
            count = min(per_development_clip, len(rows))
            for index in range(count):
                position = round(index * (len(rows) - 1) / (count - 1)) if count > 1 else 0
                selected.add((video_id, rows[position]["frame_zero_based"]))
    return selected


def validate_partition(exports):
    if len(exports) != len({row["id"] for row in exports}):
        raise ValueError("Duplicate frame identity")
    groups = defaultdict(set)
    hashes = defaultdict(set)
    for row in exports:
        expected = split_for(row["video_id"])
        if row["split"] != expected:
            raise ValueError("A video crossed its frozen partition")
        group = row["subject_group"] or row["match_id"]
        groups[group].add(expected)
        hashes[row["image_sha256"]].add(expected)
        if row["class_id"] != 0 or row["class_name"] != "active_keeper":
            raise ValueError("Wrong specialized keeper class")
    if any(len(value) > 1 for value in groups.values()):
        raise ValueError("A subject or match group crosses partitions")
    if any(len(value) > 1 for value in hashes.values()):
        raise ValueError("Identical pixels cross partitions")


def prepare(zip_path, output, pose_manifest, pose_pack=DEFAULT_PACK, matches_root=DEFAULT_MATCHES,
            per_development_clip=32):
    zip_path, output, pose_manifest, pose_pack, matches_root = [Path(p).resolve() for p in
        (zip_path, output, pose_manifest, pose_pack, matches_root)]
    if output.exists():
        raise FileExistsError("Preserve prior data exports")
    source_hash = sha256(zip_path)
    package = read_package(zip_path)
    frames = normalize_package(package)
    pose = json.loads(pose_manifest.read_text())
    if pose["source_zip_sha256"] != source_hash or not pose["source_pixels_verified"]:
        raise ValueError("Sparse pose reference source differs from the full box source")
    reference_keys = {(r["video_id"], r["frame_zero_based"]) for r in pose["exports"]}
    selected = choose_frames(frames, reference_keys, per_development_clip)
    valid_counts = Counter(split_for(row["video_id"]) for row in frames if row["bbox_status"] in VISIBLE)
    if valid_counts != {"train": 835, "development_subject_02": 275, "development_external_match": 1836}:
        raise ValueError(f"Frozen source inventory changed: {valid_counts}")
    inventory_path = pose_pack / "inventory.json"
    matches_path = matches_root / "metadata/annotations.json"
    pose_inventory = json.loads(inventory_path.read_text())
    match_inventory = {Path(r["filename"]).stem: r for r in json.loads(matches_path.read_text())["clips"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    exports, sources = [], {}
    try:
        for video_id, document in sorted(package["clips"].items()):
            source = (pose_pack if video_id in pose_inventory else matches_root) / "videos" / f"{video_id}.mp4"
            digest = sha256(source)
            if digest != document["source_sha256"] or digest != pose["sources"][video_id]["sha256"]:
                raise ValueError(f"Source identity differs: {video_id}")
            group = identity(video_id, pose_inventory, match_inventory)
            split = split_for(video_id)
            subset = {row["frame_zero_based"]: row for row in frames
                      if row["video_id"] == video_id and (video_id, row["frame_zero_based"]) in selected}
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot open {video_id}")
                width, height = [int(capture.get(prop)) for prop in
                                 (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
                fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if [width, height] != document["source_dimensions"]:
                    raise ValueError("Decoded dimensions differ from reviewed coordinates")
                for kind in ("images", "labels"):
                    (staging / kind / split).mkdir(parents=True, exist_ok=True)
                index, seen, last_pts = 0, set(), None
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if not math.isfinite(pts) or (last_pts is not None and pts <= last_pts):
                        raise ValueError("Nonmonotonic decoded source PTS")
                    last_pts = pts
                    if index in subset:
                        row = subset[index]
                        if abs(pts - row["timestamp_s"]) > 1e-6:
                            raise ValueError(f"Frame/PTS mismatch: {video_id}/{index}")
                        target = box_target(row["bbox_original_pixels"], width, height)
                        image_path = staging / "images" / split / f"{row['annotation_id']}.jpg"
                        label_path = staging / "labels" / split / f"{row['annotation_id']}.txt"
                        if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise RuntimeError("Cannot export original frame")
                        label_path.write_text("0 " + " ".join(f"{v:.10f}" for v in target) + "\n")
                        exports.append({**group, "split": split, "id": row["annotation_id"],
                            "video_id": video_id, "frame_zero_based": index, "timestamp_s": pts,
                            "decoded_pts_seconds": pts, "image": str(image_path.relative_to(staging)),
                            "label": str(label_path.relative_to(staging)), "image_sha256": sha256(image_path),
                            "label_sha256": sha256(label_path), "width": width, "height": height,
                            "bbox": row["bbox_original_pixels"], "bbox_status": row["bbox_status"],
                            "source_video_path": str(source), "source_video_sha256": digest,
                            "class_id": 0, "class_name": "active_keeper", "keeper_id": row["keeper_id"],
                            "proposal_was_interpolated": row["proposal_was_interpolated"],
                            "existing_pose_reference": (video_id, index) in reference_keys,
                            "stage": row["stage"]})
                        seen.add(index)
                    index += 1
                if index != count or seen != set(subset):
                    raise ValueError("Source frame count or selected-frame coverage changed")
                sources[video_id] = {"path": str(source), "sha256": digest, "decoded_frames": count,
                                     "width": width, "height": height, "fps": fps,
                                     "exported_frames": len(subset), "split": split}
            finally:
                capture.release()
            print(json.dumps({"exported_video": video_id, "frames": len(subset), "split": split}), flush=True)
        validate_partition(exports)
        if sha256(zip_path) != source_hash or any(sha256(v["path"]) != v["sha256"] for v in sources.values()):
            raise ValueError("Source changed during export")
        import yaml
        (staging / "dataset.yaml").write_text(yaml.safe_dump({"path": str(output),
            "train": "images/train", "val": "images/train", "names": {0: "active_keeper"}}))
        manifest = {"schema": "active_keeper_detector_final/v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_zip": str(zip_path), "source_zip_sha256": source_hash,
            "pose_manifest": str(pose_manifest), "pose_manifest_sha256": sha256(pose_manifest),
            "metadata_sha256": {str(p): sha256(p) for p in (inventory_path, matches_path)},
            "class_semantics": "the defending goalkeeper of the current penalty attempt, not generic person",
            "source_pixels_verified": True, "human_reviewed_boxes": True, "independent_blinded_truth": False,
            "all_partitions_are_development": True, "validation_yaml_uses_train_only": True,
            "valid_box_counts": dict(valid_counts), "unlocalizable_excluded": sum(r["bbox_status"] not in VISIBLE for r in frames),
            "unknown_frames_are_negative_examples": False, "evaluation_selection": {
                "existing_pose_reference_keys_included": len(reference_keys),
                "additional_uniform_samples_per_development_clip": per_development_clip,
                "uses_prediction_results": False},
            "export_counts": dict(Counter(r["split"] for r in exports)), "sources": sources,
            "exports": exports}
        save_json(staging / "manifest.json", manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-manifest", type=Path, default=ROOT / "outputs/keeper_reference_import_2026-10-28/manifest.json")
    parser.add_argument("--pose-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--matches-root", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--per-development-clip", type=int, default=32)
    args = parser.parse_args()
    result = prepare(args.zip_path, args.output, args.pose_manifest, args.pose_pack,
                     args.matches_root, args.per_development_clip)
    print(json.dumps(result["export_counts"], indent=2))


if __name__ == "__main__":
    main()
