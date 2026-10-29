"""Bind reviewed ball labels to clean original video pixels for a fixed round."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
import openpyxl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Downloads/IND5003_\u8d44\u6599\u5ba1\u9605_20261028"
DEFAULT_OUTPUT = ROOT / "outputs/external_ball_frames_2026-10-28"
FRAME_SHEET = "\u7403\u8def\u9010\u5e27"
INDEX_SHEET = "\u540e\u4fa7\u7403\u8def\u7d22\u5f15"
SOURCE_SHEET = "\u6765\u6e90\u4e0e\u539f\u8bdd"
CLEAR = {"\u6e05\u6670", "\u53ef\u89c1\uff08\u8fd0\u52a8\u6a21\u7cca\uff09"}
FRAME_HEADERS = [
    "\u5c40\u90e8\u653e\u5927\u56fe", "\u5b8c\u6574\u753b\u9762\u6838\u5bf9\u56fe\uff08\u5de6\u4e0a\u89d2\u5e27\u53f7\uff0c\u53f3\u4e0a\u89d23\u500d\u653e\u5927\uff09",
    "\u7f5a\u7403\u5e8f\u53f7", "\u7403\u5458", "\u955c\u5934\u7f16\u53f7", "\u673a\u4f4d", "\u539f\u7247\u5e27\u53f7", "\u539f\u7247\u65f6\u95f4(\u79d2)",
    "\u955c\u5934\u5185\u5e27\u53f7", "\u7403\u5fc3x(px)", "\u7403\u5fc3y(px)", "\u7403\u6846x1", "\u7403\u6846y1", "\u7403\u6846x2", "\u7403\u6846y2",
    "\u80fd\u5426\u770b\u6e05", "\u5b9a\u4f4d\u65b9\u5f0f", "\u9636\u6bb5", "\u5907\u6ce8",
]
SOURCE_CONFIGS = (
    {"key": "lev", "match_id": "lev_stu_20240817", "folder": "match10_new_delivery",
     "frames": "match_10_\u540e\u4fa7\u673a\u4f4d_\u7403\u8def\u9010\u5e27.xlsx",
     "summary": "\u52d2\u6c83\u5e93\u68eevs\u65af\u56fe\u52a0\u7279_\u70b9\u7403\u5206\u6790\u6c47\u603b.xlsx",
     "archive": "match_10_\u5e26\u5e27\u53f7\u89c6\u9891_13\u6bb5.zip", "width": 1280, "height": 720,
     "split": "train", "expected_rows": 476, "expected_boxes": 320, "first_col": 6, "local_col": 8},
    {"key": "ars", "match_id": "ars_mci_20230806", "folder": "",
     "frames": "match_03_\u963f\u68ee\u7eb3vs\u66fc\u57ce_\u7403\u8def\u9010\u5e27.xlsx",
     "summary": "\u963f\u68ee\u7eb3vs\u66fc\u57ce_\u70b9\u7403\u5206\u6790\u6c47\u603b.xlsx",
     "archive": "match_03_\u5e26\u5e27\u53f7\u89c6\u9891_7\u6bb5.zip", "width": 1920, "height": 1080,
     "split": "validation", "expected_rows": 228, "expected_boxes": 88, "first_col": 7, "local_col": 9},
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def integer(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or value < 0:
        raise ValueError(f"Invalid nonnegative integer {field}: {value!r}")
    return int(value)


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid number {field}: {value!r}")
    return float(value)


def load_sheets(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        return {sheet.title: [list(row) for row in sheet.values] for sheet in book}
    finally:
        book.close()


def normalize_rows(rows, config, workbook_hash):
    if rows[0] != FRAME_HEADERS:
        raise ValueError("Unexpected frame annotation headers")
    result, seen = [], set()
    width, height = config["width"], config["height"]
    for row_index, values in enumerate(rows[1:], 2):
        if not any(value is not None for value in values):
            continue
        if len(values) != len(FRAME_HEADERS):
            raise ValueError(f"Unexpected row width: {row_index}")
        penalty = integer(values[2], "penalty")
        shot = f"{integer(int(str(values[4])), 'shot'):04d}"
        frame = integer(values[6], "original_frame")
        if (shot, frame) in seen:
            raise ValueError(f"Repeated source annotation: {shot}/{frame}")
        seen.add((shot, frame))
        time = number(values[7], "source_time")
        if not math.isclose(time, frame / 25, abs_tol=0.00051):
            raise ValueError(f"Frame/time mismatch: {shot}/{frame}")
        local = integer(values[8], "shot_local_frame")
        center = None
        if any(value is not None for value in values[9:11]):
            center = [number(value, "center") for value in values[9:11]]
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                raise ValueError(f"Center outside original dimensions: {shot}/{frame}")
        box = None
        if any(value is not None for value in values[11:15]):
            if not all(value is not None for value in values[11:15]):
                raise ValueError(f"Incomplete box: {shot}/{frame}")
            box = [number(value, "box") for value in values[11:15]]
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid box geometry: {shot}/{frame}")
            if center is not None and not (x1 <= center[0] <= x2 and y1 <= center[1] <= y2):
                raise ValueError(f"Center outside box: {shot}/{frame}")
        disallowed_note = any(word in str(values[18] or "") for word in ("\u63d2\u503c", "\u4f30\u8ba1"))
        eligible = box is not None and values[15] in CLEAR and not disallowed_note
        event_id = f"{config['match_id']}_p{penalty:02d}"
        video_id = f"{event_id}_s{shot}"
        annotation_id = f"{video_id}_f{frame:06d}"
        result.append({"annotation_id": annotation_id, "match_id": config["match_id"], "event_id": event_id,
                       "video_id": video_id, "penalty": penalty, "shot": shot, "original_frame_zero_based": frame,
                       "source_time_seconds": time, "full_shot_local_frame_zero_based": local,
                       "center_original_pixels": center, "box_original_pixels": box, "visibility": values[15],
                       "method": values[16], "phase": values[17], "note": values[18], "eligible_box": eligible,
                       "split": config["split"], "used_as_negative": False,
                       "source": {"workbook_sha256": workbook_hash, "sheet": FRAME_SHEET,
                                  "row_1_based": row_index, "raw_values": values,
                                  "raw_row_sha256": hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()}})
    return result


def shot_mappings(summary_sheets, config, annotations):
    selected = {(row["penalty"], row["shot"]) for row in annotations}
    mappings = {}
    for values in summary_sheets[INDEX_SHEET][7:]:
        if not values or not isinstance(values[0], (int, float)):
            continue
        penalty = integer(values[0], "index_penalty")
        shot = str(values[2]).zfill(4)
        if (penalty, shot) not in selected:
            continue
        first = integer(values[config["first_col"]], "effective_first")
        last = integer(values[config["first_col"] + 1], "effective_last")
        local_first = integer(values[config["local_col"]], "effective_local_first")
        if last < first or (penalty, shot) in mappings:
            raise ValueError("Invalid or duplicate shot index")
        mappings[penalty, shot] = {"penalty": penalty, "shot": shot, "effective_first_original_frame": first,
                                    "effective_last_original_frame": last, "full_shot_first_original_frame": first - local_first}
    if set(mappings) != selected:
        raise ValueError("Missing selected shot mapping")
    for row in annotations:
        mapping = mappings[row["penalty"], row["shot"]]
        frame = row["original_frame_zero_based"]
        if not mapping["effective_first_original_frame"] <= frame <= mapping["effective_last_original_frame"]:
            raise ValueError("Annotation outside indexed shot")
        if frame - mapping["full_shot_first_original_frame"] != row["full_shot_local_frame_zero_based"]:
            raise ValueError("Full-shot frame offset does not match workbook")
        row["review_frame_zero_based"] = frame - mapping["effective_first_original_frame"]
    return mappings


def extract_review_archive(archive_path, directory, expected):
    result = {}
    directory.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            name = entry.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
                raise ValueError(f"Unsafe archive entry: {name}")
            if entry.is_dir() or not name.lower().endswith(".mp4"):
                continue
            match = re.search(r"match_(?:10|03)_P(\d+)_(\d{4})_", path.name)
            if not match:
                raise ValueError(f"Unknown MP4 filename: {name}")
            key = (int(match[1]), match[2])
            if key not in expected or key in result:
                raise ValueError(f"Unexpected or repeated review shot: {key}")
            destination = directory / f"p{key[0]:02d}_s{key[1]}.mp4"
            with archive.open(entry) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            result[key] = destination
    if set(result) != set(expected):
        raise ValueError("Review archive does not contain every selected shot")
    return result


def video_metadata(path, config):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode {path}")
        width, height = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (width, height) != (config["width"], config["height"]) or not math.isclose(fps, 25, abs_tol=1e-6) or frames <= 0:
            raise ValueError(f"Unexpected video geometry/rate: {path}: {width}x{height}/{fps}/{frames}")
        return {"path": str(path), "sha256": sha256(path), "width": width, "height": height, "fps": fps,
                "reported_frame_count": frames}
    finally:
        cap.release()


def read_frame(cap, frame_index):
    if frame_index < 0:
        raise ValueError("Negative frame index")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = cap.read()
    if not ok:
        raise ValueError(f"Cannot decode frame {frame_index}")
    return image


def scene_similarity(clean, review):
    """Compare scene pixels using fixed masks, never ball-label coordinates."""
    if clean.shape != review.shape:
        raise ValueError("Clean/review pixel dimensions differ")
    height, width = clean.shape[:2]
    scale = 480 / width
    dimensions = (480, round(height * scale))
    left = cv2.resize(clean, dimensions, interpolation=cv2.INTER_AREA)
    right = cv2.resize(review, dimensions, interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(right)
    overlay = ((g > 170) & (r < 120) & (b < 170)) | ((r > 180) & (g > 80) & (g < 225) & (b < 100)) | ((r > 190) & (g < 100) & (b < 100))
    overlay = cv2.dilate(overlay.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)).astype(bool)
    mask = ~overlay
    mask[:math.ceil(dimensions[1] * 0.25)] = False
    valid_fraction = float(mask.mean())
    if valid_fraction < 0.25:
        raise ValueError("Insufficient unobscured scene area for frame alignment")
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean(axis=2)[mask]
    cutoff = np.quantile(difference, 0.90)
    trimmed = difference[difference <= cutoff]
    return {"scene_mae": float(difference.mean()), "trimmed_scene_mae": float(trimmed.mean()),
            "scene_p90": float(cutoff), "valid_fraction": valid_fraction,
            "mask_policy": "fixed_top_quarter_and_saturated_annotation_colors_no_label_coordinates"}


def require_alignment(metrics, context):
    if metrics["scene_mae"] > 12 or metrics["trimmed_scene_mae"] > 9 or metrics["scene_p90"] > 30:
        raise ValueError(f"Clean/review source-frame alignment rejected ({context}): {metrics}")


def choose_offset(clean_cap, review_cap, mapping, review_frames, maximum_offset):
    samples = sorted({0, review_frames // 2, review_frames - 1})
    comparisons = []
    review_images = {index: read_frame(review_cap, index) for index in samples}
    for offset in range(-maximum_offset, maximum_offset + 1):
        qualities = []
        for index in samples:
            source_frame = mapping["effective_first_original_frame"] + index + offset
            try:
                qualities.append(scene_similarity(read_frame(clean_cap, source_frame), review_images[index]))
            except ValueError:
                qualities = []
                break
        if qualities:
            comparisons.append({"offset": offset, "mean_trimmed_scene_mae": float(np.mean([x["trimmed_scene_mae"] for x in qualities])),
                                "samples": [{"review_frame": index, **quality} for index, quality in zip(samples, qualities)]})
    if not comparisons:
        raise ValueError("No valid source-frame alignment candidates")
    best = min(comparisons, key=lambda value: (value["mean_trimmed_scene_mae"], abs(value["offset"])))
    for quality in best["samples"]:
        require_alignment(quality, f"shot {mapping['shot']} sample {quality['review_frame']}")
    return {"selected_frame_offset": best["offset"], "candidate_offsets": comparisons,
            "sample_review_frames": samples, "selected_mean_trimmed_scene_mae": best["mean_trimmed_scene_mae"]}


def yolo_box(box, width, height):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]


def save_comparison(path, clean, review, text):
    width = 640
    height = round(clean.shape[0] * width / clean.shape[1])
    canvas = np.zeros((height + 40, 2 * width, 3), dtype=np.uint8)
    canvas[40:, :width] = cv2.resize(clean, (width, height))
    canvas[40:, width:] = cv2.resize(review, (width, height))
    cv2.putText(canvas, text[:160], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise ValueError("Cannot write alignment comparison")


def export_round(source_root, output, maximum_offset=3):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    annotations, exports, source_records, alignments, pixel_groups = [], [], [], [], {}
    protected = {}
    try:
        for config in SOURCE_CONFIGS:
            folder = source_root / config["folder"]
            frames_path, summary_path, archive_path = [folder / config[name] for name in ("frames", "summary", "archive")]
            clean_path = source_root / "clean_sources" / f"{config['match_id']}.mp4"
            for path in (frames_path, summary_path, archive_path, clean_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Required original source missing: {path}")
                protected[path] = sha256(path)
            sheets = load_sheets(frames_path)
            summary = load_sheets(summary_path)
            rows = normalize_rows(sheets[FRAME_SHEET], config, protected[frames_path])
            if len(rows) != config["expected_rows"] or sum(row["eligible_box"] for row in rows) != config["expected_boxes"]:
                raise ValueError(f"Unexpected annotation/strict-box counts: {config['key']}")
            mappings = shot_mappings(summary, config, rows)
            review_paths = extract_review_archive(archive_path, staging / "review_alignment_sources" / config["key"], mappings)
            clean_metadata = video_metadata(clean_path, config)
            declared_source = {str(row[0]): row[1] for row in summary[SOURCE_SHEET] if len(row) > 1 and row[0] is not None and row[1] is not None}
            source_records.append({"match_id": config["match_id"], "split": config["split"], "clean_video": clean_metadata,
                                   "annotation_workbook": str(frames_path), "annotation_workbook_sha256": protected[frames_path],
                                   "summary_workbook_sha256": protected[summary_path], "review_archive_sha256": protected[archive_path],
                                   "declared_original_metadata": declared_source, "coordinate_transform": "identity_original_pixels",
                                   "used_for_training": config["split"] == "train", "used_for_checkpoint_selection": False})
            clean_cap = cv2.VideoCapture(str(clean_path))
            try:
                for key, mapping in sorted(mappings.items()):
                    penalty, shot = key
                    selected = [row for row in rows if (row["penalty"], row["shot"]) == key]
                    review_metadata = video_metadata(review_paths[key], config)
                    expected_count = mapping["effective_last_original_frame"] - mapping["effective_first_original_frame"] + 1
                    if review_metadata["reported_frame_count"] != expected_count:
                        raise ValueError(f"Unexpected review duration: {key}: {review_metadata['reported_frame_count']} != {expected_count}")
                    review_cap = cv2.VideoCapture(str(review_paths[key]))
                    try:
                        alignment = choose_offset(clean_cap, review_cap, mapping, expected_count, maximum_offset)
                        offset = alignment["selected_frame_offset"]
                        checked = []
                        indices = sorted(set(alignment["sample_review_frames"]) | {row["review_frame_zero_based"] for row in selected if row["eligible_box"]})
                        images_dir, labels_dir = staging / "images" / config["split"], staging / "labels" / config["split"]
                        images_dir.mkdir(parents=True, exist_ok=True)
                        labels_dir.mkdir(parents=True, exist_ok=True)
                        comparison_dir = staging / "alignment_samples"
                        comparison_dir.mkdir(exist_ok=True)
                        by_index = {row["review_frame_zero_based"]: row for row in selected}
                        for index in indices:
                            decoded_frame = mapping["effective_first_original_frame"] + index + offset
                            clean = read_frame(clean_cap, decoded_frame)
                            review = read_frame(review_cap, index)
                            quality = scene_similarity(clean, review)
                            require_alignment(quality, f"{config['key']} {shot} frame {index}")
                            checked.append({"review_frame_zero_based": index, "clean_frame_zero_based": decoded_frame, **quality})
                            if index in alignment["sample_review_frames"]:
                                sample_path = comparison_dir / f"{config['match_id']}_p{penalty:02d}_s{shot}_r{index:05d}.jpg"
                                save_comparison(sample_path, clean, review, f"CLEAN | REVIEW {config['match_id']} shot {shot} source {decoded_frame} offset {offset}")
                            row = by_index.get(index)
                            if row is None or not row["eligible_box"]:
                                continue
                            image_path, label_path = images_dir / f"{row['annotation_id']}.jpg", labels_dir / f"{row['annotation_id']}.txt"
                            if not cv2.imwrite(str(image_path), clean, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                                raise ValueError("Cannot export original frame")
                            normalized = yolo_box(row["box_original_pixels"], config["width"], config["height"])
                            label_path.write_text("0 " + " ".join(f"{value:.9f}" for value in normalized) + "\n", encoding="ascii")
                            pixel_hash = hashlib.sha256(clean.tobytes()).hexdigest()
                            primary = pixel_groups.setdefault(pixel_hash, row["annotation_id"])
                            row.update({"clean_frame_zero_based": decoded_frame, "selected_frame_offset": offset,
                                        "frame_alignment": quality, "exported_image": str(image_path.relative_to(staging)),
                                        "exported_label": str(label_path.relative_to(staging)), "raw_pixel_sha256": pixel_hash})
                            exports.append({"image": str(image_path.relative_to(staging)), "label": str(label_path.relative_to(staging)),
                                            "image_sha256": sha256(image_path), "label_sha256": sha256(label_path),
                                            **{field: row[field] for field in ("annotation_id", "match_id", "event_id", "video_id", "split")},
                                            "width": config["width"], "height": config["height"], "raw_pixel_sha256": pixel_hash,
                                            "deduplicated_primary": primary == row["annotation_id"], "primary_annotation_id": primary,
                                            "clean_source_sha256": clean_metadata["sha256"], "original_frame_zero_based": row["original_frame_zero_based"],
                                            "clean_frame_zero_based": decoded_frame, "selected_frame_offset": offset,
                                            "used_for_checkpoint_selection": False, "evaluation_only": config["split"] != "train"})
                        alignments.append({"match_id": config["match_id"], **mapping, **alignment, "review_video_sha256": review_metadata["sha256"],
                                           "verified_frames": checked, "verified_export_count": sum(row["eligible_box"] for row in selected)})
                        print(f"Aligned {config['key']} P{penalty} shot {shot}: offset={offset}, verified={len(checked)}, boxes={sum(row['eligible_box'] for row in selected)}", flush=True)
                    finally:
                        review_cap.release()
            finally:
                clean_cap.release()
            annotations.extend(rows)
        hashes = {}
        for record in exports:
            prior_split = hashes.setdefault(record["raw_pixel_sha256"], record["split"])
            if prior_split != record["split"]:
                raise ValueError("Identical pixels crossed train/evaluation separation")
        if any(sha256(path) != digest for path, digest in protected.items()):
            raise ValueError("Source file changed during export")
        write_json(staging / "annotations.json", {"annotations": annotations, "source_workbooks_unmodified": True,
                                                   "all_unlocalized_rows_retained": True, "unlocalized_rows_are_negatives": False})
        write_json(staging / "alignment.json", {"shots": alignments, "source_frame_indexing": "zero_based", "coordinate_transform": "identity",
                                                "alignment_uses_ball_label_coordinates": False, "training_pixels_include_review_overlays": False})
        summary = {"annotation_count": len(annotations), "strict_box_count": len(exports), "export_counts": dict(Counter(row["split"] for row in exports)),
                   "distinct_events": dict(Counter(split for _, split in {(row["event_id"], row["split"]) for row in annotations})),
                   "duplicate_pixel_exports": sum(not row["deduplicated_primary"] for row in exports),
                   "all_source_hashes_unchanged": True}
        write_json(staging / "summary.json", summary)
        write_json(staging / "manifest.json", {"schema": "external_ball_frames_round1/v1", "created_at": datetime.now(timezone.utc).isoformat(),
                                              "development_only": True, "independent_ground_truth": False,
                                              "clean_source_pixels_verified": True,
                                              "provenance": "human_reviewed_auto_refined_candidates_not_independent_ground_truth",
                                              "selection_uses_evaluation": False, "training_source": "clean_original_video_pixels",
                                              "limitations": ["Labels include human-reviewed automatic refinement; not independent ground truth.",
                                                              "ARS is held out for final fixed-epoch evaluation and must not select checkpoints.",
                                                              "Frame agreement verifies source binding, not ball-label correctness.",
                                                              "Kicking-foot, run-up and outcome narratives are not used by this ball-box export."],
                                              "sources": source_records, "summary": summary, "exports": exports})
        # Review material is evidence only and is never a dataset image directory.
        shutil.rmtree(staging / "review_alignment_sources")
        staging.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alignment-max-offset", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.alignment_max_offset <= 50:
        parser.error("alignment-max-offset must be between 0 and 50 frames")
    print(json.dumps(export_round(args.source_root, args.output_dir, args.alignment_max_offset), indent=2))


if __name__ == "__main__":
    main()
