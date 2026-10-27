"""One fixed height augmentation experiment on original source-only old36 videos.

The remote extraction manifest contains no targets. The released World Cup
evaluation is never accessed, and the Italian match remains development only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_external_ball_classifier as visual

ALLOWED_GROUPS = {"col_eng", "cro_den", "esp_rus", "rus_cro"}
CONFLICT_ID = "esp_rus_p05_koke_esp8"
ENCODER_SHA = "30d1d3539b65ebf747cfe6fc380570604b1634786b06c36cec697688e5f18ca1"


def validate_sources(data):
    if data.get("labels_present") is not False or len(data["samples"]) != 36:
        raise ValueError("Expected label-free fixed old36 source manifest")
    ids, hashes = set(), set()
    allowed = {"clip_id", "video_path", "video_sha256", "pose_path", "start_frame",
               "end_frame_exclusive", "fps", "source_frame_count", "window_method",
               "window_scan", "selected_shot_index", "labels_used"}
    for row in data["samples"]:
        name = row["clip_id"]
        if set(row) - allowed or row["labels_used"] is not False:
            raise ValueError("Labels or unknown fields in source-only manifest")
        if "_".join(name.split("_")[:2]) not in ALLOWED_GROUPS or name == CONFLICT_ID:
            raise ValueError("Unexpected or conflicting source event")
        if name in ids or row["video_sha256"] in hashes:
            raise ValueError("Duplicate source event or source video")
        ids.add(name)
        hashes.add(row["video_sha256"])
        if not (0 <= row["start_frame"] < row["end_frame_exclusive"] <= row["source_frame_count"]):
            raise ValueError("Invalid automatic shooting window")
        if row["fps"] <= 0 or row["window_method"] != "first_viable":
            raise ValueError("Invalid source-only window producer")
    if data["settings"]["frames"] != 64 or data["settings"]["max_shot_seconds"] != 20:
        raise ValueError("Frame density and duration cap must remain frozen")


def window_indices(row, settings):
    start = row["start_frame"]
    end = min(row["end_frame_exclusive"], start + int(settings["max_shot_seconds"] * row["fps"]))
    return [start + i for i in visual.sample_indices(end - start, settings["frames"])]


def extract(args):
    data = visual.read_json(args.inputs)
    validate_sources(data)
    if visual.digest(args.backbone) != ENCODER_SHA:
        raise ValueError("Frozen encoder checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    encoder = visual.load_encoder(args.backbone, device=args.device, threads=2)
    temporal_rows, first_rows, records = [], [], []
    started = time.perf_counter()
    for row in data["samples"]:
        item_started = time.perf_counter()
        video, pose_path = Path(row["video_path"]), Path(row["pose_path"])
        if visual.digest(video) != row["video_sha256"]:
            raise ValueError("Original video SHA does not match human reference source")
        pose = visual.read_json(pose_path)
        if (pose.get("reference_labels_used_for_prediction") is not False or
                pose["source_mapping"]["clip_sha256"] != row["video_sha256"] or
                pose["shot_start_frame"] != row["start_frame"] or
                pose["shot_end_frame"] != row["end_frame_exclusive"] or
                pose["shot_selection"]["method"] != "first_viable" or
                not pose["shot_selection"]["found_qualified_shot"]):
            raise ValueError("Source-only pose window binding changed")
        ids = window_indices(row, data["settings"])
        wanted, frames = set(ids), {}
        capture = cv2.VideoCapture(str(video))
        try:
            if abs(capture.get(cv2.CAP_PROP_FPS) - row["fps"]) > .01:
                raise ValueError("Source video fps differs from cached window")
            for index in range(max(ids) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Source video ended at frame {index}")
                if index in wanted:
                    frame = cv2.resize(frame, (data["settings"]["width"], data["settings"]["height"]),
                                       interpolation=cv2.INTER_AREA)
                    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise ValueError("JPEG normalization failed")
                    frames[index] = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        finally:
            capture.release()
        temporal, first, _ = visual.encode_frames([frames[i] for i in ids], encoder,
                                                   data["settings"], device=args.device)
        temporal_rows.append(temporal)
        first_rows.append(first)
        record = {**row, "sample_frame_ids": ids, "pose_file_sha256": visual.digest(pose_path),
                  "actual_video_sha256": row["video_sha256"],
                  "extraction_seconds": time.perf_counter() - item_started}
        records.append(record)
        np.savez_compressed(args.output / (row["clip_id"] + ".npz"),
                            temporal=temporal, first_frame=first, sample_frame_ids=ids)
        print(f"{len(records):02d}/36 {row['clip_id']} {record['extraction_seconds']:.2f}s", flush=True)
    np.savez_compressed(args.output / "features.npz", temporal=np.stack(temporal_rows),
                        first_frame=np.stack(first_rows))
    visual.save_json(args.output / "features.json", {"schema_version": 1, "samples": records,
        "settings": data["settings"], "labels_used": False, "heldout_accessed": False,
        "source_manifest_sha256": visual.digest(args.inputs),
        "encoder_sha256": ENCODER_SHA, "script_sha256": visual.digest(__file__),
        "visual_helper_sha256": visual.digest(visual.__file__), "device": args.device,
        "elapsed_seconds": time.perf_counter() - started,
        "features_sha256": visual.digest(args.output / "features.npz")})


def source_targets(sources, reference):
    by_id = {row["clip_id"]: row for row in reference["events"]}
    targets = []
    for row in sources:
        ref = by_id[row["clip_id"]]
        provenance = ref.get("_shot_save_annotation", {}).get("field_provenance", {}).get("shot_height", {})
        if ref["source_sha256"] != row["video_sha256"]:
            raise ValueError("Human source hash mismatch")
        if provenance.get("human_verified") is not True or provenance.get("status") != "human_annotated":
            raise ValueError("Missing explicit human height provenance")
        height = {"low": "low", "mid": "middle", "middle": "middle", "high": "high"}.get(ref["shot_height"])
        if height is None or row["clip_id"] == CONFLICT_ID:
            raise ValueError("Unresolved human height cannot train")
        targets.append(height)
    return targets


def train(args):
    from scripts import train_external_ball_height_augmented as height
    from scripts import train_external_ball_v2 as experiment
    protocol = visual.read_json(args.protocol)
    if protocol["train_count"] != 44 or protocol["worldcup_access_or_new_evaluation"] is not False:
        raise ValueError("Unexpected frozen development protocol")
    sources = visual.read_json(args.inputs)
    validate_sources(sources)
    cache = visual.read_json(args.old_features / "features.json")
    if (cache["source_manifest_sha256"] != visual.digest(args.inputs) or
            cache["features_sha256"] != visual.digest(args.old_features / "features.npz") or
            cache["encoder_sha256"] != ENCODER_SHA or cache["labels_used"] is not False or
            cache["heldout_accessed"] is not False):
        raise ValueError("Source embedding provenance mismatch")
    if [r["clip_id"] for r in cache["samples"]] != [r["clip_id"] for r in sources["samples"]]:
        raise ValueError("Source embedding row order mismatch")
    frames = visual.read_json(args.external_frames / "frames.json")
    experiment.validate_frames_manifest(frames)
    prior = visual.read_json(args.external_model / "report.json")
    if (prior["frames_manifest_sha256"] != visual.digest(args.external_frames / "frames.json") or
            prior["portable_encoder_sha256"] != ENCODER_SHA):
        raise ValueError("External embedding source binding mismatch")
    old_labels = source_targets(sources["samples"], visual.read_json(args.references))
    if Counter(old_labels) != {"low": 17, "middle": 12, "high": 7}:
        raise ValueError("Unexpected old36 training cohort")
    external = frames["samples"]
    if set(r["video_sha256"] for r in external) & set(r["video_sha256"] for r in sources["samples"]):
        raise ValueError("Duplicate event across cohorts")
    features = []
    for path in [args.external_model / "feature_cache.npz", args.old_features / "features.npz"]:
        with np.load(path, allow_pickle=False) as data:
            temporal, first = data["temporal"].astype(np.float64), data["first_frame"].astype(np.float64)
        features.append(temporal - np.tile(first, (1, 4)))
    features = np.concatenate(features)
    if features.shape != (51, 8192) or not np.isfinite(features).all():
        raise ValueError("Unexpected or invalid frozen embeddings")
    targets = [row["targets"]["ball_height_category"] for row in external] + old_labels
    train_ids = [i for i, row in enumerate(external) if row["split"] == "train"] + list(range(15, 51))
    val_ids = [i for i, row in enumerate(external) if row["split"] == "validation"]
    if len(train_ids) != 44 or len(val_ids) != 7:
        raise ValueError("Unexpected training or development counts")
    args.output.mkdir(parents=True, exist_ok=False)
    fitted = height.fit_height(features, train_ids, [targets[i] for i in train_ids])
    weights = height.export_height(*fitted)
    predictions = height.predict_height(features, weights)
    np.savez_compressed(args.output / "height_weights.npz", **weights)
    np.savez_compressed(args.output / "features.npz", first_frame_difference=features)
    true = [targets[i] for i in val_ids]
    pred = [predictions["labels"][i] for i in val_ids]
    majority = Counter(targets[i] for i in train_ids).most_common(1)[0][0]
    baseline_report = visual.read_json(args.core_model / "report.json")
    if baseline_report["event_ids"][:15] != [r["event_id"] for r in external]:
        raise ValueError("Baseline validation event order mismatch")
    baseline_pred = [baseline_report["predictions"]["labels"][i] for i in val_ids]
    german_weights_path = args.german_model / "first_frame_difference_weights.npz"
    with np.load(german_weights_path, allow_pickle=False) as old_weights:
        german_pred = visual.predict_parameters(features[val_ids], old_weights)["ball_height_category"]["labels"]
    report = {"experiment": protocol["experiment"], "phase": protocol["phase"],
        "protocol_sha256": visual.digest(args.protocol), "train_events": 44, "validation_events": 7,
        "training_class_counts": dict(Counter(targets[i] for i in train_ids)),
        "classes": height.CLASSES, "train_indices": train_ids, "validation_indices": val_ids,
        "event_ids": [r["event_id"] for r in external] + [r["clip_id"] for r in sources["samples"]],
        "reference_labels": targets, "predictions": predictions,
        "validation_rows": [{"event_id": external[i]["event_id"], "reference": targets[i],
            "new_prediction": predictions["labels"][i], "core8_baseline": baseline_report["predictions"]["labels"][i],
            "german8_baseline": german_pred[j]} for j, i in enumerate(val_ids)],
        "scores": {"old36_augmented": visual.metrics(true, pred, height.CLASSES),
            "core8_augmented_baseline": visual.metrics(true, baseline_pred, height.CLASSES),
            "german8_baseline": visual.metrics(true, german_pred, height.CLASSES),
            "majority": visual.metrics(true, [majority] * 7, height.CLASSES)},
        "backbone_trained": False, "head_trained": True, "preprocessing_train_only": True,
        "logistic_C": .1, "pca_dimensions": int(fitted[1].n_components_),
        "settings": frames["settings"], "old_settings": cache["settings"],
        "old_windows": cache["samples"], "old_features_sha256": cache["features_sha256"],
        "source_manifest_sha256": visual.digest(args.inputs), "references_sha256": visual.digest(args.references),
        "external_frames_manifest_sha256": visual.digest(args.external_frames / "frames.json"),
        "external_features_sha256": visual.digest(args.external_model / "feature_cache.npz"),
        "portable_encoder_sha256": ENCODER_SHA, "weights_sha256": visual.digest(args.output / "height_weights.npz"),
        "script_sha256": visual.digest(__file__), "height_helper_sha256": visual.digest(height.__file__),
        "visual_helper_sha256": visual.digest(visual.__file__), "held_out_test_evaluated": False,
        "worldcup_accessed": False, "promoted_to_default_pipeline": False,
        "new_independent_test_required": True, "limitations": protocol["limitations"]}
    visual.save_json(args.output / "report.json", report)
    shutil.copyfile(__file__, args.output / "training_source.py")
    print(report["scores"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--backbone", type=Path, required=True)
    extraction.add_argument("--device", default="cuda:0")
    extraction.set_defaults(run=extract)
    fitting = commands.add_parser("train")
    for name in ("protocol", "old-features", "external-frames", "external-model", "references", "core-model", "german-model"):
        fitting.add_argument("--" + name, type=Path, required=True)
    fitting.set_defaults(run=train)
    for command in (extraction, fitting):
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
