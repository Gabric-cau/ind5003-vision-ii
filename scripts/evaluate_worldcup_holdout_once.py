"""Freeze, run and score a single model-held-out World Cup evaluation.

Existing classifiers and feature extraction are imported unchanged. Prediction
loads a label-free input manifest; references are opened only after raw results
have been persisted and hashed. This evaluator never updates model parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_external_ball_classifier as ball
from scripts import train_external_ball_height_augmented as height
from scripts import train_external_camera_classifier as camera

SOURCE_SHA256 = "ae88061bca8e3a269b51b592447a462de5dfa59759e4c6b6e46eec69694816f6"
HEIGHT_LABELS = {"\u4f4e": "low", "\u4e2d": "middle", "\u9ad8": "high"}
DIRECTION_LABELS = {"\u5de6": "left", "\u4e2d": "center", "\u53f3": "right"}
MODEL_FOLDERS = {
    "ball_v2": ROOT / "outputs/external_ball_training_2026-10-26/model_v2",
    "height_v2": ROOT / "outputs/external_ball_training_2026-10-26/height_core_augmented_v2",
    "camera_v1": ROOT / "outputs/external_camera_training_2026-10-26/model_v1_final",
    "ball_v1_control": ROOT / "outputs/external_ball_training_2026-10-25/model_v1",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def camera_references(metadata, selected_indices):
    admitted, excluded = [], []
    for row in metadata["shot_timeline"]:
        if row["export_index"] not in selected_indices:
            continue
        label = row.get("camera_label")
        if label is None:
            excluded.append({"shot_id": row["original_shot_id"], "reason": "unlabelled_not_negative"})
            continue
        if row.get("camera_annotation_source") != "human_user_observation_in_chat":
            raise ValueError("Unexpected nonhuman camera reference")
        if label not in camera.LABEL_MAP:
            excluded.append({"shot_id": row["original_shot_id"], "raw_label": label,
                             "reason": "not_in_frozen_training_label_map"})
            continue
        start, end = row.get("camera_label_start_seconds"), row.get("camera_label_end_seconds_exclusive")
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, (int, float)) or not isinstance(end, (int, float))):
            excluded.append({"shot_id": row["original_shot_id"], "raw_label": label,
                             "reason": "partial_annotation_without_exact_interval"})
            continue
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError("Invalid human camera interval")
        admitted.append({"filename": Path(row["video_file"]).name,
            "shot_id": row["original_shot_id"], "class": camera.LABEL_MAP[label],
            "start_seconds": float(start), "end_seconds_exclusive": float(end),
            "raw_human_label": label, "label_scope": row["camera_label_scope"],
            "annotation_source": row["camera_annotation_source"]})
    return admitted, excluded


def assert_model_bindings(models):
    reports = {name: ball.read_json(Path(item["report_path"])) for name, item in models.items()}
    encoders = {report["portable_encoder_sha256"] for report in reports.values()}
    if len(encoders) != 1:
        raise ValueError("Selected model encoders differ")
    for name, item in models.items():
        if ball.digest(item["report_path"]) != item["report_sha256"]:
            raise ValueError(f"Frozen model report changed: {name}")
        if ball.digest(item["weights_path"]) != item["weights_sha256"]:
            raise ValueError(f"Frozen model weights changed: {name}")
    return reports


def prepare(args):
    metadata_path = args.release / "metadata/annotations.json"
    if ball.digest(metadata_path) != SOURCE_SHA256:
        raise ValueError("Source annotation bytes differ from the audited release")
    metadata = ball.read_json(metadata_path)
    clips = [row for row in metadata["clips"] if row["dataset_id"] == "world_cup_2022"]
    if [row["export_index"] for row in clips] != list(range(17, 25)):
        raise ValueError("Expected precisely the eight frozen World Cup events")
    if any(row["human_features"].get("direction_reference_frame") is not None for row in clips):
        raise ValueError("Direction-reference version changed; do not silently change scoring")
    if any(row["human_features"].get("provenance") != "human_user_observation_in_chat" for row in clips):
        raise ValueError("Unexpected event-label provenance")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, event_refs, extracted = [], [], []
    with zipfile.ZipFile(args.archive) as archive:
        for row in clips:
            path = (args.release / row["file"]).resolve()
            if not path.is_relative_to(args.release.resolve()):
                raise ValueError("Unsafe release path")
            if not path.exists():
                member = args.release.name + "/" + row["file"]
                info = archive.getinfo(member)
                if info.file_size != row["media_check"]["bytes"]:
                    raise ValueError("Archive member size mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                extracted.append(row["filename"])
            actual_hash = ball.digest(path)
            if actual_hash != row["media_check"]["sha256"]:
                raise ValueError(f"Video checksum mismatch: {path.name}")
            event_id = row["dataset_id"] + ":" + row["original_event_id"]
            inputs.append({"event_id": event_id, "filename": row["filename"], "video_path": str(path),
                           "video_sha256": actual_hash, "duration_seconds": row["duration_seconds"],
                           "fps": row["fps"], "source_frame_count": row["expected_video_frames"]})
            labels = row["human_features"]
            event_refs.append({"event_id": event_id, "filename": row["filename"], "video_sha256": actual_hash,
                "ball_height_category": HEIGHT_LABELS[labels["ball_height_label"]],
                "ball_direction_unresolved": DIRECTION_LABELS[labels["ball_direction_label"]],
                "direction_reference_frame": None, "annotation_source": labels["provenance"],
                "raw_user_observations": labels["raw_user_observations"]})
            print("verified", row["filename"], actual_hash, flush=True)
    camera_refs, camera_excluded = camera_references(metadata, set(range(17, 25)))
    if len(camera_refs) != 15 or Counter(row["class"] for row in camera_refs) != {"behind_kicker": 8, "goal_area": 7}:
        raise ValueError("Frozen strict camera reference set changed")
    models = {}
    for name, folder in MODEL_FOLDERS.items():
        report_path = folder / "report.json"
        report = ball.read_json(report_path)
        filename = {"ball_v2": "first_frame_difference_weights.npz", "height_v2": "height_weights.npz",
                    "camera_v1": "camera_weights.npz", "ball_v1_control": "classifier_weights.npz"}[name]
        expected = (report["candidates"]["first_frame_difference"]["weights_sha256"]
                    if name == "ball_v2" else report["weights_sha256"])
        models[name] = {"report_path": str(report_path), "report_sha256": ball.digest(report_path),
                       "weights_path": str(folder / filename), "weights_sha256": expected}
    reports = assert_model_bindings(models)
    backbone = MODEL_FOLDERS["ball_v1_control"] / "encoder.pt"
    if ball.digest(backbone) != reports["ball_v2"]["portable_encoder_sha256"]:
        raise ValueError("Portable encoder mismatch")
    development = ball.read_json(ROOT / "outputs/external_ball_training_2026-10-25/manifest.json")["samples"]
    camera_train = reports["camera_v1"]["samples"]
    core_train = reports["height_v2"]["admitted_cores"]
    training_hashes = {row["video_sha256"] for row in development + camera_train + core_train}
    training_hashes |= {row["old_reference_video_sha256"] for row in core_train}
    training_events = {row["event_id"] for row in development}
    training_events |= {row["match_group"] + ":" + row["event_id"] for row in camera_train}
    training_events |= {row["match_group"] + ":" + row["clip_id"] for row in core_train}
    test_hashes = {row["video_sha256"] for row in inputs}
    test_events = {row["event_id"] for row in inputs}
    overlaps = [(a["filename"], b["filename"]) for i, a in enumerate(clips) for b in clips[i + 1:]
                if max(a["source_start_seconds"], b["source_start_seconds"]) <
                min(a["source_end_seconds_exclusive"], b["source_end_seconds_exclusive"])]
    sources = metadata["sources"]
    test_sources = [row for row in sources if row["dataset_id"] == "world_cup_2022"]
    dev_sources = [row for row in sources if row["dataset_id"] in camera.ALLOWED_MATCHES]
    audit = {"actual_video_hashes_verified": True, "test_unique_video_count": len(test_hashes),
        "training_video_hash_overlap": sorted(test_hashes & training_hashes),
        "canonical_event_overlap": sorted(test_events & training_events),
        "source_recording_hash_overlap": sorted({r["sha256"] for r in test_sources} & {r["sha256"] for r in dev_sources}),
        "source_url_overlap": sorted({r["source_url"] for r in test_sources} & {r["source_url"] for r in dev_sources}),
        "within_test_source_interval_overlaps": overlaps, "independent_test_match_count": 1,
        "scope": "supplied metadata and exact hashes; not independent human event-identity adjudication"}
    if len(test_hashes) != 8 or any(audit[key] for key in (
            "training_video_hash_overlap", "canonical_event_overlap", "source_recording_hash_overlap",
            "source_url_overlap", "within_test_source_interval_overlaps")):
        raise ValueError("Holdout independence audit failed")
    ball.save_json(args.output / "inputs.json", {"samples": inputs, "human_labels_in_inference_input": False})
    ball.save_json(args.output / "references.json", {"events": event_refs, "camera_intervals": camera_refs,
        "excluded_camera_records": camera_excluded, "source_annotation_sha256": SOURCE_SHA256})
    code_paths = [ROOT / "scripts/train_external_ball_classifier.py", ROOT / "scripts/train_external_ball_v2.py",
                  ROOT / "scripts/train_external_ball_height_augmented.py", ROOT / "scripts/train_external_camera_classifier.py",
                  ROOT / "src/step_pipeline.py", Path(__file__).resolve()]
    protocol = {"schema_version": 1, "created_at_utc": utc_now(), "status": "frozen_before_first_model_inference",
        "evaluation_kind": "single_model_heldout_match", "code_sha256": {str(p): ball.digest(p) for p in code_paths},
        "models": models, "encoder_path": str(backbone), "encoder_sha256": ball.digest(backbone),
        "source_annotation_path": str(metadata_path), "source_annotation_sha256": SOURCE_SHA256,
        "source_archive": str(args.archive), "newly_extracted_files": extracted,
        "inputs_sha256": ball.digest(args.output / "inputs.json"), "references_sha256": ball.digest(args.output / "references.json"),
        "overlap_audit": audit, "event_count": 8, "test_video_duration_seconds": sum(r["duration_seconds"] for r in inputs),
        "height_policy": {"classes": height.CLASSES, "eligible_events": 8,
            "primary_metrics": ["accuracy_all_eligible", "output_coverage", "macro_f1_fixed_classes", "confusion_matrix"],
            "controls": ["frozen_v1_height_head", "training_majority_low"],
            "scope": "single-observer ordinal ball height, not physical height"},
        "direction_policy": {"prediction_only": True, "scores": None,
            "reason": "All eight supplied direction coordinate conventions are unknown",
            "no_coordinate_flip_search": True, "future_reference_resolution_must_be_independent_of_predictions": True},
        "camera_policy": {"label_map": camera.LABEL_MAP, "classes": camera.CLASSES,
            "reference_intervals": 15, "reference_seconds": sum(r["end_seconds_exclusive"] - r["start_seconds"] for r in camera_refs),
            "known_reference_classes": ["behind_kicker", "goal_area"], "third_class_generalization_assessable": False,
            "automatic_cut_threshold": 30., "maximum_prediction_window_seconds": 2.,
            "primary": "time-overlap confusion on strictly labelled reference durations",
            "secondary": "duration-majority class per human reference interval; ties become unknown",
            "localization": "same-class interval matching at temporal IoU 0.5 inside annotation mask only",
            "oracle_human_cropped_inference": False, "unlabelled_time_is_negative": False,
            "ground_truth_boundaries_used_in_prediction": False, "boundary_accuracy": "not established by partially labelled class intervals",
            "course_four_class_completion_claim": False, "slow_motion_scored": False,
            "constant_control": "behind_kicker from frozen all-32 training majority, tie resolved by class order"},
        "inference_policy": {"device": "cpu", "threads": 2, "batch_encoder_loads": 1,
            "primary_ball_window": reports["ball_v2"]["settings"],
            "camera_frame_settings": reports["camera_v1"]["settings"],
            "raw_predictions_frozen_before_scoring": True, "fitting_or_tuning_allowed": False,
            "errors_retained_in_all_event_denominator": True, "silent_sample_exclusion": False},
        "runtime_policy": {"batch_total_includes_model_load": True, "per_video_processing_excludes_shared_model_load": True,
            "summaries": ["median", "p95", "real_time_ratio"], "latency_ceiling": None},
        "interpretation": ["One unseen match with eight correlated events is a small external check",
                           "Metadata was audited before inference; this is not label-blind human annotation",
                           "No post-test model repair or model selection will be made from these results"]}
    ball.save_json(args.output / "protocol.json", protocol)
    protocol_sha = ball.digest(args.output / "protocol.json")
    ball.save_json(args.output / "protocol_freeze.json", {"protocol_sha256": protocol_sha, "frozen_at_utc": utc_now()})
    print("FROZEN_PROTOCOL", str(args.output / "protocol.json"), protocol_sha, flush=True)


def load_weights(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def predict_video(sample, encoder, parameters, protocol):
    from src.step_pipeline import _scan_camera_shots
    started = time.perf_counter()
    encoded, ball_window = ball.decode_video(sample["video_path"], protocol["inference_policy"]["primary_ball_window"])
    frames = [cv2.imdecode(frame, cv2.IMREAD_COLOR) for frame in encoded]
    temporal, first, _ = ball.encode_frames(frames, encoder, protocol["inference_policy"]["primary_ball_window"])
    difference = temporal.astype(np.float64) - np.tile(first.astype(np.float64), 4)
    current = ball.predict_parameters(difference, parameters["ball_v2"])
    current["ball_height_category"] = height.predict_height(difference, parameters["height_v2"])
    control = ball.predict_parameters(temporal, parameters["ball_v1_control"])
    ball_seconds = time.perf_counter() - started
    camera_started = time.perf_counter()
    shots = _scan_camera_shots(sample["video_path"], sample["duration_seconds"], 30.)
    windows = camera.prediction_windows(shots, 2.)
    decoded = camera.decode_windows(sample["video_path"], windows, sample["fps"])
    camera_predictions = []
    for window, (images, ids) in zip(windows, decoded):
        features, _, _ = ball.encode_frames(images, encoder, protocol["inference_policy"]["camera_frame_settings"])
        prediction = camera.predict_head(features.reshape(4, -1).mean(axis=0), parameters["camera_v1"])
        camera_predictions.append({**window, "class": prediction["labels"][0],
                                   "probabilities_uncalibrated": prediction["probabilities"][0],
                                   "sample_frame_ids": ids, "slow_motion": "unknown"})
    return {"status": "completed", "filename": sample["filename"], "event_id": sample["event_id"],
        "video_sha256": sample["video_sha256"], "ball": current,
        "frozen_v1_control": control, "ball_automatic_window": ball_window,
        "camera_windows": camera_predictions, "automatic_camera_shots": shots,
        "human_labels_used_in_prediction": False, "manual_windows_used_in_prediction": False,
        "ball_processing_seconds": ball_seconds, "camera_processing_seconds": time.perf_counter() - camera_started,
        "processing_seconds": time.perf_counter() - started,
        "encoder_frame_samples": len(frames) + sum(len(ids) for _, ids in decoded)}


def confusion_metrics(matrix, classes):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(classes), len(classes) + 1):
        raise ValueError("Confusion matrix must include an unknown-prediction column")
    total, correct = float(matrix.sum()), float(np.trace(matrix[:, :len(classes)]))
    covered = float(matrix[:, :len(classes)].sum())
    per_class = []
    for index, label in enumerate(classes):
        tp = matrix[index, index]
        actual, predicted = matrix[index].sum(), matrix[:, index].sum()
        precision = float(tp / predicted) if predicted else 0.
        recall = float(tp / actual) if actual else 0.
        f1 = float(2 * tp / (actual + predicted)) if actual + predicted else 0.
        per_class.append({"class": label, "support": float(actual), "precision": precision, "recall": recall, "f1": f1})
    supported = [row["f1"] for row in per_class if row["support"] > 0]
    return {"reference_weight": total, "correct_weight": correct, "covered_weight": covered,
        "accuracy_all_eligible": correct / total if total else None,
        "output_coverage": covered / total if total else None,
        "accuracy_covered": correct / covered if covered else None,
        "macro_f1_fixed_classes": float(np.mean([row["f1"] for row in per_class])),
        "macro_f1_reference_supported_classes": float(np.mean(supported)) if supported else None,
        "per_class": per_class, "rows": classes, "columns": classes + ["unknown"],
        "confusion_matrix": matrix.tolist()}


def category_metrics(true, predicted, classes):
    matrix = np.zeros((len(classes), len(classes) + 1))
    for actual, output in zip(true, predicted):
        matrix[classes.index(actual), classes.index(output) if output in classes else len(classes)] += 1
    return confusion_metrics(matrix, classes)


def merge_intervals(rows):
    result = []
    for row in sorted(rows, key=lambda item: item["start_seconds"]):
        item = {key: row[key] for key in ("start_seconds", "end_seconds_exclusive", "class")}
        if (result and result[-1]["class"] == item["class"] and
                abs(result[-1]["end_seconds_exclusive"] - item["start_seconds"]) <= 1e-6):
            result[-1]["end_seconds_exclusive"] = item["end_seconds_exclusive"]
        else:
            result.append(item)
    return result


def masked_localization(references, predictions):
    from scipy.optimize import linear_sum_assignment
    counts = {label: {"reference_runs": 0, "prediction_runs": 0, "matched_runs": 0} for label in camera.CLASSES}
    matches = []
    for filename in sorted({row["filename"] for row in references}):
        truth = merge_intervals([row for row in references if row["filename"] == filename])
        clipped = []
        for predicted in predictions.get(filename, []):
            for actual in truth:
                start = max(predicted["start_seconds"], actual["start_seconds"])
                end = min(predicted["end_seconds_exclusive"], actual["end_seconds_exclusive"])
                if end > start:
                    clipped.append({"start_seconds": start, "end_seconds_exclusive": end, "class": predicted["class"]})
        proposed = merge_intervals(clipped)
        for label in camera.CLASSES:
            target = [row for row in truth if row["class"] == label]
            candidate = [row for row in proposed if row["class"] == label]
            counts[label]["reference_runs"] += len(target)
            counts[label]["prediction_runs"] += len(candidate)
            if not target or not candidate:
                continue
            iou = np.zeros((len(target), len(candidate)))
            for i, actual in enumerate(target):
                for j, predicted in enumerate(candidate):
                    overlap = max(0., min(actual["end_seconds_exclusive"], predicted["end_seconds_exclusive"]) -
                                  max(actual["start_seconds"], predicted["start_seconds"]))
                    union = actual["end_seconds_exclusive"] - actual["start_seconds"] + predicted["end_seconds_exclusive"] - predicted["start_seconds"] - overlap
                    iou[i, j] = overlap / union
            # Prefer the maximum number of threshold-qualified one-to-one pairs.
            value = (iou >= .5).astype(float) * (max(len(target), len(candidate)) + 1) + iou
            selected = zip(*linear_sum_assignment(-value))
            for i, j in selected:
                if iou[i, j] >= .5:
                    counts[label]["matched_runs"] += 1
                    matches.append({"filename": filename, "class": label, "temporal_iou": float(iou[i, j])})
    for values in counts.values():
        actual, predicted, matched = values["reference_runs"], values["prediction_runs"], values["matched_runs"]
        values["precision"] = matched / predicted if predicted else 0.
        values["recall"] = matched / actual if actual else 0.
        values["f1"] = 2 * matched / (actual + predicted) if actual + predicted else 0.
    return {"scope": "class runs clipped to human-annotated time only; not full-video camera-cut accuracy",
            "temporal_iou_threshold": .5, "per_class": counts, "matches": matches,
            "boundary_errors": None, "unlabelled_region_false_positives_assessable": False}


def score_camera(references, predictions):
    classes = camera.CLASSES
    matrix = np.zeros((len(classes), len(classes) + 1))
    interval_predictions, truth, details = [], [], []
    for row in references:
        durations = np.zeros(len(classes) + 1)
        for predicted in predictions.get(row["filename"], []):
            overlap = max(0., min(row["end_seconds_exclusive"], predicted["end_seconds_exclusive"]) -
                          max(row["start_seconds"], predicted["start_seconds"]))
            if overlap:
                label = predicted["class"]
                durations[classes.index(label) if label in classes else len(classes)] += overlap
        duration = row["end_seconds_exclusive"] - row["start_seconds"]
        if durations.sum() > duration + 1e-5:
            raise ValueError("Overlapping predictions double-count labelled time")
        durations[-1] += max(0., duration - durations.sum())
        matrix[classes.index(row["class"])] += durations
        winners = np.flatnonzero(np.isclose(durations, durations.max(), atol=1e-9, rtol=0))
        chosen = classes[int(winners[0])] if len(winners) == 1 and winners[0] < len(classes) else "unknown"
        truth.append(row["class"])
        interval_predictions.append(chosen)
        details.append({**row, "automatic_window_majority_class": chosen,
                        "class_duration_seconds": dict(zip(classes + ["unknown"], durations.tolist()))})
    return {"time_weighted": confusion_metrics(matrix, classes),
            "reference_interval_majority": category_metrics(truth, interval_predictions, classes),
            "interval_details": details, "oracle_cropped_inference": False,
            "masked_localization": masked_localization(references, predictions)}


def run(args):
    protocol_path = args.output / "protocol.json"
    freeze = ball.read_json(args.output / "protocol_freeze.json")
    if ball.digest(protocol_path) != args.protocol_sha256 or args.protocol_sha256 != freeze["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch")
    protocol = ball.read_json(protocol_path)
    for path, expected in protocol["code_sha256"].items():
        if ball.digest(path) != expected:
            raise ValueError(f"Frozen code changed before inference: {path}")
    for name in ("inputs", "references"):
        if ball.digest(args.output / (name + ".json")) != protocol[name + "_sha256"]:
            raise ValueError(f"Frozen {name} changed")
    if ball.digest(protocol["encoder_path"]) != protocol["encoder_sha256"]:
        raise ValueError("Frozen encoder changed")
    assert_model_bindings(protocol["models"])
    if (args.output / "run_started.json").exists():
        raise ValueError("This holdout run has already started; automatic reruns are forbidden")
    samples = ball.read_json(args.output / "inputs.json")["samples"]
    for sample in samples:
        if ball.digest(sample["video_path"]) != sample["video_sha256"]:
            raise ValueError("Frozen test input changed")
    ball.save_json(args.output / "run_started.json", {"started_at_utc": utc_now(), "protocol_sha256": args.protocol_sha256})
    raw_folder = args.output / "raw_predictions"
    raw_folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    cv2.setNumThreads(2)
    load_started = time.perf_counter()
    encoder = ball.load_encoder(protocol["encoder_path"], threads=2)
    parameters = {name: load_weights(item["weights_path"]) for name, item in protocol["models"].items()}
    model_load_seconds = time.perf_counter() - load_started
    raw = []
    for sample in samples:
        event_started = time.perf_counter()
        try:
            result = predict_video(sample, encoder, parameters, protocol)
        except Exception as error:
            result = {"filename": sample["filename"], "event_id": sample["event_id"],
                "video_sha256": sample["video_sha256"], "status": "failed",
                "exception_type": type(error).__name__, "error": str(error),
                "processing_seconds": time.perf_counter() - event_started}
        raw.append(result)
        ball.save_json(raw_folder / (sample["filename"] + ".json"), result)
        print("predicted", sample["filename"], result["status"], f"{result['processing_seconds']:.2f}s", flush=True)
    batch_seconds = time.perf_counter() - started
    ball.save_json(args.output / "raw_predictions.json", {"protocol_sha256": args.protocol_sha256, "samples": raw})
    prediction_sha = ball.digest(args.output / "raw_predictions.json")
    ball.save_json(args.output / "predictions_frozen.json", {"frozen_at_utc": utc_now(), "predictions_sha256": prediction_sha,
        "protocol_sha256": args.protocol_sha256, "all_predictions_saved_before_scoring": True})
    # Reference data is loaded only after the immutable raw-prediction snapshot.
    references = ball.read_json(args.output / "references.json")
    by_filename = {row["filename"]: row for row in raw}
    truth, proposed, legacy = [], [], []
    for reference in references["events"]:
        prediction = by_filename[reference["filename"]]
        truth.append(reference["ball_height_category"])
        completed = prediction["status"] == "completed"
        proposed.append(prediction["ball"]["ball_height_category"]["labels"][0] if completed else "unknown")
        legacy.append(prediction["frozen_v1_control"]["ball_height_category"]["labels"][0] if completed else "unknown")
    camera_predictions = {row["filename"]: row.get("camera_windows", []) for row in raw}
    camera_control = {sample["filename"]: [{"start_seconds": 0., "end_seconds_exclusive": sample["duration_seconds"],
                                           "class": "behind_kicker"}] for sample in samples}
    times = [row["processing_seconds"] for row in raw]
    import torch
    report = {"status": "completed_once_no_post_test_model_changes", "completed_at_utc": utc_now(),
        "protocol_path": str(protocol_path), "protocol_sha256": args.protocol_sha256,
        "predictions_sha256": prediction_sha, "test_events": 8, "independent_test_matches": 1,
        "completed_predictions": sum(row["status"] == "completed" for row in raw),
        "ball_height": {"final_candidate": category_metrics(truth, proposed, height.CLASSES),
                        "frozen_v1_control": category_metrics(truth, legacy, height.CLASSES),
                        "training_majority_low": category_metrics(truth, ["low"] * len(truth), height.CLASSES),
                        "per_event": [{"filename": row["filename"], "reference": target, "prediction": prediction,
                                       "v1_prediction": control} for row, target, prediction, control in
                                      zip(references["events"], truth, proposed, legacy)]},
        "ball_direction": {"scores": None, "predictions_available": sum(row["status"] == "completed" for row in raw),
                           "scorable_reference_events": 0, "reason": protocol["direction_policy"]["reason"]},
        "camera": score_camera(references["camera_intervals"], camera_predictions),
        "camera_constant_control": score_camera(references["camera_intervals"], camera_control),
        "runtime": {"device": "cpu", "threads": 2, "python": platform.python_version(), "torch": torch.__version__,
            "opencv": cv2.__version__, "numpy": np.__version__, "model_load_seconds": model_load_seconds,
            "batch_end_to_end_seconds_including_model_load": batch_seconds,
            "per_video_processing_median_seconds": float(np.median(times)),
            "per_video_processing_p95_seconds": float(np.quantile(times, .95)),
            "real_time_ratio_batch": batch_seconds / protocol["test_video_duration_seconds"],
            "test_video_duration_seconds": protocol["test_video_duration_seconds"],
            "encoder_frame_samples": sum(row.get("encoder_frame_samples", 0) for row in raw)},
        "limitations": ["One match and eight correlated events; no broad accuracy guarantee",
                        "Height is a supplied single-observer ordinal reference, not physical height",
                        "Direction convention unknown; direction accuracy is deliberately absent",
                        "Camera metrics cover only 15 labelled intervals; unseen other-side and playback are unassessed",
                        "Automatic prediction windows are not verified camera-shot boundaries",
                        "No model retraining or selection follows these holdout results"]}
    ball.save_json(args.output / "report.json", report)
    print(json.dumps({"height": report["ball_height"]["final_candidate"],
                      "camera_time": report["camera"]["time_weighted"], "runtime": report["runtime"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=prepare)
    q = commands.add_parser("run")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--protocol-sha256", required=True)
    q.set_defaults(run=run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
