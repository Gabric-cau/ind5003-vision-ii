"""Small human-supervised spatial camera classifier, not four-class playback AI.

Only resolved human intervals from the German and Italian matches are admitted.
Unknown camera annotations are not negatives. The World Cup match stays locked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_external_ball_classifier as previous

CLASSES = ["behind_kicker", "goal_area", "other_side"]
ALLOWED_MATCHES = {"german_cup_2022", "coppa_italia_2025"}
SETTINGS = {"frames": 8, "width": 512, "height": 288, "temporal_bins": 4,
            "spatial_pool": [2, 2], "horizontal_flip": False}
LABEL_MAP = {
    "\u70b9\u7403\u8005\u540e\u4fa7\u89c6\u89d2": "behind_kicker",
    "\u70b9\u7403\u8005\u80cc\u540e\u8fdc\u666f": "behind_kicker",
    "\u7403\u95e8\u5185\u53f3\u4fa7\u673a\u4f4d": "goal_area",
    "\u7403\u95e8\u80cc\u540e\u9ad8\u8fdc\u666f\u89c6\u89d2": "goal_area",
    "\u7403\u95e8\u540e\u4fa7\u8fd1\u666f": "goal_area",
    "\u7403\u95e8\u540e\u4fa7\u89c6\u89d2": "goal_area",
    "\u7403\u95e8\u540e\u8fd1\u666f\uff0c\u53f3\u4fa7\u89c6\u89d2": "goal_area",
    "\u5e38\u89c4\u89c6\u89d2\uff08\u5de6\u4fa7\u9ad8\u673a\u4f4d\u8fdc\u666f\uff09": "other_side",
    "\u5de6\u4fa7\u9ad8\u673a\u4f4d\u8fdc\u666f": "other_side",
    "\u70b9\u7403\u8005\u53f3\u4fa7\u8fd1\u666f": "other_side",
    "\u70b9\u7403\u8005\u53f3\u4fa7\u89c6\u89d2": "other_side",
    "\u70b9\u7403\u8005\u4fa7\u5411\u89c6\u89d2": "other_side",
}


def select_intervals(metadata, release):
    clips = {row["export_index"]: row for row in metadata["clips"]
             if row["dataset_id"] in ALLOWED_MATCHES and row["filename"] != "match_09.mp4"}
    selected, excluded, seen = [], [], set()
    for row in metadata["shot_timeline"]:
        clip = clips.get(row["export_index"])
        if clip is None:
            continue
        label = row.get("camera_label")
        if label is None:
            excluded.append({"shot_id": row["original_shot_id"], "reason": "unlabelled_not_negative"})
            continue
        if row.get("camera_annotation_source") != "human_user_observation_in_chat":
            raise ValueError("Camera label lacks required human provenance")
        if label not in LABEL_MAP:
            excluded.append({"shot_id": row["original_shot_id"], "reason": "unmapped_human_category"})
            continue
        start, end = row.get("camera_label_start_seconds"), row.get("camera_label_end_seconds_exclusive")
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, (int, float)) or not isinstance(end, (int, float))):
            excluded.append({"shot_id": row["original_shot_id"], "reason": "no_exact_label_interval"})
            continue
        if not np.isfinite([start, end]).all() or not 0 <= start < end <= clip["duration_seconds"] + .05:
            raise ValueError("Camera label interval is invalid or exceeds its video")
        if row["video_file"] != clip["file"]:
            raise ValueError("Camera/video binding mismatch")
        path = (release / clip["file"]).resolve()
        if not path.is_relative_to(release.resolve()):
            raise ValueError("Video path escapes release")
        identity = (clip["media_check"]["sha256"], row["original_shot_id"])
        if identity in seen:
            raise ValueError("Duplicate annotated shot")
        seen.add(identity)
        selected.append({"shot_id": row["original_shot_id"], "event_id": clip["original_event_id"],
            "match_group": clip["dataset_id"], "video_path": str(path), "filename": clip["filename"],
            "video_sha256": clip["media_check"]["sha256"], "fps": clip["fps"],
            "start_seconds": float(start), "end_seconds_exclusive": float(end),
            "class": LABEL_MAP[label], "raw_human_label": label,
            "slow_motion_label": row.get("slow_motion_label"),
            "annotation_source": row["camera_annotation_source"]})
    if {row["match_group"] for row in selected} != ALLOWED_MATCHES:
        raise ValueError("Both development matches are required")
    return selected, excluded


def group_folds(rows):
    groups = sorted({row["match_group"] for row in rows})
    if set(groups) != ALLOWED_MATCHES:
        raise ValueError("Unexpected match group; holdouts cannot enter fitting")
    video_groups = {}
    event_groups = {}
    for row in rows:
        for mapping, key in ((video_groups, row["video_sha256"]), (event_groups, row["event_id"])):
            if key in mapping and mapping[key] != row["match_group"]:
                raise ValueError("A video or event crosses match groups")
            mapping[key] = row["match_group"]
    return [(group, [i for i, row in enumerate(rows) if row["match_group"] != group],
             [i for i, row in enumerate(rows) if row["match_group"] == group]) for group in groups]


def interval_frame_ids(start, end, fps):
    start_frame = int(np.ceil(start * fps - 1e-6))
    end_frame = int(np.ceil(end * fps - 1e-6))
    if end_frame <= start_frame:
        raise ValueError("Empty camera interval")
    # Avoid annotation/cut boundary frames: all samples are within the interval.
    return np.floor(start_frame + (np.arange(8) + .5) / 8 * (end_frame - start_frame)).astype(int).tolist()


def decode_interval(video, start, end, fps):
    ids = interval_frame_ids(start, end, fps)
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        for frame_id in ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Camera interval stopped decoding at frame {frame_id}")
            image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError("JPEG encoding failed")
            frames.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    finally:
        capture.release()
    return frames, ids


def decode_windows(video, windows, fps):
    indices = [interval_frame_ids(row["start_seconds"], row["end_seconds_exclusive"], fps) for row in windows]
    if not indices:
        return []
    needed = {index for group in indices for index in group}
    decoded = {}
    capture = cv2.VideoCapture(str(video))
    try:
        for index in range(max(needed) + 1):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Camera windows stopped decoding at frame {index}")
            if index not in needed:
                continue
            image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError("JPEG encoding failed")
            decoded[index] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    finally:
        capture.release()
    return [([decoded[index] for index in group], group) for group in indices]


def fit_head(features, training, labels):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if set(labels) != set(CLASSES):
        raise ValueError("A training match lacks one of the three spatial classes")
    scaler = StandardScaler().fit(features[training])
    pca = PCA(n_components=min(7, len(training) - 1), svd_solver="full")
    x = pca.fit_transform(scaler.transform(features[training]))
    head = LogisticRegression(C=.1, class_weight="balanced", solver="lbfgs", max_iter=2000,
                              random_state=previous.SEED)
    head.fit(x, [CLASSES.index(label) for label in labels])
    return scaler, pca, head


def export_head(scaler, pca, head):
    return {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "pca_mean": pca.mean_,
            "pca_components": pca.components_, "coef": head.coef_, "intercept": head.intercept_}


def predict_head(features, weights):
    x = (np.atleast_2d(features) - weights["scaler_mean"]) / weights["scaler_scale"]
    x = (x - weights["pca_mean"]) @ weights["pca_components"].T
    logits = x @ weights["coef"].T + weights["intercept"]
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    return {"labels": [CLASSES[i] for i in probability.argmax(axis=1)],
            "probabilities": probability.tolist()}


def all_scores(true, predicted):
    return {"spatial_three_class": previous.metrics(true, predicted, CLASSES),
            "behind_kicker_binary": previous.metrics(
                ["behind_kicker" if value == "behind_kicker" else "other" for value in true],
                ["behind_kicker" if value == "behind_kicker" else "other" for value in predicted],
                ["behind_kicker", "other"])}


def prediction_windows(shots, max_seconds=2.):
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Prediction window duration must be positive")
    windows = []
    for shot in shots:
        fps = shot["fps"]
        step = max(1, int(np.floor(max_seconds * fps)))
        for start in range(shot["shot_start_frame"], shot["shot_end_frame"], step):
            end = min(start + step, shot["shot_end_frame"])
            windows.append({"start_seconds": start / fps, "end_seconds_exclusive": end / fps,
                            "source_shot_index": shot["shot_index"],
                            "subdivided_automatic_shot": end - start < shot["shot_end_frame"] - shot["shot_start_frame"]})
    return windows


def train(args):
    metadata_path = args.release / "metadata" / "annotations.json"
    rows, excluded = select_intervals(previous.read_json(metadata_path), args.release)
    folds = group_folds(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = previous.load_encoder(args.backbone, threads=args.threads)
    verified, features = set(), []
    for row in rows:
        video = Path(row["video_path"])
        if row["video_sha256"] not in verified:
            if previous.digest(video) != row["video_sha256"]:
                raise ValueError("Camera source video checksum mismatch")
            verified.add(row["video_sha256"])
        started = time.perf_counter()
        frames, frame_ids = decode_interval(video, row["start_seconds"], row["end_seconds_exclusive"], row["fps"])
        temporal, _, _ = previous.encode_frames(frames, encoder, SETTINGS)
        features.append(temporal.reshape(4, -1).mean(axis=0).astype(np.float64))
        row.update(sample_frame_ids=frame_ids, feature_seconds=time.perf_counter() - started)
        print(row["filename"], row["shot_id"], row["class"], f"{row['feature_seconds']:.2f}s", flush=True)
    features = np.stack(features)
    np.savez_compressed(args.output / "features.npz", mean_spatial=features)
    oof, baselines = [None] * len(rows), [None] * len(rows)
    reports = []
    for group, training, validation in folds:
        labels = [rows[i]["class"] for i in training]
        fitted = fit_head(features, training, labels)
        weights = export_head(*fitted)
        np.savez_compressed(args.output / ("fold_" + group + ".npz"), **weights)
        predictions = predict_head(features[validation], weights)
        majority = max(CLASSES, key=labels.count)
        for index, label, probability in zip(validation, predictions["labels"], predictions["probabilities"]):
            oof[index] = {"label": label, "probabilities_uncalibrated": probability, "heldout_match": group}
            baselines[index] = majority
        truth = [rows[i]["class"] for i in validation]
        reports.append({"heldout_match": group, "train_indices": training, "validation_indices": validation,
                        "scores": all_scores(truth, predictions["labels"]),
                        "majority": all_scores(truth, [majority] * len(validation))})
    labels = [row["class"] for row in rows]
    final = fit_head(features, list(range(len(rows))), labels)
    np.savez_compressed(args.output / "camera_weights.npz", **export_head(*final))
    report = {"experiment": "external_spatial_camera_supervised_v1", "settings": SETTINGS,
        "classes": CLASSES, "raw_label_mapping": LABEL_MAP, "logistic_C": .1,
        "samples": rows, "excluded_in_development_matches": excluded, "folds": reports,
        "oof_predictions": oof, "scores": all_scores(labels, [row["label"] for row in oof]),
        "majority_scores": all_scores(labels, baselines),
        "class_counts": dict(Counter(labels)), "group_counts": dict(Counter(row["match_group"] for row in rows)),
        "playback_label_counts": dict(Counter(str(row["slow_motion_label"]) for row in rows)),
        "classifier_trained": True, "backbone_trained": False, "training_only_preprocessing": True,
        "oof_split": "leave_one_whole_match_out", "deployment_fit": "all admitted human camera intervals",
        "held_out_world_cup_accessed": False, "independent_test_claim": False,
        "teacher_four_classes_completed": False, "playback_classifier_trained": False,
        "temporal_boundaries_evaluated": False, "promoted_to_default_pipeline": False,
        "portable_encoder_sha256": previous.digest(args.backbone),
        "weights_sha256": previous.digest(args.output / "camera_weights.npz"),
        "source_metadata_sha256": previous.digest(metadata_path), "script_sha256": previous.digest(__file__),
        "limitations": ["Only two independent matches; repeated camera setups and replays are correlated",
                        "Human segment boundaries are provided during evaluation, not detected end to end",
                        "Goal-area does not prove behind-keeper slow motion",
                        "Other-side does not prove a 45-degree right view or a close-up",
                        "Unlabelled intervals are excluded, not treated as negatives",
                        "Unknown playback speed remains unknown; container FPS is not a slow-motion label"]}
    previous.save_json(args.output / "report.json", report)
    print({"scores": report["scores"], "majority": report["majority_scores"]}, flush=True)


def predict(args):
    from src.step_pipeline import _scan_camera_shots
    report = previous.read_json(args.model / "report.json")
    if previous.digest(args.backbone) != report["portable_encoder_sha256"]:
        raise ValueError("Encoder hash mismatch")
    weights_path = args.model / "camera_weights.npz"
    if previous.digest(weights_path) != report["weights_sha256"]:
        raise ValueError("Camera classifier hash mismatch")
    started = time.perf_counter()
    encoder = previous.load_encoder(args.backbone, threads=args.threads)
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    capture.release()
    if duration <= 0:
        raise ValueError("Unreadable video")
    shots = _scan_camera_shots(str(args.video), duration, 30.)
    windows = prediction_windows(shots, args.max_window_seconds)
    decoded_windows = decode_windows(args.video, windows, fps)
    records = []
    with np.load(weights_path, allow_pickle=False) as weights:
        for window, (frames, ids) in zip(windows, decoded_windows):
            temporal, _, _ = previous.encode_frames(frames, encoder, report["settings"])
            prediction = predict_head(temporal.reshape(4, -1).mean(axis=0), weights)
            label = prediction["labels"][0]
            records.append({**window,
                "spatial_class_candidate": label, "probabilities_uncalibrated": prediction["probabilities"][0],
                "course_class_candidate": "behind_kicker" if label == "behind_kicker" else "unknown",
                "slow_motion": "unknown", "human_annotation_used": False, "sample_frame_ids": ids})
    previous.save_json(args.output, {"video_sha256": previous.digest(args.video), "segments": records,
        "end_to_end_seconds_including_model_load": time.perf_counter() - started,
        "teacher_four_classes_completed": False, "temporal_boundaries_validated": False,
        "interval_meaning": "bounded prediction windows, not verified camera-shot boundaries",
        "maximum_prediction_window_seconds": args.max_window_seconds,
        "automatic_camera_shots": shots,
        "model_report_sha256": previous.digest(args.model / "report.json"),
        "inference_script_sha256": previous.digest(__file__),
        "manual_annotation_required": False, "experimental_candidate": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--release", type=Path, required=True)
    p.set_defaults(run=train)
    q = commands.add_parser("predict")
    q.add_argument("--video", type=Path, required=True)
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--max-window-seconds", type=float, default=2.)
    q.set_defaults(run=predict)
    for command in (p, q):
        command.add_argument("--backbone", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
