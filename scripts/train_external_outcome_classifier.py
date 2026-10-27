"""Fixed two-cohort goal/no-goal pilot using existing automatic-shot features.

Neither World Cup test targets nor its video frames enter this experiment.
The learned outcome stays an auxiliary candidate and never overrides a rule.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_external_ball_classifier as visual
from scripts import train_external_ball_height_augmented as height
from scripts import train_external_ball_v2 as motion
from scripts import evaluate_offline_video_batch as offline

CLASSES = ["no_goal", "goal"]
COHORTS = ("german8", "german8_plus_core10")


def external_goal_labels(metadata, samples):
    visual.validate_samples(samples)
    by_name = {row["filename"]: row for row in metadata["clips"]
               if row["dataset_id"] in ("german_cup_2022", "coppa_italia_2025")
               and row["filename"] != "match_09.mp4"}
    records = []
    for sample in samples:
        row = by_name[sample["filename"]]
        human = row["human_features"]
        if row["media_check"]["sha256"] != sample["video_sha256"]:
            raise ValueError("Goal reference/video hash mismatch")
        if row.get("outcome_conflict") or row.get("training_goal_label") not in (0, 1):
            raise ValueError("Conflicting or unresolved outcome cannot enter this experiment")
        if (human.get("provenance") != "human_user_observation_in_chat" or
                human.get("goal_label") != row["training_goal_label"] or
                row.get("official_goal_label") != human["goal_label"]):
            raise ValueError("Human and supplied external-result references disagree")
        records.append({"sample_id": sample["event_id"], "filename": sample["filename"],
            "dataset_id": sample["dataset_id"], "split": sample["split"],
            "video_sha256": sample["video_sha256"], "goal": CLASSES[int(human["goal_label"])],
            "reference_provenance": human["provenance"], "outcome_detail": human.get("result_detail_cn"),
            "source_audit_flags": sample.get("audit_flags", []),
            "terminal_result_visibility": "not_independently_adjudicated"})
    return records


def fit_head(features, indices, labels):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if set(labels) != set(CLASSES):
        raise ValueError("Both outcome classes must appear in training")
    scaler = StandardScaler().fit(features[indices])
    pca = PCA(n_components=min(7, len(indices) - 1), svd_solver="full")
    transformed = pca.fit_transform(scaler.transform(features[indices]))
    head = LogisticRegression(C=.1, class_weight="balanced", solver="lbfgs", max_iter=2000,
                              random_state=visual.SEED)
    head.fit(transformed, [CLASSES.index(label) for label in labels])
    return scaler, pca, head


def export_head(scaler, pca, head):
    return {"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "pca_mean": pca.mean_,
            "pca_components": pca.components_, "coef": head.coef_, "intercept": head.intercept_}


def predict_head(features, weights):
    x = (np.atleast_2d(features) - weights["scaler_mean"]) / weights["scaler_scale"]
    x = (x - weights["pca_mean"]) @ weights["pca_components"].T
    logits = np.clip((x @ weights["coef"].T + weights["intercept"]).ravel(), -700, 700)
    positive = 1 / (1 + np.exp(-logits))
    probabilities = np.stack([1 - positive, positive], axis=1)
    return {"labels": [CLASSES[i] for i in probabilities.argmax(axis=1)],
            "probabilities": probabilities.tolist()}


def score(rows, predicted):
    return offline.categorical_metrics([
        {"core_id": row["sample_id"], "reference": row["goal"], "prediction": label}
        for row, label in zip(rows, predicted)], CLASSES)


def train(args):
    frames_path = ROOT / "outputs/external_ball_training_2026-10-25/frames_v2/frames.json"
    source_manifest = ROOT / "outputs/external_ball_training_2026-10-25/manifest.json"
    manifest = visual.read_json(source_manifest)
    frames = visual.read_json(frames_path)
    motion.validate_frames_manifest(frames)
    samples = manifest["samples"]
    if [row["event_id"] for row in frames["samples"]] != [row["event_id"] for row in samples]:
        raise ValueError("External feature order differs from the audited video manifest")
    source_metadata = Path(manifest["annotation_source"]["path"])
    if visual.digest(source_metadata) != manifest["annotation_source"]["sha256"]:
        raise ValueError("External human reference bytes changed")
    external = external_goal_labels(visual.read_json(source_metadata), samples)
    core_inputs = visual.read_json(ROOT / "outputs/core_contact_training_2026-10-26/data/inputs.json")["samples"]
    rules, expected_core, rule_bindings = offline.load_completed_batch(args.core_batch)
    references, reference_bindings = offline.load_references(
        ROOT / "outputs/core_contact_training_2026-10-26/data/labels.json", expected_core)
    # This helper's reference-read ordering note belongs to batch evaluation.
    # Here the workbook outcome is deliberately a supervised training target.
    reference_bindings["usage"] = "supervised_outcome_training_and_existing_core_development_comparison"
    visual_folder = ROOT / "outputs/external_ball_training_2026-10-25/model_v1"
    motion_report = visual.read_json(ROOT / "outputs/external_ball_training_2026-10-26/model_v2/report.json")
    feature_path = visual_folder / "feature_cache.npz"
    if visual.digest(feature_path) != motion_report["prior_feature_cache_sha256"]:
        raise ValueError("Frozen external feature cache changed")
    backbone = visual_folder / "encoder.pt"
    if visual.digest(backbone) != motion_report["portable_encoder_sha256"]:
        raise ValueError("Encoder hash mismatch")
    with np.load(feature_path, allow_pickle=False) as cache:
        external_features = cache["temporal"].astype(np.float64) - np.tile(cache["first_frame"].astype(np.float64), (1, 4))
    height_folder = ROOT / "outputs/external_ball_training_2026-10-26/height_core_augmented_v2"
    height_report = visual.read_json(height_folder / "report.json")
    if height_report["event_ids"][:len(external)] != [row["sample_id"] for row in external]:
        raise ValueError("Existing core feature-cache order mismatch")
    with np.load(height_folder / "features.npz", allow_pickle=False) as cache:
        existing_features = cache["first_frame_difference"].copy()
    with np.load(height_folder / "height_weights.npz", allow_pickle=False) as weights:
        check = height.predict_height(existing_features, weights)
    np.testing.assert_allclose(check["probabilities"], height_report["predictions"]["probabilities"], atol=1e-12)
    core_cache = {row["core_id"]: (row, existing_features[len(external) + index])
                  for index, row in enumerate(height_report["admitted_cores"])}
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"experiment": "goal_no_goal_two_fixed_training_cohorts_v1", "cohorts": list(COHORTS),
        "validation": "same seven Coppa Italia events for both cohorts; match_09 quarantined",
        "descriptor": "frozen ResNet18 four temporal means minus first-frame features",
        "logistic_C": .1, "class_weight": "balanced", "pca_dimensions": 7,
        "preprocessing_fit_on_training_only": True, "manual_timeline_or_contact_used_as_input": False,
        "world_cup_targets_read_for_selection": False, "world_cup_inference_performed": False,
        "cohort_selection_before_fitting": True, "rule_outputs_never_overridden": True,
        "source_annotation_sha256": visual.digest(source_metadata),
        "external_feature_cache_sha256": visual.digest(feature_path),
        "core_feature_cache_sha256": visual.digest(height_folder / "features.npz"),
        "reference_bindings": reference_bindings, "rule_bindings": rule_bindings,
        "script_sha256": visual.digest(__file__)}
    visual.save_json(args.output / "protocol.json", protocol)
    all_features, rows = list(external_features), list(external)
    encoder = None
    for sample in core_inputs:
        cid = sample["core_id"]
        if (sample["video_sha256"] != expected_core[cid]["video_sha256"] or
                sample["video_sha256"] != references[cid]["video_sha256"]):
            raise ValueError("Core feature/reference/video mismatch")
        if cid in core_cache:
            cached, feature = core_cache[cid]
            if cached["video_sha256"] != sample["video_sha256"]:
                raise ValueError("Cached core features belong to another video")
            window, feature_origin = cached["automatic_window"], "existing_automatic_first_shot_cache"
        else:
            if visual.digest(sample["video_path"]) != sample["video_sha256"]:
                raise ValueError("Core video checksum mismatch")
            if encoder is None:
                encoder = visual.load_encoder(backbone, threads=2)
            encoded, window = visual.decode_video(sample["video_path"], frames["settings"])
            images = [cv2.imdecode(frame, cv2.IMREAD_COLOR) for frame in encoded]
            temporal, first, _ = visual.encode_frames(images, encoder, frames["settings"])
            feature = temporal.astype(np.float64) - np.tile(first.astype(np.float64), 4)
            feature_origin = "fresh_automatic_first_shot_same_frozen_encoder"
        all_features.append(feature)
        rows.append({"sample_id": cid, "clip_id": sample["clip_id"], "dataset_id": sample["match_group"],
            "split": "core_development", "video_sha256": sample["video_sha256"], "goal": references[cid]["outcome"],
            "reference_provenance": "human_filled_workbook_as_delivered", "reference_source_cell": "R" + str(references[cid]["source_row"]),
            "automatic_window": window, "feature_origin": feature_origin,
            "terminal_result_visibility": "not_independently_adjudicated",
            "existing_rule": offline.SCOREBOARD_MAP.get(rules[cid]["required_outputs"].get("goal_outcome"))})
    features = np.stack(all_features)
    if not np.isfinite(features).all() or len({row["video_sha256"] for row in rows}) != len(rows):
        raise ValueError("Nonfinite features or duplicate source videos")
    np.savez_compressed(args.output / "features.npz", first_frame_difference=features)
    training = [i for i, row in enumerate(rows) if row["split"] == "train"]
    validation = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    core = [i for i, row in enumerate(rows) if row["split"] == "core_development"]
    core_rows = [rows[i] for i in core]
    rule_labels = [row["existing_rule"] for row in core_rows]
    report = {"experiment": protocol["experiment"], "protocol_sha256": visual.digest(args.output / "protocol.json"),
        "samples": rows, "cohorts": {}, "classes": CLASSES, "feature_dimensions": int(features.shape[1]),
        "settings": frames["settings"], "portable_encoder_sha256": visual.digest(backbone),
        "classifiers_trained": True, "backbone_trained": False, "promoted_to_default_pipeline": False,
        "world_cup_reused": False, "same_italian_validation_for_both_cohorts": True,
        "core_rule_baseline": score(core_rows, rule_labels),
        "italian_rule_baseline": None,
        "italian_rule_baseline_reason": "No existing same-video rule artifacts; no expensive new full pipeline run performed",
        "limitations": ["An automatic first shot need not show the terminal result",
                        "No manual result-time crop or human contact time enters features",
                        "First-shot outcome visibility is not independently labelled; match_08 retains its incomplete-action flag",
                        "Only one no-goal appears in the seven-event validation match",
                        "Core10 is development data; augmented-cohort core scores are in-sample",
                        "A visual category prediction does not verify a goal-plane crossing or final endpoint"]}
    for cohort in COHORTS:
        indices = training if cohort == "german8" else training + core
        labels = [rows[i]["goal"] for i in indices]
        started = time.perf_counter()
        fitted = fit_head(features, indices, labels)
        weights = export_head(*fitted)
        weight_path = args.output / (cohort + "_weights.npz")
        np.savez_compressed(weight_path, **weights)
        predictions = predict_head(features, weights)
        majority = max(CLASSES, key=labels.count)
        core_model = [predictions["labels"][i] for i in core]
        fallback = [rule if rule in CLASSES else prediction for rule, prediction in zip(rule_labels, core_model)]
        result = {"train_indices": indices, "train_events": len(indices), "validation_indices": validation,
            "training_class_counts": dict(zip(CLASSES, [labels.count(label) for label in CLASSES])),
            "fit_seconds": time.perf_counter() - started, "weights_sha256": visual.digest(weight_path),
            "predictions": predictions, "validation": score([rows[i] for i in validation], [predictions["labels"][i] for i in validation]),
            "validation_training_majority": score([rows[i] for i in validation], [majority] * len(validation)),
            "training": score([rows[i] for i in indices], [predictions["labels"][i] for i in indices]),
            "core_development": {"contains_training_events": cohort != "german8", "model": score(core_rows, core_model),
                "hypothetical_keep_rule_then_model": score(core_rows, fallback), "fallback_enabled": False,
                "disagreements_with_available_rules": [row["sample_id"] for row, rule, prediction in
                    zip(core_rows, rule_labels, core_model) if rule in CLASSES and rule != prediction]}}
        report["cohorts"][cohort] = result
        print(cohort, "validation", result["validation"]["correct_events"], "/ 7",
              "core", result["core_development"]["model"]["correct_events"], "/ 10", flush=True)
    visual.save_json(args.output / "report.json", report)


def predict(args):
    report = visual.read_json(args.model / "report.json")
    if visual.digest(args.backbone) != report["portable_encoder_sha256"]:
        raise ValueError("Encoder hash mismatch")
    path = args.model / (args.cohort + "_weights.npz")
    if visual.digest(path) != report["cohorts"][args.cohort]["weights_sha256"]:
        raise ValueError("Classifier hash mismatch")
    started = time.perf_counter()
    encoded, window = visual.decode_video(args.video, report["settings"])
    images = [cv2.imdecode(frame, cv2.IMREAD_COLOR) for frame in encoded]
    encoder = visual.load_encoder(args.backbone, threads=2)
    temporal, first, _ = visual.encode_frames(images, encoder, report["settings"])
    feature = temporal.astype(np.float64) - np.tile(first.astype(np.float64), 4)
    with np.load(path, allow_pickle=False) as weights:
        prediction = predict_head(feature, weights)
    visual.save_json(args.output, {"video_sha256": visual.digest(args.video), "cohort": args.cohort,
        "outcome_candidate": prediction, "automatic_window": window, "probabilities_calibrated": False,
        "end_to_end_seconds_including_model_load": time.perf_counter() - started,
        "manual_annotations_used": False, "terminal_result_visibility_verified": False,
        "existing_rule_overridden": False, "experimental_auxiliary_only": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--core-batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=train)
    q = commands.add_parser("predict")
    q.add_argument("--video", type=Path, required=True)
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--backbone", type=Path, required=True)
    q.add_argument("--cohort", choices=COHORTS, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.set_defaults(run=predict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
