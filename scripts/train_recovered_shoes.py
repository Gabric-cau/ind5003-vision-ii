"""Evaluate recovered shoe evidence with match-isolated upstream contact weights."""
from __future__ import annotations

import argparse
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_shoe_supervised import read, write, sha, load_events, foot_metrics, landing_metrics
from src.contact_temporal_model import predict_contact, validate_prediction_bindings
from src.shoe_supervised_model import extract_evidence, fit_models, predict_event


def restrict_training_to_review(event):
    """Unreviewed time is unknown, not a negative landing example."""
    result = deepcopy(event)
    if event.get("external_foot_only"):
        result["evidence"]["landing_rows"] = []
        result["landings"] = []
        return result
    window = event["review_window"]
    low, high = window["start_frame"], window["end_frame"]
    result["evidence"]["landing_rows"] = [r for r in event["evidence"]["landing_rows"]
                                           if low <= r["frame"] <= high]
    result["landings"] = [r for r in event["landings"] if low <= r["frame"] <= high]
    return result


def restrict_scoring_to_review(event, prediction):
    result = deepcopy(prediction)
    if result.get("landings") is not None:
        window = event["review_window"]
        result["landings"] = [r for r in result["landings"]
                              if window["start_frame"] <= r["frame"] <= window["end_frame"]]
        result["candidate_count"] = len(result["landings"])
    return result


def events_for_contact_weights(original, recovered, parameters, temporal_foot=False):
    """Use the same outer-fold contact weights for every new training/validation event."""
    events = deepcopy(original)
    for event in events:
        data = recovered.get(event["clip_id"])
        if data is None:
            continue
        inputs = deepcopy(data["inputs"])
        predicted = predict_contact(data["record"], parameters)
        event["recovered_contact_prediction"] = predicted
        if predicted["frame"] is None:
            event["reason"] = "outer_fold_contact_unavailable"
            continue
        offset = data["receipt"]["lossless_shot"]["source_start_frame"]
        point = offset + predicted["frame"]
        first, end = inputs["scene_interval_frames"]
        bracket = [max(first, point - 1), min(end - 1, point + 1)]
        if not first < bracket[0] < bracket[1] < end:
            event["reason"] = "outer_fold_contact_near_shot_edge"
            continue
        inputs["contact_bracket_frames"] = bracket
        inputs["interval_frames"] = [first, bracket[0]]
        event["automatic_window"] = {"start_frame": first, "end_frame": bracket[0]}
        event["evidence"] = extract_evidence(inputs, data["poses"])
        if temporal_foot:
            from src.shoe_temporal_foot import temporal_foot_vector
            event["evidence"]["foot_vector"] = temporal_foot_vector(inputs, data["poses"])
        event["reason"] = None
    return events


def validated_folds(model_dir, core_inputs):
    groups = {r["core_id"]: r["match_group"] for r in read(core_inputs)["samples"]}
    report = read(model_dir / "report.json")
    expected = {"col_eng", "cro_den", "esp_rus", "rus_cro"}
    folds = report["folds"]
    if {f["held_out_match"] for f in folds} != expected or len(folds) != 4:
        raise ValueError("Contact model must retain four distinct match folds")
    for fold in folds:
        held_out = fold["held_out_match"]
        if (any(groups[cid] == held_out for cid in fold["training_core_ids"])
                or any(groups[cid] != held_out for cid in fold["validation_core_ids"])):
            raise ValueError("Upstream contact fold contains its held-out match")
        if sha(model_dir / fold["weights_file"]) != fold["weights_sha256"]:
            raise ValueError("Contact fold weights changed")
    return folds


def load_external(data_dir, evidence_dir, metadata):
    label_document = read(data_dir / "labels.json")
    if label_document.get("landing_supervision_available") is not False:
        raise ValueError("External augmentation must be foot-only")
    labels = {r["clip_id"]: r for r in label_document["labels"]}
    inputs = read(data_dir / "inputs.json")["samples"]
    if (len(inputs) != 15 or len(labels) != 15
            or {r["clip_id"] for r in inputs} != set(labels)):
        raise ValueError("External foot augmentation requires the exact eligible 15 events")
    events, recovered, audit = [], {}, []
    for sample in inputs:
        cid = sample["clip_id"]
        label = labels[cid]
        if (sample["match_group"] not in ("german_cup_2022", "coppa_italia_2025") or cid == "match_09"
                or label["match_group"] != sample["match_group"]
                or label["source_sha256"] != sample["video_sha256"]
                or label["provenance"] != "human_user_observation_in_chat"
                or label["striking_foot"] not in ("left", "right")):
            raise ValueError("External source/provenance or holdout exclusion mismatch")
        event = {"clip_id": cid, "match_group": sample["match_group"], "fps": sample["fps"],
            "source_sha256": sample["video_sha256"], "foot": label["striking_foot"],
            "landings": [], "review_window": None, "automatic_window": None, "external_foot_only": True,
            "evidence": {"landing_rows": [], "foot_vector": None, "supported_side_frames": {}},
            "reason": "external_evidence_unavailable"}
        directory = evidence_dir / cid
        if (directory / "shoes/inputs.json").exists():
            document = read(directory / "shoes/inputs.json")
            record = read(directory / "contact/features.json")
            if (document.get("reference_labels_read") is not False
                    or document["source_mapping"]["clip_sha256"] != sample["video_sha256"]
                    or record["video_sha256"] != sample["video_sha256"]
                    or document["poses33_sha256"] != sha(directory / "shoes/poses33.json")
                    or document["inference_interval_frames"] != [document["scene_interval_frames"][0], document["scene_interval_frames"][1] - 1]
                    or document["clock_mapping"]["source_start_frame"] != 0):
                raise ValueError("External shoe features do not bind the complete automatic original-video shot")
            validate_prediction_bindings(record, metadata)
            recovered[cid] = {"inputs": document, "record": record,
                "poses": read(directory / "shoes/poses33.json"),
                "receipt": {"lossless_shot": {"source_start_frame": 0,
                              "mapping_kind": "original_video_no_reencoding"}}}
            audit.append({"clip_id": cid, "inputs_sha256": sha(directory / "shoes/inputs.json"),
                          "poses_sha256": sha(directory / "shoes/poses33.json"),
                          "pose_model_sha256": document["model_sha256"],
                          "source_sha256": sample["video_sha256"], "foot_label": label["striking_foot"]})
        events.append(event)
    return events, recovered, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-foot", action="store_true",
                        help="Fixed contact-aligned 95-feature head, requires its corresponding inference adapter")
    parser.add_argument("--external-data", type=Path)
    parser.add_argument("--external-evidence", type=Path)
    parser.add_argument("--core-inputs", type=Path,
        default=ROOT / "outputs/core_contact_training_2026-10-26/data/inputs.json")
    args = parser.parse_args()
    if bool(args.external_data) != bool(args.external_evidence) or (args.external_data and args.temporal_foot):
        parser.error("External augmentation requires both paths and the fixed original 57-feature head")
    if args.output.exists():
        raise FileExistsError("Preserve previous model runs")
    started = time.perf_counter()
    base = ROOT / "outputs/fresh_course_experiment"
    references = base / "human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json"
    events, bindings = load_events(base / "shoe_approach33_cohort_v1", references,
        base / "shoe_landing_replay_2026-10-18_v1", base / "pivot_support_development_2026-10-22")
    if args.temporal_foot:
        from src.shoe_temporal_foot import temporal_foot_vector
        for event in events:
            if event["automatic_window"] is not None:
                directory = base / "shoe_approach33_cohort_v1/events" / event["clip_id"]
                event["evidence"]["foot_vector"] = temporal_foot_vector(
                    read(directory / "inputs.json"), read(directory / "poses33.json"))
    metadata = read(args.contact_model / "deployment_metadata.json")
    if sha(args.contact_model / "deployment_weights.npz") != metadata["weights_sha256"]:
        raise ValueError("Deployment contact weights changed")
    recovered = {}
    for event in events:
        directory = args.recovered / event["clip_id"]
        if not (directory / "shoes/inputs.json").exists():
            continue
        if event["automatic_window"] is not None:
            raise ValueError("Recovery must not overwrite one of the original 32 evidence events")
        inputs = read(directory / "shoes/inputs.json")
        if (inputs.get("reference_labels_read") is not False
                or inputs["source_mapping"]["clip_sha256"] != event["source_sha256"]
                or inputs["poses33_sha256"] != sha(directory / "shoes/poses33.json")):
            raise ValueError("Recovered evidence source or annotation boundary changed")
        receipt = read(directory / "recovery_input.json")
        record = read(directory / "contact/features.json")
        if record["video_sha256"] != receipt["lossless_shot"]["shot_sha256"]:
            raise ValueError("Contact features changed source clock")
        if receipt["source_mapping"]["clip_sha256"] != event["source_sha256"]:
            raise ValueError("Original clip binding differs from reference")
        if (inputs["inference_interval_frames"] != [inputs["scene_interval_frames"][0], inputs["scene_interval_frames"][1] - 1]
                or receipt.get("manual_annotations_used") is not False
                or not receipt["lossless_shot"]["all_pixels_verified_equal"]):
            raise ValueError("Recovery must cache the whole automatic shot without manual timing")
        validate_prediction_bindings(record, metadata)
        recovered[event["clip_id"]] = {"inputs": inputs, "record": record,
            "poses": read(directory / "shoes/poses33.json"), "receipt": receipt}
        bindings.append({"clip_id": event["clip_id"], "inputs_sha256": sha(directory / "shoes/inputs.json"),
            "poses_sha256": sha(directory / "shoes/poses33.json"), "pose_model_sha256": inputs["model_sha256"],
            "scope": "new_automatic_shot_with_per_outer_fold_contact_windows",
            "human_review_window": event["review_window"]})
    folds = validated_folds(args.contact_model, args.core_inputs)
    external_events, external_audit = [], []
    if args.external_data:
        external_events, external_recovered, external_audit = load_external(
            args.external_data, args.external_evidence, metadata)
        if any(item["pose_model_sha256"] not in {b["pose_model_sha256"] for b in bindings}
               for item in external_audit):
            raise ValueError("External pose checkpoint differs from original shoe evidence")
        recovered.update(external_recovered)
    all_events = events + external_events
    args.output.mkdir(parents=True)
    predictions, scoring, fold_audits = {}, {}, []
    with threadpool_limits(limits=2):
        for fold in folds:
            held_out = fold["held_out_match"]
            with np.load(args.contact_model / fold["weights_file"], allow_pickle=False) as parameters:
                specific = events_for_contact_weights(all_events, recovered, parameters, args.temporal_foot)
            training = [restrict_training_to_review(e) for e in specific]
            model = fit_models(training, held_out_match=held_out)
            for event in specific:
                if event["match_group"] != held_out:
                    continue
                predicted = predict_event(model, event["evidence"], event["fps"])
                predictions[event["clip_id"]] = {"clip_id": event["clip_id"],
                    "automatic_window": event["automatic_window"],
                    "recovered_contact_prediction": event.get("recovered_contact_prediction"),
                    "supported_side_frames": event["evidence"]["supported_side_frames"], **predicted}
                scoring[event["clip_id"]] = restrict_scoring_to_review(event, predicted)
            joblib.dump(model, args.output / f"fold_{held_out}.joblib")
            fold_audits.append({"held_out_match": held_out,
                "same_upstream_contact_weights_for_training_and_validation": True,
                "upstream_contact_weights_sha256": fold["weights_sha256"],
                "upstream_training_core_ids": fold["training_core_ids"],
                "training_landing_ids": model["training_landing_ids"],
                "training_foot_ids": model["training_foot_ids"],
                "validation_ids": [e["clip_id"] for e in specific if e["match_group"] == held_out]})
        with np.load(args.contact_model / "deployment_weights.npz", allow_pickle=False) as parameters:
            deployment_events = events_for_contact_weights(all_events, recovered, parameters, args.temporal_foot)
        deployment = fit_models([restrict_training_to_review(e) for e in deployment_events])
        deployment["evidence_contract"] = {"pose_model_sha256": sorted({b["pose_model_sha256"] for b in bindings}),
            "feature_code_sha256": sha(ROOT / "src/shoe_supervised_model.py"),
            "coordinate_space": "original_source_image_pixels",
            "landmarks": "mediapipe_33_normal_and_inverse_mapped_mirror",
            "automatic_windows_only": True, "labels_required_for_prediction": False}
        if args.temporal_foot:
            deployment["evidence_contract"].update(
                foot_feature_mode="contact_aligned_95_requires_temporal_adapter",
                foot_feature_code_sha256=sha(ROOT / "src/shoe_temporal_foot.py"))
        joblib.dump(deployment, args.output / "deployment.joblib")
    ordered = [scoring[e["clip_id"]] for e in events]
    old = {e["clip_id"]: e for e in read(ROOT / "outputs/shoe_supervised_2026-10-26/model_v2_audit/oof_predictions.json")["events"]}
    comparator = [restrict_scoring_to_review(e, old[e["clip_id"]]) for e in events]
    report = {"schema_version": "recovered-shoe-nested-upstream-match-folds-v1",
        "events": len(events), "new_recovered_event_ids": sorted(cid for cid in recovered if not cid.startswith("match_")),
        "external_foot_supervision_events": len(external_events), "external_evidence_audit": external_audit,
        "external_labels_sha256": sha(args.external_data / "labels.json") if args.external_data else None,
        "evaluation": "four_outer_match_folds_with_upstream_contact_weights_excluding_same_match",
        "previously_inspected_development": True, "untouched_test": False,
        "foot_feature_mode": "contact_aligned_95" if args.temporal_foot else "quantile_summary_57",
        "landing_scoring_scope": "Each original human review window; automatic predictions outside it unscored, never counted as negatives",
        "all_human_windows_and_source_clocks_unchanged": True,
        "reference_sha256": sha(references), "evidence_bindings": bindings, "folds": fold_audits,
        "foot": {"recovered": foot_metrics(events, ordered), "old32": foot_metrics(events, comparator)},
        "landings": {"recovered": landing_metrics(events, ordered), "old32": landing_metrics(events, comparator)},
        "deployment_training_landing_ids": deployment["training_landing_ids"],
        "deployment_training_foot_ids": deployment["training_foot_ids"],
        "training_and_evaluation_seconds": time.perf_counter() - started,
        "code_sha256": {"scripts/train_recovered_shoes.py": sha(Path(__file__)),
                        "src/shoe_temporal_foot.py": sha(ROOT / "src/shoe_temporal_foot.py")},
        "limitations": ["Only four previously inspected matches and one human reviewer.",
            "Recovered actor identities are automatic geometric associations, not manually verified identities.",
            "Full-shot extraction and manually reviewed scoring scopes differ and are recorded separately.",
            "Human reference absence outside its review window is not a negative label.",
            "Contact neighborhood is a sampling convention, not a statistical confidence interval."]}
    if external_events:
        comparison = read(ROOT / "outputs/shoe_supervised_2026-10-26/model_recovery_v3/oof_predictions.json")
        original39 = {r["clip_id"]: r for r in comparison["events"]}
        if set(original39) != {e["clip_id"] for e in events}:
            raise ValueError("Original-39 ablation does not cover the same evaluation cohort")
        report["foot"]["original39_only"] = foot_metrics(events, [original39[e["clip_id"]] for e in events])
    write(args.output / "report.json", report)
    write(args.output / "oof_predictions.json", {"events": [predictions[e["clip_id"]] for e in events]})
    write(args.output / "deployment_metadata.json", {"model_sha256": sha(args.output / "deployment.joblib"),
        "evidence_contract": deployment["evidence_contract"],
        "dependencies": {n: version(n) for n in ("numpy", "scikit-learn", "joblib")},
        "reference_sha256": sha(references), "development_evaluation": report["evaluation"],
        "external_labels_sha256": report["external_labels_sha256"],
        "external_evidence_audit": external_audit,
        "training_landing_ids": deployment["training_landing_ids"],
        "training_foot_ids": deployment["training_foot_ids"], "config": deployment["config"]})
    print({"foot": report["foot"], "landings": {k: {a: b for a, b in v.items() if a != "rows"}
          for k, v in report["landings"].items()}})


if __name__ == "__main__":
    main()
