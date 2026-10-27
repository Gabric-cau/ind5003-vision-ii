"""Grouped supervised shoe experiments on the original 39-event development set."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_shoe_landings import match_landings
from src.shoe_supervised_model import ShoeConfig, extract_evidence, fit_models, predict_event


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_events(dense, references, baseline, pivot):
    summary = read(dense / "summary.json")
    rows = {r["clip_id"]: r for r in summary["rows"]}
    ref_events = read(references)["events"]
    if len(ref_events) != 39 or len(rows) != 39 or len({e["clip_id"] for e in ref_events}) != 39:
        raise ValueError("Expected the complete unique original 39-event cohort")
    events, bindings = [], []
    for ref in ref_events:
        cid = ref["clip_id"]
        metadata = rows[cid]
        if ref["source_sha256"] != metadata["source_sha256"]:
            raise ValueError(f"Reference source hash mismatch: {cid}")
        review = ref["reviews"][0]
        if review.get("annotator_type") != "human" or review.get("human_verified") is not True:
            raise ValueError("Expected explicit human reference provenance")
        event = {"clip_id": cid, "match_group": "_".join(cid.split("_")[:2]),
                 "fps": ref["fps"], "source_sha256": ref["source_sha256"],
                 "foot": review["striking_foot"], "landings": review["verified_landings"],
                 "review_window": review["review_window"], "automatic_window": None,
                 "evidence": {"landing_rows": [], "foot_vector": None, "supported_side_frames": {}},
                 "reason": metadata.get("reason")}
        if metadata["status"] == "fresh_dense_pose_candidate":
            directory = dense / "events" / cid
            inputs_path, poses_path = directory / "inputs.json", directory / "poses33.json"
            if sha(inputs_path) != metadata["inputs_sha256"] or sha(poses_path) != metadata["poses33_sha256"]:
                raise ValueError(f"Frozen evidence hash mismatch: {cid}")
            inputs = read(inputs_path)
            if inputs.get("reference_labels_read") is not False:
                raise ValueError("Automatic evidence cannot have used reference labels")
            window = {"start_frame": inputs["interval_frames"][0], "end_frame": inputs["interval_frames"][1]}
            if window != ref["observable_window"] or inputs["source_mapping"]["clip_sha256"] != ref["source_sha256"]:
                raise ValueError("Automatic window/source differs from reference context")
            event["automatic_window"] = window
            event["evidence"] = extract_evidence(inputs, read(poses_path))
            bindings.append({"clip_id": cid, "inputs_sha256": sha(inputs_path), "poses_sha256": sha(poses_path),
                             "pose_model_sha256": inputs["model_sha256"]})
        for name, folder in (("baseline", baseline), ("pivot", pivot)):
            path = folder / "events" / cid / "landing_result.json"
            event[name] = read(path)
            if event[name]["source_sha256"] != ref["source_sha256"]:
                raise ValueError("Baseline source differs from supervised reference")
        events.append(event)
    return events, bindings


def foot_metrics(events, predictions):
    truths = [e["foot"] for e in events]
    guessed = [p.get("striking_foot", "unknown") for p in predictions]
    decided = [i for i, p in enumerate(guessed) if p in ("left", "right")]
    correct = sum(a == b for a, b in zip(truths, guessed))
    per_class = {side: {"references": truths.count(side),
                        "correct": sum(a == b == side for a, b in zip(truths, guessed)),
                        "recall_all_references": sum(a == b == side for a, b in zip(truths, guessed)) / truths.count(side)}
                 for side in ("left", "right")}
    return {"total": len(events), "outputs": len(decided), "correct": correct,
            "all_event_accuracy": correct / len(events), "coverage": len(decided) / len(events),
            "covered_accuracy": correct / len(decided) if decided else None,
            "macro_f1_with_abstentions": float(f1_score(truths, guessed, labels=["left", "right"], average="macro", zero_division=0)),
            "per_class": per_class,
            "confusion_rows_true_columns_predicted_left_right_unknown": confusion_matrix(truths, guessed, labels=["left", "right", "unknown"]).tolist()}


def landing_metrics(events, predictions):
    matched = missed = extra = side_correct = 0
    count_errors = []
    available = 0
    rows = []
    for event, pred in zip(events, predictions):
        expected = event["landings"]
        guessed = pred["landings"]
        comparison = match_landings(guessed or [], expected, 0.08 * event["fps"])
        matched += comparison["true_positive"]
        missed += comparison["false_negative"]
        extra += comparison["false_positive"]
        side_correct += sum(pair["predicted"]["foot"] == pair["reference"]["foot"]
                            for pair in comparison["pairs"])
        if guessed is not None:
            available += 1
            error = abs(len(guessed) - len(expected))
            count_errors.append(error)
        rows.append({"clip_id": event["clip_id"], "available": guessed is not None,
                     "reference_count": len(expected), "prediction_count": len(guessed) if guessed is not None else None,
                     "matching": comparison})
    precision = matched / (matched + extra) if matched + extra else 0.0
    recall = matched / (matched + missed) if matched + missed else 0.0
    return {"events": len(events), "output_events": available,
            "reference_landings": matched + missed, "matched": matched, "false_positive": extra,
            "false_negative_including_unavailable_events": missed, "precision": precision,
            "matching_is_temporal_only_for_compatibility_with_frozen_baseline": True,
            "correct_side_among_temporally_matched": side_correct,
            "side_accuracy_on_temporal_matches": side_correct / matched if matched else None,
            "recall_all_reference_events": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "count_mae_available": float(np.mean(count_errors)) if count_errors else None,
            "within_one_count_all_events": sum(e <= 1 for e in count_errors) / len(events),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/fresh_course_experiment"
    parser.add_argument("--dense", type=Path, default=base / "shoe_approach33_cohort_v1")
    parser.add_argument("--references", type=Path, default=base / "human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json")
    parser.add_argument("--baseline", type=Path, default=base / "shoe_landing_replay_2026-10-18_v1")
    parser.add_argument("--pivot", type=Path, default=base / "pivot_support_development_2026-10-22")
    parser.add_argument("--foot-supplement", type=Path, default=base / "shoe_approach_foot_supplement_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/shoe_supervised_2026-10-26/model_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output already exists; preserve old runs and choose a new directory")
    started = time.perf_counter()
    events, bindings = load_events(args.dense, args.references, args.baseline, args.pivot)
    supplement = read(args.foot_supplement / "summary.json")
    supplement_predictions = {}
    for row in supplement["rows"]:
        path = args.foot_supplement / "results" / f"{row['clip_id']}.json"
        if sha(path) != row["result_sha256"]:
            raise ValueError("Frozen foot supplement hash mismatch")
        supplement_predictions[row["clip_id"]] = {"striking_foot": row["candidate"]}
    if set(supplement_predictions) != {e["clip_id"] for e in events}:
        raise ValueError("Foot supplement must contain the same 39 events")
    args.output.mkdir(parents=True)
    predictions = {}
    fold_audits = []
    with threadpool_limits(limits=2):
        for group in sorted({e["match_group"] for e in events}):
            model = fit_models(events, held_out_match=group)
            validation = [e for e in events if e["match_group"] == group]
            for event in validation:
                predictions[event["clip_id"]] = predict_event(model, event["evidence"], event["fps"])
            joblib.dump(model, args.output / f"fold_{group}.joblib")
            fold_audits.append({"held_out_match": group,
                "validation_ids": [e["clip_id"] for e in validation],
                "training_landing_ids": model["training_landing_ids"], "training_foot_ids": model["training_foot_ids"]})
        deployment = fit_models(events)
        deployment["evidence_contract"] = {
            "pose_model_sha256": sorted({b["pose_model_sha256"] for b in bindings}),
            "feature_code_sha256": sha(ROOT / "src/shoe_supervised_model.py"),
            "coordinate_space": "original_source_image_pixels",
            "landmarks": "mediapipe_33_normal_and_inverse_mapped_mirror",
            "automatic_windows_only": True, "labels_required_for_prediction": False}
        joblib.dump(deployment, args.output / "deployment.joblib")
    ordered = [predictions[e["clip_id"]] for e in events]
    comparisons = {"learned": ordered, "frozen_rule": [e["baseline"] for e in events],
                   "frozen_pivot": [e["pivot"] for e in events]}
    common = [i for i in range(len(events)) if all(p[i].get("landings") is not None for p in comparisons.values())]
    report = {"schema_version": "shoe-supervised-development-v1", "events": len(events),
        "evaluation": "four_fold_leave_one_match_out_previously_inspected_development",
        "held_out_test": False, "new_neural_inference_calls": 0, "original_outputs_modified": False,
        "config": asdict(ShoeConfig()), "reference_sha256": sha(args.references),
        "evidence_bindings": bindings, "folds": fold_audits,
        "feature_extraction_and_training_seconds": time.perf_counter() - started,
        "foot_reference_counts": dict(Counter(e["foot"] for e in events)),
        "foot": {name: foot_metrics(events, pred) for name, pred in comparisons.items() if name != "frozen_pivot"},
        "landings": {name: landing_metrics(events, pred) for name, pred in comparisons.items()},
        "landing_common_event_ids": [events[i]["clip_id"] for i in common],
        "landings_common_events": {name: landing_metrics([events[i] for i in common], [pred[i] for i in common])
                                   for name, pred in comparisons.items()},
        "deployment_training_landing_ids": deployment["training_landing_ids"],
        "deployment_training_foot_ids": deployment["training_foot_ids"],
        "limitations": ["Single human reviewer; previously inspected development events.",
            "Only three left-foot references; one has no saved dense input window.",
            "Negative landing samples mean absent from supplied list; completeness was not independently certified for every list.",
            "Observed-window landing counts are not full run-up counts.",
            "Missing input events remain unavailable and in the all-event denominator.",
            "Pose labels are model predictions; supervised head does not repair wrong actor or anatomical channels.",
            "Source label clocks are bound by hashes; no labels or human timing enter feature extraction."]}
    report["foot"]["always_right_control"] = foot_metrics(events, [{"striking_foot": "right"} for _ in events])
    report["foot"]["latest_frozen_supplement"] = foot_metrics(events, [supplement_predictions[e["clip_id"]] for e in events])
    report["dependencies"] = {name: version(name) for name in ("numpy", "scikit-learn", "joblib")}
    report["code_sha256"] = {p: sha(ROOT / p) for p in ("src/shoe_supervised_model.py", "scripts/train_shoe_supervised.py")}
    write(args.output / "report.json", report)
    write(args.output / "deployment_metadata.json", {
        "model_sha256": sha(args.output / "deployment.joblib"), "evidence_contract": deployment["evidence_contract"],
        "dependencies": report["dependencies"], "reference_sha256": report["reference_sha256"],
        "development_evaluation": report["evaluation"], "training_landing_ids": deployment["training_landing_ids"],
        "training_foot_ids": deployment["training_foot_ids"], "config": report["config"]})
    write(args.output / "oof_predictions.json", {"events": [{"clip_id": e["clip_id"], "match_group": e["match_group"],
        "source_sha256": e["source_sha256"], "automatic_window": e["automatic_window"],
        "supported_side_frames": e["evidence"]["supported_side_frames"], **predictions[e["clip_id"]]} for e in events]})
    print(json.dumps({"output": str(args.output), "foot": report["foot"],
        "landings": {k: {a: b for a, b in v.items() if a != "rows"} for k, v in report["landings"].items()}}, indent=2))


if __name__ == "__main__":
    main()
