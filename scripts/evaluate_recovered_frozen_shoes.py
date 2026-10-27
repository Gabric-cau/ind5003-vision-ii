"""Isolate input-recovery gains while keeping the original shoe fold models frozen."""
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_shoe_supervised import read, write, sha, load_events, foot_metrics, landing_metrics
from scripts.train_recovered_shoes import validated_folds, events_for_contact_weights, restrict_scoring_to_review
from src.contact_temporal_model import validate_prediction_bindings
from src.shoe_supervised_model import predict_event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--shoe-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Frozen-head control already exists")
    base = ROOT / "outputs/fresh_course_experiment"
    references = base / "human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json"
    events, bindings = load_events(base / "shoe_approach33_cohort_v1", references,
        base / "shoe_landing_replay_2026-10-18_v1", base / "pivot_support_development_2026-10-22")
    metadata = read(args.contact_model / "deployment_metadata.json")
    recovered = {}
    for event in events:
        directory = args.recovered / event["clip_id"]
        if not (directory / "shoes/inputs.json").exists():
            continue
        inputs = read(directory / "shoes/inputs.json")
        receipt = read(directory / "recovery_input.json")
        record = read(directory / "contact/features.json")
        if (event["automatic_window"] is not None or inputs["reference_labels_read"] is not False
                or inputs["source_mapping"]["clip_sha256"] != event["source_sha256"]
                or inputs["poses33_sha256"] != sha(directory / "shoes/poses33.json")
                or record["video_sha256"] != receipt["lossless_shot"]["shot_sha256"]
                or inputs["inference_interval_frames"] != [inputs["scene_interval_frames"][0], inputs["scene_interval_frames"][1] - 1]):
            raise ValueError("Frozen-head recovery evidence binding mismatch")
        validate_prediction_bindings(record, metadata)
        recovered[event["clip_id"]] = {"inputs": inputs, "record": record, "receipt": receipt,
            "poses": read(directory / "shoes/poses33.json")}
    folds = validated_folds(args.contact_model, ROOT / "outputs/core_contact_training_2026-10-26/data/inputs.json")
    predictions, scores, audit = {}, {}, []
    group_by_id = {e["clip_id"]: e["match_group"] for e in events}
    for fold in folds:
        held_out = fold["held_out_match"]
        model_path = args.shoe_model / f"fold_{held_out}.joblib"
        model = joblib.load(model_path)
        if (model["held_out_match"] != held_out
                or any(group_by_id[cid] == held_out for cid in model["training_foot_ids"] + model["training_landing_ids"])):
            raise ValueError("Frozen shoe model contains validation match")
        with np.load(args.contact_model / fold["weights_file"], allow_pickle=False) as parameters:
            specific = events_for_contact_weights(events, recovered, parameters)
        for event in specific:
            if event["match_group"] != held_out:
                continue
            prediction = predict_event(model, event["evidence"], event["fps"])
            predictions[event["clip_id"]] = {"clip_id": event["clip_id"], "source_sha256": event["source_sha256"],
                "automatic_window": event["automatic_window"], **prediction}
            scores[event["clip_id"]] = restrict_scoring_to_review(event, prediction)
        audit.append({"held_out_match": held_out, "shoe_fold_model_sha256": sha(model_path),
                      "contact_fold_weights_sha256": fold["weights_sha256"],
                      "new_shoe_parameter_updates": False})
    old = {r["clip_id"]: r for r in read(args.shoe_model / "oof_predictions.json")["events"]}
    original = [e for e in events if e["clip_id"] not in recovered]
    for event in original:
        current, previous = predictions[event["clip_id"]], old[event["clip_id"]]
        if (current["striking_foot"] != previous["striking_foot"] or current["landings"] != previous["landings"]):
            raise ValueError("Frozen head changed an original available prediction")
    ordered = [scores[e["clip_id"]] for e in events]
    new_events = [e for e in events if e["clip_id"] in recovered]
    report = {"schema_version": "recovery-with-frozen-original-shoe-folds-v1", "events": 39,
        "reference_sha256": sha(references), "folds": audit,
        "new_parameter_updates": False, "original32_predictions_exactly_unchanged": True,
        "evaluation": "previously_inspected_four_match_development_with_fold_isolated_contact",
        "foot": foot_metrics(events, ordered),
        "foot_new7": foot_metrics(new_events, [scores[e["clip_id"]] for e in new_events]),
        "landings": landing_metrics(events, ordered),
        "scoring_scope": "unchanged human review windows, predictions outside reviewed scope unscored"}
    args.output.mkdir(parents=True)
    write(args.output / "report.json", report)
    write(args.output / "oof_predictions.json", {"events": [predictions[e["clip_id"]] for e in events]})
    print({"foot": report["foot"], "foot_new7": report["foot_new7"],
           "landings": {k: v for k, v in report["landings"].items() if k != "rows"}})


if __name__ == "__main__":
    main()
