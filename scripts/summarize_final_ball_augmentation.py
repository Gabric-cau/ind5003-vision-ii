"""Audit the completed final ball comparison and select a candidate on CPU only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.summarize_ball_controlled_round2 import counts, verified_rows, write_csv
from scripts.summarize_external_ball_round import (
    SCOPES, paired_changes, read_json, read_losses, same_frames, selected_rows, sha,
)
from scripts.train_ball_frame_detector import summarize

ARMS = ("control_full", "augmented_mixed")
MODELS = ("initializer", *ARMS)
MODES = ("full", "tiles")
GROUPS = {"core_development": (101, 91, 65), "arsenal_development": (88, 88, 62)}
STEPS = [130, 167, 195, 222, 250, 278, 305, 333, 361, 388]
EPOCHS = list(range(1, 11))


def sequence_sha(names):
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()


def verify_training(directory, report):
    protocol = report["protocol"]
    if read_json(directory / "protocol.json") != protocol:
        raise ValueError("Final report differs from the frozen protocol")
    if protocol["epochs_each"] != 10 or protocol["samples_per_epoch"] != 442 or protocol["expected_optimizer_steps_per_epoch"] != STEPS:
        raise ValueError("Final training budget changed")
    if sha(directory / "schedules.json") != protocol["schedule_sha256"]:
        raise ValueError("Frozen source schedule changed")
    if sha(directory / "dataset/manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("Frozen dataset manifest changed")
    schedules = read_json(directory / "schedules.json")
    records = read_json(directory / "dataset/manifest.json")["exports"]
    if set(report["candidates"]) != set(ARMS) or any(len(schedules[key]) != 10 for key in ("source", *ARMS)):
        raise ValueError("Both complete frozen training arms are required")
    if sha(Path(protocol["initial_weights"])) != protocol["initial_sha256"]:
        raise ValueError("Initializer checkpoint SHA mismatch")
    training = {}
    for arm in ARMS:
        candidate = report["candidates"][arm]
        if candidate != read_json(directory / arm / "selected_candidate.json"):
            raise ValueError("Final report differs from the arm receipt")
        if candidate["completed_epochs"] != EPOCHS or candidate["optimizer_steps"] != 388:
            raise ValueError("Incomplete epochs or optimizer updates")
        if candidate["initial_sha256"] != protocol["initial_sha256"] or sha(Path(candidate["weights"])) != candidate["weights_sha256"]:
            raise ValueError("Candidate checkpoint SHA mismatch")
        if candidate["weights_sha256"] == protocol["initial_sha256"] or not candidate["changed_parameter_names"] or candidate["maximum_observed_parameter_delta"] <= 0:
            raise ValueError("Actual parameter updates are unverified")
        if candidate["checkpoint_selection"] != "fixed_last_epoch_10" or candidate["current_round_scores_seen_before_checkpoint_freeze"] is not False:
            raise ValueError("Fixed checkpoint selection is unverified")
        sampling = candidate["sampling"]
        if sampling != read_json(directory / arm / "actual_sampling.json") or [r["epoch"] for r in sampling] != EPOCHS:
            raise ValueError("Actual sampling receipt is incomplete or differs")
        for index, row in enumerate(sampling):
            source, variants = schedules["source"][index], schedules[arm][index]
            if len(source) != 442 or [records[name]["source_name"] for name in variants] != source:
                raise ValueError("Variant schedule changes the paired source sequence")
            if row["source_sequence_sha256"] != sequence_sha(source) or row["variant_sequence_sha256"] != sequence_sha(variants):
                raise ValueError("Actual model input sequence differs from frozen schedule")
            if (row["samples"], row["old"], row["new"], row["optimizer_steps"]) != (442, 221, 221, STEPS[index]):
                raise ValueError("Actual source or optimizer budget differs")
            checks = row["batch_checks"]
            if (checks["batches"], checks["remaining_targets"], checks["empty_after_framework_augmentation"]) != (442, 442, 0):
                raise ValueError("Positive training targets were not preserved")
            expected_photo = (index + 1) * 442 if arm == "augmented_mixed" else 0
            if row["photometric_calls_cumulative"] != expected_photo:
                raise ValueError("Photometric intervention receipt differs from budget")
        expected_photo = 4420 if arm == "augmented_mixed" else 0
        if candidate["photometric_transform_calls"] != expected_photo:
            raise ValueError("Photometric intervention did not complete")
        training[arm] = {"epochs": 10, "optimizer_steps": 388,
                         "training_seconds": candidate["training_seconds"], "sampling": sampling,
                         "losses": read_losses(directory / arm / "fit/results.csv", EPOCHS)}
    return training


def event_metrics(indexed):
    grouped = defaultdict(list)
    for row in indexed.values():
        grouped[row["event_id"]].append(row)
    original = {event: {scope: summarize([row for row in rows if scope == "all_frames" or row["deduplicated_primary"]])
                        for scope in SCOPES} for event, rows in grouped.items()}
    per_event = {event: counts([row for row in rows if row["deduplicated_primary"]])
                 for event, rows in sorted(grouped.items())}
    macro = sum(value["recall"] for value in per_event.values()) / len(per_event)
    return original, per_event, macro


def choose_candidate(groups):
    def metric(group, model):
        return next(row["deduplicated_frames"] for row in groups[group]["comparisons"]
                    if row["model"] == model and row["mode"] == "tiles")

    def mean(model):
        return sum((Fraction(metric(group, model)["hits"], metric(group, model)["denominator"])
                    for group in GROUPS), Fraction()) / len(GROUPS)

    reasons, historical, eligible = {}, [], []
    for arm in ARMS:
        failures = []
        passes_history = all(metric(group, arm)["hits"] >= threshold for group, (_, _, threshold) in GROUPS.items())
        if passes_history:
            historical.append(arm)
        else:
            failures.append("Does not reach both historical gates: core 65/91 and ARS 62/88")
        for group in GROUPS:
            if metric(group, arm)["hits"] < metric(group, "initializer")["hits"]:
                failures.append(f"Regresses on {group} versus this run's initializer")
        if mean(arm) <= mean("initializer"):
            failures.append("Equal-group mean does not strictly improve over this run's initializer")
        reasons[arm] = failures or ["Passes both gates, no group regresses, and equal-group mean strictly improves"]
        if not failures:
            eligible.append(arm)
    selected = max(eligible, key=lambda model: (mean(model), model == "control_full")) if eligible else "initializer"
    historical_selected = max(historical, key=mean) if historical else None
    return {"selected_model": selected, "eligible_new_models": eligible, "reasons": reasons,
            "equal_group_mean_recall": {model: float(mean(model)) for model in MODELS},
            "runner_historical_decision": {"eligible_arms": historical, "selected_candidate": historical_selected},
            "rule": "Tiles: historical dual gate; no group below current initializer; strictly higher equal-group mean; new-arm ties prefer control; otherwise keep initializer",
            "selection_status": "development_candidate_pending_full_video_smoke", "default_pipeline_changed": False,
            "independent_test_evaluated": False, "overall_project_accuracy_measured": False}


def build_summary(directory):
    directory = Path(directory).resolve()
    report = read_json(directory / "report.json")
    if report.get("training_completed") is not True:
        raise ValueError("Round is incomplete: final training is not confirmed")
    if report.get("independent_test_evaluated") is not False or report.get("overall_project_accuracy_measured") is not False:
        raise ValueError("This comparison must remain exposed development localization")
    training = verify_training(directory, report)
    protocol = report["protocol"]
    expected_hash = {"initializer": protocol["initial_sha256"], **{arm: report["candidates"][arm]["weights_sha256"] for arm in ARMS}}
    if set(report["results"]) != set(GROUPS):
        raise ValueError("Missing or unexpected evaluation group")
    sources = {str(directory / "report.json"): sha(directory / "report.json")}
    groups = {}
    for group, (all_count, unique_count, _) in GROUPS.items():
        if set(report["results"][group]) != set(MODELS):
            raise ValueError("Every group requires initializer and both new candidates")
        comparisons, bound, reference = [], {}, None
        for model in MODELS:
            if set(report["results"][group][model]) != set(MODES):
                raise ValueError("Both frozen inference modes are required")
            for mode in MODES:
                path = directory / f"{group}_{model}_{mode}.json"
                prediction = read_json(path)
                if {key: value for key, value in prediction.items() if key != "rows"} != report["results"][group][model][mode]:
                    raise ValueError("Raw prediction and final report differ")
                if prediction["weights_sha256"] != expected_hash[model] or prediction["mode"] != mode:
                    raise ValueError("Evaluation checkpoint or mode binding differs")
                if prediction["prediction_uses_reference"] is not False or (prediction["image_size"], prediction["confidence"], prediction["nms_iou"], prediction["class_id"]) != (1280, .05, .6, 0):
                    raise ValueError("Frozen reference-free inference settings differ")
                if prediction["inference_code_sha256"] != protocol["code_sha256"]["src/ball_specialist.py"]:
                    raise ValueError("Frozen inference code differs")
                indexed = verified_rows(prediction)
                if reference is not None:
                    same_frames(reference, indexed)
                reference, bound[model, mode] = indexed, indexed
                original, by_event, macro = event_metrics(indexed)
                if prediction["metrics"]["by_event"] != original or not math.isclose(prediction["metrics"]["macro_event_localization_recall_iou_0_5"], macro, abs_tol=1e-12):
                    raise ValueError("Stored event metrics differ from raw predictions")
                elapsed = prediction["seconds_including_model_load"]
                if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Invalid inference elapsed time")
                comparisons.append({"model": model, "mode": mode,
                    **{scope: counts(selected_rows(indexed, scope)) for scope in SCOPES},
                    "by_event": by_event, "macro_event_localization_recall_iou_0_5": macro,
                    "seconds_including_model_load": elapsed, "prediction_file": str(path)})
                sources[str(path)] = sha(path)
        if len(reference) != all_count or len(selected_rows(reference, "deduplicated_frames")) != unique_count:
            raise ValueError("Frozen evaluation frame counts differ")
        groups[group] = {"all_frames": all_count, "deduplicated_frames": unique_count,
                         "event_count": len({row["event_id"] for row in reference.values()}), "comparisons": comparisons,
                         "paired_changes_from_initializer": {arm: {mode: paired_changes(bound["initializer", mode], bound[arm, mode], "deduplicated_frames")
                                                                   for mode in MODES} for arm in ARMS}}
    decision = choose_candidate(groups)
    if decision["runner_historical_decision"] != {key: report[key] for key in ("eligible_arms", "selected_candidate")}:
        raise ValueError("Runner selection does not match its frozen historical rule")
    selected = decision["selected_model"]
    decision["weights"] = protocol["initial_weights"] if selected == "initializer" else report["candidates"][selected]["weights"]
    decision["weights_sha256"] = expected_hash[selected]
    return {"schema": "final-ball-augmentation-posthoc/v1", "source_directory": str(directory), "source_sha256": sources,
            "training_completed": True, "training": training, "groups": groups, "release_selection": decision,
            "evaluation_scope": "exposed_development_positive_frame_localization_only",
            "independent_test_evaluated": False, "overall_project_accuracy_measured": False,
            "metric": "Top-ranked ball box IoU >=0.5 on selected positive frames; missing prediction is a miss",
            "oracle_scope": "Any-candidate recall is a reference-selected diagnostic upper bound, not deployed accuracy",
            "timing_scope": "Group inference seconds include model loading; not full-video or full-pipeline latency"}


def markdown(summary):
    decision = summary["release_selection"]
    lines = ["# Final Ball Augmentation Review", "", f"Selected development candidate: `{decision['selected_model']}`.", "",
        summary["metric"] + ".", "", "Exposed development retests only; no independent test or whole-project accuracy is established.", "",
        "| Group | Model | Mode | Dedup hits/N | Recall | Candidate oracle/N | Event macro recall | Run seconds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group, data in summary["groups"].items():
        for row in data["comparisons"]:
            count = row["deduplicated_frames"]
            lines.append(f"| {group} | {row['model']} | {row['mode']} | {count['hits']}/{count['denominator']} | {count['recall']:.2%} | "
                         f"{count['oracle_candidate_hits']}/{count['denominator']} | {row['macro_event_localization_recall_iou_0_5']:.2%} | {row['seconds_including_model_load']:.2f} |")
    lines += ["", "## Selection", "", decision["rule"] + ".", ""]
    lines += [f"- {arm}: " + "; ".join(reasons) + "." for arm, reasons in decision["reasons"].items()]
    lines += ["", "Both arms completed 10 epochs and 388 actual optimizer updates using identical source sequences.", "",
              "The mixed arm tests a combined crop, contrast, brightness and rotation intervention. It does not create new independent events or true 3D viewpoints.", "",
              summary["oracle_scope"] + ". " + summary["timing_scope"] + ".", "",
              "Selection does not change deployment. A source-bound full-video smoke comparison remains required.", ""]
    return "\n".join(lines)


def write_review(directory, output):
    summary = build_summary(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", summary), ("release_selection.json", summary["release_selection"])):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    comparisons, events, paired = [], [], []
    for group, data in summary["groups"].items():
        for row in data["comparisons"]:
            common = {"group": group, "model": row["model"], "mode": row["mode"]}
            comparisons.append({**common, **row["deduplicated_frames"],
                "macro_event_localization_recall_iou_0_5": row["macro_event_localization_recall_iou_0_5"],
                "seconds_including_model_load": row["seconds_including_model_load"]})
            events.extend({**common, "event_id": event, **value} for event, value in row["by_event"].items())
        for arm, modes in data["paired_changes_from_initializer"].items():
            for mode, values in modes.items():
                paired.extend({"group": group, "model": arm, "mode": mode, "event_id": event, **value}
                              for event, value in values.items())
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / "event_metrics.csv", events)
    write_csv(output / "paired_event_changes.csv", paired)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = write_review(args.run, args.output)
    print(json.dumps(summary["release_selection"], indent=2))


if __name__ == "__main__":
    main()
