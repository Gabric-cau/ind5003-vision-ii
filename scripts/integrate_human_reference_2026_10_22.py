"""Import an unchanged human reference and report single-reviewer diagnostics."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.evaluate_shoe_landings import evaluate, read, sha256
from src.fresh_eda import load_bundle


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/fresh_course_experiment"
FIELDS = ("shot_height", "save_height", "ball_horizontal", "save_direction")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def single_landing_summary(evaluation):
    diagnostics = evaluation["individual_review_diagnostics"]
    eligible = [row for row in diagnostics if "comparison" in row]
    totals = {key: sum(row["comparison"][key] for row in eligible)
              for key in ("true_positive", "false_positive", "false_negative")}
    count_error = sum(row["count_absolute_error"] for row in eligible)
    return {
        "events_with_comparable_predictions": len(eligible),
        "reviewed_events_without_comparable_predictions": len(diagnostics) - len(eligible),
        "prediction_coverage": ratio(len(eligible), len(diagnostics)),
        "reviewed_landings_in_comparable_windows": sum(row["reviewed_visible_count"] for row in eligible),
        "predicted_landings_in_comparable_windows": sum(row["visible_candidate_count"] for row in eligible),
        **totals,
        "precision_against_supplied_list": ratio(totals["true_positive"], totals["true_positive"] + totals["false_positive"]),
        "recall_against_supplied_list": ratio(totals["true_positive"], totals["true_positive"] + totals["false_negative"]),
        "visible_list_count_mae": ratio(count_error, len(eligible)),
        "full_runup_count_mae": None,
        "scope": "One supplied human list; not second-reviewer consensus, held-out accuracy, or verified full run-up counts.",
    }


def categorical_summary(records, prediction_key, reference_key):
    decided = [row for row in records if row[prediction_key] != "unknown" and row[reference_key] != "unknown"]
    correct = sum(row[prediction_key] == row[reference_key] for row in decided)
    return {
        "events": len(records), "compared": len(decided), "agreed": correct,
        "agreement_on_decided": ratio(correct, len(decided)),
        "coverage": ratio(len(decided), len(records)),
        "disagreements": [row["clip_id"] for row in decided if row[prediction_key] != row[reference_key]],
        "scope": "Single-reviewer categorical development agreement; not held-out accuracy.",
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "human_evaluation_2026-10-22")
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = read(source / "SHA256_MANIFEST.json")
    for name, digest in manifest.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != digest:
            raise ValueError(f"Invalid imported artifact: {name}")
    reference_path = source / "penalty_annotations_39_human_v040.json"
    reference = read(reference_path)
    events = {event["clip_id"]: event for event in reference["events"]}
    if len(events) != 39 or len(reference["events"]) != 39:
        raise ValueError("The human reference must contain 39 unique events")
    reviewers = {review["reviewer_id"] for event in events.values() for review in event["reviews"]}
    if len(reviewers) != 1 or reference["independent_human_reviewer_count"] != 1:
        raise ValueError("This diagnostic route is specifically for the supplied single reviewer")
    expected_summary = read(source / "final_annotation_summary.json")
    field_counts = {}
    for field in FIELDS:
        counts = Counter(event[field] for event in events.values())
        recorded = expected_summary["field_summary"][field]["value_counts"]
        if any(counts[value] != count for value, count in recorded.items()):
            raise ValueError(f"Human field totals differ: {field}")
        field_counts[field] = dict(counts)
    if {cid for cid, event in events.items() if event["shot_height"] == "unknown"} != {
        "esp_rus_p09_aspas_esp17", "rus_cro_p05_fernandes_rus2"
    }:
        raise ValueError("The explicit outside-goal recoding policy has changed")
    rows, _, _ = load_bundle(EXPERIMENT / "final_bundle_v3_shoe_fusion")
    if {row["clip_id"] for row in rows} != set(events):
        raise ValueError("Human and prediction cohorts differ")
    for row in rows:
        if row["source_sha256"] != events[row["clip_id"]]["source_sha256"]:
            raise ValueError("Human reference uses different video bytes")
    supplement_dir = EXPERIMENT / "shoe_approach_foot_supplement_v2"
    supplement = read(supplement_dir / "summary.json")
    feet = {}
    for item in supplement["rows"]:
        if sha256(supplement_dir / "results" / f"{item['clip_id']}.json") != item["result_sha256"]:
            raise ValueError("Foot supplement bytes have changed")
        feet[item["clip_id"]] = item["candidate"]
    if set(feet) != set(events):
        raise ValueError("Foot supplement cohort differs")
    freeze = EXPERIMENT / "shoe_landing_replay_2026-10-18_v1/prediction_freeze.json"
    evaluation = evaluate(freeze, reference_path)
    args.output.mkdir(parents=True, exist_ok=True)
    imported = args.output / "reference_source"
    imported.mkdir(exist_ok=True)
    for name in [*manifest, "SHA256_MANIFEST.json"]:
        destination = imported / name
        if destination.exists() and sha256(destination) != sha256(source / name):
            raise ValueError(f"Refusing to replace a different imported reference: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != (source / name).resolve():
            shutil.copyfile(source / name, destination)
    direction = {"image_left": "left", "image_right": "right", "center": "center", "unknown": "unknown"}
    comparisons = []
    for row in rows:
        event = events[row["clip_id"]]
        review = event["reviews"][0]
        if event["_shot_save_annotation"]["direction_coordinate_system"] != "source_image_x_right":
            raise ValueError("Keeper direction labels must use the source-image convention")
        comparisons.append({
            "clip_id": row["clip_id"], "event_id": row["event_id"],
            "source_sha256": event["source_sha256"], "reviewer_id": review["reviewer_id"],
            "predicted_striking_foot": feet[row["clip_id"]], "human_striking_foot": review["striking_foot"],
            "predicted_keeper_direction": direction[row["keeper_direction_image_candidate"]],
            "human_keeper_direction": event["save_direction"],
            "human_shot_height": event["shot_height"], "human_save_height": event["save_height"],
            "human_ball_horizontal": event["ball_horizontal"],
            "human_in_window_landings": len(review["verified_landings"]),
            "human_outside_window_observations": len(review.get("outside_window_human_observations", [])),
            "prediction_is_human_label": False,
        })
    with (args.output / "per_event_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    summary = {
        "schema_version": "vision2-single-human-reference-integration-v1", "date": "2026-10-22",
        "reviewer_count": 1, "reviewer_ids_as_declared": sorted(reviewers),
        "reviewed_events": evaluation["reviewed_events"], "human_field_cells_completed": 156,
        "source_artifact_hashes_verified": len(manifest), "source_manifest_sha256": sha256(source / "SHA256_MANIFEST.json"),
        "reference_sha256": sha256(reference_path), "prediction_freeze_sha256": sha256(freeze),
        "source_video_hashes_matched": 39, "source_reference_unchanged": True,
        "human_in_window_landings": sum(row["human_in_window_landings"] for row in comparisons),
        "human_outside_window_observations": sum(row["human_outside_window_observations"] for row in comparisons),
        "field_counts": field_counts,
        "single_reviewer_landing_diagnostics": single_landing_summary(evaluation),
        "supplement_striking_foot": categorical_summary(comparisons, "predicted_striking_foot", "human_striking_foot"),
        "keeper_image_direction": categorical_summary(comparisons, "predicted_keeper_direction", "human_keeper_direction"),
        "jointly_reviewed_events": evaluation["jointly_reviewed_landing_metrics"]["evaluated_events"],
        "held_out_test": False, "second_reviewer_consensus": False, "new_neural_inference_calls": 0,
        "automatic_features_replaced_with_human_labels": False,
        "limitations": [
            "The supplied package attributes labels to one human; this import does not independently authenticate identity or viewing history.",
            "The annotator had seen aggregate candidate counts; review was not fully blinded.",
            "Most full-approach visibility assessments were AI-assisted, not an additional human review.",
            "Landing diagnostics compare the supplied lists within frozen windows; list completeness is not separately human-confirmed for every clip.",
            "Qualitative shot/save labels have no exact evidence frame or metric-height calibration.",
            "Two outside-goal shots are reviewed unknown labels, not unfinished annotation.",
        ],
    }
    write_json(args.output / "evaluation_consensus.json", evaluation)
    write_json(args.output / "integration_summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    foot_metrics = summary["supplement_striking_foot"]
    keeper_metrics = summary["keeper_image_direction"]
    for ax, prediction_key, reference_key, labels, title in (
        (axes[0], "predicted_striking_foot", "human_striking_foot", ["left", "right"], f"Foot supplement: {foot_metrics['agreed']} / {foot_metrics['compared']} decided agree"),
        (axes[1], "predicted_keeper_direction", "human_keeper_direction", ["left", "center", "right"], f"Keeper direction: {keeper_metrics['agreed']} / {keeper_metrics['compared']} decided agree"),
    ):
        predicted_labels = labels + ["unknown"]
        matrix = [[sum(row[reference_key] == label and row[prediction_key] == prediction for row in comparisons)
                   for prediction in predicted_labels] for label in labels]
        ax.imshow(matrix, cmap="Greens", vmin=0)
        for i, values in enumerate(matrix):
            for j, value in enumerate(values):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=13)
        ax.set_xticks(range(len(predicted_labels)), predicted_labels, rotation=20)
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlabel="Automatic candidate", ylabel="Supplied human label", title=title)
    fig.text(.06, .02, "One reviewer, same development videos. Unknown predictions remain visible. Not held-out accuracy.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(args.output / "single_reviewer_categorical_comparison.png", dpi=180)
    plt.close(fig)
    eligible = [row for row in evaluation["individual_review_diagnostics"] if "comparison" in row]
    order = {row["clip_id"]: row["event_id"] for row in rows}
    eligible.sort(key=lambda row: order[row["clip_id"]])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = [order[row["clip_id"]] for row in eligible]
    ax.bar([value - .2 for value in x], [row["reviewed_visible_count"] for row in eligible], width=.4, label="Human supplied list", color="#527d9f")
    ax.bar([value + .2 for value in x], [row["visible_candidate_count"] for row in eligible], width=.4, label="Automatic proposals", color="#bd644e")
    ax.set(xlabel="Canonical event ID (27 comparable windows)", ylabel="Within-window landings", title="The repaired rule still misses most human-listed landings")
    ax.legend()
    fig.text(.08, .015, "12 unavailable predictions excluded; not zero-filled. 0.08 s matching tolerance. No full-run-up accuracy claim.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(args.output / "single_reviewer_landing_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
