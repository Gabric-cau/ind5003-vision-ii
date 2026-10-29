"""Summarize a completed ball-detector round without model inference or label loading."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.train_ball_frame_detector import summarize


GROUPS = {"arsenal_new_match": "ARS New Match", "core_development_retest": "Old Core Development Retest"}
MODELS = ("pretrained", "previous_refit", "new_round1")
MODES = ("full", "tiles")
SCOPES = ("all_frames", "deduplicated_frames")
LOSS_FIELDS = ("train/box_loss", "train/cls_loss", "train/dfl_loss")


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Round is incomplete: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def indexed_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prediction rows must be a nonempty list")
    indexed = {}
    for row in rows:
        key = (row["video_id"], row["annotation_id"])
        if key in indexed or not all(isinstance(value, str) and value for value in (*key, row["event_id"])):
            raise ValueError("Frame and event identities must be present and unique")
        if not isinstance(row["deduplicated_primary"], bool) or not isinstance(row["prediction_available"], bool):
            raise ValueError("Availability and deduplication flags must be booleans")
        iou = row["top1_iou"]
        if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not math.isfinite(iou) or not 0 <= iou <= 1:
            raise ValueError("Invalid top-ranked IoU")
        if row["prediction_available"] != bool(row["candidates"]) or (not row["prediction_available"] and iou != 0):
            raise ValueError("Missing prediction must remain a miss")
        indexed[key] = row
    return indexed


def same_frames(first, second):
    if first.keys() != second.keys():
        raise ValueError("Prediction frame identities differ between comparisons")
    for key in first:
        for field in ("image", "event_id", "reference_bbox", "deduplicated_primary"):
            if first[key][field] != second[key][field]:
                raise ValueError(f"Frame binding differs: {key}: {field}")


def selected_rows(indexed, scope):
    return [row for row in indexed.values() if scope == "all_frames" or row["deduplicated_primary"]]


def compact_metrics(rows):
    metrics = summarize(rows)
    return {"hits": metrics["top1_iou_0_5_hits"], "denominator": metrics["annotated_frames"],
            "recall": metrics["top1_localization_recall_iou_0_5"],
            "prediction_available_frames": metrics["prediction_available_frames"]}


def paired_changes(previous, new, scope):
    same_frames(previous, new)
    events = defaultdict(lambda: {"denominator": 0, "both_correct": 0, "corrected": 0,
                                  "regressed": 0, "both_incorrect": 0})
    for key, old in previous.items():
        if scope == "deduplicated_frames" and not old["deduplicated_primary"]:
            continue
        current = new[key]
        before, after = old["top1_iou"] >= .5, current["top1_iou"] >= .5
        change = ("both_correct" if after else "regressed") if before else ("corrected" if after else "both_incorrect")
        event = events[old["event_id"]]
        event["denominator"] += 1
        event[change] += 1
    for event in events.values():
        event["previous_hits"] = event["both_correct"] + event["regressed"]
        event["new_hits"] = event["both_correct"] + event["corrected"]
        event["net_hit_change"] = event["corrected"] - event["regressed"]
    return dict(sorted(events.items()))


def read_losses(path, expected_epochs):
    if not path.is_file():
        raise FileNotFoundError(f"Round is incomplete: missing {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Training loss CSV has no header")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        if not {"epoch", *LOSS_FIELDS} <= set(reader.fieldnames):
            raise ValueError("Training loss CSV is missing required fields")
        rows = []
        for record in reader:
            epoch = float(record["epoch"])
            if not epoch.is_integer():
                raise ValueError("Invalid loss epoch")
            losses = {field: float(record[field]) for field in LOSS_FIELDS}
            if not all(math.isfinite(value) and value >= 0 for value in losses.values()):
                raise ValueError("Training loss must be finite and nonnegative")
            rows.append({"epoch": int(epoch), **losses})
    if [row["epoch"] for row in rows] != expected_epochs:
        raise ValueError("Training loss CSV is incomplete or contains duplicate/out-of-order epochs")
    return rows


def build_summary(directory):
    report_path = directory / "report.json"
    report = read_json(report_path)
    if report.get("training_completed") is not True or report.get("detector_weights_updated") is not True:
        raise ValueError("Round is incomplete: completed training and weight updates are not confirmed")
    protocol, candidate = report["protocol"], report["candidate"]
    epochs = protocol["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Protocol must specify a positive epoch count")
    expected_epochs = list(range(1, epochs + 1))
    if candidate["completed_epochs"] != expected_epochs:
        raise ValueError("Candidate does not confirm every planned training epoch")
    if set(report["results"]) != set(GROUPS) or protocol["evaluation_modes"] != list(MODES):
        raise ValueError("Unexpected evaluation groups or inference modes")
    losses_path = directory / "fit/results.csv"
    losses = read_losses(losses_path, expected_epochs)
    expected_weights = {"pretrained": protocol["initial_weights_sha256"],
                        "previous_refit": protocol["previous_weights_sha256"],
                        "new_round1": candidate["weights_sha256"]}
    sources = {"report.json": sha(report_path), "fit/results.csv": sha(losses_path)}
    groups = {}
    for group in GROUPS:
        results, bound_rows, comparisons = report["results"][group], {}, []
        if set(results) != set(MODELS):
            raise ValueError("Every group requires all three model comparisons")
        reference = None
        for model in MODELS:
            if set(results[model]) != set(MODES):
                raise ValueError("Every model requires both inference modes")
            for mode in MODES:
                filename = f"{group}_{model}_{mode}.json"
                path = directory / filename
                prediction = read_json(path)
                if {key: value for key, value in prediction.items() if key != "rows"} != results[model][mode]:
                    raise ValueError(f"Prediction and final report disagree: {filename}")
                if prediction["mode"] != mode or prediction["weights_sha256"] != expected_weights[model]:
                    raise ValueError(f"Model or mode binding mismatch: {filename}")
                elapsed = prediction["seconds_including_model_load"]
                if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Inference elapsed time is invalid")
                indexed = indexed_rows(prediction["rows"])
                if reference is not None:
                    same_frames(reference, indexed)
                reference = indexed
                metrics = {}
                for scope in SCOPES:
                    selected = selected_rows(indexed, scope)
                    if summarize(selected) != prediction["metrics"][scope]:
                        raise ValueError(f"Reported metrics disagree with prediction rows: {filename}: {scope}")
                    metrics[scope] = compact_metrics(selected)
                comparisons.append({"model": model, "mode": mode, **metrics,
                                    "seconds_including_model_load": elapsed, "prediction_file": filename})
                bound_rows[model, mode] = indexed
                sources[filename] = sha(path)
        expected_frames = protocol["new_match_evaluation_frames" if group == "arsenal_new_match" else "regression_frames"]
        if len(reference) != expected_frames:
            raise ValueError("Evaluation frame count differs from the frozen protocol")
        groups[group] = {"all_frames": len(reference),
                         "deduplicated_frames": len(selected_rows(reference, "deduplicated_frames")),
                         "event_count": len({row["event_id"] for row in reference.values()}),
                         "comparisons": comparisons,
                         "paired_previous_to_new": {mode: {scope: paired_changes(
                             bound_rows["previous_refit", mode], bound_rows["new_round1", mode], scope)
                             for scope in SCOPES} for mode in MODES}}
    return {"schema_version": "external-ball-round-summary-v1", "source_directory": str(directory.resolve()),
            "source_sha256": sources, "training_completed": True,
            "training": {"epochs": epochs, "frames": protocol["training_frames"],
                         "seconds": candidate["training_seconds"], "losses": losses},
            "groups": groups, "overall_project_accuracy_measured": False,
            "metric": "Top-ranked ball box IoU >= 0.5 on selected positive frames; missing prediction is a miss",
            "limitations": protocol["limitations"],
            "paired_comparison": "Same event/frame and mode, previous_refit to new_round1; frame-level changes, not event accuracy",
            "timing_scope": "Whole prediction run including model load; no per-event timing was measured"}


def markdown(summary):
    lines = ["# Ball Detector Round Summary", "", summary["metric"] + ".",
             "This is detector localization recall, not overall project accuracy or false-positive precision.", "",
             f"Training: {summary['training']['epochs']} epochs, {summary['training']['frames']} frames, "
             f"{summary['training']['seconds']:.2f} seconds.", "",
             "![Training loss](training_loss.png)", "",
             "Only training losses are plotted; validation/mAP placeholders in the training CSV are not evaluation results."]
    for group, data in summary["groups"].items():
        lines += ["", "## " + GROUPS[group], "",
                  f"{data['all_frames']} annotated frames, {data['deduplicated_frames']} deduplicated frames, {data['event_count']} events.", "",
                  "| Model | Mode | All hits/N | Recall | Available | Dedup hits/N | Recall | Available | Run seconds |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in data["comparisons"]:
            all_frames, unique = row["all_frames"], row["deduplicated_frames"]
            lines.append(f"| {row['model']} | {row['mode']} | {all_frames['hits']}/{all_frames['denominator']} | "
                         f"{all_frames['recall']:.2%} | {all_frames['prediction_available_frames']} | "
                         f"{unique['hits']}/{unique['denominator']} | {unique['recall']:.2%} | "
                         f"{unique['prediction_available_frames']} | {row['seconds_including_model_load']:.2f} |")
        lines += ["", "### Paired Frame Changes", "", "previous_refit -> new_round1, matched within each event and inference mode.", "",
                  "| Mode | Frame scope | Event | N | Old hits | New hits | Both correct | Corrected | Regressed | Both incorrect | Net hits |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for mode, scopes in data["paired_previous_to_new"].items():
            for scope, events in scopes.items():
                for event, row in events.items():
                    lines.append(f"| {mode} | {scope} | {event} | {row['denominator']} | {row['previous_hits']} | "
                                 f"{row['new_hits']} | {row['both_correct']} | {row['corrected']} | {row['regressed']} | "
                                 f"{row['both_incorrect']} | {row['net_hit_change']:+d} |")
    lines += ["", "## Interpretation", "", summary["timing_scope"] + ".", ""]
    lines += ["- " + limitation for limitation in summary["limitations"]]
    return "\n".join(lines) + "\n"


def write_summary(directory, output):
    summary = build_summary(directory)
    if output.exists():
        raise FileExistsError("Choose a new summary output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for field in LOSS_FIELDS:
        axis.plot([row["epoch"] for row in summary["training"]["losses"]],
                  [row[field] for row in summary["training"]["losses"]], label=field)
    axis.set(xlabel="Epoch", ylabel="Training loss", title="Training loss (fit only)")
    axis.grid(alpha=.25)
    axis.legend()
    figure.savefig(output / "training_loss.png", dpi=180)
    plt.close(figure)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed round directory containing report.json")
    parser.add_argument("--output", type=Path, help="New output directory; defaults to INPUT/summary")
    args = parser.parse_args()
    directory = args.input.resolve()
    output = args.output.resolve() if args.output else directory / "summary"
    try:
        summary = write_summary(directory, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot summarize this round: {error}\n")
    print(json.dumps({"output": str(output), "groups": {group: {key: value[key] for key in
        ("all_frames", "deduplicated_frames", "event_count")} for group, value in summary["groups"].items()},
        "overall_project_accuracy_measured": False}, indent=2))


if __name__ == "__main__":
    main()
