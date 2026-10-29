"""Audit completed controlled ball runs and render post-hoc reviews on CPU only."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.summarize_external_ball_round import (
    LOSS_FIELDS, SCOPES, indexed_rows, paired_changes, read_json, read_losses,
    same_frames, selected_rows, sha,
)
from scripts.train_ball_frame_detector import score_frame, summarize
from scripts.visualize_external_ball_round import image_paths, select_examples, top_candidate


ARMS = ("A_round1_natural", "B_round1_balanced", "C_previous_balanced")
BASELINES = ("previous_refit", "new_round1")
MODELS = BASELINES + ARMS
MODES = ("full", "tiles")
GROUPS = {"arsenal_development": "arsenal_new_match", "core_development": "core_development_retest"}
LABELS = {"previous_refit": "Previous", "new_round1": "Round 1", **dict(zip(ARMS, ("A natural", "B balanced", "C previous")))}


def verified_rows(prediction):
    indexed = indexed_rows(prediction["rows"])
    for row in indexed.values():
        top_candidate(row)
        measured = score_frame(row["reference_bbox"], row["candidates"])
        for field in ("top1_iou", "any_candidate_iou"):
            if not math.isclose(measured[field], row[field], abs_tol=1e-9):
                raise ValueError(f"Candidate-derived {field} disagrees with stored prediction")
        if "candidate_count" in row and row["candidate_count"] != measured["candidate_count"]:
            raise ValueError("Candidate count disagrees with candidate inventory")
    for scope in SCOPES:
        if summarize(selected_rows(indexed, scope)) != prediction["metrics"][scope]:
            raise ValueError(f"Prediction metrics disagree with raw rows: {scope}")
    return indexed


def counts(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty frame scope")
    hits = sum(row["top1_iou"] >= .5 for row in rows)
    oracle = sum(row["any_candidate_iou"] >= .5 for row in rows)
    if oracle < hits:
        raise ValueError("Candidate oracle cannot be below top-ranked hits")
    return {"hits": hits, "denominator": len(rows), "recall": hits / len(rows),
            "oracle_candidate_hits": oracle, "oracle_candidate_recall": oracle / len(rows),
            "candidate_ranking_gap": oracle - hits,
            "no_adequate_candidate_frames": len(rows) - oracle,
            "prediction_available_frames": sum(row["prediction_available"] for row in rows)}


def verify_mps_comparison(protocol, baseline_protocol):
    if protocol["device"] != "mps" or baseline_protocol["device"] != "mps":
        raise ValueError("Historical comparison requires both runs on MPS")
    evaluation = protocol["evaluation"]
    if evaluation["scope"] != "exposed_development_retests" or evaluation["independent_test"] is not False:
        raise ValueError("Round 2 must remain exposed development")
    if not evaluation["all_arms_frozen_before_scoring"] or evaluation["reference_used_in_inference"]:
        raise ValueError("Reference-free frozen-arm comparison is not established")
    same = {"imgsz": baseline_protocol["evaluation_image_size"], "batch": baseline_protocol["evaluation_batch"],
            "confidence": baseline_protocol["confidence"], "nms_iou": baseline_protocol["nms_iou"],
            "modes": baseline_protocol["evaluation_modes"]}
    if any(evaluation[key] != value for key, value in same.items()):
        raise ValueError("Historical MPS inference settings differ")
    return {"same_device": "mps", "same_inference_settings": same,
            "training_hyperparameters_are_not_identical": True,
            "historical_comparison_is_not_a_matched_training_budget_experiment": True}


def verify_receipts(report):
    protocol = report["protocol"]
    if set(report["candidates"]) != set(ARMS) or set(protocol["arms"]) != set(ARMS):
        raise ValueError("All three frozen arms are required")
    expected_epochs = list(range(1, protocol["epochs_each"] + 1))
    for arm, candidate in report["candidates"].items():
        if candidate["completed_epochs"] != expected_epochs or candidate["optimizer_steps"] != protocol["expected_optimizer_steps"]:
            raise ValueError("Incomplete or unequal optimizer budget")
        if not candidate["changed_parameter_names"] or candidate["maximum_observed_parameter_delta"] <= 0:
            raise ValueError("Actual parameter updates are unverified")
        if candidate["round2_scores_seen_before_checkpoint_freeze"] is not False or candidate["checkpoint_selection"] != f"fixed_last_epoch_{protocol['epochs_each']}":
            raise ValueError("Fixed checkpoint selection is unverified")
        expected_initial = protocol["previous_weights_sha256"] if arm == ARMS[2] else protocol["round1_weights_sha256"]
        if candidate["initial_sha256"] != expected_initial:
            raise ValueError("Arm initial checkpoint identity mismatch")
        sampling = candidate["sampling"]
        if [row["epoch"] for row in sampling] != expected_epochs:
            raise ValueError("Actual sampling epochs are incomplete")
        for row in sampling:
            if row["samples"] != protocol["samples_per_epoch"] or row["old"] + row["new"] != row["samples"]:
                raise ValueError("Actual source sampling budget differs")
        if sampling[-1]["optimizer_steps"] != candidate["optimizer_steps"]:
            raise ValueError("Sampling and receipt optimizer counts differ")
    if report["candidates"][ARMS[1]]["sampling"] != report["candidates"][ARMS[2]]["sampling"]:
        raise ValueError("Balanced arms used different observed sampling")
    return expected_epochs


def frozen_decision(groups):
    def metric(group, arm):
        return next(row["deduplicated_frames"] for row in groups[group]["comparisons"]
                    if row["model"] == arm and row["mode"] == "tiles")
    eligible = [arm for arm in ARMS if metric("core_development", arm)["hits"] >= 65
                and metric("arsenal_development", arm)["hits"] >= 62]
    selected = max(eligible, key=lambda arm: metric("core_development", arm)["recall"]
                   + metric("arsenal_development", arm)["recall"]) if eligible else None
    return {"eligible_arms": eligible, "selected_candidate": selected,
            "criterion": "Frozen development rule: tiles core hits >=65 and ARS hits >=62; then equal-group recall mean",
            "independent_performance_estimate": False}


def build_summary(directory, baseline_directory):
    directory, baseline_directory = Path(directory), Path(baseline_directory)
    report_path, baseline_path = directory / "report.json", baseline_directory / "report.json"
    report, baseline = read_json(report_path), read_json(baseline_path)
    if report.get("training_completed") is not True or baseline.get("training_completed") is not True:
        raise ValueError("Both completed final reports are required")
    if report.get("independent_test_evaluated") is not False:
        raise ValueError("Round 2 evaluation scope changed")
    protocol, baseline_protocol = report["protocol"], baseline["protocol"]
    if protocol["round1_report_sha256"] != sha(baseline_path) or report["baseline_results"] != baseline["results"]:
        raise ValueError("Historical MPS report identity changed")
    if protocol["round1_weights_sha256"] != baseline["candidate"]["weights_sha256"] or protocol["previous_weights_sha256"] != baseline_protocol["previous_weights_sha256"]:
        raise ValueError("Historical checkpoint identity differs")
    device_comparison = verify_mps_comparison(protocol, baseline_protocol)
    expected_epochs = verify_receipts(report)
    if set(report["results"]) != set(GROUPS):
        raise ValueError("Unexpected controlled evaluation groups")
    sources = {str(report_path.resolve()): sha(report_path), str(baseline_path.resolve()): sha(baseline_path)}
    training, bound, groups = {}, {}, {}
    for arm in ARMS:
        loss_path = directory / arm / "fit/results.csv"
        training[arm] = {"epochs": len(expected_epochs), "optimizer_steps": report["candidates"][arm]["optimizer_steps"],
                         "training_seconds": report["candidates"][arm]["training_seconds"],
                         "losses": read_losses(loss_path, expected_epochs),
                         "sampling": report["candidates"][arm]["sampling"]}
        sources[str(loss_path.resolve())] = sha(loss_path)
    for group, baseline_group in GROUPS.items():
        if set(report["results"][group]) != set(ARMS):
            raise ValueError("Missing completed arm evaluation")
        comparisons, reference = [], None
        for model in MODELS:
            old = model in BASELINES
            parent, named_group = (baseline_directory, baseline_group) if old else (directory, group)
            declared = baseline["results"][named_group][model] if old else report["results"][group][model]
            if set(declared) != set(MODES):
                raise ValueError("Both frozen inference modes are required")
            expected_hash = (baseline_protocol["previous_weights_sha256"] if model == BASELINES[0]
                             else baseline["candidate"]["weights_sha256"] if model == BASELINES[1]
                             else report["candidates"][model]["weights_sha256"])
            for mode in MODES:
                path = parent / f"{named_group}_{model}_{mode}.json"
                prediction = read_json(path)
                if {key: value for key, value in prediction.items() if key != "rows"} != declared[mode]:
                    raise ValueError(f"Final report and prediction disagree: {path.name}")
                if prediction["weights_sha256"] != expected_hash or prediction["mode"] != mode:
                    raise ValueError("Prediction checkpoint/mode binding differs")
                if prediction["prediction_uses_reference"] is not False:
                    raise ValueError("Prediction used its evaluation reference")
                indexed = verified_rows(prediction)
                if reference is not None:
                    same_frames(reference, indexed)
                reference = indexed
                elapsed = prediction["seconds_including_model_load"]
                if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Invalid inference elapsed time")
                comparisons.append({"model": model, "mode": mode,
                                    **{scope: counts(selected_rows(indexed, scope)) for scope in SCOPES},
                                    "seconds_including_model_load": elapsed, "prediction_file": str(path.resolve()),
                                    "inference_code_sha256": prediction["inference_code_sha256"]})
                bound[group, model, mode] = indexed
                sources[str(path.resolve())] = sha(path)
        if len({row["inference_code_sha256"] for row in comparisons}) != 1:
            raise ValueError("Inference implementation differs across historical and controlled runs")
        expected_count = baseline_protocol["new_match_evaluation_frames" if group == "arsenal_development" else "regression_frames"]
        if len(reference) != expected_count:
            raise ValueError("Evaluation frame inventory differs from frozen baseline")
        groups[group] = {"all_frames": len(reference), "deduplicated_frames": len(selected_rows(reference, "deduplicated_frames")),
                         "event_count": len({row["event_id"] for row in reference.values()}), "comparisons": comparisons,
                         "paired_changes": {base: {arm: {mode: {scope: paired_changes(bound[group, base, mode], bound[group, arm, mode], scope)
                             for scope in SCOPES} for mode in MODES} for arm in ARMS} for base in BASELINES}}
    decision = frozen_decision(groups)
    if decision["eligible_arms"] != report["eligible_arms"] or decision["selected_candidate"] != report["selected_candidate"]:
        raise ValueError("Final report selection disagrees with its frozen rule")
    summary = {"schema": "controlled-ball-round2-posthoc/v1", "training_completed": True,
               "source_sha256": sources, "device_comparison": device_comparison, "training": training,
               "groups": groups, "frozen_decision_verified": decision, "evaluation_scope": "exposed_development_retests",
               "independent_test_evaluated": False, "overall_project_accuracy_measured": False,
               "metric": "Top-ranked ball box IoU >=0.5 on selected positive reference frames; missing prediction is a miss",
               "oracle_scope": "Reference-selected best candidate at the same frozen proposal settings; diagnostic upper bound, not deployed inference",
               "timing_scope": "Prediction run seconds include model loading; no per-event latency was measured",
               "limitations": protocol["limitations"]}
    return summary, bound


def markdown(summary):
    lines = ["# Controlled Ball Development Review", "", summary["metric"] + ".",
             "All comparisons are exposed development retests, not independent tests or overall project accuracy.", "",
             "![Recall and proposal oracle](recall_vs_oracle.png)", "", "![Training losses](training_loss.png)", "",
             "| Group | Model | Mode | Dedup hits/N | Recall | Candidate oracle/N | Ranking gap | All hits/N | Run seconds |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group, data in summary["groups"].items():
        for row in data["comparisons"]:
            unique, all_rows = row["deduplicated_frames"], row["all_frames"]
            lines.append(f"| {group} | {row['model']} | {row['mode']} | {unique['hits']}/{unique['denominator']} | {unique['recall']:.2%} | "
                         f"{unique['oracle_candidate_hits']}/{unique['denominator']} | {unique['candidate_ranking_gap']} | "
                         f"{all_rows['hits']}/{all_rows['denominator']} | {row['seconds_including_model_load']:.2f} |")
    lines += ["", "## Interpretation", "", summary["oracle_scope"] + ".", "", summary["timing_scope"] + ".", "",
              "Historical MPS runs use matching inference settings, but their training budgets and warmup differ.", "",
              "Per-event matched-frame corrections/regressions are in paired_event_changes.csv; these are not whole-event success rates.", "",
              "Contact sheets compare each arm to round 1 in tiles mode using deterministic event-balanced examples; they are post-hoc illustrations.", ""]
    lines += ["- " + limitation for limitation in summary["limitations"]]
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_charts(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#666666", "#159c96", "#d76757", "#c99926", "#477fba")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, (group, data) in enumerate(summary["groups"].items()):
        for row_index, mode in enumerate(MODES):
            axis = axes[row_index, col]
            metrics = [next(row["deduplicated_frames"] for row in data["comparisons"] if row["model"] == model and row["mode"] == mode) for model in MODELS]
            axis.bar(range(len(MODELS)), [row["recall"] for row in metrics], color=colors)
            for index, row in enumerate(metrics):
                axis.plot([index, index], [row["recall"], row["oracle_candidate_recall"]], color="black", linewidth=1)
                axis.scatter([index], [row["oracle_candidate_recall"]], facecolors="none", edgecolors="black", zorder=3)
                axis.text(index, min(1.03, row["oracle_candidate_recall"] + .025), f"{row['hits']}/{row['denominator']}", ha="center", fontsize=9)
            axis.set(xticks=range(len(MODELS)), xticklabels=[LABELS[m] for m in MODELS], ylim=(0, 1.12), ylabel="Localization recall",
                     title=f"{group} | {mode}")
            axis.grid(axis="y", alpha=.2)
    figure.suptitle("Exposed development: top1 bars and candidate-oracle circles", fontsize=14)
    figure.savefig(output / "recall_vs_oracle.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, arm in zip(axes, ARMS):
        rows = summary["training"][arm]["losses"]
        for field in LOSS_FIELDS:
            axis.plot([row["epoch"] for row in rows], [row[field] for row in rows], label=field)
        axis.set(title=arm, xlabel="Epoch", ylabel="Training loss")
        axis.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    figure.suptitle("Fixed training budget; fit losses only, not validation accuracy")
    figure.savefig(output / "training_loss.png", dpi=160)
    plt.close(figure)


def render_contact_sheet(examples, paths, title, destination):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (1440, 1010), "white")
    drawing = ImageDraw.Draw(canvas)
    drawing.text((12, 8), title + " | exposed development; cyan=reference orange=round1 magenta=arm", fill="black")
    for category_index, category in enumerate(("corrected", "regressed", "both_incorrect")):
        chosen = [row for row in examples if row["category"] == category]
        for index in range(3):
            x, y = index * 480, 35 + category_index * 325
            if index >= len(chosen):
                drawing.text((x + 10, y + 10), category + ": no additional eligible frame", fill="black")
                continue
            row = chosen[index]
            with Image.open(paths[row["image"]]) as source:
                image = source.convert("RGB")
            width, height = image.size
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            boxes = [(row["reference_bbox"], "#00bcd4")]
            for key, color in (("previous_refit", "#f28e2b"), ("new_round1", "#d22b8a")):
                if row[key]["top1"] is not None:
                    boxes.append((row[key]["top1"]["bbox"], color))
            for box, color in boxes:
                draw.rectangle(box, outline=color, width=3)
            overlay.thumbnail((470, 263))
            canvas.paste(overlay, (x + 5, y + 50))
            rx1, ry1, rx2, ry2 = row["reference_bbox"]
            cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            zoom_box = (max(0, round(cx - 80)), max(0, round(cy - 50)), min(width, round(cx + 80)), min(height, round(cy + 50)))
            zoom = image.crop(zoom_box).resize((200, 125))
            zoom_draw = ImageDraw.Draw(zoom)
            for box, color in boxes:
                mapped = [(box[0] - zoom_box[0]) * 200 / (zoom_box[2] - zoom_box[0]),
                          (box[1] - zoom_box[1]) * 125 / (zoom_box[3] - zoom_box[1]),
                          (box[2] - zoom_box[0]) * 200 / (zoom_box[2] - zoom_box[0]),
                          (box[3] - zoom_box[1]) * 125 / (zoom_box[3] - zoom_box[1])]
                zoom_draw.rectangle(mapped, outline=color, width=2)
            canvas.paste(zoom, (x + 270, y + 185))
            drawing.text((x + 5, y), f"{category} {row['event_id']}", fill="black")
            drawing.text((x + 5, y + 15), row["annotation_id"][:72], fill="black")
            drawing.text((x + 5, y + 30), f"R1 IoU {row['previous_refit']['top1_iou']:.3f}; arm IoU {row['new_round1']['top1_iou']:.3f}", fill="black")
    canvas.save(destination)


def write_review(directory, baseline_directory, output, image_roots=None):
    if output.exists():
        raise FileExistsError("Choose a new review directory; existing results are preserved")
    summary, bound = build_summary(directory, baseline_directory)
    selections, image_maps = {}, {}
    if image_roots is not None:
        for group in GROUPS:
            for arm in ARMS:
                selected, inventory = select_examples(bound[group, "new_round1", "tiles"], bound[group, arm, "tiles"])
                paths = image_paths(Path(image_roots[group]), selected)
                for row in selected:
                    row["current_image_sha256"] = sha(paths[row["image"]])
                selections[group, arm] = {"selected": selected, "inventory": inventory,
                                          "baseline_model": "new_round1", "candidate_model": arm,
                                          "reused_record_field_mapping": {"previous_refit": "new_round1", "new_round1": arm}}
                image_maps[group, arm] = paths
    output.mkdir(parents=True)
    render_charts(summary, output)
    comparison_rows, pair_rows = [], []
    for group, data in summary["groups"].items():
        for row in data["comparisons"]:
            for scope in SCOPES:
                comparison_rows.append({"group": group, "model": row["model"], "mode": row["mode"], "scope": scope,
                                        **row[scope], "seconds_including_model_load": row["seconds_including_model_load"]})
        for base, arms in data["paired_changes"].items():
            for arm, modes in arms.items():
                for mode, scopes in modes.items():
                    for scope, events in scopes.items():
                        pair_rows.extend({"group": group, "baseline": base, "arm": arm, "mode": mode, "scope": scope,
                                          "event_id": event, **change} for event, change in events.items())
    write_csv(output / "comparisons.csv", comparison_rows)
    write_csv(output / "paired_event_changes.csv", pair_rows)
    selection_report = {"scope": "posthoc_exposed_development", "selection_rule": "Deduplicated; deterministic sorted-event round-robin, max3/category vs round1 tiles",
                        "image_identity": "Unique filename lookup; current hashes recorded. Prediction files do not carry expected image hashes.", "groups": {}}
    for (group, arm), data in selections.items():
        name = f"{group}_{arm}_contact_sheet.png"
        render_contact_sheet(data["selected"], image_maps[group, arm], f"{group} {arm}", output / name)
        selection_report["groups"].setdefault(group, {})[arm] = {**data, "contact_sheet": name}
    (output / "selection.json").write_text(json.dumps(selection_report, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(markdown(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ars-images", type=Path, default=ROOT / "outputs/external_ball_frames_2026-10-28/images/validation")
    parser.add_argument("--core-images", type=Path, default=ROOT / "outputs/ball_frame_evaluation_2026-10-27/yolo/images/test")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()
    roots = None if args.no_contact_sheets else {"arsenal_development": args.ars_images, "core_development": args.core_images}
    try:
        summary = write_review(args.input.resolve(), args.round1.resolve(), args.output.resolve(), roots)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot review this completed round: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "scope": summary["evaluation_scope"],
                      "decision": summary["frozen_decision_verified"], "overall_project_accuracy_measured": False}, indent=2))


if __name__ == "__main__":
    main()
