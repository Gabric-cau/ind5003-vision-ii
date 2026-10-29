"""Create deterministic post-hoc ball-error sheets from completed predictions on CPU."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.summarize_external_ball_round import GROUPS, indexed_rows, read_json, same_frames, selected_rows, sha
from scripts.train_ball_frame_detector import score_frame, summarize


CATEGORIES = ("corrected", "regressed", "both_incorrect")
COLORS = {"reference": "#00bcd4", "previous_refit": "#f28e2b", "new_round1": "#d22b8a"}


def top_candidate(row):
    for candidate in row["candidates"]:
        box = candidate["bbox"]
        values = [candidate["confidence"], *box]
        if (len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
                or not 0 <= candidate["confidence"] <= 1 or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("Invalid candidate box or confidence")
    measured = score_frame(row["reference_bbox"], row["candidates"])
    if not math.isclose(measured["top1_iou"], row["top1_iou"], abs_tol=1e-9):
        raise ValueError("Stored top1 IoU disagrees with the highest-confidence candidate")
    return max(row["candidates"], key=lambda candidate: candidate["confidence"], default=None)


def select_examples(previous, new):
    same_frames(previous, new)
    buckets = {category: defaultdict(list) for category in CATEGORIES}
    for key in sorted(previous):
        old, current = previous[key], new[key]
        old_top, new_top = top_candidate(old), top_candidate(current)
        if not old["deduplicated_primary"]:
            continue
        before, after = old["top1_iou"] >= .5, current["top1_iou"] >= .5
        category = "regressed" if before and not after else "corrected" if after and not before else "both_incorrect"
        if before and after:
            continue
        buckets[category][old["event_id"]].append({"category": category, "event_id": old["event_id"],
            "video_id": old["video_id"], "annotation_id": old["annotation_id"], "image": old["image"],
            "deduplicated_primary": True, "reference_bbox": old["reference_bbox"],
            "previous_refit": {"top1": old_top, "top1_iou": old["top1_iou"]},
            "new_round1": {"top1": new_top, "top1_iou": current["top1_iou"]}})
    selected, counts = [], {}
    for category, events in buckets.items():
        counts[category] = {"eligible_frames": sum(map(len, events.values())), "eligible_events": len(events)}
        queues = {event: deque(rows) for event, rows in sorted(events.items())}
        chosen = []
        while len(chosen) < 3 and any(queues.values()):
            for queue in queues.values():
                if queue and len(chosen) < 3:
                    chosen.append(queue.popleft())
        selected.extend(chosen)
        counts[category]["selected_frames"] = len(chosen)
    return selected, counts


def image_paths(root, selected):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError("Original-image root is missing: " + str(root))
    requested = {row["image"] for row in selected}
    if any(not isinstance(name, str) or Path(name).name != name or "\\" in name
           or Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"} for name in requested):
        raise ValueError("Prediction image must be a plain image filename")
    matches = defaultdict(list)
    for path in root.rglob("*"):
        if path.name in requested and path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError("Selected image resolves outside its supplied root")
            matches[path.name].append(path)
    if any(len(matches[name]) != 1 for name in requested):
        raise ValueError("Each selected image must have exactly one filename match in its supplied root")
    return {name: matches[name][0] for name in requested}


def prepare_selection(directory, image_roots):
    from PIL import Image

    report = read_json(directory / "report.json")
    if report.get("training_completed") is not True or report.get("detector_weights_updated") is not True:
        raise ValueError("Final round report is not complete")
    expected_epochs = list(range(1, report["protocol"]["epochs"] + 1))
    if report["candidate"]["completed_epochs"] != expected_epochs:
        raise ValueError("Final report does not confirm all planned epochs")
    output = {"schema_version": "external-ball-posthoc-error-review-v1", "mode": "tiles",
              "diagnosis": "Post-hoc diagnosis of already observed evaluation data; not an independent test or overall project accuracy",
              "selection_rule": "Deduplicated primary frames; per category, round-robin sorted event IDs and then sorted (video_id, annotation_id), maximum 3",
              "image_identity": "Unique filename and coordinate binding to supplied image roots; predictions contain no expected image hash. Current image hashes are recorded, not independent pixel-identity proof.",
              "reference_scope": "Assisted localization reference boxes, not independently verified ground truth",
              "source_sha256": {"report.json": sha(directory / "report.json")}, "groups": {}}
    images = {}
    for group in GROUPS:
        predictions = {}
        for model in ("previous_refit", "new_round1"):
            filename = f"{group}_{model}_tiles.json"
            path = directory / filename
            prediction = read_json(path)
            if {key: value for key, value in prediction.items() if key != "rows"} != report["results"][group][model]["tiles"]:
                raise ValueError("Prediction and final report disagree: " + filename)
            expected_hash = report["protocol"]["previous_weights_sha256"] if model == "previous_refit" else report["candidate"]["weights_sha256"]
            if prediction["mode"] != "tiles" or prediction["weights_sha256"] != expected_hash:
                raise ValueError("Prediction mode or model binding mismatch")
            indexed = indexed_rows(prediction["rows"])
            expected_count = report["protocol"]["new_match_evaluation_frames" if group == "arsenal_new_match" else "regression_frames"]
            if len(indexed) != expected_count:
                raise ValueError("Prediction frame count differs from the completed protocol")
            for scope in ("all_frames", "deduplicated_frames"):
                if summarize(selected_rows(indexed, scope)) != prediction["metrics"][scope]:
                    raise ValueError("Prediction rows disagree with reported metrics: " + scope)
            predictions[model] = indexed
            output["source_sha256"][filename] = sha(path)
        selected, counts = select_examples(predictions["previous_refit"], predictions["new_round1"])
        paths = image_paths(image_roots[group], selected)
        for row in selected:
            path = paths[row["image"]]
            with Image.open(path) as source:
                image = source.convert("RGB").copy()
            width, height = image.size
            box = row["reference_bbox"]
            if not (len(box) == 4 and all(math.isfinite(value) for value in box)
                    and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError("Reference box does not fit the selected image")
            row.update(image_path=str(path.resolve()), image_sha256=sha(path), width=width, height=height)
            images[group, row["video_id"], row["annotation_id"]] = image
        output["groups"][group] = {"counts": counts, "selected": selected,
                                  "contact_sheet": group + "_contact_sheet.png"}
    return output, images


def render_sheet(group, data, images, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(3, 3, figsize=(18, 12))
    figure.subplots_adjust(top=.86, bottom=.06, hspace=.55, wspace=.03)
    figure.suptitle(GROUPS[group] + " | tiles\nPost-hoc diagnosis of already observed evaluation data", fontsize=15, y=.99)
    figure.legend(handles=[Line2D([0], [0], color=COLORS[name], linewidth=2, linestyle=style, label=label)
                           for name, style, label in (("reference", "-", "Assisted reference"),
                               ("previous_refit", "--", "Previous refit top1"), ("new_round1", ":", "New round top1"))],
                  loc="upper center", bbox_to_anchor=(.5, .935), ncol=3)
    for category, row_axes in zip(CATEGORIES, axes):
        examples = [row for row in data["selected"] if row["category"] == category]
        for index, axis in enumerate(row_axes):
            axis.set_axis_off()
            if index >= len(examples):
                axis.text(.5, .5, category + "\nNo additional eligible frame", ha="center", va="center", transform=axis.transAxes)
                continue
            row = examples[index]
            image = images[group, row["video_id"], row["annotation_id"]]
            axis.imshow(image)
            boxes = [("reference", row["reference_bbox"], "-", 2.5)]
            for name, style in (("previous_refit", "--"), ("new_round1", ":")):
                if row[name]["top1"] is not None:
                    boxes.append((name, row[name]["top1"]["bbox"], style, 2.0))
            inset = axis.inset_axes([.54, .03, .43, .43])
            inset.imshow(image)
            x0, y0, x1, y1 = row["reference_bbox"]
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            crop_w, crop_h = max(160, (x1 - x0) * 8), max(100, (y1 - y0) * 8)
            inset.set(xlim=(max(0, center_x - crop_w / 2), min(row["width"], center_x + crop_w / 2)),
                      ylim=(min(row["height"], center_y + crop_h / 2), max(0, center_y - crop_h / 2)), xticks=[], yticks=[])
            inset.set_title("Reference region", fontsize=7, color="black", backgroundcolor="white")
            for name, box, style, width in boxes:
                for target in (axis, inset):
                    target.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                               fill=False, edgecolor=COLORS[name], linestyle=style, linewidth=width))
            status = []
            for name, label in (("previous_refit", "old"), ("new_round1", "new")):
                value = f"{row[name]['top1_iou']:.3f}" if row[name]["top1"] is not None else "missing (0.000)"
                status.append(label + " IoU=" + value)
            title = category + " | " + row["event_id"] + "\n" + row["annotation_id"]
            axis.set_title("\n".join(textwrap.fill(line, width=65) for line in title.splitlines()) + "\n" + "; ".join(status), fontsize=8)
    figure.text(.5, .02, "IoU >= 0.50 defines a hit. Deduplicated, event-balanced selected examples; not a representative error rate.", ha="center", fontsize=10)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def visualize(directory, image_roots, output):
    if output.exists():
        raise FileExistsError("Choose a new review output directory; existing results are preserved")
    selection, images = prepare_selection(directory, image_roots)
    output.mkdir(parents=True)
    for group, data in selection["groups"].items():
        render_sheet(group, data, images, output / data["contact_sheet"])
    (output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed round directory")
    parser.add_argument("--ars-images", type=Path, required=True, help="Original ARS evaluation images, searched by unique basename")
    parser.add_argument("--core-images", type=Path, required=True, help="Original core retest images, searched by unique basename")
    parser.add_argument("--output", type=Path, required=True, help="New directory for two contact sheets and selection.json")
    args = parser.parse_args()
    try:
        result = visualize(args.input.resolve(), {"arsenal_new_match": args.ars_images,
            "core_development_retest": args.core_images}, args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot visualize this round: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "selected_frames": {
        group: len(data["selected"]) for group, data in result["groups"].items()}}, indent=2))


if __name__ == "__main__":
    main()
