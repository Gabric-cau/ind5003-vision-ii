"""Audit completed keeper training and select a development candidate on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.import_manual_pose_joints import save_json, sha256
from scripts.train_keeper_detector_final import (
    ARMS, CONFIDENCE, EPOCHS, IMAGE_SIZE, load_manifest, optimizer_budget,
    schedules_for, score_predictions, sequence_digest,
)

GENERIC = "pretrained_person_candidate_control"
MODELS = (GENERIC, *ARMS)
STEPS = [247, 317, 369, 421, 473, 525, 578, 630, 682, 734, 786, 838]
EPOCH_NUMBERS = list(range(1, 13))


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Final keeper run is incomplete: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value, *, minimum=0.):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= minimum)


def valid_digests(values):
    if not isinstance(values, dict) or not values:
        return False
    for value in values.values():
        if isinstance(value, str):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                return False
        elif isinstance(value, dict):
            if (not isinstance(value.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
                    or not isinstance(value.get("shape"), list)):
                return False
        else:
            return False
    return True


def verify_training(directory, report):
    protocol = report["protocol"]
    if read_json(directory / "protocol.json") != protocol:
        raise ValueError("Final report differs from the frozen protocol")
    if (protocol["epochs"], protocol["training_frames"], protocol["expected_optimizer_steps_per_epoch"]) != (12, 835, STEPS):
        raise ValueError("Frozen training budget changed")
    if EPOCHS != 12 or optimizer_budget() != STEPS:
        raise ValueError("Installed training budget no longer matches this experiment")
    if (protocol["checkpoint_selection"] != "fixed_last_epoch_12"
            or protocol["all_candidates_frozen_before_development_evaluation"] is not True
            or protocol["validation_yaml_uses_train_only"] is not True
            or protocol["human_boxes_used_for_inference"] is not False
            or protocol["independent_test"] is not False
            or protocol["whole_pipeline_accuracy_measured"] is not False):
        raise ValueError("Frozen selection or development isolation is unverified")
    manifest_path = Path(protocol["manifest"])
    if sha256(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Frozen keeper manifest changed")
    manifest, samples, dataset = load_manifest(manifest_path)
    if sha256(dataset) != protocol["dataset_yaml_sha256"]:
        raise ValueError("Frozen training YAML changed")
    initial = Path(protocol["initial_weights"]).resolve()
    if sha256(initial) != protocol["initial_weights_sha256"]:
        raise ValueError("Initializer checkpoint SHA mismatch")
    schedule_path = directory / "schedules.json"
    schedules = read_json(schedule_path)
    expected = schedules_for([Path(row["image"]).name for row in samples if row["split"] == "train"])
    if sha256(schedule_path) != protocol["schedule_sha256"] or schedules != expected:
        raise ValueError("Frozen training sample sequence changed")
    candidates = report["candidates"]
    if set(candidates) != set(ARMS) or read_json(directory / "frozen_candidates.json") != candidates:
        raise ValueError("Both complete frozen candidates are required")
    training = {}
    for arm in ARMS:
        candidate = candidates[arm]
        if read_json(directory / arm / "candidate.json") != candidate:
            raise ValueError("Final report differs from the arm receipt")
        if candidate["completed_epochs"] != EPOCH_NUMBERS or candidate["optimizer_steps"] != STEPS[-1]:
            raise ValueError("Incomplete epochs or optimizer updates")
        weights = Path(candidate["weights"]).resolve()
        if (weights != (directory / arm / "fit/weights/last.pt").resolve()
                or sha256(weights) != candidate["weights_sha256"]
                or candidate["initial_sha256"] != protocol["initial_weights_sha256"]
                or Path(candidate["initial_weights"]).resolve() != initial):
            raise ValueError("Candidate checkpoint SHA or source binding differs")
        if (candidate["weights_sha256"] == candidate["initial_sha256"]
                or not candidate["changed_parameters"]
                or not finite_number(candidate["maximum_observed_parameter_delta"])
                or candidate["maximum_observed_parameter_delta"] <= 0
                or not valid_digests(candidate["initial_parameter_digests"])):
            raise ValueError("Actual parameter updates or initial parameters are unverified")
        if (candidate["checkpoint_selection"] != "fixed_last_epoch_12"
                or candidate["development_scored_before_checkpoint_freeze"] is not False
                or candidate["promoted_to_default"] is not False
                or candidate["photometric_enabled"] is not (arm == "photometric")):
            raise ValueError("Candidate selection or augmentation differs from protocol")
        sampling = candidate["sampling"]
        if read_json(directory / arm / "actual_sampling.json") != sampling or len(sampling) != 12:
            raise ValueError("Actual sampling receipt is incomplete or differs")
        for index, row in enumerate(sampling):
            if (row["epoch"], row["samples"], row["unique_images"], row["optimizer_steps"], row["sequence_sha256"]) != (
                    index + 1, 835, 835, STEPS[index], sequence_digest(schedules[index])):
                raise ValueError("Actual sample sequence or optimizer budget differs")
            targets = row["target_counts"]
            fallback = targets["fallback_frames"]
            if ((targets["frames"], targets["targets"]) != (835, 835)
                    or isinstance(fallback, bool) or not isinstance(fallback, int) or not 0 <= fallback <= 835):
                raise ValueError("Positive training targets were not preserved")
        photo = read_json(directory / arm / "photometric_receipt.json")
        if photo["batches_augmented"] != (10020 if arm == "photometric" else 0):
            raise ValueError("Photometric intervention did not complete")
        if not finite_number(candidate["training_seconds"]):
            raise ValueError("Invalid training elapsed time")
        training[arm] = {"epochs": 12, "optimizer_steps": 838,
                         "training_seconds": candidate["training_seconds"],
                         "weights": str(weights), "weights_sha256": candidate["weights_sha256"]}
    if (candidates["control"]["initial_parameter_digests"] != candidates["photometric"]["initial_parameter_digests"]
            or candidates["control"]["sampling"] != candidates["photometric"]["sampling"]):
        raise ValueError("Arms have different initial parameters or actual sampling")
    return manifest, samples, training


def verify_predictions(prediction, samples):
    if (prediction["reference_labels_used_for_prediction"] is not False
            or prediction["mode"] != "full_original_image"
            or (prediction["image_size"], prediction["confidence"]) != (IMAGE_SIZE, CONFIDENCE)):
        raise ValueError("Frozen reference-free inference settings differ")
    if not finite_number(prediction["seconds"]):
        raise ValueError("Invalid inference elapsed time")
    indexed = {row["id"]: row for row in samples}
    for row in prediction["rows"]:
        if row["id"] not in indexed or not isinstance(row["detections"], list):
            raise ValueError("Invalid prediction frame or detections")
        frame = indexed[row["id"]]
        for item in row["detections"]:
            box, confidence = item["bbox"], item["confidence"]
            if (not finite_number(confidence, minimum=CONFIDENCE) or confidence > 1
                    or len(box) != 4 or not all(finite_number(v) for v in box)
                    or not 0 <= box[0] < box[2] <= frame["width"]
                    or not 0 <= box[1] < box[3] <= frame["height"]):
                raise ValueError("Invalid detector confidence or bounding box")


def choose_candidate(scores, groups):
    if len(groups) != 2 or "train" in groups or len(set(groups)) != 2:
        raise ValueError("Selection requires exactly two development groups")
    macros = {}
    for arm in ARMS:
        macros[arm] = {}
        for group in groups:
            clips = defaultdict(list)
            for row in scores[arm]["rows"]:
                if row["split"] == group:
                    clips[row["video_id"]].append(row)
            if not clips:
                raise ValueError("A development group has no evaluation frames")
            macros[arm][group] = sum((Fraction(sum(row["top1_hit"] for row in rows), len(rows))
                                      for rows in clips.values()), Fraction()) / len(clips)
    means = {arm: sum(macros[arm].values(), Fraction()) / 2 for arm in ARMS}
    regressions = [group for group in groups if macros["photometric"][group] < macros["control"][group]]
    improves = means["photometric"] > means["control"]
    selected = "photometric" if not regressions and improves else "control"
    return {"selected_model": selected, "groups_used_for_selection": list(groups),
            "macro_clip_top1_by_group": {arm: {group: float(value) for group, value in values.items()}
                                         for arm, values in macros.items()},
            "equal_group_mean_macro_clip_top1": {arm: float(value) for arm, value in means.items()},
            "photometric_regressing_groups": regressions, "photometric_strict_mean_improvement": improves,
            "rule": "Photometric only if both development clip-macro top1 recalls do not decrease and their equal-weight mean strictly increases; otherwise control",
            "selection_status": "development_candidate_pending_full_video_smoke",
            "default_pipeline_changed": False, "independent_test": False,
            "whole_pipeline_accuracy_measured": False, "training_diagnostics_used_for_selection": False}


def build_summary(directory):
    directory = Path(directory).resolve()
    report = read_json(directory / "report.json")
    if report.get("training_completed") is not True:
        raise ValueError("Final keeper run is incomplete")
    if (report.get("default_pipeline_changed") is not False or report.get("independent_test") is not False
            or report.get("full_video_role_tracking_measured") is not False):
        raise ValueError("Report must remain development-only keeper localization")
    _, samples, training = verify_training(directory, report)
    evaluation = [row for row in samples if row["split"] != "train" or row["existing_pose_reference"]]
    groups = sorted({row["split"] for row in evaluation} - {"train"})
    protocol = report["protocol"]
    if (protocol["evaluation_frame_ids"] != [row["id"] for row in evaluation]
            or protocol["evaluation_counts"] != dict(Counter(row["split"] for row in evaluation))):
        raise ValueError("Frozen evaluation inventory differs from the manifest")
    if set(report["evaluation"]) != set(MODELS):
        raise ValueError("All three completed model evaluations are required")
    digests = {GENERIC: protocol["initial_weights_sha256"],
               **{arm: training[arm]["weights_sha256"] for arm in ARMS}}
    sources = {}
    paths = [directory / name for name in ("report.json", "protocol.json", "schedules.json", "frozen_candidates.json")]
    paths += [Path(protocol["manifest"]), Path(protocol["manifest"]).parent / "dataset.yaml"]
    scores, comparisons, diagnostics = {}, [], {}
    for model in MODELS:
        prediction_path, score_path = (directory / f"{model}_{suffix}.json" for suffix in ("predictions", "scores"))
        prediction = read_json(prediction_path)
        verify_predictions(prediction, evaluation)
        score = score_predictions(prediction["rows"], evaluation, specialized=model != GENERIC)
        if read_json(score_path) != score:
            raise ValueError("Stored keeper scores differ from recomputed raw predictions")
        expected = {"scores": {key: value for key, value in score.items() if key != "rows"},
                    "seconds": prediction["seconds"], "weights_sha256": digests[model]}
        if report["evaluation"][model] != expected:
            raise ValueError("Final report evaluation or checkpoint binding differs")
        scores[model] = score
        comparisons.extend({"model": model, "split": group, **score["by_split"][group],
                            "all_evaluation_inference_seconds": prediction["seconds"]} for group in groups)
        diagnostics[model] = {"all_including_training": score["all"],
                              "training_reference_frames": score["by_split"].get("train"),
                              "existing_pose_references_including_training": score["existing_pose_references"]}
        paths.extend((prediction_path, score_path))
    decision = choose_candidate(scores, groups)
    decision.update({key: training[decision["selected_model"]][key] for key in ("weights", "weights_sha256")})
    for arm in ARMS:
        paths.extend(directory / arm / name for name in ("candidate.json", "actual_sampling.json", "photometric_receipt.json"))
    for path in paths:
        sources[str(path)] = sha256(path)
    return {"schema": "final-keeper-training-summary/v1", "source_directory": str(directory),
            "source_sha256": sources, "training": training, "selection": decision,
            "development_comparisons": comparisons, "diagnostics_not_used_for_selection": diagnostics,
            "metric": "Highest-confidence active-keeper box IoU >= 0.5; missing prediction is a miss",
            "evaluation_scope": "Exposed development positive frames; no independent test or whole-project accuracy",
            "generic_person_scope": "Any-candidate overlap only, not goalkeeper-role top1 accuracy",
            "timing_scope": f"Inference over all {len(evaluation)} frozen evaluation frames, excluding model load; not full-video latency"}


def markdown(summary):
    choice = summary["selection"]
    lines = ["# Final Keeper Training Review", "", f"Selected standalone development candidate: `{choice['selected_model']}`.", "",
             summary["evaluation_scope"] + ".", "", summary["metric"] + ".", "",
             "| Development group | Model | Top1 hits/frames | Clip-macro top1 | Any-candidate hits/frames |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in summary["development_comparisons"]:
        top = "-" if row["top1_hits"] is None else f"{row['top1_hits']}/{row['frames']}"
        macro = "-" if row["macro_clip_top1_recall_iou_0_5"] is None else f"{row['macro_clip_top1_recall_iou_0_5']:.2%}"
        lines.append(f"| {row['split']} | {row['model']} | {top} | {macro} | {row['any_candidate_hits']}/{row['frames']} |")
    lines += ["", choice["rule"] + ".", "",
              "Both arms completed 12 epochs and 838 verified optimizer updates with identical initialization and sampling.", "",
              "Training-reference diagnostics and the aggregate containing them are excluded from selection.", "",
              summary["generic_person_scope"] + ". " + summary["timing_scope"] + ".", "",
              "This selects an experimental detector only. No default pipeline change, temporal identity, pose, direction, height or whole-project improvement is established.", ""]
    return "\n".join(lines)


def write_review(directory, output):
    summary = build_summary(directory)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "summary.json", summary)
    save_json(output / "selection.json", summary["selection"])
    (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    rows = summary["development_comparisons"]
    with (output / "comparisons.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_review(args.run, args.output)["selection"], indent=2))


if __name__ == "__main__":
    main()
