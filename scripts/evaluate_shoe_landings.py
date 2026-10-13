"""Freeze shoe proposals and compare only explicitly supplied frame reviews."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re


SCHEMA = "shoe-landing-reference-v1"
FRAME_CLOCK = "clip_local_zero_based"
FEET = {"left", "right", "unknown"}
STATUSES = {"pending", "observable", "unobservable", "ambiguous"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _fps(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("fps must be positive and finite")
    return float(value)


def _window(value, name="observable_window", allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain inclusive start_frame and end_frame")
    start = _integer(value.get("start_frame"), f"{name}.start_frame")
    end = _integer(value.get("end_frame"), f"{name}.end_frame")
    if end < start:
        raise ValueError(f"{name} has reversed frame bounds")
    return {"start_frame": start, "end_frame": end}


def _landings(value, window, name):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list or null")
    if window is None:
        raise ValueError(f"{name} cannot be known without a frame window")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{name} entries must be objects")
        frame = _integer(item.get("frame"), f"{name}.frame")
        foot = item.get("foot")
        if foot not in FEET:
            raise ValueError(f"{name}.foot must be left, right, or unknown")
        if not window["start_frame"] <= frame <= window["end_frame"]:
            raise ValueError(f"{name} frame outside declared window")
        result.append({"frame": frame, "foot": foot})
    if result != sorted(result, key=lambda item: (item["frame"], item["foot"])):
        raise ValueError(f"{name} must be ordered by frame and foot")
    if len({(item["frame"], item["foot"]) for item in result}) != len(result):
        raise ValueError(f"{name} contains duplicate events")
    return result


def validate_prediction(value):
    if not isinstance(value, dict):
        raise ValueError("prediction must be an object")
    clip_id = value.get("clip_id")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError("prediction clip_id is required")
    source_hash = value.get("source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source_sha256 must identify the analyzed clip bytes")
    if value.get("frame_clock", FRAME_CLOCK) != FRAME_CLOCK:
        raise ValueError("prediction frame_clock must be clip_local_zero_based")
    fps = _fps(value.get("fps"))
    window = _window(value.get("observable_window"), allow_none=True)
    frame_count = value.get("frame_count")
    if frame_count is not None:
        _integer(frame_count, "frame_count")
        if frame_count == 0:
            raise ValueError("frame_count must be positive when supplied")
        if window is not None and window["end_frame"] >= frame_count:
            raise ValueError("observable_window exceeds clip frame_count")
    if not isinstance(value.get("full_approach_visible"), bool):
        raise ValueError("full_approach_visible must be explicitly boolean")
    landings = _landings(value.get("landings"), window, "landings")
    count = value.get("candidate_count")
    if count is not None:
        _integer(count, "candidate_count")
    if (landings is None) != (count is None):
        raise ValueError("unknown landings and candidate_count must both be null")
    if landings is not None and count != len(landings):
        raise ValueError("candidate_count must equal the number of landing proposals")
    foot = value.get("striking_foot", "unknown")
    if foot not in FEET:
        raise ValueError("striking_foot must be left, right, or unknown")
    complete = value.get("count_complete", False)
    if not isinstance(complete, bool):
        raise ValueError("count_complete must be boolean")
    return {"clip_id": clip_id, "source_sha256": source_hash, "fps": fps,
            "frame_clock": FRAME_CLOCK, "observable_window": window,
            "frame_count": frame_count,
            "full_approach_visible": value["full_approach_visible"],
            "count_complete": complete, "candidate_count": count,
            "landings": landings, "striking_foot": foot}


def freeze_predictions(paths, output: Path):
    entries, clip_ids, source_windows = [], set(), set()
    for path in sorted(paths):
        value = validate_prediction(read(path))
        clip_id = value["clip_id"]
        if clip_id in clip_ids:
            raise ValueError(f"duplicate prediction clip_id: {clip_id}")
        window = value["observable_window"]
        key = (value["source_sha256"], value["fps"],
               window["start_frame"] if window else None, window["end_frame"] if window else None)
        if key in source_windows:
            raise ValueError("duplicate source and frame window would double-count one input")
        clip_ids.add(clip_id)
        source_windows.add(key)
        entries.append({"clip_id": clip_id, "path": os.path.relpath(path.resolve(), output.parent.resolve()),
                        "sha256": sha256(path)})
    if not entries:
        raise ValueError("no landing_result.json predictions found")
    result = {"schema_version": "shoe-landing-prediction-freeze-v1",
              "created_at": datetime.now(timezone.utc).isoformat(), "predictions": entries,
              "held_out_test": False, "development_cohort": True,
              "scope": "original course footage; proposal hashes frozen, not human accuracy",
              "chronology_limit": "A hash checks byte integrity; it cannot prove when a person first saw labels."}
    write_new(output, result)
    return result


def load_frozen_predictions(freeze: Path):
    manifest = read(freeze)
    if manifest.get("schema_version") != "shoe-landing-prediction-freeze-v1":
        raise ValueError("unsupported prediction freeze schema")
    predictions, source_windows = {}, set()
    for entry in manifest.get("predictions", []):
        path = (freeze.parent / entry["path"]).resolve()
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"prediction changed after freezing: {entry['clip_id']}")
        value = validate_prediction(read(path))
        if value["clip_id"] != entry["clip_id"] or value["clip_id"] in predictions:
            raise ValueError("prediction freeze contains mismatched or duplicate clip_id")
        window = value["observable_window"]
        key = (value["source_sha256"], value["fps"],
               window["start_frame"] if window else None, window["end_frame"] if window else None)
        if key in source_windows:
            raise ValueError("prediction freeze contains duplicate source and frame window")
        source_windows.add(key)
        predictions[value["clip_id"]] = value
    if not predictions:
        raise ValueError("prediction freeze is empty")
    return predictions


def reference_template(freeze: Path):
    predictions = load_frozen_predictions(freeze)
    events = []
    for clip_id, value in sorted(predictions.items()):
        events.append({"clip_id": clip_id, "source_sha256": value["source_sha256"],
                       "fps": value["fps"], "frame_clock": FRAME_CLOCK,
                       "frame_count": value["frame_count"],
                       "observable_window": value["observable_window"],
                       "reviews": [{"reviewer_id": "", "reviewed_at": None,
                                    "annotator_type": None, "human_verified": False,
                                    "review_status": "pending", "striking_foot": "unknown",
                                    "review_window": value["observable_window"],
                                    "full_approach_visible": None, "verified_landings": None,
                                    "notes": ""}]})
    return {"schema_version": SCHEMA, "prediction_freeze_sha256": sha256(freeze),
            "annotation_scope": "Independent human frame review; no labels have been filled automatically.",
            "events": events}


def validate_references(reference, predictions, freeze_hash):
    if not isinstance(reference, dict):
        raise ValueError("reference must be an object")
    if reference.get("schema_version") != SCHEMA:
        raise ValueError("unsupported reference schema")
    if reference.get("prediction_freeze_sha256") != freeze_hash:
        raise ValueError("references do not identify this prediction freeze")
    events, seen = {}, set()
    if not isinstance(reference.get("events"), list):
        raise ValueError("reference events must be a list")
    for event in reference["events"]:
        if not isinstance(event, dict):
            raise ValueError("reference event must be an object")
        clip_id = event.get("clip_id")
        if clip_id in seen:
            raise ValueError(f"duplicate reference clip_id: {clip_id}")
        if clip_id not in predictions:
            raise ValueError(f"reference has no frozen prediction: {clip_id}")
        seen.add(clip_id)
        prediction = predictions[clip_id]
        if event.get("source_sha256") != prediction["source_sha256"]:
            raise ValueError(f"reference source hash differs: {clip_id}")
        if event.get("frame_clock") != FRAME_CLOCK:
            raise ValueError(f"reference frame_clock differs: {clip_id}")
        if not math.isclose(_fps(event.get("fps")), prediction["fps"], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"reference fps differs: {clip_id}")
        if event.get("frame_count") != prediction["frame_count"]:
            raise ValueError(f"reference frame_count differs: {clip_id}")
        window = _window(event.get("observable_window"), allow_none=True)
        if window != prediction["observable_window"]:
            raise ValueError(f"reference source window differs: {clip_id}")
        reviews = event.get("reviews")
        if not isinstance(reviews, list) or not reviews:
            raise ValueError("each reference event needs at least one review row")
        reviewer_ids, validated = set(), []
        for review in reviews:
            if not isinstance(review, dict):
                raise ValueError("review must be an object")
            status = review.get("review_status")
            foot = review.get("striking_foot")
            if status not in STATUSES or foot not in FEET:
                raise ValueError("invalid review_status or striking_foot")
            reviewer_id = review.get("reviewer_id")
            if not isinstance(reviewer_id, str):
                raise ValueError("reviewer_id must be a string")
            if reviewer_id.strip():
                if reviewer_id.strip() in reviewer_ids:
                    raise ValueError("duplicate reviewer_id within one event")
                reviewer_ids.add(reviewer_id.strip())
            review_window = _window(review.get("review_window"), "review_window", allow_none=True)
            if window is not None and (review_window is None or review_window["start_frame"] < window["start_frame"] or review_window["end_frame"] > window["end_frame"]):
                raise ValueError("review_window must stay within the frozen source window")
            if review_window is not None and prediction["frame_count"] is not None and review_window["end_frame"] >= prediction["frame_count"]:
                raise ValueError("review_window exceeds clip frame_count")
            full = review.get("full_approach_visible")
            landings = _landings(review.get("verified_landings"), review_window, "verified_landings")
            if status == "pending":
                if (reviewer_id.strip() or review.get("reviewed_at") is not None or
                    review.get("human_verified") is not False or review.get("annotator_type") is not None or
                    foot != "unknown" or landings is not None or full is not None):
                    raise ValueError("pending review cannot contain fabricated reviewer or verified labels")
            else:
                if not reviewer_id.strip() or review.get("annotator_type") != "human" or review.get("human_verified") is not True:
                    raise ValueError("reviewed references require explicit human provenance")
                try:
                    timestamp = datetime.fromisoformat(review.get("reviewed_at", ""))
                except (TypeError, ValueError) as exc:
                    raise ValueError("reviewed_at must be an ISO timestamp") from exc
                if timestamp.tzinfo is None:
                    raise ValueError("reviewed_at needs a timezone")
                if not isinstance(full, bool):
                    raise ValueError("reviewed full_approach_visible must be boolean")
                if review_window is None and (status == "observable" or foot != "unknown"):
                    raise ValueError("observable labels require an explicit reviewed frame window")
                if status == "observable" and landings is None:
                    raise ValueError("observable landings require a checked list; use [] only for observed zero")
                if status != "observable" and landings is not None:
                    raise ValueError("ambiguous or unobservable landings must remain null")
            validated.append({**review, "reviewer_id": reviewer_id.strip(), "review_window": review_window,
                              "verified_landings": landings})
        events[clip_id] = validated
    return events


def match_landings(predicted, reference, tolerance_frames):
    """Maximum-cardinality, minimum-error ordered matching; each event is used once."""
    if not math.isfinite(tolerance_frames) or tolerance_frames < 0:
        raise ValueError("timing tolerance must be finite and nonnegative")
    predicted = sorted(predicted, key=lambda item: item["frame"])
    reference = sorted(reference, key=lambda item: item["frame"])
    table = [[(0, 0.0, ()) for _ in range(len(reference) + 1)] for _ in range(len(predicted) + 1)]
    for i, prediction in enumerate(predicted, 1):
        for j, expected in enumerate(reference, 1):
            choices = [table[i - 1][j], table[i][j - 1]]
            error = abs(prediction["frame"] - expected["frame"])
            if error <= tolerance_frames:
                count, total_error, pairs = table[i - 1][j - 1]
                choices.append((count + 1, total_error + error, pairs + ((i - 1, j - 1),)))
            table[i][j] = min(choices, key=lambda value: (-value[0], value[1], value[2]))
    pairs = [{"predicted": predicted[i], "reference": reference[j],
              "absolute_frame_error": abs(predicted[i]["frame"] - reference[j]["frame"])}
             for i, j in table[-1][-1][2]]
    return {"true_positive": len(pairs), "false_positive": len(predicted) - len(pairs),
            "false_negative": len(reference) - len(pairs), "pairs": pairs}


def _landing_consensus(reviews, tolerance_frames):
    reviewed = [row for row in reviews if row["review_status"] != "pending"]
    if len(reviewed) < 2:
        return None, "fewer_than_two_reviewers"
    if any(row["review_status"] != "observable" for row in reviewed):
        return None, "not_jointly_observable"
    representative = sorted(reviewed, key=lambda row: row["reviewer_id"])[0]
    for other in reviewed:
        if (other["review_window"] != representative["review_window"] or
            other["full_approach_visible"] != representative["full_approach_visible"]):
            return None, "window_or_full_approach_disagreement"
    for index, first in enumerate(reviewed):
        for other in reviewed[index + 1:]:
            match = match_landings(first["verified_landings"], other["verified_landings"], tolerance_frames)
            if match["false_positive"] or match["false_negative"]:
                return None, "landing_sequence_disagreement"
            if any(pair["predicted"]["foot"] != pair["reference"]["foot"] for pair in match["pairs"]):
                return None, "landing_side_disagreement"
    return representative, "jointly_reviewed_agreement"


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate(freeze: Path, references: Path, tolerance_sec=0.08, reviewer_tolerance_sec=0.08):
    if (not math.isfinite(tolerance_sec) or tolerance_sec < 0 or
        not math.isfinite(reviewer_tolerance_sec) or reviewer_tolerance_sec < 0):
        raise ValueError("tolerances must be finite and nonnegative")
    predictions = load_frozen_predictions(freeze)
    rows = validate_references(read(references), predictions, sha256(freeze))
    comparisons, individual_reviews, count_errors, full_count_errors = [], [], [], []
    tp = fp = fn = known_side_pairs = correct_side_pairs = 0
    timing_errors, foot_agreed = [], []
    reviewed_foot_events = foot_predicted = foot_correct = foot_disagreements = 0
    landing_reference_events = 0
    for clip_id, prediction in sorted(predictions.items()):
        reviews = rows.get(clip_id, [])
        for review in reviews:
            diagnostic = {"clip_id": clip_id, "reviewer_id": review["reviewer_id"],
                          "review_status": review["review_status"],
                          "striking_foot": review["striking_foot"]}
            if review["review_status"] == "observable" and prediction["landings"] is not None:
                window = review["review_window"]
                subset = [item for item in prediction["landings"]
                          if window["start_frame"] <= item["frame"] <= window["end_frame"]]
                diagnostic["comparison"] = match_landings(subset, review["verified_landings"], tolerance_sec * prediction["fps"])
                diagnostic["visible_candidate_count"] = len(subset)
                diagnostic["reviewed_visible_count"] = len(review["verified_landings"])
                diagnostic["count_absolute_error"] = abs(len(subset) - len(review["verified_landings"]))
            individual_reviews.append(diagnostic)
        known_foot = [row["striking_foot"] for row in reviews
                      if row["review_status"] != "pending" and row["striking_foot"] != "unknown"]
        if known_foot:
            reviewed_foot_events += 1
        if len(set(known_foot)) > 1:
            foot_disagreements += 1
        foot_consensus = known_foot[0] if len(known_foot) >= 2 and len(set(known_foot)) == 1 else None
        if foot_consensus:
            foot_agreed.append(clip_id)
            if prediction["striking_foot"] != "unknown":
                foot_predicted += 1
                foot_correct += prediction["striking_foot"] == foot_consensus
        consensus, reason = _landing_consensus(reviews, reviewer_tolerance_sec * prediction["fps"])
        landing_reference_events += consensus is not None
        comparison = {"clip_id": clip_id, "landing_reference_status": reason,
                      "known_foot_reviewers": len(known_foot), "striking_foot_consensus": foot_consensus,
                      "prediction_available": prediction["landings"] is not None}
        if consensus and prediction["landings"] is not None:
            window = consensus["review_window"]
            subset = [item for item in prediction["landings"]
                      if window["start_frame"] <= item["frame"] <= window["end_frame"]]
            expected = consensus["verified_landings"]
            match = match_landings(subset, expected, tolerance_sec * prediction["fps"])
            error = abs(len(subset) - len(expected))
            count_errors.append(error)
            tp += match["true_positive"]
            fp += match["false_positive"]
            fn += match["false_negative"]
            for pair in match["pairs"]:
                timing_errors.append(pair["absolute_frame_error"] / prediction["fps"])
                if pair["predicted"]["foot"] != "unknown" and pair["reference"]["foot"] != "unknown":
                    known_side_pairs += 1
                    correct_side_pairs += pair["predicted"]["foot"] == pair["reference"]["foot"]
            full_eligible = (consensus["full_approach_visible"] and prediction["full_approach_visible"] and
                             prediction["count_complete"])
            if full_eligible:
                full_count_errors.append(error)
            comparison.update(reference_window=window, reviewed_visible_count=len(expected),
                              predicted_visible_count=len(subset), count_absolute_error=error,
                              full_runup_metric_eligible=full_eligible,
                              representative_reviewer_id=consensus["reviewer_id"], matching=match)
        comparisons.append(comparison)
    known_prediction_feet = sum(value["striking_foot"] != "unknown" for value in predictions.values())
    return {"schema_version": "shoe-landing-evaluation-v1", "held_out_test": False,
            "development_cohort": True, "evaluation_scope": "Same-video development agreement, not held-out accuracy.",
            "human_review_provenance": "Reviewer declarations are supplied by people; this script cannot authenticate their identity or independence.",
            "prediction_freeze_sha256": sha256(freeze), "reference_sha256": sha256(references),
            "evaluator_sha256": sha256(Path(__file__)), "timing_tolerance_sec": tolerance_sec,
            "reviewer_agreement_tolerance_sec": reviewer_tolerance_sec,
            "events": len(predictions), "reference_events_supplied": len(rows),
            "reviewed_events": sum(any(row["review_status"] != "pending" for row in event) for event in rows.values()),
            "pending_review_rows": sum(row["review_status"] == "pending" for event in rows.values() for row in event),
            "proposal_coverage": {"known_count_events": sum(value["candidate_count"] is not None for value in predictions.values()),
                                  "unknown_count_events": sum(value["candidate_count"] is None for value in predictions.values()),
                                  "striking_foot_candidates": known_prediction_feet,
                                  "striking_foot_coverage": _ratio(known_prediction_feet, len(predictions))},
            "jointly_reviewed_landing_metrics": {"evaluated_events": len(count_errors),
                "jointly_reviewed_reference_events": landing_reference_events,
                "unknown_prediction_reference_events": landing_reference_events - len(count_errors),
                "prediction_coverage_on_jointly_reviewed_events": _ratio(len(count_errors), landing_reference_events),
                "visible_window_count_mae": _ratio(sum(count_errors), len(count_errors)),
                "true_positive": tp, "false_positive": fp, "false_negative": fn,
                "event_precision": _ratio(tp, tp + fp), "event_recall": _ratio(tp, tp + fn),
                "matched_timing_mae_sec": _ratio(sum(timing_errors), len(timing_errors)),
                "matched_known_side_pairs": known_side_pairs, "matched_known_side_agreement": _ratio(correct_side_pairs, known_side_pairs),
                "full_runup_evaluated_events": len(full_count_errors),
                "full_runup_count_mae": _ratio(sum(full_count_errors), len(full_count_errors))},
            "jointly_reviewed_striking_foot_metrics": {"events_with_any_known_human_foot": reviewed_foot_events,
                "reviewer_disagreement_events": foot_disagreements, "consensus_events": len(foot_agreed),
                "predicted_on_consensus_events": foot_predicted,
                "coverage_on_consensus_events": _ratio(foot_predicted, len(foot_agreed)),
                "agreement_on_known_predictions": _ratio(foot_correct, foot_predicted)},
            "individual_review_diagnostics": individual_reviews, "per_event": comparisons,
            "limitations": ["Empty or pending references produce null agreement metrics; they do not establish accuracy.",
                "Landing precision, recall and MAE are conditional on available predictions; unavailable windows are reported separately in coverage.",
                "Detection matching uses time only; anatomical side agreement is reported separately.",
                "A visible-window count is not a complete run-up count.",
                "Two reviewers must agree within tolerance before an event enters aggregate landing metrics."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freezing = subparsers.add_parser("freeze")
    inputs = freezing.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--predictions-dir", type=Path)
    inputs.add_argument("--prediction-json", type=Path)
    freezing.add_argument("--output", type=Path, required=True)
    template = subparsers.add_parser("template")
    template.add_argument("--freeze", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--freeze", type=Path, required=True)
    evaluation.add_argument("--references", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--tolerance-sec", type=float, default=0.08)
    evaluation.add_argument("--reviewer-tolerance-sec", type=float, default=0.08)
    args = parser.parse_args()
    if args.command == "freeze":
        paths = list(args.predictions_dir.rglob("landing_result.json")) if args.predictions_dir else [args.prediction_json]
        value = freeze_predictions(paths, args.output)
    elif args.command == "template":
        value = reference_template(args.freeze)
        write_new(args.output, value)
    else:
        value = evaluate(args.freeze, args.references, args.tolerance_sec, args.reviewer_tolerance_sec)
        write_new(args.output, value)
    print(json.dumps({"output": str(args.output), "command": args.command,
                      "schema_version": value["schema_version"]}, indent=2))


if __name__ == "__main__":
    main()
