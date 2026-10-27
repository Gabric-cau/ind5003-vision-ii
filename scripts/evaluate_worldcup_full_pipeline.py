"""Score the frozen full-video World Cup batch after integrity checks.

The batch is label-free at inference time.  References are opened only after
the completed batch, event receipts, analysis hashes and copied source videos
have all been verified by ``load_completed_batch``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_offline_video_batch import (
    SCOREBOARD_MAP,
    categorical_metrics,
    load_completed_batch,
    number,
    sha,
)


HEIGHTS = ("low", "middle", "high")


def read_references(path: Path, samples: dict):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "worldcup8-full-pipeline-references-v1":
        raise ValueError("Unexpected World Cup reference schema")
    rows = {row["core_id"]: row for row in document.get("samples", [])}
    if set(rows) != set(samples) or len(rows) != 8:
        raise ValueError("References do not cover exactly the completed batch")
    for core_id, sample in samples.items():
        row = rows[core_id]
        if row.get("video_sha256") != sample.get("video_sha256"):
            raise ValueError(f"Reference/video hash mismatch for {core_id}")
        if row.get("outcome_conflict"):
            raise ValueError(f"Conflicted outcome reference for {core_id}")
        if row.get("goal_reference") not in ("goal", "no_goal"):
            raise ValueError(f"Invalid goal reference for {core_id}")
        if row.get("foot") not in ("left", "right"):
            raise ValueError(f"Invalid foot reference for {core_id}")
        if row.get("keeper_ordinal_height") not in HEIGHTS:
            raise ValueError(f"Invalid keeper height reference for {core_id}")
    return rows, {"references_sha256": sha(path), "schema_version": document["schema_version"]}


def keeper_height(analysis):
    required = analysis.get("required_outputs", {}).get("keeper_dive_height")
    if isinstance(required, dict):
        value = required.get("value")
        if value in HEIGHTS:
            return value
    auxiliary = analysis.get("auxiliary_models", {}).get("trained_keeper") or {}
    value = (auxiliary.get("prediction") or {}).get("height_model_candidate")
    return value if value in HEIGHTS else None


def event_rows(predictions, references):
    rows = []
    foot_rows, goal_rows, height_rows = [], [], []
    for core_id in sorted(predictions):
        analysis = predictions[core_id]
        reference = references[core_id]
        required = analysis.get("required_outputs", {})
        foot = required.get("kicking_foot")
        foot = foot if foot in ("left", "right") else None
        goal = SCOREBOARD_MAP.get(required.get("goal_outcome"))
        height = keeper_height(analysis)
        foot_rows.append({"core_id": core_id, "reference": reference["foot"], "prediction": foot})
        goal_rows.append({"core_id": core_id, "reference": reference["goal_reference"], "prediction": goal})
        height_rows.append({"core_id": core_id, "reference": reference["keeper_ordinal_height"], "prediction": height})
        rows.append({
            "core_id": core_id,
            "clip_id": reference["clip_id"],
            "video_sha256": reference["video_sha256"],
            "foot_reference": reference["foot"],
            "foot_prediction": foot,
            "goal_reference": reference["goal_reference"],
            "goal_prediction": goal,
            "keeper_height_reference": reference["keeper_ordinal_height"],
            "keeper_height_prediction": height,
            "end_to_end_seconds": analysis.get("end_to_end_seconds"),
            "video_fps": analysis.get("source", {}).get("fps"),
            "video_frame_count": analysis.get("source", {}).get("frame_count"),
            "manual_annotations_used": analysis.get("manual_annotations_used"),
            "all_stages_executed_successfully": analysis.get("all_stages_executed_successfully"),
            "selected_output_sources": analysis.get("selected_output_sources", {}),
        })
    metrics = {
        "kicking_foot": categorical_metrics(foot_rows, ("left", "right")),
        "goal_outcome": categorical_metrics(goal_rows, ("goal", "no_goal")),
        "keeper_ordinal_height": categorical_metrics(height_rows, HEIGHTS),
    }
    return rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; earlier reports are preserved")
    predictions, samples, batch_bindings = load_completed_batch(args.batch_dir.resolve())
    # This is intentionally after the full prediction-integrity audit.
    references, reference_bindings = read_references(args.references.resolve(), samples)
    rows, metrics = event_rows(predictions, references)
    runtimes = [row["end_to_end_seconds"] for row in rows if number(row["end_to_end_seconds"])]
    ratios = [row["end_to_end_seconds"] / (row["video_frame_count"] / row["video_fps"])
              for row in rows if number(row["end_to_end_seconds"])
              and number(row["video_frame_count"]) and number(row["video_fps"])
              and row["video_fps"] > 0]
    report = {
        "schema_version": "worldcup8-full-pipeline-evaluation-v1",
        "evaluation_kind": "first_full_pipeline_scoring_on_frozen_worldcup_match",
        "held_out_for_primary_modules": True,
        "match_count": 1,
        "event_count": len(rows),
        "all_predictions_finished_before_reference_read": True,
        "batch_bindings": batch_bindings,
        "reference_bindings": reference_bindings,
        "metrics": metrics,
        "runtime": {
            "events": len(runtimes),
            "median_seconds": statistics.median(runtimes) if runtimes else None,
            "p95_seconds": statistics.quantiles(runtimes, n=20, method="inclusive")[18]
            if len(runtimes) > 1 else (runtimes[0] if runtimes else None),
            "maximum_seconds": max(runtimes) if runtimes else None,
            "median_real_time_ratio": statistics.median(ratios) if ratios else None,
            "per_event_seconds": runtimes,
        },
        "unscored": {
            "keeper_dive_direction": "all reference coordinate conventions are null",
            "visible_runup_steps": "full-runup references do not match automatic observed-window scope",
            "top_down_trajectory": "no independent coordinate path reference",
            "ball_trajectory_and_endpoint": "no independent centre/terminal-coordinate reference",
            "camera_four_class_intervals": "reference release has no complete four-class interval truth",
        },
        "limitations": [
            "Eight correlated events come from one match and are a small external check.",
            "References were supplied in the audited release; this is not label-blind human annotation.",
            "Unknown predictions count as misses in strict metrics; conditional metrics use only available outputs.",
            "The World Cup match was previously inspected for ball-height and camera metadata; those auxiliary results are not claimed as a fresh independent test.",
            "No model selection or threshold tuning followed this evaluation.",
        ],
        "events": rows,
    }
    args.output.mkdir(parents=True)
    output = args.output / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "metrics": {
        key: {k: value for k, value in metric.items() if k not in ("rows", "per_class", "excluded_comparisons")}
        for key, metric in metrics.items()}, "runtime": report["runtime"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
