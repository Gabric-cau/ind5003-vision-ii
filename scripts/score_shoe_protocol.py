"""Score already frozen foot predictions; references never enter inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_fresh_shoe_approach_experiment import read, sha, write


def metrics(rows, key):
    available = sum(row[key] in ("left", "right") for row in rows)
    correct = sum(row[key] == row["reference"] for row in rows)
    per_class = {}
    for side in ("left", "right"):
        tp = sum(row[key] == side == row["reference"] for row in rows)
        fp = sum(row[key] == side != row["reference"] for row in rows)
        fn = sum(row[key] != side == row["reference"] for row in rows)
        per_class[side] = {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
    return {"total": len(rows), "correct": correct, "available": available,
        "strict_accuracy_unknown_counts_as_wrong": correct / len(rows), "coverage": available / len(rows),
        "conditional_accuracy": correct / available if available else None,
        "macro_f1_left_right": sum(row["f1"] for row in per_class.values()) / 2,
        "balanced_accuracy_unknown_counts_as_wrong": sum(row["recall"] for row in per_class.values()) / 2,
        "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "references", "baseline-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing scores")
    report_path = args.replay / "report.json"
    report = read(report_path)
    references = {row["core_id"]: row for row in read(args.references)["samples"]}
    rows = []
    for event in report["events"]:
        event_id = event["event_id"]
        reference = references[event_id]
        prediction_path = args.replay / event_id / "shoes/prediction.json"
        before_path = args.baseline_root / "events" / event_id / "attempt_001/pipeline/trained_shoes/prediction.json"
        prediction, before = read(prediction_path), read(before_path)
        if (reference["video_sha256"] != event["source_sha256"]
                or prediction["source_sha256"] != event["source_sha256"]
                or before["source_sha256"] != event["source_sha256"]
                or sha(prediction_path) != event["prediction_sha256"]):
            raise ValueError("Prediction/source binding changed before scoring")
        rows.append({"event_id": event_id, "source_sha256": event["source_sha256"],
            "before": before["striking_foot"], "after": prediction["striking_foot"],
            "reference": reference["foot"], "prediction_sha256": sha(prediction_path),
            "baseline_prediction_sha256": sha(before_path), "binding_method": prediction["binding_method"],
            "pose_cache": prediction.get("pose_cache"), "actual_inference_calls": prediction["actual_inference_calls"]})
    output = {"schema_version": "frozen-foot-development-score-v1", "events": rows,
        "before": metrics(rows, "before"), "after": metrics(rows, "after"),
        "references_sha256": sha(args.references), "replay_report_sha256": sha(report_path),
        "evaluation_scope": "development agreement on previously inspected eight-event cohort, not independent test accuracy",
        "unknown_policy": "included in strict denominator; false negative for its reference class",
        "inference_completed_before_scoring": True, "scoring_code_sha256": sha(Path(__file__))}
    write(args.output, output)
    print({key: output[key] for key in ("before", "after")})


if __name__ == "__main__":
    main()
