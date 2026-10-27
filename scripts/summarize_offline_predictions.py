"""Export source-bound video predictions and descriptive plots without labels."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return Path(path).read_text(encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows_from_batch(folder):
    summary = json.loads(read(folder / "summary.json"))
    if summary.get("complete") is not True:
        raise ValueError("Summarize a completed batch; failed runs remain in their own receipts")
    rows = []
    for receipt in summary["events"]:
        path = (folder / receipt["analysis_path"]).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != receipt["analysis_sha256"]:
            raise ValueError("Saved analysis does not match the completed batch")
        data = json.loads(read(path))
        if (data["input_sha256"] != receipt["video_sha256"]
                or data.get("manual_annotations_used") is not False
                or data.get("all_stages_executed_successfully") is not True):
            raise ValueError("Invalid source-bound label-free analysis")
        fields, source = data["required_outputs"], data["source"]
        keeper = data.get("auxiliary_models", {}).get("trained_keeper", {}).get("prediction", {})
        height, endpoint = fields["keeper_dive_height"], fields["ball_final_position"]
        elapsed = float(data["end_to_end_seconds"])
        duration = source["frame_count"] / source["fps"]
        rows.append({
            "core_id": receipt["core_id"], "source_sha256": data["input_sha256"],
            "video_duration_seconds": duration, "runtime_seconds": elapsed,
            "runtime_per_video_second": elapsed / duration,
            "contact_time_seconds": data["trained_contact"]["prediction"]["time_seconds"],
            "kicking_foot": fields["kicking_foot"],
            "observed_window_steps": fields["visible_runup_steps"],
            "step_source": data["selected_output_sources"].get("visible_runup_steps", "phase_rule"),
            "keeper_pre_movement": fields["keeper_pre_kick_movement"],
            "keeper_direction": fields["keeper_dive_direction"],
            "keeper_strict_height": height.get("strict_mask_pose_category") if isinstance(height, dict) else height,
            "keeper_ordinal_height_auxiliary": keeper.get("height_model_candidate", "unknown"),
            "goal_outcome": fields["goal_outcome"],
            "ball_observed_points": len(fields["ball_trajectory"]),
            "ball_terminal_position_candidate": (endpoint.get("event_final_position")
                                                   if isinstance(endpoint, dict) else endpoint) is not None,
            "top_down_projected_points": len(fields["runup_top_down_trajectory"]["points"]),
            "camera_intervals": len(data["camera_intervals"]),
            "camera_known_intervals": sum(s["task_class_candidate"] != "unknown"
                                          for s in data["camera_intervals"]),
            "analysis_sha256": receipt["analysis_sha256"],
        })
    if len(rows) != summary["expected_events"]:
        raise ValueError("Incomplete prediction population")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new descriptive-report directory")
    rows = rows_from_batch(args.batch_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    args.output.mkdir(parents=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    available = lambda value: value not in (None, "unknown")
    availability = {
        "Kicking foot": sum(available(r["kicking_foot"]) for r in rows),
        "Observed-window steps": sum(r["observed_window_steps"] is not None for r in rows),
        "Keeper pre-movement": sum(available(r["keeper_pre_movement"]) for r in rows),
        "Keeper dive direction": sum(available(r["keeper_direction"]) for r in rows),
        "Qualitative keeper height": sum(available(r["keeper_ordinal_height_auxiliary"]) for r in rows),
        "Ball trace (nonempty)": sum(r["ball_observed_points"] > 0 for r in rows),
        "Terminal ball candidate": sum(r["ball_terminal_position_candidate"] for r in rows),
        "Top-down points (nonempty)": sum(r["top_down_projected_points"] > 0 for r in rows),
        "Goal outcome": sum(available(r["goal_outcome"]) for r in rows),
    }
    times = np.array([r["runtime_seconds"] for r in rows])
    durations = np.array([r["video_duration_seconds"] for r in rows])
    report = {
        "schema_version": "offline-prediction-descriptive-report-v1", "events": len(rows),
        "batch_summary_sha256": digest(args.batch_dir / "summary.json"),
        "manual_reference_labels_read": False, "accuracy_computed": False,
        "coverage_is_not_correctness": True, "available_candidates_by_output": availability,
        "runtime": {"median_seconds": float(np.median(times)), "p95_seconds": float(np.percentile(times, 95)),
                    "total_seconds": float(times.sum()), "total_video_seconds": float(durations.sum()),
                    "aggregate_runtime_per_video_second": float(times.sum() / durations.sum())},
        "prediction_distributions": {key: dict(Counter(str(r[key]) for r in rows))
             for key in ("kicking_foot", "keeper_direction", "keeper_strict_height", "goal_outcome")},
        "limitations": ["Known training events: deployment integration, not held-out accuracy.",
                        "Observed-window counts are not full approach counts.",
                        "Nonempty traces/projections do not establish correct geometry.",
                        "Labels and conflicts are evaluated separately after predictions freeze."],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    names = list(availability)
    values = list(availability.values())
    axes[0].barh(names, values, color="#2b8c7e")
    axes[0].set_xlim(0, len(rows) + 1)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Events with an automatic candidate (not accuracy)")
    axes[0].set_title("Output availability")
    for i, count in enumerate(values):
        axes[0].text(count + .12, i, f"{count}/{len(rows)}", va="center", fontsize=10)
    x = np.arange(len(rows))
    axes[1].bar(x, times, color="#446e9b", label="End-to-end runtime")
    axes[1].plot(x, durations, color="#bb5b42", marker="o", label="Video duration")
    axes[1].set_xticks(x, [r["core_id"] for r in rows], rotation=45, ha="right")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Runtime including model loading and decoding")
    axes[1].legend(frameon=False)
    fig.suptitle("Core-video integration: automatic predictions only", fontsize=14)
    fig.savefig(args.output / "coverage_and_runtime.png", dpi=180)
    fig.savefig(args.output / "coverage_and_runtime.pdf")
    plt.close(fig)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
