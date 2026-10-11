"""Recompute descriptive analysis from a frozen fresh-course experiment bundle."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.final_results import sha256
from src.fresh_results import read_json, summarize, write_json


NUMERIC = {"source_fps", "event_id", "pose_actor_baseline", "pose_peak_frame_baseline", "contact_frame_candidate",
           "contact_actor_candidate", "steps_pose_baseline", "steps_phase_candidate", "ball_trajectory_observations",
           "ground_supported_selected_frames", "ground_pre_contact_points", "camera_intervals",
           "camera_target_class_candidates", "camera_slow_motion_candidates", "camera_timeline_coverage",
           "ball_endpoint_last_frame", "steps_sensitivity_min", "steps_sensitivity_max", "visible_approach_duration_sec"}
CONTINUOUS = {"source_fps", "camera_timeline_coverage", "visible_approach_duration_sec"}
BOOLEAN = {"contact_actor_reselected", "phase_camera_chain_complete", "keeper_pre_camera_chain_complete",
           "external_reference_source_supported", "human_verified", "physical_keeper_height_verified",
           "event_final_ball_position_verified", "ground_metric_accuracy_verified"}


def load_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != "fresh-course-bundle-v1" or manifest.get("events") != 39:
        raise ValueError("Fresh EDA needs the frozen 39-event schema")
    expected = {"features_39.csv", "features_39.json", "summary.json", "provenance.json"}
    if set(manifest.get("artifacts", {})) != expected:
        raise ValueError("Fresh bundle artifact set is incomplete")
    for name, digest in manifest["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Fresh bundle artifact hash mismatch: {name}")
    structured = read_json(directory / "features_39.json")
    ids = [event["clip_id"] for event in structured["events"]]
    if len(ids) != 39 or len(set(ids)) != 39:
        raise ValueError("Fresh bundle has duplicate or missing events")
    with (directory / "features_39.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 39 or [row["clip_id"] for row in rows] != ids:
        raise ValueError("CSV and structured fresh cohort identities differ")
    for row in rows:
        for name in NUMERIC:
            value = row[name]
            number = float(value) if value else None
            if number is not None and (not np.isfinite(number) or number < 0):
                raise ValueError(f"Invalid fresh numeric measurement: {name}")
            if name not in CONTINUOUS and number is not None and not number.is_integer():
                raise ValueError(f"Non-integer frame or count: {name}")
            if name == "source_fps" and (number is None or number <= 0):
                raise ValueError("A positive source frame rate is required")
            row[name] = number if name in CONTINUOUS else int(number) if number is not None else None
        for name in BOOLEAN:
            if row[name] not in ("", "True", "False"):
                raise ValueError(f"Invalid fresh boolean measurement: {name}")
            row[name] = {"": None, "True": True, "False": False}[row[name]]
        if any(row[column] not in ("unknown", "goal_likely", "no_goal_likely")
               for column in ("ball_rule_outcome", "scoreboard_rule_outcome", "combined_rule_outcome")):
            raise ValueError("Unsupported fresh outcome-rule label")
        if row["external_outcome_reference"] not in ("goal", "no_goal") or row["external_reference_source_supported"] is not True:
            raise ValueError("Unbound or unsupported outcome reference")
        if any(row[name] is not False for name in ("human_verified", "physical_keeper_height_verified",
                                                   "event_final_ball_position_verified", "ground_metric_accuracy_verified")):
            raise ValueError("Automatic outputs cannot establish human verification")
    summary = summarize(rows)
    if summary != read_json(directory / "summary.json"):
        raise ValueError("Stored fresh summary differs from independent recomputation")
    return rows, structured, summary


def descriptive_analysis(rows):
    groups = {}
    for outcome in ("goal", "no_goal"):
        cohort = [row for row in rows if row["external_outcome_reference"] == outcome]
        values = [row["steps_phase_candidate"] for row in cohort if row["steps_phase_candidate"] is not None]
        durations = [row["visible_approach_duration_sec"] for row in cohort if row["visible_approach_duration_sec"] is not None]
        groups[outcome] = {"events": len(cohort), "step_candidates": len(values),
                          "step_candidate_median": float(np.median(values)) if values else None,
                          "step_candidate_range": [min(values), max(values)] if values else None,
                          "visible_duration_candidates": len(durations),
                          "visible_duration_median_sec": float(np.median(durations)) if durations else None}
    sensitivity = [row["steps_sensitivity_max"] - row["steps_sensitivity_min"] for row in rows
                   if row["steps_sensitivity_min"] is not None and row["steps_sensitivity_max"] is not None]
    return {"outcome_groups": groups, "sensitivity_events": len(sensitivity),
            "sensitivity_median_range": float(np.median(sensitivity)) if sensitivity else None,
            "sensitivity_unchanged_events": sum(value == 0 for value in sensitivity),
            "causal_interpretation_supported": False, "human_verified": False,
            "limitations": "Candidate and observed-window comparisons, with outcome-dependent missingness; not validated true step counts"}


def figures(rows, structured, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 170})
    summary = summarize(rows)
    availability = summary["candidate_availability"]
    labels = ["Ball contact", "Striking foot", "Phase steps", "Keeper pre-movement", "Keeper direction",
              "Ball trace", "Ground pre-contact", "Target camera view", "Relative slow motion",
              "Strict keeper height", "Stationary ball end", "Observed image height",
              "Broadcast outcome", "Combined outcome"]
    fig, ax = plt.subplots(figsize=(9, 6))
    values = list(availability.values())
    bars = ax.barh(labels[::-1], values[::-1], color="#297e78")
    ax.bar_label(bars, fmt="%d / 39", padding=4)
    ax.set_xlim(0, 44); ax.set_xlabel("Events with an automatic candidate")
    ax.set_title("Fresh course-video candidate coverage")
    fig.text(.12, .015, "Availability does not establish identity, metric accuracy or human verification.", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1)); fig.savefig(output_dir / "01_fresh_coverage.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    usable = [r for r in rows if r["steps_phase_candidate"] is not None]
    if usable:
        axes[0].scatter([r["steps_pose_baseline"] for r in usable], [r["steps_phase_candidate"] for r in usable],
                        s=45, alpha=.8, color="#c95746")
        limit = max(max(r["steps_pose_baseline"], r["steps_phase_candidate"]) for r in usable) + 1
        axes[0].plot([0, limit], [0, limit], ":", color="#777777")
        axes[0].set_xlim(0, limit); axes[0].set_ylim(0, limit)
    else:
        axes[0].text(.5, .5, "No contact-bound phase count", ha="center", transform=axes[0].transAxes)
    axes[0].set_xlabel("Pose-only baseline candidate"); axes[0].set_ylabel("Contact-bound phase candidate")
    axes[0].set_title(f"Paired step proposals ({len(usable)} events)")
    deltas = [(r["contact_frame_candidate"] - r["pose_peak_frame_baseline"]) / r["source_fps"] for r in rows
              if r["contact_frame_candidate"] is not None and r["pose_peak_frame_baseline"] is not None]
    if deltas:
        axes[1].hist(deltas, bins=min(12, max(3, len(deltas))), color="#4576a1", edgecolor="white")
    axes[1].set_xlabel("Independent contact minus pose peak (seconds)")
    axes[1].set_ylabel("Events"); axes[1].set_title("Temporal anchor differences")
    fig.text(.12, .01, "Method differences are diagnostics, not step-count error or contact accuracy.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(output_dir / "02_fresh_steps_timing.png"); plt.close(fig)

    matrix = np.zeros((2, 3), dtype=int)
    for row in rows:
        matrix[("goal", "no_goal").index(row["external_outcome_reference"]),
               ("goal_likely", "no_goal_likely", "unknown").index(row["ball_rule_outcome"])] += 1
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.imshow(matrix, cmap="Greens", vmin=0)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18)
    ax.set_xticks(range(3), ["Likely goal", "Likely no-goal", "Abstained"])
    ax.set_yticks(range(2), ["Goal", "No goal"])
    ax.set_xlabel("Automatic ball rule"); ax.set_ylabel("Audited FIFA named-event reference")
    ax.set_title("Fresh rules and external outcome references")
    fig.text(.12, .01, "Development cohort; source-video event identity has not been independently human verified.", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, 1)); fig.savefig(output_dir / "03_fresh_ball_references.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels, values, durations = [], [], []
    for outcome in ("goal", "no_goal"):
        observed = [row for row in rows if row["external_outcome_reference"] == outcome and row["steps_phase_candidate"] is not None]
        if observed:
            labels.append(f'{outcome.replace("_", " ").title()}\n{len(observed)} / {sum(r["external_outcome_reference"] == outcome for r in rows)} events')
            values.append([row["steps_phase_candidate"] for row in observed])
    if values:
        axes[0].boxplot(values, tick_labels=labels, widths=.45)
        for index, group in enumerate(values):
            jitter = np.linspace(-.1, .1, len(group))
            axes[0].scatter(index + 1 + jitter, group, color="#c15b48", s=22, alpha=.75)
    axes[0].set_ylabel("Visible approach step candidates"); axes[0].set_title("Candidate counts by audited outcome")
    for row in rows:
        if row["steps_sensitivity_min"] is not None:
            axes[1].plot([row["steps_sensitivity_min"], row["steps_sensitivity_max"]], [row["event_id"]]*2,
                         color="#287e78", marker=".")
    axes[1].set_xlabel("Sensitive-to-strict approach counts"); axes[1].set_ylabel("Canonical event ID")
    axes[1].set_title("Parameter sensitivity retained")
    fig.text(.1, .012, "Missing counts are excluded, not zero-filled. Observed associations do not establish a scoring strategy.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(output_dir / "04_fresh_approach_outcomes.png"); plt.close(fig)
    write_json(output_dir / "descriptive_questions.json", descriptive_analysis(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, column, title in zip(axes, ("scoreboard_rule_outcome", "combined_rule_outcome"),
                                ("Source scoreboard icons", "Ball + scoreboard with conflict abstention"), strict=True):
        matrix = np.zeros((2, 3), int)
        for row in rows:
            matrix[("goal", "no_goal").index(row["external_outcome_reference"]),
                   ("goal_likely", "no_goal_likely", "unknown").index(row[column])] += 1
        ax.imshow(matrix, cmap="Greens", vmin=0)
        for i in range(2):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=17)
        ax.set_xticks(range(3), ["Goal", "No goal", "Unknown"])
        ax.set_yticks(range(2), ["Ref. goal", "Ref. no goal"])
        ax.set_title(title, fontsize=11)
    fig.text(.08, .01, "Frozen source-image predictions; references joined afterwards. Development agreement, not held-out accuracy.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(output_dir / "05_fresh_scoreboard_outcomes.png"); plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, structured, summary = load_bundle(args.bundle_dir)
    figures(rows, structured, args.output_dir)
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
