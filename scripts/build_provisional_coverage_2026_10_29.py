"""Build a clearly labelled provisional coverage view without overwriting strict results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected unique non-empty clip_id rows: {path}")
    return {row["clip_id"]: row for row in rows}


def build(v1: dict[str, dict[str, str]], v3: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    if set(v1) != set(v3):
        raise ValueError("v1 and v3 event inventories differ")
    result = []
    for clip_id in sorted(v3):
        old, current = v1[clip_id], v3[clip_id]
        legacy_height = old.get("keeper_legacy_height_candidate", "unknown")
        observed_height = old.get("keeper_observed_image_height_candidate", "unknown")
        last_frame = current.get("ball_endpoint_last_frame", "")
        result.append({
            "clip_id": clip_id,
            "strict_keeper_height": current.get("keeper_strict_height_candidate", "unknown"),
            "strict_keeper_pose_height": current.get("keeper_pose_ordinal_height_candidate", "unknown"),
            "legacy_keeper_height_candidate": legacy_height,
            "observed_image_height_candidate": observed_height,
            "provisional_keeper_height": legacy_height if legacy_height != "unknown" else observed_height,
            "ball_endpoint_status": current.get("ball_endpoint_status", "unknown"),
            "last_visible_ball_frame": last_frame,
            "strict_event_final_ball_position_verified": current.get(
                "event_final_ball_position_verified", "False"
            ),
            "provisional_endpoint_kind": "last_visible_observation" if last_frame else "unknown",
            "provisional_warning": (
                "legacy_or_observed_image_candidate; not strict height accuracy"
                if legacy_height != "unknown" or observed_height != "unknown"
                else "no_height_candidate"
            ),
        })
    return result


def write(output: Path, rows: list[dict[str, str]]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "provisional_features_39.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    height = [row for row in rows if row["provisional_keeper_height"] != "unknown"]
    visible = [row for row in rows if row["last_visible_ball_frame"]]
    summary = {
        "schema": "provisional-coverage-view/v1",
        "event_count": len(rows),
        "provisional_keeper_height_candidates": len(height),
        "provisional_keeper_height_values": {"low_or_mid": len(height)},
        "provisional_last_visible_ball_observations": len(visible),
        "strict_keeper_height_verified": sum(row["strict_keeper_height"] != "unknown" for row in rows),
        "strict_event_final_ball_position_verified": sum(
            row["strict_event_final_ball_position_verified"] == "True" for row in rows
        ),
        "independent_accuracy": False,
        "provisional_values_are_not_accuracy": True,
        "source_files": {
            "legacy_diagnostic": "outputs/fresh_course_experiment/final_bundle_v1/features_39.csv",
            "strict_current_bundle": "outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.csv",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Provisional Coverage View",
        "",
        f"The existing strict keeper-height result remains {summary['strict_keeper_height_verified']}/39.",
        f"This separate view exposes {summary['provisional_keeper_height_candidates']}/39 legacy or observed image-plane height candidates.",
        f"It also exposes {summary['provisional_last_visible_ball_observations']}/39 last-visible ball observations.",
        "",
        "These values are provisional observations, not verified keeper-height accuracy or event-final ball positions.",
        "A last visible ball frame may be occluded or before the true terminal state. The legacy height field is an older image-plane candidate, not the strict current height output.",
        "Do not overwrite the strict columns or report these counts as model accuracy.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, default=Path("outputs/fresh_course_experiment/final_bundle_v1/features_39.csv"))
    parser.add_argument("--strict", type=Path, default=Path("outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion/features_39.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, build(read_rows(args.legacy), read_rows(args.strict)))


if __name__ == "__main__":
    main()
