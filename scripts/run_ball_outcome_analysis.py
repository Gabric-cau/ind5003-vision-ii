"""Run ball tracking and join verified penalty outcomes for canonical clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Extract ball trajectories and observed outcomes.")
    result.add_argument("--step-diagnostics", type=Path, default=ROOT / "outputs" / "step_diagnostics.json")
    result.add_argument("--kick-frames", type=Path, default=ROOT / "data" / "annotations" / "penalty_kick_frames.csv")
    result.add_argument("--outcomes", type=Path, default=ROOT / "data" / "annotations" / "penalty_outcomes.csv")
    result.add_argument("--output-csv", type=Path, default=ROOT / "data" / "processed" / "penalty_ball_outcome_features.csv")
    result.add_argument("--diagnostics-json", type=Path, default=ROOT / "outputs" / "ball_outcome_diagnostics.json")
    result.add_argument("--goalkeeper-diagnostics", type=Path, default=ROOT / "outputs" / "goalkeeper_diagnostics.json")
    result.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "models" / "weights" / "yolo11n_object_detection.pt",
    )
    result.add_argument("--clip-id", action="append", dest="clip_ids")
    result.add_argument("--post-frames", type=int, default=32)
    result.add_argument("--allow-unverified-annotations", action="store_true", help="Exploratory run using AI estimates; not ground truth.")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    from ultralytics import YOLO
    from src.ball_pipeline import process_ball_outcomes

    model = YOLO(str(args.weights))
    rows = process_ball_outcomes(
        str(args.step_diagnostics), str(args.kick_frames), str(args.outcomes),
        str(args.output_csv), str(args.diagnostics_json), model,
        goalkeeper_diagnostics_json=str(args.goalkeeper_diagnostics),
        post_frames=args.post_frames, only_clip_ids=args.clip_ids,
        allow_unverified_annotations=args.allow_unverified_annotations,
    )
    accepted = sum(row["ball_track_status"] == "auto_accepted" for row in rows)
    goals = sum(row["outcome"] == "goal" for row in rows)
    print(f"Rows written: {len(rows)}")
    print(f"Ball auto-accepted: {accepted}; review required: {len(rows) - accepted}")
    print(f"Outcome labels: {goals} goal; {len(rows) - goals} no_goal")
    print(f"CSV: {args.output_csv}")
    print(f"Diagnostics: {args.diagnostics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
