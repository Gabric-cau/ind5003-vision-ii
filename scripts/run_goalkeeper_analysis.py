"""Run goalkeeper features for all canonical penalty clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract goalkeeper features from retained pose evidence.")
    parser.add_argument("--step-diagnostics", type=Path, default=PROJECT_ROOT / "outputs" / "step_diagnostics.json")
    parser.add_argument("--kick-frames", type=Path, default=PROJECT_ROOT / "data" / "annotations" / "penalty_kick_frames.csv")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "penalty_slicing_manifest.yaml")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "data" / "processed" / "penalty_goalkeeper_features.csv")
    parser.add_argument("--diagnostics-json", type=Path, default=PROJECT_ROOT / "outputs" / "goalkeeper_diagnostics.json")
    parser.add_argument("--fallback-pose-json", type=Path, default=PROJECT_ROOT / "outputs" / "goalkeeper_pose_fallback.json")
    parser.add_argument("--track-reviews", type=Path, default=PROJECT_ROOT / "data" / "annotations" / "penalty_goalkeeper_track_reviews.csv")
    parser.add_argument("--pre-window-sec", type=float, default=1.5)
    parser.add_argument("--post-window-sec", type=float, default=1.0)
    parser.add_argument("--allow-unverified-annotations", action="store_true", help="Exploratory run using AI estimates; not ground truth.")
    parser.add_argument("--field-mask", action="store_true", help="Exclude people and prefer field background during camera stabilization.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from src.goalkeeper_pipeline import process_goalkeeper_diagnostics

    results = process_goalkeeper_diagnostics(
        str(args.step_diagnostics), str(args.kick_frames), str(args.manifest),
        str(args.output_csv), str(args.diagnostics_json),
        fallback_pose_json=str(args.fallback_pose_json),
        track_reviews_csv=str(args.track_reviews),
        pre_window_sec=args.pre_window_sec, post_window_sec=args.post_window_sec,
        allow_unverified_annotations=args.allow_unverified_annotations,
        use_field_mask=args.field_mask,
    )
    accepted = sum(bool(row["auto_accepted"]) for row in results)
    print(f"Rows written: {len(results)}")
    print(f"Auto-accepted: {accepted}")
    print(f"Review required: {len(results) - accepted}")
    print(f"CSV: {args.output_csv}")
    print(f"Diagnostics: {args.diagnostics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
