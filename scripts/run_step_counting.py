"""Run manifest-driven penalty run-up step counting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "penalty_slicing_manifest.yaml"
DEFAULT_CLIPS_DIR = PROJECT_ROOT / "data" / "raw_clips"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "penalty_step_counts.csv"
DEFAULT_DIAGNOSTICS_JSON = PROJECT_ROOT / "outputs" / "step_diagnostics.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "weights" / "yolo11n_pose.pt"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count each penalty taker's run-up steps from the canonical clip manifest."
    )
    parser.add_argument("--field-mask", action="store_true", help="Experimental HSV on-field candidate prior; accuracy not yet validated.")
    parser.add_argument("--shot-selection", choices=("first_viable", "first"), default="first_viable",
                        help="Search camera shots for a qualifying pose track; 'first' retains the legacy policy.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Ultralytics pose-model path or model name (default: %(default)s).",
    )
    parser.add_argument(
        "--imgsz",
        type=_positive_int,
        default=640,
        help="YOLO inference image size (default: %(default)s).",
    )
    parser.add_argument(
        "--conf",
        type=_probability,
        default=0.10,
        help="YOLO detection confidence threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Penalty slicing manifest (default: %(default)s).",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=DEFAULT_CLIPS_DIR,
        help="Directory containing canonical penalty clips (default: %(default)s).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination for per-penalty step counts (default: %(default)s).",
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_JSON,
        help="Destination for detailed diagnostics (default: %(default)s).",
    )
    return parser


def _summary_counts(results: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int]:
    rows = len(results)
    auto_accepted = sum(bool(result.get("auto_accepted")) for result in results)
    review_required = sum(bool(result.get("review_required")) for result in results)
    failures = sum(
        str(result.get("status", "")).strip().lower() == "failed"
        or bool(result.get("error"))
        for result in results
    )
    return rows, auto_accepted, review_required, failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        parser.error(
            "Ultralytics is required. Install it in this environment before running step counting."
        )
        raise AssertionError("argparse exits above") from exc

    from src.step_pipeline import process_manifest_steps

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading pose model once: {args.model}")
    model = YOLO(args.model)
    print(f"Processing clips listed in: {args.manifest}")

    results = process_manifest_steps(
        manifest_path=str(args.manifest),
        clips_dir=str(args.clips_dir),
        model=model,
        output_csv=str(args.output_csv),
        diagnostics_json=str(args.diagnostics_json),
        predict_kwargs={"imgsz": args.imgsz, "conf": args.conf},
        use_field_mask=args.field_mask,
        shot_selection=args.shot_selection,
    )
    result_rows = list(results or [])
    rows, auto_accepted, review_required, failures = _summary_counts(result_rows)

    print("\nStep-counting summary")
    print(f"  Rows written:       {rows}")
    print(f"  Auto-accepted:      {auto_accepted}")
    print(f"  Review required:    {review_required}")
    print(f"  Failures:           {failures}")
    print(f"  CSV:                {args.output_csv}")
    print(f"  Diagnostics:        {args.diagnostics_json}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
