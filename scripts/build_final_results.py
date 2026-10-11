"""Build the canonical analysis bundle from explicit, hashed experiment inputs."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.final_results import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/final_analysis.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_analysis"))
    parser.add_argument("--raw-clips-dir", type=Path, help="Verify all original video hashes before building")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else args.root / args.output_dir
    evaluation = build_bundle(args.root, args.config, output_dir, args.raw_clips_dir)
    print(json.dumps({"output_dir": str(output_dir), "total_events": evaluation["total_events"],
                      "ball_rule_reference_comparison": evaluation["ball_rule_reference_comparison"],
                      "accuracy_validated": evaluation["accuracy_validated"]}, indent=2))


if __name__ == "__main__":
    main()
