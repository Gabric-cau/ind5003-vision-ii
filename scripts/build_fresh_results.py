"""Freeze one source-bound, 39-event fresh experiment bundle."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fresh_results import STAGE_PATHS, build_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for stage in STAGE_PATHS:
        parser.add_argument(f"--{stage}-dir", type=Path, required=stage == "pose")
    args = parser.parse_args()
    result = build_bundle(args.source_manifest, args.reference_audit,
                          {stage: getattr(args, f"{stage}_dir") for stage in STAGE_PATHS}, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
