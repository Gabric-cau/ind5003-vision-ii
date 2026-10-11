"""Add attempt-level count semantics while preserving frozen v1/v2 evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_terminal_footwork import load, save, sha, PREFIXES
from scripts.validate_terminal_footwork import check_result
from src.style_aware_steps import analyze_style_aware_steps

OUTPUT = ROOT / "outputs/style_aware_steps_2026-10-09"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/terminal_footwork_2026-10-09")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("Output must not overwrite frozen inputs")
    for folder in ("classified", "classified_refined"):
        paths = [p for p in sorted((args.input_dir / folder).glob("*.json")) if p.stem.startswith(PREFIXES)]
        if len(paths) != 39:
            parser.error(f"Expected 39 inputs in {folder}; found {len(paths)}")
        results = []
        for path in paths:
            r = load(path)
            errors = check_result(r)
            if errors:
                raise ValueError(f"Frozen input {path.name}: {errors}")
            r["step_analysis"] = analyze_style_aware_steps(r)
            r["schema_version"] = "penalty-phase-experiment-v3-style-aware-steps"
            r["step_analysis_provenance"] = {"source_path": str(path), "source_sha256": sha(path),
                "code_sha256": {n: sha(ROOT / n) for n in ("src/style_aware_steps.py", "scripts/run_style_aware_steps.py")},
                "new_pose_inference": False, "new_labels_created": False}
            save(args.output_dir / folder / path.name, r)
            results.append(r)
        summary = {"clips": len(results), "attempt_styles": dict(Counter(r["step_analysis"]["attempt_style"] for r in results)),
            "count_available": sum(r["step_analysis"]["candidate_step_count"] is not None for r in results),
            "undetermined_with_count": sum(r["step_analysis"]["attempt_style"] == "undetermined" and r["step_analysis"]["candidate_step_count"] is not None for r in results),
            "undetermined_without_count": sum(r["step_analysis"]["attempt_style"] == "undetermined" and r["step_analysis"]["candidate_step_count"] is None for r in results),
            "all_nominal_counts_preserved": all(r["step_analysis"]["candidate_step_count"] == r["footwork_result"]["counts"]["nominal_approach_candidates"] for r in results),
            "human_verified": False, "accuracy_validated": False,
            "rows": [{"clip_id": r["clip_id"], "pose_source": r["pose_source"], **r["step_analysis"]} for r in results]}
        save(args.output_dir / folder / "summary.json", summary)
        print(folder, {k:v for k,v in summary.items() if k != "rows"}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
