"""Validate style-aware reporting without claiming step/classification accuracy."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_terminal_footwork import load, save, sha, PREFIXES
from scripts.validate_terminal_footwork import check_result
from scripts.validate_penalty_phase_experiment import tests as historical_tests
from src.style_aware_steps import analyze_style_aware_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/style_aware_steps_2026-10-09")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "outputs/phase_optimization_2026-10-09/review/steps")
    args = parser.parse_args()
    errors, total = [], 0
    for folder in ("classified", "classified_refined"):
        paths = [p for p in (args.run_dir/folder).glob("*.json") if p.stem.startswith(PREFIXES)]
        if len(paths) != 39:
            errors.append(f"{folder}: missing outputs")
        for path in paths:
            total += 1
            r = load(path)
            errors.extend(f"{path.name}: {x}" for x in check_result(r))
            p = r["step_analysis_provenance"]
            source = Path(p["source_path"])
            if sha(source) != p["source_sha256"]:
                errors.append(f"{path.name}: source hash")
            old = load(source)
            if any(r[k] != v for k,v in old.items() if k != "schema_version"):
                errors.append(f"{path.name}: old evidence changed")
            for name, expected in p["code_sha256"].items():
                if sha(ROOT/name) != expected:
                    errors.append(f"{path.name}: stale code {name}")
            s = r["step_analysis"]
            if s != analyze_style_aware_steps(r):
                errors.append(f"{path.name}: numerical replay mismatch")
            if s["candidate_step_count"] != r["footwork_result"]["counts"]["nominal_approach_candidates"]:
                errors.append(f"{path.name}: nominal count changed")
            if s["human_verified"] or s["accuracy_validated"] or s["verified_step_count"] is not None:
                errors.append(f"{path.name}: unjustified verification")
            if s["candidate_step_count"] is not None and sum(v or 0 for v in s["count_partition"].values()) != s["candidate_step_count"]:
                errors.append(f"{path.name}: count partition")
            if s["attempt_style"] != "terminal_adjustment_candidate" and s["count_partition"]["terminal_adjustment"] is not None:
                errors.append(f"{path.name}: forced terminal count")
    suite = unittest.TestSuite()
    for name in ("test_terminal_footwork", "test_style_aware_steps"):
        spec = importlib.util.spec_from_file_location(name, ROOT/"tests"/f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    history = historical_tests()
    media = load(args.review_dir / "media_validation.json")
    if media["clips"] != 39 or not media["full_decode_passed"]:
        errors.append("missing full-length playback evidence")
    for row in media["reports"]:
        if sha(Path(row["source"])) != row["source_sha256"] or sha(Path(row["playback_copy"])) != row["playback_sha256"]:
            errors.append("playback hash mismatch")
    report = {"passed": not errors and outcome.wasSuccessful() and history["passed"], "errors": errors,
        "structural_and_replay_results": total, "terminal_and_style_tests": outcome.testsRun,
        "historical_tests": history["tests_run"], "total_tests": outcome.testsRun+history["tests_run"],
        "media_hashes_rechecked": len(media["reports"]), "human_verified": False, "accuracy_validated": False,
        "scope": "Synthetic mechanics, count conservation, input preservation, replay and provenance; not ground-truth accuracy."}
    save(args.run_dir/"validation.json", report)
    print(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
