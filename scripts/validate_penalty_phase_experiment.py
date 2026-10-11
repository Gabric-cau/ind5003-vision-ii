"""Structural and regression checks; intentionally not an accuracy evaluator."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2

TEST_FILES = ["test_kicker_step_count.py", "test_ball_outcome.py", "test_goalkeeper_analysis.py",
    "test_step_shot_selection.py", "test_penalty_phases.py", "test_phase_step_events.py",
    "test_motion_stabilization.py", "test_pose_refinement.py", "test_kick_contact.py"]


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def tests():
    suite = unittest.TestSuite()
    counts = {}
    for name in TEST_FILES:
        spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / "tests" / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sub = unittest.defaultTestLoader.loadTestsFromModule(module)
        for key, fn in inspect.getmembers(module, inspect.isfunction):
            if key.startswith("test_") and fn.__module__ == module.__name__:
                if inspect.signature(fn).parameters:
                    raise ValueError(f"Fixture-dependent test needs pytest: {name}:{key}")
                sub.addTest(unittest.FunctionTestCase(fn))
        counts[name] = sub.countTestCases()
        suite.addTest(sub)
    outcome = unittest.TextTestRunner(verbosity=1).run(suite)
    return {"passed": outcome.wasSuccessful(), "tests_run": outcome.testsRun,
            "failures": len(outcome.failures), "errors": len(outcome.errors), "by_file": counts,
            "scope": "Selected historical and new synthetic tests; not real-video accuracy or the entire repository pytest suite."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/phase_optimization_2026-10-09")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    errors, dirs = [], {}
    for folder in ("classified", "classified_refined"):
        results = [load(p) for p in (args.run_dir / folder).glob("*.json") if p.stem.startswith(("col_eng_p", "cro_den_p", "esp_rus_p", "rus_cro_p"))]
        if len(results) != 39:
            errors.append(f"{folder}: expected39, got{len(results)}")
        for r in results:
            cid = r["clip_id"]
            start, end = r["shot_start_frame"], r["shot_end_frame_exclusive"]
            phase = r["phase_result"]
            if not 0 <= start < end <= r["source_frame_count"]:
                errors.append(f"{folder}/{cid}: shot bounds")
            if r["human_verified"] or r["accuracy_validated"] or phase["human_verified"]:
                errors.append(f"{folder}/{cid}: unjustified verification")
            frame_labels = phase["frame_labels"]
            if any(not start <= row["frame"] < end for row in frame_labels):
                errors.append(f"{folder}/{cid}: labels outside shot")
            frames = [x["frame"] for x in frame_labels]
            if frames and frames != list(range(frames[0], frames[-1] + 1)):
                errors.append(f"{folder}/{cid}: non-contiguous explicit timeline")
            expanded = []
            for seg in phase["segments"]:
                expanded.extend((f, seg["phase"]) for f in range(seg["start_frame"], seg["end_frame"]))
            if expanded != [(x["frame"], x["phase"]) for x in frame_labels]:
                errors.append(f"{folder}/{cid}: segments differ from timeline")
            if dict(Counter(e["phase"] for e in phase["events"])) != phase["phase_event_counts"]:
                errors.append(f"{folder}/{cid}: count mismatch")
            if r["contact"]["status"] == "abstain" and phase["status"] != "abstained":
                errors.append(f"{folder}/{cid}: contact abstention ignored")
            window = r["contact"].get("interval_frames")
            if window and any(e["frame"] > window[1] for events in r["event_proposal"]["variants"].values() for e in events):
                errors.append(f"{folder}/{cid}: event beyond contact uncertainty window")
            for name, sha in r["provenance"]["code_sha256"].items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                    errors.append(f"{folder}/{cid}: stale code {name}")
            image = cv2.imread(str(args.run_dir / folder / f"{cid}_qa.jpg"))
            if image is None or image.shape[1] != 960:
                errors.append(f"{folder}/{cid}: QA decode")
        dirs[folder] = {"clips": len(results), "phase_status": dict(Counter(r["phase_result"]["status"] for r in results)),
                        "contact_status": dict(Counter(r["contact"]["status"] for r in results))}
    unit = tests() if not args.skip_tests else {"skipped": True}
    media_path = args.run_dir / "review/media_validation.json"
    media = load(media_path) if media_path.exists() else None
    if not media or media["clips"] != 39 or not media["full_decode_passed"]:
        errors.append("39 full-length media conversions not yet validated")
    report = {"structural_validation_passed": not errors and unit.get("passed", args.skip_tests),
              "unit_regression_tests": unit, "directories": dirs, "errors": errors,
              "media_full_decode_clips": media["clips"] if media else 0,
              "human_verified": False, "accuracy_validated": False,
              "accuracy_unavailable_reason": "No independent exact step/phase/contact ground truth; one user's approximate Cuadrado assessment is not a benchmark."}
    (args.run_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
    return 0 if report["structural_validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
