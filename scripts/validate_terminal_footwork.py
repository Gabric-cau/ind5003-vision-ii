"""Verify frozen inputs, preserved counts, terminal partitions, and tests."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_terminal_footwork import OUTPUT, PREFIXES, load, save, sha
from scripts.validate_penalty_phase_experiment import tests as historical_tests


def check_result(r):
    errors = []
    p = r["terminal_provenance"]
    for file_key, hash_key in (("source_result_path", "source_result_sha256"),
        ("selected_pose_path", "selected_pose_sha256"), ("camera_path", "camera_sha256")):
        if sha(Path(p[file_key])) != p[hash_key]:
            errors.append(f"input hash: {file_key}")
    for name, expected in p["code_sha256"].items():
        if sha(ROOT / name) != expected:
            errors.append(f"code hash: {name}")
    old = load(Path(p["source_result_path"]))
    for field in ("phase_result", "event_proposal", "count_sensitivity", "contact", "baseline", "review_overlay", "user_reference"):
        if r.get(field) != old.get(field):
            errors.append(f"historical field changed: {field}")
    f = r["footwork_result"]
    c, t = f["counts"], f["terminal"]
    nominal = old["count_sensitivity"].get("central", {}).get("approach_candidates_including_precontact_strike_window")
    if c["nominal_approach_candidates"] != nominal:
        errors.append("nominal count changed")
    if f["human_verified"] or f["accuracy_validated"] or r["human_verified"] or r["accuracy_validated"]:
        errors.append("unjustified verification")
    if f["ankle_control"]["status"] != "not_measurable":
        errors.append("unjustified ankle control claim")
    expected_frames = sorted(e["frame"] for e in old["event_proposal"].get("events", []))
    if [e["frame"] for e in f["events"]] != expected_frames:
        errors.append("events lost or invented")
    if nominal is not None:
        total = sum(c[k] or 0 for k in ("before_terminal_candidates", "terminal_adjustment_candidates",
            "no_distinct_split_candidates", "unresolved_prestrike_candidates", "contact_guard_candidates", "pose_gap_candidates"))
        if total != nominal or f["partition_sum"] != nominal:
            errors.append("partition mismatch")
    if t["status"] == "candidate_change":
        if not r["shot_start_frame"] <= t["start_frame"] < t["end_frame_exclusive"] <= r["contact"]["interval_frames"][0]:
            errors.append("terminal bounds")
        if not t["evidence"] or c["terminal_adjustment_candidates"] < 3:
            errors.append("insufficient sustained-change support")
        for interval in t["intervals"]:
            if interval["start_frame"] >= t["start_frame"] and not interval["pose_supported"]:
                errors.append("terminal inference bridges gap")
    elif t["start_frame"] is not None or c["terminal_adjustment_candidates"] is not None:
        errors.append("abstention shown as terminal count")
    json.dumps(r, allow_nan=False)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=OUTPUT)
    parser.add_argument("--review-dir", type=Path, default=ROOT / "outputs/phase_optimization_2026-10-09/review/terminal")
    args = parser.parse_args()
    errors, directory_reports = [], {}
    for folder in ("classified", "classified_refined"):
        paths = [p for p in sorted((args.run_dir/folder).glob("*.json")) if p.stem.startswith(PREFIXES)]
        if len(paths) != 39:
            errors.append(f"{folder}: expected 39 files, got {len(paths)}")
        for path in paths:
            errors.extend(f"{folder}/{path.stem}: {e}" for e in check_result(load(path)))
        directory_reports[folder] = len(paths)
    spec = importlib.util.spec_from_file_location("test_terminal_footwork", ROOT / "tests/test_terminal_footwork.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    history = historical_tests()
    media = load(args.review_dir / "media_validation.json")
    for row in media["reports"]:
        if sha(Path(row["source"])) != row["source_sha256"] or sha(Path(row["playback_copy"])) != row["playback_sha256"]:
            errors.append(f"media hash mismatch: {row['clip_id']}")
    if media["clips"] != 39 or not media["full_decode_passed"]:
        errors.append("media evidence incomplete")
    passed = not errors and new.wasSuccessful() and history["passed"]
    report = {"passed": passed, "errors": errors, "structural_clips": directory_reports,
        "new_unit_tests": {"run": new.testsRun, "passed": new.wasSuccessful()}, "historical_regression": history,
        "total_tests": new.testsRun + history["tests_run"], "media_hashes_rechecked": len(media["reports"]),
        "media_decode_scope": "Previously fully decoded copies; source and playback hashes rechecked this run.",
        "human_verified": False, "accuracy_validated": False}
    save(args.run_dir / "validation.json", report)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
