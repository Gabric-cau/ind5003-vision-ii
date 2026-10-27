"""Package an explicit saved-result scope and verify it after extraction."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from src.final_results import sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/fresh_course_experiment"
OUTPUT = EXPERIMENT / "submission_review_2026-10-22"
DIRECTORIES = (
    "src", "scripts", "tests", "configs", "docs", "data/raw_clips",
    "data/course_context_clips_2026-10-12",
    "outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion",
    "outputs/fresh_course_experiment/final_eda_v3_shoe_fusion",
    "outputs/fresh_course_experiment/shoe_landing_replay_2026-10-18_v1",
    "outputs/fresh_course_experiment/shoe_approach_foot_supplement_v2",
    "outputs/fresh_course_experiment/review_v3_shoe_fusion_ui3",
    "outputs/fresh_course_experiment/human_evaluation_2026-10-22",
    "outputs/fresh_course_experiment/shoe_approach33_cohort_v1",
    "outputs/fresh_course_experiment/shoe_approach_analysis_v2",
    "outputs/fresh_course_experiment/pivot_support_development_2026-10-22",
    "outputs/fresh_course_experiment/feature_gap_audit_2026-10-22",
    "outputs/fresh_course_experiment/submission_review_2026-10-21",
    "outputs/fresh_course_experiment/submission_review_2026-10-22",
)
TESTS = (
    "tests/test_single_human_reference_integration.py", "tests/test_shoe_landing_evaluation.py",
    "tests/test_fresh_eda.py", "tests/test_fresh_results.py", "tests/test_replay_shoe_landings.py",
    "tests/test_shoe_landing_events.py", "tests/test_shoe_pivot_support.py",
)


def main():
    archive = OUTPUT / "IND5003_Saved_Result_Review_2026-10-22.zip"
    receipt_path = OUTPUT / "package_verification.json"
    paths = {ROOT / "README.md", ROOT / "REVIEW_PACKAGE_README.md", *ROOT.glob("requirements*.txt")}
    for directory in DIRECTORIES:
        paths.update(path for path in (ROOT / directory).rglob("*") if path.is_file())
    paths = sorted(path for path in paths if not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts)
                   and path.suffix not in {".pyc", ".zip"} and path != receipt_path)
    if any(path.is_symlink() or not path.resolve().is_relative_to(ROOT) for path in paths):
        raise ValueError("The explicit package scope may not follow external links")
    manifest = {str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in paths}
    prefix = "IND5003_Saved_Result_Review_2026-10-22"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in paths:
            zipped.write(path, f"{prefix}/{path.relative_to(ROOT)}")
        zipped.writestr(f"{prefix}/REVIEW_PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    checks = []
    with tempfile.TemporaryDirectory(prefix="ind5003-review-verify-") as temporary:
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("ZIP CRC validation failed")
            zipped.extractall(temporary)
        extracted = Path(temporary) / prefix
        for name, expected in manifest.items():
            path = extracted / name
            if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Extracted artifact differs: {name}")
        commands = [
            [sys.executable, "-m", "src.fresh_eda", "--bundle-dir", "outputs/fresh_course_experiment/final_bundle_v3_shoe_fusion", "--output-dir", "_verification/eda"],
            [sys.executable, "-m", "scripts.integrate_human_reference_2026_10_22", "--source", "outputs/fresh_course_experiment/human_evaluation_2026-10-22/reference_source", "--output", "_verification/human"],
            [sys.executable, "-m", "scripts.run_pivot_support_development_2026_10_22"],
            [sys.executable, "-m", "scripts.audit_feature_gaps_2026_10_22"],
            [sys.executable, "-m", "pytest", "-q", *TESTS],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=extracted, env={**os.environ, "PYTHONPATH": str(extracted)},
                                    text=True, capture_output=True, timeout=120, check=True)
            checks.append({"command": command[1:], "exit_code": result.returncode,
                           "stdout": result.stdout, "stderr": result.stderr})
        human = json.loads((extracted / "_verification/human/integration_summary.json").read_text())
        original = json.loads((EXPERIMENT / "human_evaluation_2026-10-22/integration_summary.json").read_text())
        if human != original:
            raise ValueError("Extracted single-reviewer comparison differs")
        summary = json.loads((extracted / "_verification/eda/summary.json").read_text())
        expected = json.loads((EXPERIMENT / "final_bundle_v3_shoe_fusion/summary.json").read_text())
        if summary != expected:
            raise ValueError("Extracted EDA differs")
        for name in ("pivot_support_development_2026-10-22", "feature_gap_audit_2026-10-22"):
            actual = json.loads((extracted / "outputs/fresh_course_experiment" / name / "summary.json").read_text())
            expected = json.loads((EXPERIMENT / name / "summary.json").read_text())
            if actual != expected:
                raise ValueError(f"Extracted replay/audit differs: {name}")
    receipt = {
        "date": "2026-10-22", "status": "saved_result_review_package", "submitted": False,
        "archive": archive.name, "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
        "packaged_artifacts_verified_after_extraction": len(manifest), "zip_crc_valid": True,
        "separate_directory_reproduction": True, "eda_summary_reproduced": True,
        "human_diagnostics_reproduced": True, "fresh_dependency_installation_tested": False,
        "saved_pose_pivot_replay_reproduced": True, "feature_gap_audit_reproduced": True,
        "full_raw_video_extraction_reproduced": False, "new_neural_inference_calls": 0,
        "python": sys.version, "checks": checks,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
