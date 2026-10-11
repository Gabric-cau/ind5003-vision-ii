"""Check run completeness and coordinate integrity, never count accuracy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/shot_repair_2026-10-09/all_clips_final")
    args = parser.parse_args()
    with (ROOT / "data/processed/penalty_step_counts.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline = list(csv.DictReader(handle))
    code_hash = hashlib.sha256((ROOT / "src/step_pipeline.py").read_bytes()).hexdigest()
    errors, rows = [], []
    for old in baseline:
        clip_id = old["clip_id"]
        path = args.run_dir / f"{clip_id}.json"
        if not path.exists():
            errors.append(f"{clip_id}: missing result")
            continue
        result = json.loads(path.read_text(encoding="utf8"))
        start, end = result["shot_start_frame"], result["shot_end_frame"]
        if not (0 <= start < end <= result["source_frame_count"]):
            errors.append(f"{clip_id}: invalid shot bounds")
        frame_ids = [p["frame"] for p in result["pose_frames"]]
        if frame_ids != list(range(start, end)):
            errors.append(f"{clip_id}: pose frame IDs do not match selected shot")
        coordinate_frames = list(result["selected_pose_sequence"].get("frames", []))
        coordinate_frames.extend(o["frame"] for t in result["tracks"] for o in t["observations"])
        coordinate_frames.extend(result["step_frames"])
        coordinate_frames.extend(f for f in [result["kick_frame"], result["runup_start_frame"]] if f is not None)
        if any(not start <= frame < end for frame in coordinate_frames):
            errors.append(f"{clip_id}: frame coordinate leaked outside the shot")
        if result["step_count"] is not None and result["step_count"] != len(result["step_frames"]):
            errors.append(f"{clip_id}: inconsistent step count")
        if any(not result["runup_start_frame"] <= frame <= result["kick_frame"] for frame in result["step_frames"]):
            errors.append(f"{clip_id}: step lies outside the run-up interval")
        if result["field_mask_diagnostics"] is not None or result["run_metadata"]["field_mask"]:
            errors.append(f"{clip_id}: mask unexpectedly enabled")
        if result["run_metadata"].get("step_pipeline_sha256") != code_hash:
            errors.append(f"{clip_id}: result was produced by a different pipeline source")
        if result["shot_selection"]["human_verified"] or result["run_metadata"]["human_verified"]:
            errors.append(f"{clip_id}: unjustified human verification")
        qa_image = cv2.imread(str(args.run_dir / f"{clip_id}_qa.jpg"))
        if qa_image is None or qa_image.shape[1] != 960:
            errors.append(f"{clip_id}: QA image missing or not decodable at expected width")
        rows.append({"clip_id": clip_id, "old_status": old["status"], "status": result["status"],
                     "old_step_count": int(old["step_count"]) if old["step_count"] else None,
                     "step_count": result["step_count"], "review_required": result["review_required"],
                     "qualified_shot": result["shot_selection"]["found_qualified_shot"],
                     "shot_start_frame": start, "shot_end_frame": end})
    report = {
        "structural_validation_passed": not errors, "expected_rows": len(baseline), "actual_rows": len(rows),
        "successful_extractions": sum(r["status"] == "ok" for r in rows),
        "old_failures_recovered": sum(r["old_status"] == "failed" and r["status"] == "ok" for r in rows),
        "new_failures": [r["clip_id"] for r in rows if r["old_status"] == "ok" and r["status"] != "ok"],
        "review_required": sum(r["review_required"] for r in rows),
        "qualified_shots": sum(r["qualified_shot"] for r in rows),
        "unqualified_shot_fallbacks": [r["clip_id"] for r in rows if not r["qualified_shot"]],
        "changed_existing_counts": [r for r in rows if r["old_status"] == "ok" and r["step_count"] != r["old_step_count"]],
        "mask_used": False, "human_verified": False,
        "accuracy_validated": False, "errors": errors,
        "note": "Extraction completeness and coordinate consistency are not accuracy; all counts remain unverified predictions.",
    }
    (args.run_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
