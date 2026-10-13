"""Join saved experiments without turning references or proxies into predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(rows: list[dict], expected: list[str], name: str,
            require_all: bool = True) -> dict[str, dict]:
    result = {}
    for row in rows:
        clip_id = row.get("clip_id")
        if clip_id not in expected:
            raise ValueError(f"{name}: non-canonical clip_id {clip_id!r}")
        if clip_id in result:
            raise ValueError(f"{name}: duplicate clip_id {clip_id}")
        result[clip_id] = row
    if require_all and set(result) != set(expected):
        raise ValueError(f"{name}: missing canonical clips {sorted(set(expected) - set(result))}")
    return result


def canonical_events(manifest: dict) -> list[dict]:
    events = []
    for match_id, match in manifest.items():
        for penalty in match["penalties"]:
            events.append({
                "clip_id": f"{match_id}_p{penalty['number']:02d}_{penalty['kicker']}",
                "match_id": match_id,
                "event_id": len(events) + 1,
                "kicker": penalty["kicker"], "team": penalty["team"],
                "penalty_number": penalty["number"],
                "slice_start_sec": penalty["start_sec"],
                "slice_end_sec": penalty["end_sec"],
            })
    if len(events) != 39 or len({e["clip_id"] for e in events}) != 39:
        raise ValueError("The course manifest must contain exactly 39 unique events")
    return events


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric measurement: {value!r}")
    return number


def _flag(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"Expected a JSON boolean flag, got {value!r}")
    return value


def _known(value: Any) -> bool:
    return value is not None and value != "" and value != "unknown"


def compare_ball_references(rows: list[dict]) -> dict:
    mapping = {"goal_likely": "goal", "no_goal_likely": "no_goal"}
    confusion = {ref: {pred: 0 for pred in ("goal", "no_goal", "unknown")}
                 for ref in ("goal", "no_goal")}
    conflicts = []
    covered = matching = 0
    for row in rows:
        reference = row["reference_outcome"]
        prediction = mapping.get(row["ball_rule_outcome"], "unknown")
        confusion[reference][prediction] += 1
        if prediction != "unknown":
            covered += 1
            if prediction == reference:
                matching += 1
            else:
                conflicts.append({"clip_id": row["clip_id"], "event_id": row["event_id"],
                                  "prediction": prediction, "reference": reference})
    return {"total": len(rows), "covered": covered, "coverage": covered / len(rows),
            "matching": matching, "covered_agreement": matching / covered if covered else None,
            "all_event_agreement": matching / len(rows), "confusion_matrix": confusion,
            "conflicts": conflicts, "interpretation": "development_reference_agreement_not_test_accuracy"}


def build_bundle(root: Path, config_path: Path, output_dir: Path,
                 raw_clips_dir: Path | None = None) -> dict:
    root = root.resolve()
    if raw_clips_dir is not None and not raw_clips_dir.is_absolute():
        raw_clips_dir = root / raw_clips_dir
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported final-analysis configuration schema")
    input_records: dict[str, dict] = {}
    missing_optional = []

    def record(path: Path) -> None:
        try:
            name = path.resolve().relative_to(root).as_posix()
        except ValueError:
            location_hash = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
            name = f"external/{location_hash}/{path.name}"
        input_records[name] = {"path": name, "sha256": sha256(path)}

    def input_path(key: str, optional: bool = False) -> Path | None:
        value = config.get(key)
        path = Path(value) if value else None
        if path is not None and not path.is_absolute():
            path = root / path
        if path is None or not path.is_file():
            if optional:
                missing_optional.append({"key": key, "path": value, "status": "unavailable"})
                return None
            raise FileNotFoundError(f"Required input {key}: {path}")
        record(path)
        return path

    def json_input(key: str) -> Any:
        return json.loads(input_path(key).read_text(encoding="utf-8-sig"))

    record(config_path)
    manifest = yaml.safe_load(input_path("manifest").read_text(encoding="utf-8"))
    events = canonical_events(manifest)
    expected = [e["clip_id"] for e in events]
    canonical = indexed(json_input("canonical_results")["rows"], expected, "canonical results")
    heights = indexed(json_input("goalkeeper_height_summary")["rows"], expected, "height summary")
    baseline = indexed(read_csv(input_path("goalkeeper_baseline_csv")), expected, "keeper baseline")
    outcomes = indexed(read_csv(input_path("outcome_references")), expected, "outcome references")
    foot_ref_data = json_input("foot_references")
    foot_refs = indexed(foot_ref_data["reviews"], expected, "foot references", False)
    foot_eval = json_input("foot_development_evaluation")
    if foot_eval.get("held_out_test") is not False or foot_eval.get("development_cohort") is not True:
        raise ValueError("The saved foot development cohort cannot be labelled held-out")
    if foot_eval.get("reference_sha256") != sha256(input_path("foot_references")):
        raise ValueError("Foot development comparison does not match the supplied reference artifact")
    foot_experiments = indexed(foot_eval["records"], expected, "foot development records", False)
    if set(foot_experiments) != set(foot_refs):
        raise ValueError("Foot development records and supplied references have different scopes")
    for clip_id, experiment in foot_experiments.items():
        shortcut = experiment.get("shortcut", {})
        if shortcut.get("reviewer_labels_used_for_prediction") is not False:
            raise ValueError(f"Foot candidate has missing/unsafe reference-use provenance: {clip_id}")
        if experiment.get("user_reviewed_kicking_foot") != foot_refs[clip_id]["kicking_foot"]:
            raise ValueError(f"Foot comparison reference mismatch: {clip_id}")
    camera_path = input_path("camera_intervals", optional=True)
    input_path("camera_intervals_json", optional=True)
    camera_rows = []
    camera_counts = Counter()
    camera_known_counts = Counter()
    if camera_path:
        camera_rows = read_csv(camera_path)
        camera_segment_ids = set()
        for item in camera_rows:
            clip_id = item.get("clip_id")
            if clip_id not in expected:
                raise ValueError(f"Camera intervals: non-canonical clip_id {clip_id!r}")
            key = (clip_id, item["segment_id"])
            if key in camera_segment_ids:
                raise ValueError(f"Duplicate camera segment: {key}")
            camera_segment_ids.add(key)
            start, end = _number(item["start_time_sec"]), _number(item["end_time_sec"])
            if start is None or end is None or start < 0 or end <= start:
                raise ValueError(f"Invalid camera interval: {key}")
            camera_counts[clip_id] += 1
            if item["task_class_candidate"] != "unknown":
                camera_known_counts[clip_id] += 1

    ball_dir = Path(config["ball_results_dir"])
    if not ball_dir.is_absolute():
        ball_dir = root / ball_dir
    rows = []
    for event in events:
        clip_id = event["clip_id"]
        saved, height, gk, reference = canonical[clip_id], heights[clip_id], baseline[clip_id], outcomes[clip_id]
        ball_path = ball_dir / f"{clip_id}.json"
        if not ball_path.is_file():
            raise FileNotFoundError(f"Missing ball result: {ball_path}")
        record(ball_path)
        ball = json.loads(ball_path.read_text(encoding="utf-8"))
        if ball.get("clip_id") != clip_id or ball.get("event_id") != event["event_id"]:
            raise ValueError(f"Ball result identity mismatch: {clip_id}")
        if ball.get("outcome_labels_used_for_algorithm") is not False:
            raise ValueError(f"Ball prediction has missing/unsafe reference-use provenance: {clip_id}")
        outcome = ball["rule_decision"]["rule_outcome"]
        if outcome not in ("goal_likely", "no_goal_likely", "unknown"):
            raise ValueError(f"Unsupported ball outcome: {outcome!r}")
        if reference["outcome"] not in ("goal", "no_goal"):
            raise ValueError(f"Unsupported reference outcome: {clip_id}")
        source_hash = ball.get("source_sha256")
        if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hash):
            raise ValueError(f"Missing source video SHA-256: {clip_id}")
        source_verified = False
        if raw_clips_dir is not None:
            video = raw_clips_dir / f"{clip_id}.mp4"
            if not video.is_file() or sha256(video) != source_hash:
                raise ValueError(f"Source video does not match stored experiment: {clip_id}")
            source_verified = True
        trajectory = ball.get("trajectory", [])
        last = trajectory[-1] if trajectory else None
        foot = saved["foot"]
        steps = saved["steps"]
        sensitivity = steps.get("parameter_range") or [None, None]
        # Display references separately even when the original release included overrides.
        direction = height.get("automatic_dive_direction", "unknown")
        row = {**event,
               "kicking_foot_candidate": foot.get("nominal_v4_candidate", "unknown"),
               "kicking_foot_stable_candidate": foot.get("parameter_stable_candidate", "unknown"),
               "kicking_foot_reference": foot_refs.get(clip_id, {}).get("kicking_foot"),
               "kicking_foot_reference_source": foot_ref_data["source"] if clip_id in foot_refs else None,
               "kicking_foot_development_candidate": foot_experiments.get(clip_id, {}).get("shortcut", {}).get("experimental_kicking_foot_candidate"),
               "step_count_candidate": steps.get("v4_candidate_count"),
               "step_count_verified": None,
               "step_count_reference_recorded": steps.get("verified_count"),
               "step_count_verification_status": "no_independent_step_review_artifact",
               "step_count_sensitivity_min": sensitivity[0],
               "step_count_sensitivity_max": sensitivity[-1],
               "partial_runup": _flag(steps.get("partial_approach_known")),
               "runup_ground_plane_validated": _flag(saved["homography"].get("validated_ground_plane")),
               "runup_ground_plane_units": "unavailable_without_validated_ground_homography",
               "goalkeeper_pre_movement_candidate": gk.get("gk_prekick_movement_level", "unknown"),
               "goalkeeper_pre_speed_px_s_candidate": _number(gk.get("gk_prekick_mean_speed_px_s")),
               "goalkeeper_pre_movement_validated": False,
               "goalkeeper_pre_movement_status": "legacy_candidate_requires_recompute_after_stabilization_fix",
               "goalkeeper_direction_candidate": direction,
               "goalkeeper_direction_coordinate_system": "source_image_x_right",
               "goalkeeper_direction_validated": False,
               "goalkeeper_identity_verified": _flag(height.get("identity_verified")),
               "goalkeeper_height_candidate": height.get("phase_height_candidate", "unknown"),
               "goalkeeper_height_proxy_candidate": height.get("silhouette_height_proxy_candidate", "unknown"),
               "goalkeeper_height_legacy_candidate": height.get("previous_height", "unknown"),
               "goalkeeper_height_validated": False,
               "ball_rule_outcome": outcome,
               "ball_unknown_reason": ball["rule_decision"].get("reason"),
               "ball_trajectory_point_count": len(trajectory),
               "ball_trajectory_validated": False,
               "ball_trajectory_coordinate_system": "source_image_pixels",
               "ball_last_observed_x_px": _number(last.get("x")) if last else None,
               "ball_last_observed_y_px": _number(last.get("y")) if last else None,
               "ball_last_observed_frame": last.get("frame") if last else None,
               "ball_final_position_verified": False,
               "ball_event_final_x_px": None, "ball_event_final_y_px": None,
               "ball_event_final_position_status": "event_end_not_verified",
               "whole_ball_crossing_established": _flag(ball["rule_decision"].get("whole_ball_crossing_established")),
               "reference_outcome": reference["outcome"],
               "reference_outcome_detail": reference.get("outcome_detail"),
               "ball_reference_source": reference.get("source"),
               "ball_reference_evidence_url": reference.get("evidence_url"),
               "ball_reference_source_verified": False,
               "source_sha256": source_hash,
               "source_video_hash_verified": source_verified,
               "camera_interval_count": camera_counts.get(clip_id, 0) if camera_path else None,
               "camera_known_course_class_interval_count": camera_known_counts.get(clip_id, 0) if camera_path else None,
               "camera_interval_status": "candidate_intervals_classes_unverified" if camera_path else "supplement_not_available",
               "review_required": True,
               "prediction_source_ball": ball_path.relative_to(root).as_posix() if ball_path.is_relative_to(root) else ball_path.name,
               "prediction_source_keeper": config["goalkeeper_height_summary"],
               "prediction_source_steps_foot": config["canonical_results"],
               }
        rows.append(row)

    source_hashes = {row["clip_id"]: row["source_sha256"] for row in rows}
    for item in camera_rows:
        if item["source_sha256"] != source_hashes[item["clip_id"]]:
            raise ValueError(f"Camera and ball results use different source videos: {item['clip_id']}")

    candidate_columns = ["kicking_foot_candidate", "kicking_foot_stable_candidate", "step_count_candidate",
                         "goalkeeper_pre_movement_candidate", "goalkeeper_direction_candidate",
                         "goalkeeper_height_candidate", "goalkeeper_height_proxy_candidate", "ball_rule_outcome"]
    evaluation = {"schema_version": 1, "total_events": 39, "data_scope": "development",
                  "held_out_test": False, "independent_holdout": False, "accuracy_validated": False,
                  "reference_source_verified": False, "reference_sources_independently_verified": False,
                  "ball_rule_reference_comparison": compare_ball_references(rows),
                  "candidate_coverage": {key: sum(_known(r[key]) for r in rows) for key in candidate_columns},
                  "validated_coverage": {
                      "step_count_verified": sum(r["step_count_verified"] is not None for r in rows),
                      "runup_ground_plane_validated": sum(r["runup_ground_plane_validated"] for r in rows),
                      "goalkeeper_direction_validated": 0, "goalkeeper_height_validated": 0,
                      "goalkeeper_pre_movement_validated": 0, "ball_trajectory_validated": 0,
                      "ball_final_position_verified": 0,
                      "kicking_foot_user_reference_records": len(foot_refs),
                  },
                  "foot_development_comparison": {"total": len(foot_experiments),
                      "comparisons": foot_eval["comparisons"], "held_out_test": False,
                      "interpretation": "method_selection_cohort_not_generalization_accuracy"},
                  "missing_optional_inputs": missing_optional,
                  "camera_interval_diagnostics": {
                      "available": bool(camera_path), "total_intervals": len(camera_rows),
                      "clips_with_intervals": len(camera_counts),
                      "clips_with_course_class_candidates": len(camera_known_counts),
                      "task_class_candidate_counts": dict(Counter(item["task_class_candidate"] for item in camera_rows)),
                      "playback_status_counts": dict(Counter(item["playback_status"] for item in camera_rows)),
                      "human_verified_intervals": sum(item["human_verified"].lower() == "true" for item in camera_rows),
                      "interpretation": "saved_candidate_intervals_not_new_inference_or_class_accuracy"},
                  "research_questions": {
                      "RQ1": "How does feature observability vary across the four matches?",
                      "RQ2": "How sensitive are candidate step counts to parameter choices and partial run-ups?",
                      "RQ3": "When do ball rules abstain or disagree with stored outcome references?"},
                  "limitations": ["Stored results reused; no new raw-video inference in this bundle",
                                  "Raw source hashes are claims from upstream until individually checked",
                                  "Keeper pre-movement must be recomputed with the repaired stabilizer",
                                  "Strict phase heights and verified event endpoints remain unavailable",
                                  "No independent holdout accuracy can be calculated"]}
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty analysis directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(name: str, data: dict) -> None:
        (output_dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")

    table_path = output_dir / "features_39.csv"
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json("features_39.json", {"schema_version": 1, "rows": rows})
    if camera_rows:
        with (output_dir / "camera_intervals_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(camera_rows[0]))
            writer.writeheader()
            writer.writerows(camera_rows)
    write_json("evaluation.json", evaluation)
    write_json("provenance.json", {"schema_version": 1, "artifact_type": "final_analysis_candidate_bundle",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "table": {"path": table_path.name, "sha256": sha256(table_path)},
                "canonical_clip_ids": expected, "inputs": list(input_records.values()),
                "missing_optional_inputs": missing_optional,
                "raw_source_verification": "all_39_hashes_match" if raw_clips_dir else "not_performed",
                "held_out_test": False})
    write_json("manifest.json", {"schema_version": "vision2-final-analysis-v1", "clip_ids": expected,
                "features_csv_sha256": sha256(table_path),
                "evaluation_json_sha256": sha256(output_dir / "evaluation.json")})
    with (output_dir / "human_review_template.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["clip_id", "source_sha256", "review_status", "reviewer", "review_date",
                  "source_video_hash_verified", "runup_start_frame", "contact_frame", "event_end_frame",
                  "runup_complete", "step_count_reference", "kicking_foot_reference",
                  "goalkeeper_identity_reference", "goalkeeper_direction_image_reference",
                  "goalkeeper_height_reference", "ball_final_x_reference", "ball_final_y_reference",
                  "outcome_reference", "evidence_frames", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"clip_id": r["clip_id"], "source_sha256": r["source_sha256"],
                          "review_status": "pending", "source_video_hash_verified": False} for r in rows)
    return evaluation
