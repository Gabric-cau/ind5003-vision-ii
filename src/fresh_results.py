"""Bind fresh experiment outputs to one source cohort and summarize candidates."""
from collections import Counter
import csv
import json
import math
from pathlib import Path

from src.final_results import sha256


STAGE_PATHS = {"pose": "{clip_id}.json", "ball": "results/{clip_id}.json",
               "foot": "results/{clip_id}.json", "phase": "{clip_id}.json",
               "pitch": "{clip_id}.json", "camera": "clips/{clip_id}.json",
               "endpoint": "results/{clip_id}.json", "height": "{clip_id}.json", "scoreboard": "{clip_id}.json"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def contact_timing_audit(contact, source):
    frame, stored = contact.get("contact_frame"), contact.get("contact_time_sec")
    clip_time = frame / source["fps"] if frame is not None else None
    source_time = ((source["source_start_frame"] + frame) / source["fps"]
                   if frame is not None and source.get("source_start_frame") is not None else None)
    consistent = math.isclose(stored, clip_time, rel_tol=0, abs_tol=1e-9) if stored is not None and clip_time is not None else None
    return {"candidate_frame": frame, "candidate_clip_time_sec": clip_time,
            "candidate_source_time_sec": source_time, "producer_stored_time_sec": stored,
            "producer_time_matches_final_frame": consistent, "time_basis": "final_candidate_frame_and_source_fps",
            "producer_artifact_modified": False}


def bind_stage(stage, data, source, pose_hash, manifest_hash):
    if data.get("clip_id") != source["clip_id"]:
        raise ValueError(f"{stage}: clip ID mismatch")
    identity = data.get("inference_identity") or data.get("calibration_identity") or data.get("provenance") or {}
    mapping = data.get("source_mapping") or {}
    source_hash = (identity.get("source_sha256") or identity.get("source_clip_sha256")
                   or mapping.get("clip_sha256") or data.get("source_sha256"))
    if source_hash != source["clip_sha256"]:
        raise ValueError(f"{stage}: source bytes mismatch")
    if identity.get("source_manifest_sha256") not in (None, manifest_hash):
        raise ValueError(f"{stage}: source manifest mismatch")
    if stage == "pose":
        if data.get("new_neural_inference") is not True or data.get("reference_labels_used_for_prediction") is not False:
            raise ValueError("Pose must originate from fresh source-only neural inference")
    elif stage != "scoreboard":
        bound_pose = identity.get("pose_sha256") or data.get("pose_sha256") or (data.get("pose_input") or {}).get("sha256")
        if bound_pose != pose_hash:
            raise ValueError(f"{stage}: different or unbound pose producer")
    if stage == "scoreboard" and (data.get("source_video_sha256") != source.get("source_video_sha256")
                                  or data.get("frame_index_basis") != "original_full_video"
                                  or data.get("outcome_labels_read") is not False):
        raise ValueError("Scoreboard must be bound to original source frames without outcome labels")
    if data.get("human_verified") is not False:
        raise ValueError(f"{stage}: human verification needs a separate review record")
    if data.get("reference_labels_used_for_prediction") is True or data.get("reference_labels_read") is True:
        raise ValueError(f"{stage}: prediction must not depend on outcome references")
    if stage == "endpoint" and (data.get("final_position") is not None or data.get("final_position_verified") is True):
        raise ValueError("An automatic endpoint candidate cannot verify an event-final position")
    if stage == "height":
        if data.get("physical_height_metres") is not None:
            raise ValueError("Image-plane keeper evidence cannot establish physical height")
        signal = data.get("observed_image_plane_signal") or {}
        if signal.get("reference_depth_assumption_failed") is True and (
                signal.get("candidate") != "unknown" or signal.get("supported") is not False
                or data.get("observed_image_plane_height_candidate") != "unknown"):
            raise ValueError("Invalid height reference cannot support an observed category")
    if stage == "foot" and data.get("schema_version") == "fresh-foot-pose33-fusion-v1":
        from src.shoe_pose33_fusion import validate_fused_foot
        validate_fused_foot(data, source, pose_hash, manifest_hash)
    return data


def compact_stage(stage, data):
    if not data:
        return {"status": "not_run", "reason": "experiment_component_not_supplied"}
    fields = {
        "pose": ["selected_track_id", "selected_track_qualified", "kick_frame", "step_count",
                 "shot_start_frame", "shot_end_frame", "shot_selection", "qa_warnings"],
        "ball": ["independent_contact_search", "ball_departure", "rule_decision", "trajectory", "keeper",
                 "final_position", "event_cache_sha256", "contact_search_sha256", "source_cache_sha256"],
        "foot": ["kicking_foot_candidate", "timing", "temporal_evidence", "local_seed_identity_ambiguous_frames",
                 "normal_pose_frames", "mirrored_pose_frames", "upstream_scene_rejections"],
        "phase": ["candidate_step_count", "verified_step_count", "contact", "phase_result", "count_sensitivity",
                  "step_analysis", "footwork_result", "camera_motion", "analysis_actor_track_id"],
        "pitch": ["supported_frame_count", "supported_frame_count_in_manual_slice",
                  "supported_frame_count_in_selected_pose_frames", "pre_kick_contact_candidate_count",
                  "sparse_ground_contact_candidates", "ground_contact_candidate_count", "pose_input", "independent_contact"],
        "camera": ["timeline_coverage", "contact_anchor"],
        "endpoint": ["last_observed_frame", "last_observed_position", "stationary_end_candidate",
                     "boundary_evidence", "contact_bracket_frames", "final_position", "final_position_verified"],
        "height": ["strict_height_candidate", "ordinal_pose_height_candidate", "pose_only_phase",
                   "mask_phase", "reference", "mask_provenance", "physical_height_metres",
                   "observed_image_plane_signal", "observed_image_plane_height_candidate"],
        "scoreboard": ["scoreboard_outcome", "witness_images", "search_interval_frames", "frame_index_basis",
                       "source_video_sha256", "scoreboard_protocol_sha256"],
    }
    result = {key: data[key] for key in fields[stage] if key in data}
    result.update(status=data.get("status", "completed_candidate_experiment"), reason=data.get("reason"),
                  human_verified=False, accuracy_validated=False)
    if stage == "ball":
        contact = (data.get("independent_contact_search") or {}).get("selected_candidate") or {}
        result["contact_timing_audit"] = contact_timing_audit(contact, data["source_mapping"])
    if stage == "camera":
        result["segments"] = [{key: segment[key] for key in (
            "start_frame", "end_frame_exclusive", "start_sec", "end_sec_exclusive", "view_candidate",
            "task_class_candidate", "playback_evidence", "course_source_start_sec", "course_source_end_sec_exclusive")
            if key in segment} for segment in data["segments"]]
    return result


def combine_outcomes(ball, scoreboard):
    allowed = {"goal_likely", "no_goal_likely"}
    if ball in allowed and scoreboard in allowed:
        return (ball, "ball_and_scoreboard_consensus") if ball == scoreboard else ("unknown", "source_evidence_conflict")
    if scoreboard in allowed:
        return scoreboard, "scoreboard_only"
    if ball in allowed:
        return ball, "ball_only"
    return "unknown", "insufficient_source_evidence"


def feature_row(source, stages, reference):
    pose, ball, foot, phase, pitch, camera, endpoint, height, scoreboard = (stages.get(name) or {} for name in STAGE_PATHS)
    search = ball.get("independent_contact_search") or {}
    contact = search.get("selected_candidate") or {}
    clock = contact_timing_audit(contact, source)
    keeper = ball.get("keeper") or {}
    segments = camera.get("segments") or []
    counts = [row["approach_candidates_including_precontact_strike_window"]
              for row in (phase.get("count_sensitivity") or {}).values()
              if isinstance(row.get("approach_candidates_including_precontact_strike_window"), int)]
    approach = [row for row in (phase.get("phase_result") or {}).get("segments", [])
                if row.get("phase") in ("initiation", "approach")]
    ball_outcome = (ball.get("rule_decision") or {}).get("rule_outcome", "unknown")
    scoreboard_outcome = (scoreboard.get("scoreboard_outcome") or {}).get("rule_outcome", "unknown")
    combined_outcome, combined_source = combine_outcomes(ball_outcome, scoreboard_outcome)
    return {"clip_id": source["clip_id"], "event_id": source["event_id"], "match_id": source["match_id"],
            "kicker": source["kicker"], "team": source["team"], "source_sha256": source["clip_sha256"],
            "source_fps": source["fps"],
            "pose_actor_baseline": pose.get("selected_track_id"), "pose_peak_frame_baseline": pose.get("kick_frame"),
            "contact_frame_candidate": contact.get("contact_frame"),
            "contact_time_clip_sec_candidate": clock["candidate_clip_time_sec"],
            "contact_time_source_sec_candidate": clock["candidate_source_time_sec"],
            "contact_producer_time_consistent": clock["producer_time_matches_final_frame"],
            "contact_actor_candidate": contact.get("contact_associated_track_id"),
            "contact_status": search.get("status", "not_run"),
            "contact_actor_reselected": (contact["contact_associated_track_id"] != pose.get("selected_track_id")) if contact else None,
            "kicking_foot_candidate": foot.get("kicking_foot_candidate", "unknown"),
            "foot_status": foot.get("status", "not_run"),
            "steps_pose_baseline": pose.get("step_count"), "steps_phase_candidate": phase.get("candidate_step_count"),
            "steps_sensitivity_min": min(counts) if counts else None,
            "steps_sensitivity_max": max(counts) if counts else None,
            "visible_approach_duration_sec": ((contact["contact_frame"] - min(r["start_frame"] for r in approach))
                                             / source["fps"]) if approach and contact else None,
            "phase_status": phase.get("status", "not_run"),
            "phase_camera_chain_complete": (phase.get("camera_motion") or {}).get("reference_chain_complete"),
            "keeper_pre_movement_candidate": keeper.get("gk_prekick_movement_level", "unknown"),
            "keeper_pre_camera_chain_complete": keeper.get("gk_prekick_stabilization_complete"),
            "keeper_direction_image_candidate": keeper.get("dive_direction", "unknown"),
            "keeper_legacy_height_candidate": keeper.get("dive_height", "unknown"),
            "keeper_strict_height_candidate": height.get("strict_height_candidate", "unknown"),
            "keeper_pose_ordinal_height_candidate": height.get("ordinal_pose_height_candidate", "unknown"),
            "keeper_observed_image_height_candidate": height.get("observed_image_plane_height_candidate", "unknown"),
            "keeper_height_status": height.get("reason", "not_run"),
            "ball_trajectory_observations": len(ball["trajectory"]) if "trajectory" in ball else None,
            "ball_rule_outcome": ball_outcome,
            "scoreboard_rule_outcome": scoreboard_outcome,
            "combined_rule_outcome": combined_outcome,
            "combined_outcome_source": combined_source,
            "ball_endpoint_status": endpoint.get("status", "not_run"),
            "ball_endpoint_last_frame": endpoint.get("last_observed_frame"),
            "ground_supported_selected_frames": pitch.get("supported_frame_count_in_selected_pose_frames", pitch.get("supported_frame_count")),
            "ground_pre_contact_points": pitch.get("pre_kick_contact_candidate_count"),
            "camera_intervals": len(segments) if camera else None,
            "camera_target_class_candidates": sum(s["task_class_candidate"] != "unknown" for s in segments),
            "camera_slow_motion_candidates": sum((s.get("playback_evidence") or {}).get("status") == "slow_motion_candidate" for s in segments),
            "camera_timeline_coverage": camera.get("timeline_coverage"),
            "external_outcome_reference": reference["reference_outcome"],
            "external_reference_source_supported": reference["reference_supported"],
            "human_verified": False, "physical_keeper_height_verified": False,
            "event_final_ball_position_verified": False, "ground_metric_accuracy_verified": False}


def evaluate_outcomes(rows, column):
    confusion, disagreements = Counter(), []
    lookup = {"goal_likely": "goal", "no_goal_likely": "no_goal"}
    for row in rows:
        prediction = lookup.get(row[column])
        if prediction is not None:
            truth = row["external_outcome_reference"]
            confusion[(truth, prediction)] += 1
            if prediction != truth:
                disagreements.append(row["clip_id"])
    covered = sum(confusion.values())
    agree = sum(value for (truth, pred), value in confusion.items() if truth == pred)
    return {"decided": covered, "abstained": len(rows) - covered,
            "reference_agreements": agree, "covered_reference_agreement": agree / covered if covered else None,
            "decision_coverage": covered / len(rows) if rows else None,
            "confusion": [{"reference": truth, "prediction": pred, "events": value}
                          for (truth, pred), value in sorted(confusion.items())],
            "disagreements": disagreements, "held_out_test": False, "accuracy_validated": False,
            "interpretation": "Development agreement against audited named-event references; video event identity is unverified"}


def summarize(rows):
    availability = {
        "contact": sum(r["contact_frame_candidate"] is not None for r in rows),
        "foot": sum(r["kicking_foot_candidate"] in ("left", "right") for r in rows),
        "phase_steps": sum(r["steps_phase_candidate"] is not None for r in rows),
        "keeper_pre_movement": sum(r["keeper_pre_movement_candidate"] not in (None, "unknown") for r in rows),
        "keeper_direction_image": sum(r["keeper_direction_image_candidate"] not in (None, "unknown") for r in rows),
        "ball_nonempty_trace": sum((r["ball_trajectory_observations"] or 0) > 0 for r in rows),
        "ground_pre_contact_points": sum((r["ground_pre_contact_points"] or 0) > 0 for r in rows),
        "camera_target_class": sum(r["camera_target_class_candidates"] > 0 for r in rows),
        "camera_slow_motion": sum(r["camera_slow_motion_candidates"] > 0 for r in rows),
        "keeper_strict_height": sum(r["keeper_strict_height_candidate"] in ("low", "mid", "high") for r in rows),
        "ball_stationary_end": sum(r["ball_endpoint_status"] == "stationary_end_candidate" for r in rows),
        "keeper_observed_image_height": sum(r["keeper_observed_image_height_candidate"] in ("low", "mid", "high") for r in rows),
        "scoreboard_outcome": sum(r["scoreboard_rule_outcome"] != "unknown" for r in rows),
        "combined_outcome": sum(r["combined_rule_outcome"] != "unknown" for r in rows),
    }
    return {"events": len(rows), "candidate_availability": availability,
            "matches": dict(Counter(r["match_id"] for r in rows)),
            "ball_rules": dict(Counter(r["ball_rule_outcome"] for r in rows)),
            "scoreboard_rules": dict(Counter(r["scoreboard_rule_outcome"] for r in rows)),
            "combined_rules": dict(Counter(r["combined_rule_outcome"] for r in rows)),
            "ball_evaluation": evaluate_outcomes(rows, "ball_rule_outcome"),
            "scoreboard_evaluation": evaluate_outcomes(rows, "scoreboard_rule_outcome"),
            "combined_evaluation": evaluate_outcomes(rows, "combined_rule_outcome"),
            "verified_measurement_events": 0, "unknown_is_zero": False,
            "scope": "Candidate extraction from hashed course context clips; no guaranteed marks or human ground truth"}


def build_bundle(source_manifest, audit_path, stage_dirs, output_dir):
    source_manifest, audit_path, output_dir = map(Path, (source_manifest, audit_path, output_dir))
    source, audit = read_json(source_manifest), read_json(audit_path)
    sources = source["clips"]
    if len(sources) != 39 or len({r["clip_id"] for r in sources}) != 39:
        raise ValueError("The final course bundle needs exactly 39 distinct canonical source events")
    refs = {r["clip_id"]: r for r in audit["records"]}
    if set(refs) != {r["clip_id"] for r in sources} or len(audit["records"]) != 39:
        raise ValueError("Outcome audit and source cohort do not match")
    if any(not r["reference_supported"] or r["reference_outcome"] not in ("goal", "no_goal") for r in refs.values()):
        raise ValueError("Each external outcome reference needs explicit primary-source support")
    if output_dir.exists():
        raise FileExistsError("Use a new directory to preserve the frozen experiment bundle")
    manifest_hash = sha256(source_manifest)
    rows, events, inputs = [], [], []
    for source_record in sources:
        cid = source_record["clip_id"]
        if refs[cid]["match_id"] != source_record["match_id"]:
            raise ValueError("Reference match identity mismatch")
        stages, artifacts = {}, {}
        pose_hash = None
        for stage, pattern in STAGE_PATHS.items():
            directory = stage_dirs.get(stage)
            if directory is None:
                if stage == "pose":
                    raise ValueError("Fresh pose input is required")
                continue
            path = Path(directory) / pattern.format(clip_id=cid)
            data = read_json(path)
            stages[stage] = bind_stage(stage, data, source_record, pose_hash, manifest_hash)
            if stage == "foot" and data.get("schema_version") == "fresh-foot-pose33-fusion-v1":
                from src.shoe_pose33_fusion import validate_fused_ball_contract
                if "ball" not in stages:
                    raise ValueError("Foot fusion needs the bound ball stage in the same bundle")
                validate_fused_ball_contract(data, stages["ball"])
            digest = sha256(path)
            if stage == "pose":
                pose_hash = digest
            artifacts[stage] = {"path": str(path), "sha256": digest}
        rows.append(feature_row(source_record, stages, refs[cid]))
        events.append({"clip_id": cid, "source_mapping": source_record,
                       "components": {name: compact_stage(name, stages.get(name)) for name in STAGE_PATHS},
                       "artifact_bindings": artifacts, "external_reference": refs[cid]})
        inputs.append({"clip_id": cid, "artifacts": artifacts})
    output_dir.mkdir(parents=True)
    write_json(output_dir / "features_39.json", {"schema_version": "fresh-course-bundle-v1", "events": events})
    with (output_dir / "features_39.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output_dir / "summary.json", summarize(rows))
    write_json(output_dir / "provenance.json", {"source_manifest": {"path": str(source_manifest), "sha256": manifest_hash},
        "reference_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "component_inputs": inputs, "builder_sha256": sha256(Path(__file__)),
        "new_neural_inference_by_builder": False, "reference_labels_used_for_prediction": False,
        "reference_joined_after_prediction": True, "human_verified": False})
    write_json(output_dir / "manifest.json", {"schema_version": "fresh-course-bundle-v1", "events": 39,
        "artifacts": {name: sha256(output_dir / name) for name in
                      ("features_39.json", "features_39.csv", "summary.json", "provenance.json")}})
    return summarize(rows)
