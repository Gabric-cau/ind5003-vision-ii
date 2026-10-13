"""Reanalyse fresh pose with independent ball contact and full-shot camera motion."""
from collections import Counter
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write
from scripts.run_penalty_phase_experiment import classify, counts_for, render_sheet, sequence
from src.fresh_cache_provenance import validate_cached_neural_producer
from src.motion_stabilization import apply_transforms, estimate_background_motion
from src.penalty_phases import classify_penalty_phases
from src.phase_step_events import propose_step_events
from src.style_aware_steps import analyze_style_aware_steps
from src.terminal_footwork import TerminalConfig, analyze_terminal_footwork


def _unknown_contact(fps, reason):
    return {"status": "unknown", "contact_frame": None, "contact_time_sec": None,
            "interval_frames": None, "confidence": 0., "fps": fps,
            "contact_associated_track_id": None, "human_verified": False,
            "source": "fresh_independent_ball_contact_or_unknown", "warnings": [reason]}


def _bound_contact(ball_dir, source, pose_path, pose, manifest_hash):
    """Verify fresh source/pose/search/cache links before accepting a candidate."""
    fps = float(pose["fps"])
    cid = source["clip_id"]
    paths = {"result": ball_dir / "results" / (cid + ".json"),
             "search": ball_dir / "contact_search" / (cid + ".json"),
             "event": ball_dir / "event_windows" / (cid + ".json"),
             "detections": ball_dir / "detections" / (cid + ".json")}
    provenance = {"files": {}, "status": "unknown"}
    if not paths["result"].exists():
        return _unknown_contact(fps, "fresh_ball_result_unavailable"), provenance
    result = json.loads(paths["result"].read_text())
    provenance["files"]["result"] = {"path": str(paths["result"].resolve()), "sha256": _hash(paths["result"])}
    identity = result.get("inference_identity", {})
    if (result.get("clip_id") != cid or identity.get("source_sha256") != source["clip_sha256"]
            or identity.get("pose_sha256") != _hash(pose_path)
            or identity.get("source_manifest_sha256") != manifest_hash):
        raise ValueError("Fresh ball source/pose/manifest identity mismatch")
    if result.get("reference_labels_used_for_prediction") is not False:
        raise ValueError("Fresh ball result is not source-only")
    if result.get("status") != "candidate_event" or not validate_cached_neural_producer(result, ball_dir):
        return _unknown_contact(fps, "independent_ball_contact_not_established"), provenance
    provenance["neural_inference_this_run"] = result.get("new_neural_inference") is True
    provenance["fresh_neural_cache_used"] = result.get("fresh_neural_cache_used") is True
    provenance["original_neural_inference_producer"] = result.get("original_neural_inference_producer")
    bindings = {"search": "contact_search_sha256", "event": "event_cache_sha256", "detections": "source_cache_sha256"}
    for name, field in bindings.items():
        if not paths[name].exists() or result.get(field) != _hash(paths[name]):
            raise ValueError(f"Fresh ball artifact hash mismatch: {name}")
        provenance["files"][name] = {"path": str(paths[name].resolve()), "sha256": result[field]}
    search = json.loads(paths["search"].read_text())
    event = json.loads(paths["event"].read_text())
    raw_cache = json.loads(paths["detections"].read_text())
    raw_identity = (result.get("original_neural_inference_producer") or {}).get("inference_identity", identity)
    if (raw_cache.get("clip_id") != cid or raw_cache.get("source_sha256") != source["clip_sha256"]
            or raw_cache.get("fresh_model_inference") is not True or raw_cache.get("inference_identity") != raw_identity):
        raise ValueError("Fresh raw neural cache identity mismatch")
    candidate = search.get("selected_candidate")
    if (search.get("status") != "candidate_supported" or not candidate
            or search.get("old_contact_or_outcome_labels_used") is not False):
        return _unknown_contact(fps, "independent_contact_search_abstained"), provenance
    if event.get("inference_identity") != identity or event.get("pose_sha256") != identity["pose_sha256"]:
        raise ValueError("Fresh event cache identity mismatch")
    if search.get("full_selected_shot_search") != [pose["shot_start_frame"], pose["shot_end_frame"]]:
        raise ValueError("Contact search is not the full selected pose shot")
    frame = candidate.get("contact_frame")
    score = candidate.get("score")
    interval = candidate.get("interval_frames")
    departures = candidate.get("ball_departure_frames")
    bracket = candidate.get("contact_bracket_frames")
    actor = candidate.get("contact_associated_track_id")
    actor_record = next((row for row in pose.get("track_candidates", []) if row.get("track_id") == actor), {})
    field_support = actor_record.get("field_support")
    actor_allowed = (isinstance(actor, int) and not isinstance(actor, bool)
                     and actor in search.get("eligible_actor_ids", [])
                     and actor not in search.get("goal_line_excluded_actor_ids", [])
                     and actor not in (pose.get("goal_line_role_exclusions") or {}).get("excluded_track_ids", [])
                     and any(track.get("track_id") == actor for track in pose.get("tracks", []))
                     and isinstance(field_support, (int, float)) and not isinstance(field_support, bool)
                     and np.isfinite(field_support) and field_support > .20
                     and candidate.get("actor_goal_line_excluded") is False
                     and search.get("actor_identity_ambiguous") is False)
    if (isinstance(frame, bool) or not isinstance(frame, int) or not pose["shot_start_frame"] <= frame < pose["shot_end_frame"]
            or not isinstance(score, (float, int)) or isinstance(score, bool) or not np.isfinite(score) or not .60 <= score <= 1.
            or not actor_allowed or candidate.get("ankle_approach_supported") is not True
            or not departures or candidate.get("camera_chain_measured") is not True
            or candidate.get("contact_timing_status") != "bounded_stationary_departure_candidate"
            or candidate.get("goalward_departure_support", {}).get("status") != "supported_goalward_departure_candidate"
            or "stationary_ball_persists_after_candidate" in candidate.get("warnings", [])):
        return _unknown_contact(fps, "independent_contact_candidate_inadequate"), provenance
    if (not isinstance(interval, list) or len(interval) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in interval)
            or interval != bracket or not interval[0] < interval[1]
            or not pose["shot_start_frame"] <= interval[0] < interval[1] < pose["shot_end_frame"]
            or interval[1] - interval[0] > max(2, int(np.floor(.15 * fps)))
            or frame != sum(interval) // 2):
        raise ValueError("Invalid independent contact candidate interval")
    if any(isinstance(value, bool) or not isinstance(value, int) or not frame <= value < pose["shot_end_frame"]
           for value in departures):
        raise ValueError("Invalid causal ball departure frames")
    raw_observations = {(int(fid), row.get("detection_id")): row
                        for fid, rows in raw_cache.get("detections", {}).items() for row in rows}
    stationary = candidate.get("stationary_ball_observations") or []
    moving = candidate.get("departure_ball_observations") or []
    frames_by_kind = {}
    for name, observations in (("stationary", stationary), ("moving", moving)):
        frames_by_kind[name] = []
        for observation in observations:
            fid = observation.get("frame")
            if isinstance(fid, bool) or not isinstance(fid, int) or not pose["shot_start_frame"] <= fid < pose["shot_end_frame"]:
                raise ValueError("Contact observation outside selected shot")
            raw = raw_observations.get((fid, observation.get("detection_id")))
            xy = np.asarray(observation.get("xy"), float)
            if (raw is None or xy.shape != (2,) or not np.all(np.isfinite(xy))
                    or not np.allclose(xy, [raw["x"], raw["y"]], rtol=0., atol=1e-6)):
                raise ValueError("Contact observation does not match raw source detection")
            frames_by_kind[name].append(fid)
    if (len(set(frames_by_kind["stationary"])) < 3 or not frames_by_kind["moving"]
            or max(frames_by_kind["stationary"]) != interval[0] or min(frames_by_kind["moving"]) != interval[1]
            or any(fid >= interval[1] for fid in frames_by_kind["stationary"])
            or sorted(frames_by_kind["moving"]) != sorted(departures)):
        raise ValueError("Contact bracket is not the actual stationary-to-departure observation interval")
    if (event.get("kick_frame") != frame or event.get("contact", {}).get("contact_frame") != frame
            or event.get("contact", {}).get("contact_associated_track_id") != actor
            or any(event.get("contact", {}).get(key) != value for key, value in candidate.items())):
        raise ValueError("Contact search/event selected actor mismatch")
    bounded = [max(pose["shot_start_frame"], interval[0]), min(pose["shot_end_frame"] - 1, interval[1])]
    warnings = list(candidate.get("warnings", [])) + ["independent_contact_is_automatic_candidate_not_verified_truth",
                                                         "confidence_is_heuristic_not_calibrated"]
    if bounded != interval:
        warnings.append("contact_interval_truncated_at_selected_shot_boundary")
    contact = {"status": "tentative", "contact_frame": frame, "contact_time_sec": frame / fps,
               "interval_frames": bounded, "original_candidate_interval_frames": interval,
               "confidence": float(score), "fps": fps,
               "contact_associated_track_id": actor,
               "human_verified": False, "source": "fresh_global_stationary_ball_and_ankle_departure_probe",
               "contact_bracket_frames": interval, "exact_contact_frame_established": False,
               "contact_frame_is_estimated": True,
               "timing_interval_interpretation": "last_observed_stationary_to_first_observed_goalward_departure",
               "ball_departure_frames": departures, "warnings": warnings}
    provenance["status"] = "source_bound_independent_contact_candidate"
    provenance["original_pose_selected_track_id"] = pose["selected_track_id"]
    provenance["independent_contact_actor_track_id"] = actor
    provenance["actor_reselected_by_ball_evidence"] = actor != pose["selected_track_id"]
    return contact, provenance


def _prefix_motion(motion, start, end):
    """A transform after the first missing/invalid link stays invalid forever."""
    transforms = {}
    valid = True
    first_invalid = None
    valid_count = 0
    for frame in range(start, end):
        raw = motion.get("transforms", {}).get(str(frame), motion.get("transforms", {}).get(frame))
        matrix = np.asarray(raw, float) if raw is not None else None
        usable = (matrix is not None and matrix.shape == (3, 3) and np.all(np.isfinite(matrix))
                  and abs(np.linalg.det(matrix)) > 1e-10)
        valid = valid and usable
        if not valid and first_invalid is None:
            first_invalid = frame
        transforms[str(frame)] = matrix.tolist() if valid else None
        valid_count += int(valid)
    complete = valid_count == end - start and motion.get("reference_chain_complete") is True
    result = {**motion, "transforms": transforms, "reference_chain_complete": complete,
              "first_invalid_reference_frame": first_invalid, "valid_reference_frame_count": valid_count,
              "requested_frame_count": end - start, "shot_interval": [start, end]}
    result["status"] = "supported" if complete else "partial" if valid_count > 1 else "unavailable"
    return result


def _phase_with_prefix(seq, contact, motion, events):
    if motion["status"] == "supported":
        return classify(seq, contact, motion, events)
    stabilized = (apply_transforms(seq["hips"], seq["frames"], motion["transforms"])
                  if seq["frames"] else np.empty((0, 2)))
    anchor = contact["contact_frame"] if contact["status"] == "tentative" else None
    return classify_penalty_phases(seq["frames"], contact["fps"], seq["hips"], seq["left_ankles"],
        seq["right_ankles"], seq["heights"], confidences=np.asarray(seq["confidences"]).reshape(-1, 3),
        stabilized_hips=stabilized, camera_motion_status=motion["status"], kick_frame=anchor,
        kick_confidence=contact["confidence"] if anchor is not None else 0., step_events=events)


def _terminal(seq, phase, contact, proposal, motion):
    stabilized = (apply_transforms(seq["hips"], seq["frames"], motion["transforms"])
                  if seq["frames"] else np.empty((0, 2)))
    kwargs = {"frame_ids": seq["frames"], "fps": contact["fps"], "stabilized_hips": stabilized,
              "box_heights": seq["heights"], "phase_result": phase, "contact": contact,
              "camera_status": motion["status"],
              "candidate_axis_reliable": proposal.get("axis_energy_fraction", 0.) >= .65}
    central = proposal.get("variants", {}).get("central", [])
    result = analyze_terminal_footwork(**kwargs, events=central)
    def brief(value):
        return {"status": value["terminal"]["status"], "start_frame": value["terminal"]["start_frame"],
                "tags": value["terminal"]["tags"], "counts": value["counts"], "warnings": value["warnings"]}
    event_variants = {name: brief(analyze_terminal_footwork(**kwargs, events=events))
                      for name, events in proposal.get("variants", {}).items()}
    cfg = TerminalConfig()
    profiles = {"permissive": replace(cfg, rhythm_short_ratio=.85, rhythm_long_ratio=1.3,
                min_rhythm_change_seconds=.04, progress_short_ratio=.75, min_progress_change_box_heights=.04, min_split_gain=.15),
                "central": cfg,
                "conservative": replace(cfg, rhythm_short_ratio=.70, rhythm_long_ratio=1.65,
                min_rhythm_change_seconds=.08, progress_short_ratio=.55, min_progress_change_box_heights=.08, min_split_gain=.35)}
    rule_variants = {name: brief(analyze_terminal_footwork(**kwargs, events=central, config=config))
                     for name, config in profiles.items()}
    contacts = {}
    interval = contact.get("interval_frames")
    if interval and phase["kick_anchor"]["usable"]:
        for name, frame in (("early", interval[0]), ("nominal", contact["contact_frame"]), ("late", interval[1])):
            alternate = {**contact, "contact_frame": frame}
            contacts[name] = brief(analyze_terminal_footwork(**{**kwargs, "contact": alternate}, events=central))
    variants = [*event_variants.values(), *rule_variants.values(), *contacts.values()]
    signatures = {(value["status"], tuple(value["tags"])) for value in variants}
    starts = [value["start_frame"] for value in variants if value["start_frame"] is not None]
    result["sensitivity"] = {"event_prominence": event_variants, "change_rules": rule_variants,
        "contact_anchor": contacts, "status_and_tags_agree": len(signatures) == 1,
        "candidate_start_range_frames": [min(starts), max(starts)] if starts else None,
        "meaning": "Fixed engineering variants, not accuracy or confidence intervals; no user count selection"}
    return result


def _analyse_pose(pose, contact, motion):
    actor = (contact["contact_associated_track_id"] if contact["status"] == "tentative"
             else pose["selected_track_id"])
    track = next((track for track in pose.get("tracks", [])
                  if track.get("track_id") == actor), {"observations": []})
    seq = sequence(track)
    proposal = (propose_step_events(seq["frames"], pose["fps"], seq["hips"], seq["left_ankles"],
                  seq["right_ankles"], seq["heights"], seq["confidences"],
                  kick_frame=contact["contact_frame"], event_end_frame=(contact.get("interval_frames") or [None, None])[1])
                if seq["frames"] else {"events": [], "variants": {}, "warnings": ["selected_pose_track_unavailable"]})
    phase = _phase_with_prefix(seq, contact, motion, proposal["events"])
    footwork = _terminal(seq, phase, contact, proposal, motion)
    result = {"phase_result": phase, "contact": contact, "event_proposal": proposal,
              "analysis_actor_track_id": actor, "original_pose_selected_track_id": pose["selected_track_id"],
              "actor_reselected_by_independent_ball_evidence": actor != pose["selected_track_id"],
              "footwork_result": footwork,
              "count_sensitivity": {name: counts_for(events, phase, contact.get("interval_frames"))
                                    for name, events in proposal.get("variants", {}).items()}}
    result["step_analysis"] = analyze_style_aware_steps(result)
    result["candidate_step_count"] = result["step_analysis"]["candidate_step_count"]
    result["verified_step_count"] = None
    return result, track


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--ball-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.resolve() in {args.pose_dir.resolve(), args.ball_dir.resolve()}:
        parser.error("Output must not overwrite inference inputs")
    manifest_path = args.clips_dir / "source_manifest.json"
    manifest_hash = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = {record["clip_id"]: record for record in manifest["clips"]}
    if len(sources) != len(manifest["clips"]):
        raise ValueError("Duplicate source clip IDs")
    clip_ids = args.clips or list(sources)
    if len(set(clip_ids)) != len(clip_ids) or any(cid not in sources for cid in clip_ids):
        raise ValueError("Requested clip IDs must be distinct source manifest members")
    code_paths = ("src/motion_stabilization.py", "src/penalty_phases.py", "src/phase_step_events.py",
                  "src/terminal_footwork.py", "src/style_aware_steps.py",
                  "src/fresh_cache_provenance.py",
                  "scripts/run_penalty_phase_experiment.py", "scripts/run_fresh_pitch_experiment.py",
                  "scripts/run_fresh_phase_experiment.py")
    code_hashes = {name: _hash(ROOT / name) for name in code_paths}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "camera_motion").mkdir(exist_ok=True)
    rows = []
    cv2.setNumThreads(2)
    for cid in clip_ids:
        source = sources[cid]
        video = args.clips_dir / source["clip_filename"]
        if _hash(video) != source["clip_sha256"]:
            raise ValueError(f"Source clip hash mismatch: {cid}")
        pose_path = args.pose_dir / (cid + ".json")
        pose, pose_provenance = _pose_input(pose_path, source)
        if pose is None:
            result = {"clip_id": cid, "event_id": source["event_id"], "status": "abstain_pose",
                      "reason": pose_provenance["status"], "candidate_step_count": None,
                      "verified_step_count": None, "pose_input": pose_provenance,
                      "contact": _unknown_contact(float(source["fps"]), "fresh_pose_scene_not_established"),
                      "provenance": {"source_manifest_sha256": manifest_hash,
                                     "source_clip_sha256": source["clip_sha256"], "code_sha256": code_hashes}}
        else:
            if pose.get("reference_labels_used_for_prediction") is not False:
                raise ValueError("Fresh pose result is not source-only")
            if not np.isfinite(pose["fps"]) or not np.isclose(pose["fps"], source["fps"]):
                raise ValueError("Source/pose fps mismatch")
            start, end = int(pose["shot_start_frame"]), int(pose["shot_end_frame"])
            if not 0 <= start < end <= source["frame_count"]:
                raise ValueError("Selected pose shot outside source clip")
            contact, contact_provenance = _bound_contact(args.ball_dir, source, pose_path, pose, manifest_hash)
            identity = {"source_manifest_sha256": manifest_hash, "source_clip_sha256": source["clip_sha256"],
                        "pose_sha256": pose_provenance["sha256"], "contact_provenance": contact_provenance,
                        "code_sha256": code_hashes, "shot_interval": [start, end], "opencv_version": cv2.__version__}
            target = args.output_dir / (cid + ".json")
            camera_path = args.output_dir / "camera_motion" / (cid + ".json")
            if args.resume and target.exists():
                previous = json.loads(target.read_text())
                if previous.get("inference_identity") == identity:
                    if (not camera_path.exists() or previous.get("camera_motion_sha256") != _hash(camera_path)
                            or json.loads(camera_path.read_text()).get("inference_identity") != identity):
                        raise ValueError(f"Existing camera motion artifact mismatch: {cid}")
                    result = previous
                    row = {key: result.get(key) for key in ("clip_id", "event_id", "status", "candidate_step_count")}
                    row.update(contact_status=result["contact"]["status"], contact_frame=result["contact"]["contact_frame"],
                               pose_input_status=pose_provenance["status"],
                               camera_reference_complete=result.get("camera_motion", {}).get("reference_chain_complete"))
                    rows.append(row)
                    print(cid, "reused_source_bound_phase", flush=True)
                    continue
                raise ValueError(f"Existing phase result identity mismatch: {cid}")
            boxes = {}
            for track in pose.get("tracks", []):
                for observation in track.get("observations", []):
                    boxes.setdefault(int(observation["frame"]), []).append(observation["bbox"])
            motion = _prefix_motion(estimate_background_motion(str(video), start, end, boxes_by_frame=boxes), start, end)
            motion["inference_identity"] = identity
            _write(camera_path, motion)
            analysis, track = _analyse_pose(pose, contact, motion)
            result = {**analysis, "clip_id": cid, "event_id": source["event_id"], "video_path": str(video.resolve()),
                      "fps": pose["fps"], "shot_start_frame": start, "shot_end_frame_exclusive": end,
                      "status": "candidate_phase" if analysis["candidate_step_count"] is not None else "abstain_approach_count",
                      "pose_input": pose_provenance, "contact_provenance": contact_provenance,
                      "inference_identity": identity, "provenance": identity,
                      "camera_motion": {key: value for key, value in motion.items() if key not in ("transforms", "pairs")},
                      "camera_motion_sha256": _hash(camera_path),
                      "pose_only_baseline": {"candidate_step_count": pose.get("step_count"),
                         "candidate_step_frames": pose.get("step_frames"), "pose_motion_peak_frame": pose.get("kick_frame"),
                         "meaning": "Fresh pose-only proposal for comparison; never used as independent contact"}}
            render_sheet(result, track["observations"], args.output_dir / (cid + "_qa.jpg"))
        result.update(schema_version="fresh-phase-experiment-v1", human_verified=False, accuracy_validated=False,
                      old_reference_labels_used=False, user_step_labels_used=False, new_neural_inference=False,
                      scope="Fresh inference reanalysis; automatic contact, motion, phase and step candidates")
        _write(args.output_dir / (cid + ".json"), result)
        row = {key: result.get(key) for key in ("clip_id", "event_id", "status", "candidate_step_count")}
        row.update(contact_status=result["contact"]["status"], contact_frame=result["contact"]["contact_frame"],
                   pose_input_status=pose_provenance["status"],
                   camera_reference_complete=result.get("camera_motion", {}).get("reference_chain_complete"))
        rows.append(row)
        _write(args.output_dir / "progress.json", {"completed": len(rows), "total": len(clip_ids), "last_clip": cid})
        print(json.dumps(row), flush=True)
    _write(args.output_dir / "phase_experiment_summary.json", {"cohort_size": len(clip_ids), "cohort_clip_ids": clip_ids,
             "rows": rows, "count_available": sum(row["candidate_step_count"] is not None for row in rows),
             "status_counts": dict(Counter(row["status"] for row in rows)),
             "human_verified": False, "accuracy_validated": False, "verified_step_count": None,
             "old_reference_labels_used": False, "user_step_labels_used": False,
             "source_manifest_sha256": manifest_hash, "code_sha256": code_hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
