"""Sparse, same-frame ground-contact hypotheses on source-hashed fresh clips."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.pitch_calibration import (PITCH_LINES, _project, calibrate_pitch_frame,
                                   foot_contact_candidate, project_ground_contacts)
from src.step_pipeline import _histogram_for_cut, _scan_camera_shots


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _pose_input(path, source):
    if path is None or not path.exists():
        return None, {"status": "unavailable", "sha256": None}
    before_hash = _hash(path)
    try:
        pose = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None, {"status": "incomplete_json", "sha256": before_hash}
    if _hash(path) != before_hash:
        return None, {"status": "changed_during_read", "sha256": None}
    provenance = {"status": None, "path": str(path.resolve()), "sha256": before_hash}
    identity = pose.get("inference_identity", {})
    settings = identity.get("settings", {})
    if pose.get("clip_id") != source["clip_id"] or identity.get("source_sha256") != source["clip_sha256"]:
        return None, dict(provenance, status="source_identity_mismatch")
    if settings.get("use_field_mask") is not True or settings.get("require_goal_context") is not True:
        return None, dict(provenance, status="pose_selection_scene_guards_missing")
    if settings.get("event_slice_policy") != "selected_shot_must_overlap_manual_course_slice_contact_remains_automatic":
        return None, dict(provenance, status="pose_manual_slice_overlap_policy_missing")
    if pose.get("selected_track_qualified") is not True:
        return None, dict(provenance, status="selected_person_not_qualified")
    selection = pose.get("shot_selection", {})
    attempt = next((attempt for attempt in selection.get("attempts", [])
                    if attempt.get("shot_index") == selection.get("selected_shot_index")), {})
    if (selection.get("found_qualified_shot") is not True or attempt.get("qualifies") is not True
            or attempt.get("scene_rejections") or attempt.get("selected_track_id") != pose.get("selected_track_id")):
        return None, dict(provenance, status="selected_pose_shot_scene_rejected")
    role = attempt.get("role_context") or {}
    if role.get("status") == "candidate_supported" and role.get("seed_track_id") == pose.get("selected_track_id"):
        return None, dict(provenance, status="selected_person_is_goal_line_occupant")
    if pose.get("new_neural_inference") is not True:
        return None, dict(provenance, status="fresh_neural_inference_not_recorded")
    return pose, dict(provenance, status="source_bound_automatic_pose_candidate",
                      model_sha256=identity.get("model_sha256"),
                      pose_algorithm_sha256=identity.get("algorithm_sha256"),
                      selected_track_id=pose.get("selected_track_id"), human_verified=False)


def _sample_frames(shots, stride):
    frames = set()
    for shot in shots:
        start = int(shot["shot_start_frame"])
        end = int(shot.get("camera_shot_end_frame", shot["shot_end_frame"]))
        if end <= start:
            continue
        frames.update(range(start, end, stride))
        frames.update([min(end - 1, start + 2), (start + end - 1) // 2, max(start, end - 3)])
    return sorted(frames)


def _plot(path, clip_id, candidates, fps):
    canvas = np.full((630, 760, 3), 250, np.uint8)
    def pixel(point):
        x, y = point
        return int(round(380 + x * 15)), int(round(85 + y * 24))
    for x in range(-20, 21, 5):
        cv2.line(canvas, pixel((x, 0)), pixel((x, 20)), (226, 226, 226), 1)
        cv2.putText(canvas, str(x), (pixel((x, 20))[0] - 12, 590), cv2.FONT_HERSHEY_SIMPLEX, .4, (85, 85, 85), 1)
    for y in range(0, 21, 5):
        cv2.line(canvas, pixel((-22, y)), pixel((22, y)), (226, 226, 226), 1)
        cv2.putText(canvas, str(y), (16, pixel((0, y))[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, .4, (85, 85, 85), 1)
    for ends in PITCH_LINES.values():
        cv2.line(canvas, pixel(ends[0]), pixel(ends[1]), (90, 125, 85), 2)
    cv2.line(canvas, pixel((-3.66, 0)), pixel((3.66, 0)), (40, 40, 40), 5)
    cv2.circle(canvas, pixel((0, 11)), 3, (70, 70, 70), -1)
    frame_ids = [record["frame_id"] for record in candidates]
    low, high = (min(frame_ids), max(frame_ids)) if frame_ids else (0, 1)
    for record in candidates:
        ratio = (record["frame_id"] - low) / max(1, high - low)
        color = cv2.cvtColor(np.array([[[int(115 * (1 - ratio)), 210, 220]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.circle(canvas, pixel(record["coordinates_m_hypothesis"]), 4, tuple(map(int, color)), -1)
    cv2.putText(canvas, clip_id, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (35, 35, 35), 1)
    cv2.putText(canvas, f"{len(candidates)} sparse sole/contact hypotheses; metric accuracy unverified", (20, 53),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (75, 75, 75), 1)
    cv2.putText(canvas, "x / y: m_hypothesis    No interpolation across missing frames", (20, 618),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (75, 75, 75), 1)
    if frame_ids:
        cv2.putText(canvas, f"clip time {low / fps:.2f}s to {high / fps:.2f}s", (430, 590),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (75, 75, 75), 1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Cannot write diagnostic plot: {path}")


def _diagnostic_frame(path, frame, calibration, contact):
    output = frame.copy()
    candidate = calibration.get("candidate")
    if candidate:
        for name, ends in PITCH_LINES.items():
            points, valid = _project(candidate["pitch_to_image_hypothesis"], np.linspace(*ends, 150))
            if np.all(valid) and np.max(np.abs(points)) < 100000:
                color = (0, 210, 255) if name in calibration["fit_line_names"] else (255, 180, 0)
                cv2.polylines(output, [np.rint(points).astype(np.int32)], False, color, 2)
    if contact and contact.get("sole_pixel"):
        cv2.circle(output, tuple(np.rint(contact["sole_pixel"]).astype(int)), 6, (255, 0, 255), -1)
    cv2.putText(output, calibration["status"] + ": " + calibration["reason"], (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"Cannot write geometry frame: {path}")


def _run_clip(source, video, pose, identity, previous=None, independent_contact=None):
    core_keys = ("source_manifest_sha256", "source_clip_sha256", "algorithm_sha256", "goal_geometry_sha256",
                 "field_masks_sha256", "opencv_version")
    if previous is not None:
        old_identity = previous.get("calibration_identity", {})
        if any(old_identity.get(key) != identity.get(key) for key in core_keys):
            raise ValueError(f"Calibration cache identity mismatch: {source['clip_id']}")
        shots = previous["camera_shots"]
        calibrations = {record["frame_id"]: record for record in previous["frame_calibrations"]}
    else:
        shots = _scan_camera_shots(str(video), max_shot_sec=120., scene_cut_threshold=30.)
        calibrations = {}
    observations = {}
    contact_supported = independent_contact is not None and independent_contact.get("status") == "tentative"
    if pose is not None and contact_supported:
        track = next((track for track in pose.get("tracks", [])
                      if track.get("track_id") == independent_contact["contact_associated_track_id"]), None)
        if track:
            for observation in track.get("observations", []):
                frame_id = int(observation["frame"])
                if frame_id in observations:
                    raise ValueError("Duplicate selected-person pose frame")
                observations[frame_id] = observation
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode source clip: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Invalid source clip fps")
    projections = []
    diagnostics = []
    frames = set(calibrations) if previous is not None else set(_sample_frames(shots, identity["settings"]["frame_stride"]))
    sample_origins = {frame_id: "cached_camera_shot_sampling" if previous is not None else "fresh_camera_shot_sampling"
                      for frame_id in frames}
    if pose is not None:
        selected = [{"shot_start_frame": pose["shot_start_frame"], "shot_end_frame": pose["shot_end_frame"]}]
        for frame_id in _sample_frames(selected, identity["settings"]["frame_stride"]):
            if frame_id not in frames:
                sample_origins[frame_id] = "fresh_selected_pose_shot_sampling"
            frames.add(frame_id)
    if contact_supported:
        contact_frame = independent_contact["contact_frame"]
        nearby = [contact_frame + offset for offset in (-15, -6, -3, 0, 3, 6, 15)]
        nearby.extend(independent_contact["interval_frames"])
        for frame_id in nearby:
            if pose["shot_start_frame"] <= frame_id < pose["shot_end_frame"]:
                if frame_id not in frames:
                    sample_origins[frame_id] = "independent_contact_near_sampling"
                frames.add(frame_id)
    frames = sorted(frames)
    try:
        for frame_id in frames:
            record = calibrations.get(frame_id)
            need_pixels = record is None or frame_id in observations or len(diagnostics) < 2
            frame = None
            if need_pixels:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = capture.read()
                if not ok:
                    calibrations[frame_id] = {"frame_id": frame_id, "status": "abstain",
                                              "reason": "frame_decode_failed", "calibration": None}
                    continue
                decoded_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if decoded_frame != frame_id:
                    raise ValueError(f"Unexpected decoder frame mapping: {frame_id} -> {decoded_frame}")
                pixels_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                if record and record.get("decoded_pixels_sha256") != pixels_hash:
                    raise ValueError("Decoded pixels differ from source-bound calibration")
                if record is None:
                    calibration = calibrate_pitch_frame(frame, frame_id=frame_id)
                    record = {"frame_id": frame_id, "source_frame_id": source["source_start_frame"] + frame_id,
                              "decoded_pixels_sha256": pixels_hash, "status": calibration["status"],
                              "reason": calibration["reason"], "calibration": calibration,
                              "sample_origin": sample_origins[frame_id],
                              "calibration_core_identity": {key: identity[key] for key in core_keys}}
                    calibrations[frame_id] = record
            calibration = record.get("calibration")
            if calibration is None or calibration["status"] != "candidate_supported":
                continue
            contact = None
            if frame_id in observations and frame is not None:
                contact = foot_contact_candidate(frame, observations[frame_id])
                record["contact_check"] = contact
                if contact["status"] == "contact_candidate":
                    projected = project_ground_contacts(calibration, [contact["sole_pixel"]],
                                frame_id=frame_id, point_kind="foot_ground_contact", on_ground=[True])
                    coordinates = projected["coordinates_m_hypothesis"][0]
                    if coordinates is not None:
                        early, late = independent_contact["interval_frames"]
                        phase = ("pre_kick_candidate" if frame_id < early else "contact_interval_candidate"
                                 if frame_id <= late else "post_kick_candidate")
                        projections.append({"frame_id": frame_id, "source_frame_id": record["source_frame_id"],
                                            "coordinates_m_hypothesis": coordinates, "foot": contact["foot"],
                                            "sole_pixel": contact["sole_pixel"], "phase": phase,
                                            "actor_track_id": independent_contact["contact_associated_track_id"],
                                            "independent_contact_frame": independent_contact["contact_frame"],
                                            "ground_contact_verified": False, "metric_accuracy_verified": False})
            if frame is not None and len(diagnostics) < 2:
                diagnostics.append((frame_id, frame.copy(), calibration, contact))
    finally:
        capture.release()
    slice_start = source.get("original_slice_start_frame_in_clip", 0)
    slice_end = source.get("original_slice_end_frame_in_clip", source.get("frame_count", max(frames, default=0) + 1))
    for frame_id, record in calibrations.items():
        record["relative_to_manual_slice"] = ("before" if frame_id < slice_start else
                                               "after" if frame_id >= slice_end else "within")
    return {"clip_id": source["clip_id"], "event_id": source["event_id"], "fps": fps,
            "calibration_identity": identity, "camera_shots": shots,
            "reused_calibration_identity": previous.get("calibration_identity") if previous is not None else None,
            "camera_shot_sampling_identity": (previous.get("camera_shot_sampling_identity") or previous.get("calibration_identity"))
                    if previous is not None else identity,
            "new_camera_scanner_applied_to_cached_shots": False if previous is not None else None,
            "frame_calibrations": [calibrations[frame] for frame in sorted(calibrations)],
            "sampled_frame_count": len(frames),
            "supported_frame_count": sum(record["status"] == "candidate_supported" for record in calibrations.values()),
            "supported_frame_count_in_manual_slice": sum(record["status"] == "candidate_supported"
                    and record["relative_to_manual_slice"] == "within" for record in calibrations.values()),
            "supported_frame_count_in_selected_pose_frames": sum(record["status"] == "candidate_supported"
                    and frame_id in observations for frame_id, record in calibrations.items()),
            "sparse_ground_contact_candidates": projections, "ground_contact_candidate_count": len(projections),
            "pre_kick_contact_candidate_count": sum(record["phase"] == "pre_kick_candidate" for record in projections),
            "metric_accuracy_verified": False, "ground_contact_verified": False,
            "complete_runup_verified": False, "trajectory_interpolated": False,
            "coordinate_unit": "m_hypothesis", "scope": "Independent same-frame geometry and image sole-contact hypotheses"}, diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--ball-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--reuse-calibrations", action="store_true")
    args = parser.parse_args(argv)
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    if args.ball_dir is not None and args.pose_dir is None:
        parser.error("--ball-dir requires --pose-dir")
    manifest_path = args.clips_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = {record["clip_id"]: record for record in manifest["clips"]}
    if len(sources) != len(manifest["clips"]):
        raise ValueError("Duplicate source clip IDs")
    clip_ids = args.clips or list(sources)
    if len(set(clip_ids)) != len(clip_ids) or any(clip not in sources for clip in clip_ids):
        raise ValueError("Requested clip IDs must be distinct source manifest members")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"source_manifest_sha256": _hash(manifest_path),
              "algorithm_sha256": _hash(ROOT / "src/pitch_calibration.py"),
              "camera_scanner_sha256": hashlib.sha256((inspect.getsource(_scan_camera_shots)
                                                        + inspect.getsource(_histogram_for_cut)).encode()).hexdigest(),
              "goal_geometry_sha256": _hash(ROOT / "src/goal_geometry.py"),
              "field_masks_sha256": _hash(ROOT / "src/field_masks.py"),
              "independent_contact_reader_sha256": _hash(ROOT / "scripts/run_fresh_phase_experiment.py"),
              "script_sha256": _hash(Path(__file__)), "opencv_version": cv2.__version__,
              "settings": {"frame_stride": args.frame_stride, "camera_cuts_threshold": 30.}}
    summaries = []
    for clip_id in clip_ids:
        source = sources[clip_id]
        video = args.clips_dir / source["clip_filename"]
        if _hash(video) != source["clip_sha256"]:
            raise ValueError(f"Source clip hash mismatch: {clip_id}")
        identity = dict(hashes, source_clip_sha256=source["clip_sha256"])
        pose_path = args.pose_dir / (clip_id + ".json") if args.pose_dir else None
        pose, pose_provenance = _pose_input(pose_path, source)
        independent_contact = None
        contact_provenance = {"status": "independent_contact_input_unavailable"}
        if pose is not None and args.ball_dir is not None:
            from scripts.run_fresh_phase_experiment import _bound_contact
            independent_contact, contact_provenance = _bound_contact(args.ball_dir, source, pose_path, pose,
                                                                    hashes["source_manifest_sha256"])
        output_path = args.output_dir / (clip_id + ".json")
        previous = None
        if args.reuse_calibrations and output_path.exists():
            previous = json.loads(output_path.read_text())
            previous_artifact_hash = _hash(output_path)
            for record in previous["frame_calibrations"]:
                record.pop("contact_check", None)
        result, diagnostics = _run_clip(source, video, pose, identity, previous, independent_contact)
        result["pose_input"] = pose_provenance
        result["independent_contact"] = independent_contact
        result["contact_provenance"] = contact_provenance
        result["reused_calibration_artifact_sha256"] = previous_artifact_hash if previous is not None else None
        _plot(args.output_dir / (clip_id + "_ground.png"), clip_id,
              result["sparse_ground_contact_candidates"], result["fps"])
        for frame_id, frame, calibration, contact in diagnostics:
            _diagnostic_frame(args.output_dir / f"{clip_id}_frame{frame_id:06d}.jpg", frame, calibration, contact)
        _write(output_path, result)
        summary = {key: result[key] for key in ("clip_id", "event_id", "sampled_frame_count",
                                                "supported_frame_count", "supported_frame_count_in_manual_slice",
                                                "supported_frame_count_in_selected_pose_frames",
                                                "ground_contact_candidate_count", "pre_kick_contact_candidate_count")}
        summary["pose_input_status"] = pose_provenance["status"]
        summaries.append(summary)
        _write(args.output_dir / "progress.json", {"completed": len(summaries), "total": len(clip_ids),
                                                    "last_clip": clip_id})
        print(json.dumps(summary), flush=True)
    report = {"schema_version": 1, "calibration_provenance": hashes, "cohort_clip_ids": clip_ids,
              "cohort_size": len(clip_ids), "metric_accuracy_verified": False,
              "human_reference_used": False, "ground_contact_verified": False,
              "clips_with_supported_geometry": sum(record["supported_frame_count"] > 0 for record in summaries),
              "clips_with_geometry_in_manual_slice": sum(record["supported_frame_count_in_manual_slice"] > 0
                                                            for record in summaries),
              "clips_with_geometry_in_selected_pose_frames": sum(record["supported_frame_count_in_selected_pose_frames"] > 0
                                                                   for record in summaries),
              "clips_with_ground_contact_candidates": sum(record["ground_contact_candidate_count"] > 0 for record in summaries),
              "scope": "Coverage of automatic geometric/contact candidates, not an accuracy evaluation",
              "records": summaries}
    _write(args.output_dir / "pitch_experiment_summary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
