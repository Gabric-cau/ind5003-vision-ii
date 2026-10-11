"""Compare two independently checked pitch hypotheses on identical source frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_fresh_pitch_experiment import _diagnostic_frame, _hash, _plot, _pose_input, _sample_frames, _write
from src.joint_pitch_calibration import calibrate_joint_pitch_frame
from src.pitch_calibration import calibrate_pitch_frame, foot_contact_candidate, project_ground_contacts


def _runup_segments(points, requested_frames):
    order = {frame: index for index, frame in enumerate(sorted(set(requested_frames)))}
    segments = []
    previous = None
    for point in sorted(points, key=lambda row: row["frame_id"]):
        frame = point["frame_id"]
        if frame not in order or frame == previous:
            raise ValueError("Ground candidate must have a distinct sampled source frame")
        if previous is None or order[frame] != order[previous] + 1:
            segments.append([])
        segments[-1].append(point)
        previous = frame
    return segments


def _plot_runup(path, clip_id, segments, fps):
    points = [point for segment in segments for point in segment]
    _plot(path, clip_id, points, fps)
    canvas = cv2.imread(str(path))
    for segment in segments:
        for first, second in zip(segment, segment[1:]):
            a, b = (np.array([380 + row["coordinates_m_hypothesis"][0] * 15,
                              85 + row["coordinates_m_hypothesis"][1] * 24]) for row in (first, second))
            for start in np.arange(0., 1., .2):
                cv2.line(canvas, tuple(np.rint(a + (b - a) * start).astype(int)),
                         tuple(np.rint(a + (b - a) * min(1., start + .1)).astype(int)), (120, 120, 120), 1)
    for point in points:
        xy = tuple(np.rint([380 + point["coordinates_m_hypothesis"][0] * 15,
                            85 + point["coordinates_m_hypothesis"][1] * 24]).astype(int))
        cv2.circle(canvas, xy, 3, (30, 120, 205), -1)
    cv2.rectangle(canvas, (0, 575), (canvas.shape[1], canvas.shape[0]), (250, 250, 250), -1)
    for x in range(-20, 21, 5):
        cv2.putText(canvas, str(x), (380 + x * 15 - 12, 590), cv2.FONT_HERSHEY_SIMPLEX, .4, (85, 85, 85), 1)
    if points:
        cv2.putText(canvas, f"clip time {points[0]['frame_id'] / fps:.2f}s to {points[-1]['frame_id'] / fps:.2f}s", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (75, 75, 75), 1)
    cv2.putText(canvas, "m_hypothesis; pre-contact sole samples; dashed guides break at missing samples", (20, 619),
                cv2.FONT_HERSHEY_SIMPLEX, .43, (70, 70, 70), 1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError("Cannot write sparse run-up candidate plot")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--ball-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--frame-stride", type=int, default=30)
    args = parser.parse_args(argv)
    if args.frame_stride < 1 or (args.frames is not None and any(frame < 0 for frame in args.frames)):
        parser.error("Sampling needs positive stride and nonnegative frames")
    if args.ball_dir is not None and args.pose_dir is None:
        parser.error("--ball-dir requires --pose-dir")
    inputs = [path.resolve() for path in (args.clips_dir, args.pose_dir, args.ball_dir) if path is not None]
    if args.output_dir.resolve() in inputs:
        parser.error("Output must not overwrite input directories")
    manifest_path = args.clips_dir / "source_manifest.json"
    manifest_hash = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = {source["clip_id"]: source for source in manifest["clips"]}
    ids = args.clips or list(sources)
    if len(sources) != len(manifest["clips"]) or len(set(ids)) != len(ids) or any(cid not in sources for cid in ids):
        raise ValueError("Clip IDs must be distinct manifest members")
    code_names = ("src/pitch_calibration.py", "src/joint_pitch_calibration.py", "src/goal_geometry.py",
                  "src/field_masks.py",
                  "scripts/run_fresh_pitch_experiment.py", "scripts/run_joint_pitch_calibration_pilot.py")
    if args.ball_dir is not None:
        code_names += ("scripts/run_fresh_phase_experiment.py", "src/fresh_cache_provenance.py")
    code_hashes = {name: _hash(ROOT / name) for name in code_names}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    summaries = []
    for cid in ids:
        source = sources[cid]
        video = args.clips_dir / source["clip_filename"]
        if _hash(video) != source["clip_sha256"]:
            raise ValueError(f"Source hash mismatch: {cid}")
        pose_path = args.pose_dir / (cid + ".json") if args.pose_dir is not None else None
        pose, pose_provenance = _pose_input(pose_path, source)
        contact = None
        contact_provenance = {"status": "independent_contact_input_unavailable"}
        if pose is not None and args.ball_dir is not None:
            contact, contact_provenance = _bound_contact(args.ball_dir, source, pose_path, pose, manifest_hash)
        if args.frames is not None:
            frames = sorted(set(args.frames))
            sampling_policy = "explicit_frame_ids_for_diagnostic_comparison_not_truth_labels"
        elif pose is not None:
            frames = _sample_frames([{"shot_start_frame": pose["shot_start_frame"],
                                    "shot_end_frame": pose["shot_end_frame"]}], args.frame_stride)
            sampling_policy = "qualified_fresh_selected_pose_shot"
        else:
            frames = _sample_frames([{"shot_start_frame": source["original_slice_start_frame_in_clip"],
                                    "shot_end_frame": source["original_slice_end_frame_in_clip"]}], args.frame_stride)
            sampling_policy = "manual_course_slice_sampling_boundaries_not_semantic_event_truth"
        if contact is not None and contact["status"] == "tentative":
            frames = sorted(set(frames) | {frame for frame in [contact["contact_frame"] + offset for offset in (-15, -6, -3, 0, 3, 6, 15)]
                                        if pose["shot_start_frame"] <= frame < pose["shot_end_frame"]})
            actor = contact["contact_associated_track_id"]
            track = next(track for track in pose["tracks"] if track["track_id"] == actor)
            observations = {row["frame"]: row for row in track["observations"]}
        else:
            observations = {}
        if any(not 0 <= frame < source["frame_count"] for frame in frames):
            raise ValueError("Sample frame outside source clip")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), source["fps"]):
            cap.release()
            raise ValueError("Source decode/fps mismatch")
        records = []
        try:
            for fid in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, image = cap.read()
                if not ok:
                    records.append({"frame_id": fid, "status": "decode_failed", "baseline": None, "joint": None})
                    continue
                if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != fid:
                    raise ValueError("Decoded source frame identity mismatch")
                old = calibrate_pitch_frame(image, frame_id=fid)
                joint = calibrate_joint_pitch_frame(image, frame_id=fid)
                sole = foot_contact_candidate(image, observations[fid]) if fid in observations else None
                coordinates = {"baseline": None, "joint": None}
                if sole is not None and sole["status"] == "contact_candidate":
                    for name, geometry in (("baseline", old), ("joint", joint)):
                        projected = project_ground_contacts(geometry, [sole["sole_pixel"]], frame_id=fid,
                                                            point_kind="foot_ground_contact", on_ground=[True])
                        coordinates[name] = projected["coordinates_m_hypothesis"][0]
                record = {"frame_id": fid, "source_frame_id": source["source_start_frame"] + fid,
                          "decoded_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                          "baseline": old, "joint": joint, "foot_contact_candidate": sole,
                          "coordinates_m_hypothesis": coordinates,
                          "actor_track_id": contact["contact_associated_track_id"] if observations else None}
                records.append(record)
                for name, geometry in (("baseline", old), ("joint", joint)):
                    _diagnostic_frame(args.output_dir / f"{cid}_f{fid:06d}_{name}.jpg", image, geometry, sole)
        finally:
            cap.release()
        if _hash(video) != source["clip_sha256"]:
            raise ValueError("Source changed during comparison")
        projections = []
        for row in records:
            xy = row.get("coordinates_m_hypothesis", {}).get("joint")
            if xy is None:
                continue
            early, late = contact["contact_bracket_frames"]
            sole = row["foot_contact_candidate"]
            phase = "pre_kick_candidate" if row["frame_id"] < early else "contact_interval_candidate" if row["frame_id"] <= late else "post_kick_candidate"
            projections.append({"frame_id": row["frame_id"], "source_frame_id": row["source_frame_id"],
                "coordinates_m_hypothesis": xy, "sole_pixel": sole["sole_pixel"], "foot": sole["foot"],
                "phase": phase, "actor_track_id": row["actor_track_id"], "independent_contact_frame": contact["contact_frame"],
                "calibration_mode": "joint", "ground_contact_verified": False, "metric_accuracy_verified": False})
        pre_contact = [point for point in projections if point["phase"] == "pre_kick_candidate"]
        segments = _runup_segments(pre_contact, frames)
        plot_path = args.output_dir / (cid + "_ground.png")
        _plot_runup(plot_path, cid, segments, source["fps"])
        supported = [row for row in records if (row["joint"] or {}).get("status") == "candidate_supported"]
        identity = {"source_sha256": source["clip_sha256"], "source_manifest_sha256": manifest_hash,
                    "code_sha256": code_hashes, "opencv_version": cv2.__version__}
        result = {"clip_id": cid, "event_id": source["event_id"], "source_mapping": source,
                  "sampling_policy": sampling_policy, "sampled_frame_ids": frames, "frames": records,
                  "pose_input": pose_provenance, "contact": contact, "contact_provenance": contact_provenance,
                  "independent_contact": contact, "sparse_ground_contact_candidates": projections,
                  "ground_contact_candidate_count": len(projections), "pre_kick_contact_candidate_count": len(pre_contact),
                  "sampled_frame_count": len(records), "supported_frame_count": len(supported),
                  "supported_frame_count_in_manual_slice": sum(source["original_slice_start_frame_in_clip"] <= row["frame_id"]
                      < source["original_slice_end_frame_in_clip"] for row in supported),
                  "supported_frame_count_in_selected_pose_frames": sum(row["frame_id"] in observations for row in supported),
                  "status": "candidate_ground_geometry" if supported else "abstain_ground_geometry",
                  "pre_contact_path_segments_frame_ids": [[point["frame_id"] for point in segment] for segment in segments],
                  "path_connections": "sample-to-sample display guides; unsampled intermediate positions are unknown",
                  "trajectory_interpolated": False, "ground_contact_verified": False, "coordinate_unit": "m_hypothesis",
                  "runup_plot": {"path": plot_path.name, "sha256": _hash(plot_path)},
                  "inference_identity": identity, "human_verified": False, "metric_accuracy_verified": False,
                  "accuracy_validated": False, "complete_runup_verified": False,
                  "old_reference_labels_used": False, "valid_for_other_frames": False}
        _write(args.output_dir / (cid + ".json"), result)
        summary = {"clip_id": cid, "event_id": source["event_id"], "sampled_frames": len(records)}
        summary.update(ground_contact_candidate_count=len(projections), pre_kick_contact_candidate_count=len(pre_contact))
        for name in ("baseline", "joint"):
            summary[name + "_supported_frames"] = sum((row[name] or {}).get("status") == "candidate_supported" for row in records)
            summary[name + "_reason_counts"] = dict(Counter((row[name] or {}).get("reason", "decode_failed") for row in records))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        _write(args.output_dir / "progress.json", {"completed": len(summaries), "total": len(ids), "last_clip": cid})
    _write(args.output_dir / "joint_geometry_summary.json", {"cohort_clip_ids": ids, "cohort_size": len(ids),
               "rows": summaries, "code_sha256": code_hashes, "source_manifest_sha256": manifest_hash,
               "scope": "Paired same-frame automatic fit diagnostics; not measured visual or metric accuracy",
               "human_verified": False, "metric_accuracy_verified": False, "accuracy_validated": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
