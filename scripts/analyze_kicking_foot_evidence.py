"""Replay temporal foot evidence from frozen poses, without new neural inference."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kicking_foot_evidence import kicking_foot_evidence


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def render_source_sheet(raw, result, match_path: Path, offset: float, fps: float, destination: Path) -> dict:
    """Show original video observations with their existing pose-channel overlay."""
    normal = {row["frame"]: row for row in raw if not row["reflected"]}
    first, departure = result["contact_bracket_frames"]
    selected = sorted({max(0, first - 4), first - 2, first, departure,
                       departure + 1, departure + 2, departure + 4, departure + 6} & normal.keys())
    capture = cv2.VideoCapture(str(match_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode source match: {match_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    panels, frames = [], []
    for local_frame in selected:
        source_frame = round((offset + local_frame / fps) * source_fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Source frame unavailable: {source_frame}")
        row = normal[local_frame]
        points, confidence = np.asarray(row["keypoints"]), np.asarray(row["confidence"])
        for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12)):
            if min(confidence[a], confidence[b]) >= .4:
                cv2.line(image, tuple(points[a].astype(int)), tuple(points[b].astype(int)), (0, 220, 0), 2)
        for index, name, color in ((15, "L", (230, 180, 0)), (16, "R", (0, 110, 240))):
            if confidence[index] >= .4:
                position = tuple(points[index].astype(int))
                cv2.circle(image, position, 4, color, -1)
                cv2.putText(image, name, position, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        box = np.asarray(row["bbox"])
        height = box[3] - box[1]
        x0, y0 = np.maximum([0, 0], (box[:2] - [.45 * height, .2 * height]).astype(int))
        x1, y1 = np.minimum([image.shape[1], image.shape[0]],
                            (box[2:] + [.45 * height, .35 * height]).astype(int))
        crop = image[y0:y1, x0:x1]
        if not crop.size:
            raise ValueError("Existing pose box is outside the recovered source frame")
        scale = min(384 / crop.shape[1], 280 / crop.shape[0])
        tile = cv2.resize(crop, None, fx=scale, fy=scale)
        panel = np.full((320, 384, 3), 245, np.uint8)
        left, top = (384 - tile.shape[1]) // 2, (280 - tile.shape[0]) // 2
        panel[top:top + tile.shape[0], left:left + tile.shape[1]] = tile
        cv2.putText(panel, f"local {local_frame} | match {source_frame/source_fps:.3f}s",
                    (8, 302), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1)
        panels.append(panel)
        frames.append({"local_pose_frame": local_frame, "match_frame": source_frame,
                       "manifest_offset_sec": offset, "source_fps": source_fps})
    capture.release()
    sheet = np.full((70 + 320 * max(1, (len(panels) + 3) // 4), 1536, 3), 255, np.uint8)
    cv2.putText(sheet, result["clip_id"], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 1)
    cv2.putText(sheet, f"Temporal hypothesis: {result['candidate']} | L/R are cached pose channels, unverified",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 4)
        sheet[70 + row * 320:70 + (row + 1) * 320, column * 384:(column + 1) * 384] = panel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet):
        raise ValueError(f"Could not write source evidence image: {destination}")
    return {"source_video": str(match_path), "displayed_frames": frames,
            "mapping": "slicing manifest offset + local pose frame/source FPS",
            "source_clip_equivalence_independently_verified": False,
            "scope": "source observation sheet, not human annotation or fresh pose inference"}


def camera_diagnosis(path: Path) -> dict:
    camera = read(path)
    segments = [segment for clip in camera["clips"] for segment in clip["segments"]]
    return {"source": str(path), "source_sha256": digest(path), "segments": len(segments),
            "view_candidates": dict(Counter(segment["view_candidate"] for segment in segments)),
            "task_unknown_by_view": dict(Counter(segment["view_candidate"] for segment in segments
                                                if segment["task_class_candidate"] == "unknown")),
            "playback_reasons": dict(Counter(segment["playback_evidence"].get("reason", "not_recorded")
                                            for segment in segments)),
            "role_playback_reasons": dict(Counter(role.get("reason", "not_recorded") for segment in segments
                 for role in segment["playback_evidence"].get("role_results", {}).values())),
            "sample_reasons": dict(Counter(reason for segment in segments for reason in segment.get("sample_reasons", []))),
            "human_verified_segments": sum(bool(segment["human_verified"]) for segment in segments),
            "interpretation": "saved decision-reason audit; no fresh camera inference or accuracy evaluation"}


def step_foot_diagnosis(path: Path) -> dict:
    rows = read(path)["rows"]
    changes = [{"clip_id": row["clip_id"], "baseline_count": row["steps"]["baseline_v3_candidate_count"],
                "updated_count": row["steps"]["v4_candidate_count"], "sensitivity_range": row["steps"]["parameter_range"],
                "partial_runup": row["steps"]["partial_approach_known"], "contact_status": row["contact"]["status"]}
               for row in rows if row["steps"]["baseline_v3_candidate_count"] != row["steps"]["v4_candidate_count"]]
    return {"source": str(path), "source_sha256": digest(path), "canonical_events": len(rows),
            "step_changes": changes,
            "new_nonmissing_step_candidates": sum(row["baseline_count"] is None for row in changes),
            "changed_nonmissing_step_candidates": sum(row["baseline_count"] is not None for row in changes),
            "left_right_foot_flips": [row["clip_id"] for row in rows
                 if row["foot"]["baseline_candidate"] in {"left", "right"}
                 and row["foot"]["nominal_v4_candidate"] != row["foot"]["baseline_candidate"]],
            "contact_status": dict(Counter(row["contact"]["status"] for row in rows)),
            "high_step_candidates": [{"clip_id": row["clip_id"], "steps": row["steps"], "contact": row["contact"]}
                                     for row in rows if row["steps"]["v4_candidate_count"] >= 10],
            "interpretation": "saved candidate differences, not demonstrated improvements or independent accuracy"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/goal_right_foot_redetection_2026-10-11/run_v2"
    parser.add_argument("--foot-results-dir", type=Path, default=base / "results")
    parser.add_argument("--pose-dir", type=Path, default=base / "poses")
    parser.add_argument("--camera-json", type=Path, default=ROOT / "outputs/camera_timing_2026-10-10/release/camera_timings_candidates.json")
    parser.add_argument("--slicing-manifest", type=Path, default=ROOT / "configs/penalty_slicing_manifest.yaml")
    parser.add_argument("--canonical-results", type=Path, default=ROOT / "outputs/optimization_evening_2026-10-09/integration/canonical_results_39.json")
    parser.add_argument("--ball-results-dir", type=Path, help="Optional matching frozen ball-mask records for frame-matched source-pixel proximity")
    parser.add_argument("--source-video-dir", type=Path, help="Optional original shootout videos for source-frame sheets")
    parser.add_argument("--reference-json", type=Path, help="Optional development comparison, loaded only after predictions are written")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/foot_evidence_2026-10-12/cached_pose_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve frozen experiments")
    camera = read(args.camera_json)
    fps_by_id = {clip["clip_id"]: clip["fps"] for clip in camera["clips"]}
    manifest = yaml.safe_load(args.slicing_manifest.read_text(encoding="utf-8"))
    slices = {f"{key}_p{item['number']:02}_{item['kicker']}": (value["video_filename"], item["start_sec"])
              for key, value in manifest.items() for item in value["penalties"]}
    rows, inputs, source_hashes = [], [], {}
    for path in sorted(args.foot_results_dir.glob("*.json")):
        old = read(path)
        clip_id = old["clip_id"]
        pose_path = args.pose_dir / f"{clip_id}.json"
        raw = read(pose_path)
        normal = [row for row in raw if not row["reflected"]]
        mirrored = [row for row in raw if row["reflected"]]
        balls = None
        ball_path = args.ball_results_dir / f"{clip_id}.json" if args.ball_results_dir else None
        if ball_path:
            ball = read(ball_path)
            if ball["source_sha256"] != old["source_video_sha256"]:
                raise ValueError("Frozen ball and pose results use different source clip bytes")
            variant = ball["variants"]["mask_and_radius_35"]
            selected = {point["frame"]: point for point in variant["seed"] + variant["trajectory"]}
            balls = [{"frame": frame, "xy": [point["x"], point["y"]]} for frame, point in sorted(selected.items())]
        result = kicking_foot_evidence(normal, mirrored, old["contact_bracket_frames"], fps_by_id[clip_id],
                                       ball_observations=balls)
        result.update(clip_id=clip_id, original_strict_candidate=old["automatic_candidate"]["candidate"],
                      input_sha256={str(path): digest(path), str(pose_path): digest(pose_path)},
                      new_neural_inference=False, data_scope="development_cached_pose_replay")
        if ball_path:
            result["input_sha256"][str(ball_path)] = digest(ball_path)
            result["ball_observation_scope"] = "saved selected seed/trajectory image-pixel candidates; identity not human verified"
        if args.source_video_dir:
            filename, offset = slices[clip_id]
            source_path = args.source_video_dir / filename
            if filename not in source_hashes:
                source_hashes[filename] = digest(source_path)
            result["source_observation_sheet"] = render_source_sheet(
                raw, result, source_path, offset, fps_by_id[clip_id], args.output_dir / "evidence" / f"{clip_id}.jpg")
        save(args.output_dir / "results" / f"{clip_id}.json", result)
        rows.append(result)
        inputs.extend(result["input_sha256"].items())
    if not rows:
        raise ValueError("No frozen foot-result/pose pairs were found")
    summary = {"schema_version": 1, "events": len(rows), "scope": "development_cached_pose_replay",
               "new_neural_inference": False, "candidate_counts": dict(Counter(row["candidate"] for row in rows)),
               "status_counts": dict(Counter(row["status"] for row in rows)),
               "original_strict_candidate_counts": dict(Counter(row["original_strict_candidate"] for row in rows)),
               "accuracy_validated": False, "held_out_test": False, "reference_labels_read_during_prediction": False}
    save(args.output_dir / "summary.json", summary)
    save(args.output_dir / "camera_diagnosis.json", camera_diagnosis(args.camera_json))
    save(args.output_dir / "step_foot_diagnosis.json", step_foot_diagnosis(args.canonical_results))
    save(args.output_dir / "protocol.json", {"module_sha256": digest(ROOT / "src/kicking_foot_evidence.py"),
         "runner_sha256": digest(Path(__file__)), "input_sha256": dict(inputs), "source_match_sha256": source_hashes,
         "parameters": rows[0]["parameters"], "source_video_dir": str(args.source_video_dir) if args.source_video_dir else None,
         "reviewer_labels_used_for_inference": False, "existing_poses_reused": True,
         "predictions_frozen_before_runner_comparison": True, "preregistered_independent_experiment": False,
         "cohort_and_upstream_references_previously_inspected": True,
         "original_review_messages_independently_verified": False})
    for source in [Path(__file__), ROOT / "src/kicking_foot_evidence.py"]:
        snapshot = args.output_dir / "source_snapshot" / source.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    with (args.output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "candidate", "status", "original_strict_candidate"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    if args.reference_json:
        reference = read(args.reference_json)
        lookup = {row["clip_id"]: row["kicking_foot"] for row in reference["reviews"]}
        compared = [row for row in rows if row["clip_id"] in lookup and row["candidate"] != "unknown"]
        save(args.output_dir / "development_comparison.json", {
            "reference_sha256": digest(args.reference_json), "reference_source": "upstream_annotation_claim_not_independently_verified",
            "events_with_reference": sum(row["clip_id"] in lookup for row in rows), "covered": len(compared),
            "matching": sum(row["candidate"] == lookup[row["clip_id"]] for row in compared),
            "conflicts": [{"clip_id": row["clip_id"], "candidate": row["candidate"], "reference": lookup[row["clip_id"]]}
                         for row in compared if row["candidate"] != lookup[row["clip_id"]]],
            "held_out_test": False, "accuracy_validated": False,
            "interpretation": "development diagnostic agreement, never generalization accuracy"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
