"""Recover shoe evidence using an automatic shooting shot and a frozen contact model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time

import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def pixel_sha(frame):
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def export_automatic_shot(video, source, pose, destination):
    """Copy a machine-selected interval, checking every decoded BGR pixel afterwards."""
    if sha(video) != source["clip_sha256"]:
        raise ValueError("Original video differs from source manifest")
    if (pose.get("reference_labels_used_for_prediction") is not False
            or pose.get("source_mapping", {}).get("clip_sha256") != source["clip_sha256"]
            or pose.get("source_frame_count") != source["frame_count"]
            or abs(pose["fps"] - source["fps"]) > .001):
        raise ValueError("Automatic shot lacks source-only clock provenance")
    start, end = pose["shot_start_frame"], pose["shot_end_frame"]
    if not 0 <= start < end <= source["frame_count"]:
        raise ValueError("Automatic shot lies outside its source")
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"FFV1"),
                             source["fps"], (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("FFV1 lossless writer unavailable")
    hashes = []
    try:
        for frame in range(end):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Original video ended before automatic shot")
            if frame >= start:
                hashes.append(pixel_sha(image))
                writer.write(image)
    finally:
        cap.release()
        writer.release()
    cap = cv2.VideoCapture(str(destination))
    try:
        if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(hashes)
                or abs(cap.get(cv2.CAP_PROP_FPS) - source["fps"]) > .001):
            raise ValueError("Lossless shot clock changed")
        for expected in hashes:
            ok, image = cap.read()
            if not ok or pixel_sha(image) != expected:
                raise ValueError("Lossless shot pixels differ from original source")
    finally:
        cap.release()
    return {"source_sha256": source["clip_sha256"], "shot_sha256": sha(destination),
            "source_start_frame": start, "source_end_frame_exclusive": end,
            "source_frame_count": source["frame_count"], "fps": source["fps"],
            "width": width, "height": height, "decoded_frame_sha256": hashes,
            "selection": "previous_source_bound_automatic_pose_shot",
            "human_annotations_used": False, "all_pixels_verified_equal": True}


def associate_measured_actor(tracks, actor, offset, fps=30.):
    """Identify a unique old dense track through overlapping measured boxes."""
    observed = {r["frame"] + offset: np.asarray(r["bbox"], float) for r in actor["observations"]}
    scores = []
    for track in tracks:
        overlaps = []
        for row in track["observations"]:
            if row["frame"] not in observed:
                continue
            a, b = np.asarray(row["bbox"], float), observed[row["frame"]]
            intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
            union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
            overlaps.append(intersection / union if union > 0 else 0.)
        if overlaps:
            scores.append({"track_id": track["track_id"], "shared_frames": len(overlaps),
                           "median_iou": float(np.median(overlaps)),
                           "strong_matches": sum(v >= .5 for v in overlaps)})
    scores.sort(key=lambda r: (-r["median_iou"], -r["strong_matches"]))
    accepted = [r for r in scores if r["shared_frames"] >= 3 and r["strong_matches"] >= 3
                and r["median_iou"] >= .5]
    selected_ids = [r["track_id"] for r in accepted] if len(accepted) == 1 else []
    # Detector track fragmentation can produce a short overlapping hand-off.
    # Each fragment must independently agree with the same new measured actor.
    if len(accepted) == 2:
        old = [{r["frame"]: np.asarray(r["bbox"], float) for r in t["observations"]}
               for t in tracks if t["track_id"] in {a["track_id"] for a in accepted}]
        shared = sorted(set(old[0]) & set(old[1]))
        overlaps = []
        for frame in shared:
            a, b = old[0][frame], old[1][frame]
            intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
            union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
            overlaps.append(intersection / union if union > 0 else 0.)
        if (0 < len(shared) <= max(1, round(.15 * fps)) and np.median(overlaps) >= .35
                and min(overlaps) >= .2):
            selected_ids = [a["track_id"] for a in accepted]
    selected = selected_ids[0] if len(selected_ids) == 1 else None
    return selected, {"selected_old_track_id": selected, "candidates": scores,
                      "selected_old_track_ids": selected_ids,
                      "method": "unique_3_shared_measured_boxes_median_iou_at_least_0.5",
                      "human_labels_used": False}


def build_bound(bundle, contact, evidence, original_pose=None):
    """Map only measured contact-stage observations back to the original clip clock."""
    source, receipt = bundle["source_mapping"], bundle["lossless_shot"]
    if (contact["video_sha256"] != receipt["shot_sha256"]
            or evidence["video_sha256"] != receipt["shot_sha256"]):
        raise ValueError("Contact evidence is not bound to the verified shot")
    point = contact["prediction"].get("frame")
    actor_id = contact.get("diagnostics", {}).get("actor", {}).get("actor_id")
    if point is None or actor_id is None:
        return None, "automatic_contact_or_actor_unavailable"
    actor = next((t for t in evidence["tracks"] if t["track_id"] == actor_id), None)
    if actor is None:
        return None, "contact_actor_missing_from_measured_tracks"
    offset = receipt["source_start_frame"]
    fps = receipt["fps"]
    shot_end = min(receipt["source_end_frame_exclusive"],
                   offset + contact["automatic_window"]["end_frame_exclusive"])
    contact_frame = offset + int(point)
    bracket = [max(offset, contact_frame - 1), min(shot_end - 1, contact_frame + 1)]
    # Cache the whole automatic shot so later outer folds can change contact time
    # without selecting pose evidence with a model trained on their validation match.
    start, end = offset, shot_end - 1
    if not start < bracket[0] < bracket[1] <= end:
        return None, "contact_too_close_to_automatic_shot_edge"
    association = {"selected_old_track_id": None, "human_labels_used": False}
    candidates = []
    if original_pose is not None:
        selected, association = associate_measured_actor(original_pose["tracks"], actor, offset, fps)
        for old_track in original_pose["tracks"]:
            if old_track["track_id"] not in association["selected_old_track_ids"]:
                continue
            candidates.extend((r["frame"], r, "original_measured_pose_geometrically_reassociated")
                              for r in old_track["observations"])
    candidates.extend((offset + r["frame"], r, "new_measured_contact_pose") for r in actor["observations"])
    observations, observation_sources = {}, {}
    for fid, row, measured_source in candidates:
        if row.get("interpolated") is True or row.get("is_interpolated") is True:
            raise ValueError("Interpolated poses cannot anchor neural shoe evidence")
        if start <= fid <= end:
            box = np.asarray(row["bbox"], float)
            if (box.shape == (4,) and np.isfinite(box).all()
                    and 0 <= box[0] < box[2] <= receipt["width"]
                    and 0 <= box[1] < box[3] <= receipt["height"]):
                observations[fid] = row
                observation_sources[fid] = measured_source
    if len(observations) < 5:
        return None, "insufficient_measured_actor_observations"
    from src.pose_refinement import crop_bounds
    bounds = np.asarray([crop_bounds(r["bbox"], receipt["width"], receipt["height"], margin=.45)
                         for r in observations.values()], int)
    fixed = [int(bounds[:, 0].min()), int(bounds[:, 1].min()),
             int(bounds[:, 2].max()), int(bounds[:, 3].max())]
    rivals = {}
    for track in evidence["tracks"]:
        if track["track_id"] == actor_id:
            continue
        for row in track["observations"]:
            rivals.setdefault(offset + row["frame"], []).append(row["bbox"])
    plan = [{"frame": frame, "crop_xyxy": fixed,
             "anchor_bbox": observations.get(frame, {}).get("bbox"),
             "anchor_source": observation_sources.get(frame, "missing_no_interpolation"),
             "rival_source_boxes": rivals.get(frame, [])} for frame in range(start, end + 1)]
    transforms = {str(offset + int(k)): v for k, v in evidence["transforms"].items()}
    return {"source_mapping": source, "video": Path(bundle["video_path"]), "fps": fps,
            "width": receipt["width"], "height": receipt["height"],
            "scene_interval_frames": [offset, shot_end], "interval_frames": [start, bracket[0]],
            "inference_interval_frames": [start, end], "contact_bracket_frames": bracket,
            "actor_track_id": actor_id, "original_pose_selected_track_id": bundle["old_selected_actor"],
            "dense_actor_association": association,
            "plan": plan, "fixed_roi": fixed, "original_camera_motion": {"transforms": transforms},
            "original_phase_result": {"status": "not_used", "approach_start_frame": None},
            "contact": {"status": "learned_candidate", "contact_frame": contact_frame,
                        "contact_associated_track_id": actor_id,
                        "bracket_is_sampling_neighborhood_not_confidence_interval": True}}, None


def infer_shoes(args):
    from scripts.run_fresh_shoe_approach_experiment import infer_event
    from src.shoe_supervised_model import extract_evidence
    bundle = read(args.event_dir / "recovery_input.json")
    if sha(bundle["video_path"]) != bundle["source_mapping"]["clip_sha256"]:
        raise ValueError("Original video changed before shoe extraction")
    contact = read(args.event_dir / "contact/prediction.json")
    evidence = read(args.event_dir / "contact/evidence.json")
    if sha(bundle["pose_path"]) != bundle["pose_sha256"]:
        raise ValueError("Original measured pose changed during recovery")
    original_pose = read(bundle["pose_path"])
    bound, reason = build_bound(bundle, contact, evidence, original_pose)
    directory = args.event_dir / "shoes"
    if directory.exists():
        raise FileExistsError("Shoe evidence already exists")
    directory.mkdir()
    if bound is None:
        write(directory / "summary.json", {"status": "unavailable", "reason": reason})
        return
    started = time.perf_counter()
    from src.kick_contact import _camera_transforms
    cap = cv2.VideoCapture(bundle["video_path"])
    images, ids = [], []
    try:
        for fid in range(bound["scene_interval_frames"][1]):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Cannot decode full automatic shot for camera motion")
            if fid >= bound["scene_interval_frames"][0]:
                images.append(image)
                ids.append(fid)
    finally:
        cap.release()
    matrices, camera_diagnostics = _camera_transforms(images, ids, original_pose)
    failed_starts = [pair[1] for pair in camera_diagnostics["failed_pairs"]]
    first_failed = min(failed_starts) if failed_starts else float("inf")
    bound["original_camera_motion"] = {"transforms": {
        str(fid): matrix.tolist() if fid < first_failed else None for fid, matrix in zip(ids, matrices)},
        "camera_diagnostics": camera_diagnostics}
    del images
    records, hashes, quality = infer_event(bound, args.pose33_model)
    write(directory / "poses33.json", records)
    inputs = {k: bound[k] for k in ("source_mapping", "fps", "scene_interval_frames", "interval_frames",
        "inference_interval_frames", "contact_bracket_frames", "actor_track_id", "original_pose_selected_track_id",
        "original_camera_motion", "original_phase_result", "contact", "dense_actor_association")}
    inputs.update(model_sha256=sha(args.pose33_model), reference_labels_read=False,
        new_neural_inference=True, human_verified=False, decoded_frame_hashes=hashes,
        poses33_sha256=sha(directory / "poses33.json"), coordinate_space="original_source_image_pixels",
        extraction_scope="automatic_shooting_shot_before_predicted_contact_not_manual_review_window",
        recovery_input_sha256=sha(args.event_dir / "recovery_input.json"),
        contact_prediction_sha256=sha(args.event_dir / "contact/prediction.json"))
    write(directory / "inputs.json", inputs)
    features = extract_evidence(inputs, records)
    write(directory / "summary.json", {"status": "fresh_recovered_pose_candidate", **quality,
        "supported_side_frames": features["supported_side_frames"],
        "foot_features_available": features["foot_vector"] is not None,
        "landing_feature_rows": len(features["landing_rows"]),
        "automatic_window": inputs["interval_frames"], "runtime_seconds": time.perf_counter() - started,
        "inputs_sha256": sha(directory / "inputs.json"), "poses33_sha256": sha(directory / "poses33.json")})


def recover(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh experiment output directory")
    args.output.mkdir(parents=True)
    config = read(args.config)
    for key in ("contact_model", "pose_model", "ball_model", "pose33_model", "shoe_python"):
        if not Path(config[key]).is_absolute():
            config[key] = str(args.config.parent / config[key])
    source_rows = {r["clip_id"]: r for r in read(args.manifest)["clips"]}
    summaries = []
    for cid in args.clips:
        directory = args.output / cid
        directory.mkdir()
        started = time.perf_counter()
        try:
            source = source_rows[cid]
            pose_path = args.pose_dir / (cid + ".json")
            pose = read(pose_path)
            video = args.clips_dir / source["clip_filename"]
            shot = directory / "automatic_shot.avi"
            if args.reuse_contact:
                previous = args.reuse_contact / cid
                old_input = read(previous / "recovery_input.json")
                if (old_input["source_mapping"] != source or old_input["pose_sha256"] != sha(pose_path)
                        or sha(video) != source["clip_sha256"]):
                    raise ValueError("Reused contact has different source or shot selector")
                receipt = old_input["lossless_shot"]
                if sha(previous / "automatic_shot.avi") != receipt["shot_sha256"]:
                    raise ValueError("Reused lossless shot changed")
                shutil.copytree(previous / "contact", directory / "contact")
            else:
                receipt = export_automatic_shot(video, source, pose, shot)
            write(directory / "recovery_input.json", {"source_mapping": source, "video_path": str(video),
                "lossless_shot": receipt, "pose_path": str(pose_path), "pose_sha256": sha(pose_path),
                "old_selected_actor": pose["selected_track_id"], "manual_annotations_used": False})
            command = [sys.executable, str(args.runtime_root / "scripts/predict_video_contact.py"),
                "--video", str(shot), "--model", config["contact_model"], "--ball-model", config["ball_model"],
                "--pose-model", config["pose_model"], "--output", str(directory / "contact"),
                "--device", args.device, "--threads", "2"]
            if not args.reuse_contact:
                with (directory / "contact.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = [config["shoe_python"], str(Path(__file__).resolve()), "--runtime-root", str(args.runtime_root),
                "--event-dir", str(directory), "--pose33-model", config["pose33_model"]]
            with (directory / "shoes.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = read(directory / "shoes/summary.json")
        except Exception as error:
            result = {"status": "failed", "reason": str(error)}
        result = {"clip_id": cid, "total_runtime_seconds": time.perf_counter() - started, **result}
        summaries.append(result)
        write(args.output / "summary.json", {"manual_annotations_used": False, "rows": summaries})
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--event-dir", type=Path)
    parser.add_argument("--pose33-model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-contact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(args.runtime_root))
    cv2.setNumThreads(2)
    if args.event_dir:
        infer_shoes(args)
    else:
        if any(v is None for v in (args.manifest, args.pose_dir, args.clips_dir, args.clips, args.config, args.output)):
            parser.error("Recovery requires manifest, pose-dir, clips-dir, clips, config and output")
        recover(args)


if __name__ == "__main__":
    main()
