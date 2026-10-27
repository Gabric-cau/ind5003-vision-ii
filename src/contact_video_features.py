"""Frozen visual evidence for contact-time learning, with no annotation inputs."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.runtime import file_sha256
from src.step_pipeline import (_scan_camera_shots, _extract_result_detections,
                               _NearestCentreBoxTracker, _json_safe)
from src.kick_contact import (_camera_transforms, _all_track_samples,
                              _score_all_tracks, _xy_transform)

SETTINGS = {"stride": 2, "max_shot_sec": 12.0, "cut_threshold": 30.0,
            "ball_imgsz": 1280, "pose_imgsz": 640, "batch_size": 4,
            "ball_conf": .04, "pose_conf": .15}
FEATURE_NAMES = [
    "pose_visible", "left_ankle_visible", "right_ankle_visible",
    "left_ankle_confidence", "right_ankle_confidence",
    "left_ankle_seed_distance", "right_ankle_seed_distance",
    "ankle_separation", "left_ankle_speed", "right_ankle_speed", "hip_speed",
    "left_knee_extension", "right_knee_extension", "body_lean",
    "seed_available", "ball_visible", "ball_confidence",
    "ball_seed_distance", "ball_speed", "ball_nearest_ankle_distance",
    "ball_near_seed", "camera_pair_reliable", "body_height_fraction",
    "pose_detection_confidence",
]


def angle(a, b, c):
    u, v = a - b, c - b
    norm = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1)) / np.pi) if norm > 1e-8 else 0.


def rule_baseline(candidates, camera, end):
    selected = candidates[0] if candidates else None
    support_end = max(selected["interval_frames"] + (selected["ball_departure_frames"] or [0])) if selected else end
    reliable = not any(pair[0] <= support_end for pair in camera["failed_pairs"])
    ambiguous = bool(selected and any(c["contact_associated_track_id"] != selected["contact_associated_track_id"]
                     and c["score"] >= selected["score"] - .07 for c in candidates[1:]))
    if selected and selected["score"] >= .75 and selected["ball_departure_frames"] and selected["ankle_approach_supported"] and reliable and not ambiguous:
        status = "supported"
    elif selected and selected["score"] >= .50:
        status = "tentative"
    else:
        status, selected = "abstain", None
    return {"time_seconds": selected["contact_time_sec"] if selected else None,
            "status": status, "score": selected["score"] if selected else None,
            "method": "existing_causal_first_departure_rule_on_same_stride2_neural_evidence",
            "full_original_pipeline": False, "sparse_to_dense_refinement": False,
            "camera_reliable_through_support": reliable, "identity_ambiguous": ambiguous}


def geometric_features(ids, fps, height, tracks, balls, transforms, candidates, camera):
    track_by_id = {t["track_id"]: t for t in tracks}
    usable_tracks = [t for t in tracks if len(t["observations"]) >= 3]
    if candidates:
        actor_id = candidates[0]["contact_associated_track_id"]
        seed = np.asarray(candidates[0]["stationary_ball_reference_xy"], float)
        actor_method = "highest_ranked_existing_rule_actor"
    else:
        def movement(track):
            observations = track["observations"]
            centers = np.array([[(o["bbox"][0] + o["bbox"][2]) / 2,
                                 (o["bbox"][1] + o["bbox"][3]) / 2] for o in observations])
            scale = np.median([o["bbox"][3] - o["bbox"][1] for o in observations])
            return float(np.linalg.norm(centers[-1] - centers[0]) / max(scale, 1))
        actor_id = max(usable_tracks, key=movement)["track_id"] if usable_tracks else None
        seed, actor_method = None, "largest_normalized_track_displacement_fallback"
    actor = track_by_id.get(actor_id, {"observations": []})
    observations = {o["frame"]: o for o in actor["observations"]}
    balls_by_frame = defaultdict(list)
    for ball in balls:
        stable = _xy_transform(np.asarray([ball["xy"]]), transforms[ball["frame"]])[0]
        balls_by_frame[ball["frame"]].append({**ball, "stable_xy": stable})
    failed = {pair[1] for pair in camera["failed_pairs"]}
    rows, previous_points, previous_ball = [], None, None
    previous_scale = 1.
    proximity = []
    for i, fid in enumerate(ids):
        values = dict.fromkeys(FEATURE_NAMES, 0.)
        values["camera_pair_reliable"] = float(fid not in failed)
        values["seed_available"] = float(seed is not None)
        observation = observations.get(fid)
        points = valid = None
        scale = previous_scale
        dt = (fid - ids[i - 1]) / fps if i else 0
        if observation:
            scale = max(20., observation["bbox"][3] - observation["bbox"][1])
            points = _xy_transform(np.asarray(observation["keypoints"]), transforms[fid])
            confidence = np.asarray(observation["keypoint_confidences"])
            valid = confidence >= .25
            values.update(pose_visible=1., body_height_fraction=scale / height,
                          pose_detection_confidence=float(observation.get("box_confidence") or 0.))
            for side, index in (("left", 15), ("right", 16)):
                values[side + "_ankle_visible"] = float(valid[index])
                values[side + "_ankle_confidence"] = float(confidence[index])
                if valid[index] and seed is not None:
                    values[side + "_ankle_seed_distance"] = min(5., float(np.linalg.norm(points[index] - seed) / scale))
                    proximity.append((values[side + "_ankle_seed_distance"], fid))
                if valid[index] and previous_points is not None and previous_points[1][index] and dt:
                    values[side + "_ankle_speed"] = min(20., float(np.linalg.norm(points[index] - previous_points[0][index]) / scale / dt))
            if valid[15] and valid[16]:
                values["ankle_separation"] = float(np.linalg.norm(points[15] - points[16]) / scale)
            if all(valid[[11, 12]]) and previous_points is not None and all(previous_points[1][[11, 12]]) and dt:
                values["hip_speed"] = min(20., float(np.linalg.norm(points[[11, 12]].mean(0) - previous_points[0][[11, 12]].mean(0)) / scale / dt))
            for side, joints in (("left", [11, 13, 15]), ("right", [12, 14, 16])):
                if all(valid[joints]):
                    values[side + "_knee_extension"] = angle(*points[joints])
            if all(valid[[5, 6, 11, 12]]):
                shoulder = points[[5, 6]].mean(0)
                hip = points[[11, 12]].mean(0)
                values["body_lean"] = abs(float(shoulder[0] - hip[0])) / scale
        options = balls_by_frame[fid]
        if options:
            if seed is not None:
                ball = min(options, key=lambda b: np.linalg.norm(b["stable_xy"] - seed))
            else:
                ball = max(options, key=lambda b: b["confidence"])
            values.update(ball_visible=1., ball_confidence=ball["confidence"])
            if seed is not None:
                dist = float(np.linalg.norm(ball["stable_xy"] - seed) / scale)
                values["ball_seed_distance"] = min(5., dist)
                values["ball_near_seed"] = float(dist < .15)
            if previous_ball is not None and dt:
                values["ball_speed"] = min(30., float(np.linalg.norm(ball["stable_xy"] - previous_ball) / scale / dt))
            if points is not None and any(valid[[15, 16]]):
                values["ball_nearest_ankle_distance"] = min(5., min(float(np.linalg.norm(points[j] - ball["stable_xy"]) / scale)
                                                                  for j in (15, 16) if valid[j]))
            previous_ball = ball["stable_xy"]
        else:
            previous_ball = None
        previous_points = (points, valid) if points is not None else None
        previous_scale = scale
        rows.append([float(values[n]) for n in FEATURE_NAMES])
    return rows, {"actor_id": actor_id, "actor_selection_method": actor_method,
                  "seed_xy": seed.tolist() if seed is not None else None,
                  "actor_visible_samples": len(observations)}, {
        "time_seconds": min(proximity)[1] / fps if proximity else None,
        "method": "minimum_ankle_to_stationary_seed_distance", "status": "candidate" if proximity else "abstain"}


def extract_contact_record(sample, ball_model, pose_model, model_paths, *, video_root=None,
                           device="cpu", settings=None):
    settings = dict(settings or SETTINGS)
    if any("reference" in key or "contact_seconds" in key for key in sample):
        raise ValueError("Annotation data cannot enter the visual extractor")
    started = time.perf_counter()
    video = Path(video_root) / (sample["core_id"] + ".mp4") if video_root else Path(sample["video_path"])
    if file_sha256(video) != sample["video_sha256"]:
        raise ValueError("Video checksum does not match label-free input manifest")
    shot = _scan_camera_shots(str(video), settings["max_shot_sec"], settings["cut_threshold"])[0]
    if not np.isclose(shot["fps"], sample["fps"]):
        raise ValueError("FPS mismatch")
    ids = list(range(shot["shot_start_frame"], shot["shot_end_frame"], settings["stride"]))
    cap = cv2.VideoCapture(str(video))
    frames = []
    try:
        wanted = set(ids)
        for index in range(shot["shot_end_frame"]):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Video decode ended inside the automatic shot")
            if index in wanted:
                frames.append(frame)
    finally:
        cap.release()
    tracker = _NearestCentreBoxTracker(max_frame_gap=3 * settings["stride"])
    balls, pose_warnings = [], []
    inference_started = time.perf_counter()
    for start in range(0, len(ids), settings["batch_size"]):
        batch = frames[start:start + settings["batch_size"]]
        batch_ids = ids[start:start + settings["batch_size"]]
        results = pose_model.predict(batch, imgsz=settings["pose_imgsz"], conf=settings["pose_conf"],
                                     device=device, verbose=False)
        for fid, result in zip(batch_ids, results):
            detections, warnings = _extract_result_detections(result, fid)
            tracker.update(fid, detections)
            pose_warnings.extend(warnings)
        results = ball_model.predict(batch, imgsz=settings["ball_imgsz"], conf=settings["ball_conf"],
                                     classes=[32], device=device, verbose=False)
        for fid, result in zip(batch_ids, results):
            if result.boxes is None:
                continue
            for bbox, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x0, y0, x1, y1 = map(float, bbox)
                w, h = x1 - x0, y1 - y0
                if min(w, h) < 2 or max(w, h) > sample["height"] * .1:
                    continue
                balls.append({"frame": fid, "xy": [(x0 + x1) / 2, (y0 + y1) / 2],
                              "bbox": [x0, y0, x1, y1], "confidence": float(confidence),
                              "diameter": (w + h) / 2})
    neural_seconds = time.perf_counter() - inference_started
    pose = {"tracks": list(tracker.tracks.values()), "fps": sample["fps"],
            "shot_start_frame": shot["shot_start_frame"], "shot_end_frame": shot["shot_end_frame"]}
    matrices, camera = _camera_transforms(frames, ids, pose)
    transforms = dict(zip(ids, matrices))
    candidates = _score_all_tracks(balls, _all_track_samples(pose), transforms,
                                   sample["fps"], settings["stride"], "causal_first_departure")
    features, actor, ankle_baseline = geometric_features(ids, sample["fps"], sample["height"],
        pose["tracks"], balls, transforms, candidates, camera)
    record = {"schema_version": 1, "core_id": sample["core_id"], "clip_id": sample["clip_id"],
        "match_group": sample["match_group"], "video_sha256": sample["video_sha256"], "fps": sample["fps"],
        "window": {"start_frame": shot["shot_start_frame"], "end_frame_exclusive": shot["shot_end_frame"],
                   "automatic": True, "method": "existing_histogram_and_spatial_camera_cut"},
        "frame_ids": ids, "time_seconds": [i / sample["fps"] for i in ids],
        "feature_names": FEATURE_NAMES, "features": features,
        "baselines": {"existing_rule": rule_baseline(candidates, camera, shot["shot_end_frame"]),
                      "ankle_proximity": ankle_baseline},
        "provenance": {"human_labels_used": False, "old_contact_used": False,
                       "manual_camera_boundaries_used": False, "pretrained_models_updated": False,
                       "offline_future_frames_allowed": True,
                       "model_sha256": {Path(p).name: file_sha256(p) for p in model_paths},
                       "code_sha256": {name: file_sha256(Path(__file__).parents[1] / name) for name in
                          ("src/contact_video_features.py", "src/kick_contact.py", "src/step_pipeline.py")}},
        "settings": settings, "runtime": {"device": device, "neural_seconds": neural_seconds,
                      "total_seconds_excluding_model_load": time.perf_counter() - started},
        "diagnostics": {"camera": camera, "actor": actor, "pose_warnings": sorted(set(pose_warnings)),
                        "sampled_frames": len(ids), "ball_detections": len(balls),
                        "ball_detected_frames": len({b["frame"] for b in balls}),
                        "tracks": len(pose["tracks"]), "rule_candidates": len(candidates)}}
    evidence = {"core_id": sample["core_id"], "video_sha256": sample["video_sha256"],
                "tracks": pose["tracks"], "balls": balls, "candidate_contacts": candidates,
                "camera": camera, "transforms": {str(k): v.tolist() for k, v in transforms.items()}}
    return _json_safe(record), _json_safe(evidence)
