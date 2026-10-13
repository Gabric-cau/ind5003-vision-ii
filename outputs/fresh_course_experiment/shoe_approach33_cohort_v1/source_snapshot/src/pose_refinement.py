"""Optional cropped-pose ablation anchored to a ball-associated cached actor.

This refines coordinates, not identity or contact ground truth. Original caches
are not mutated. Detections are returned in original pixels and absolute frames.
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REQUIRED = [11, 12, 15, 16]


def crop_bounds(box: Sequence[float], width: int, height: int, margin: float = .35) -> tuple[int, int, int, int]:
    """Expand by body height, clamp to image; never wrap negative slices."""
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Expected a finite xyxy box")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0 or margin < 0:
        raise ValueError("Invalid box, image shape or margin")
    pad = max(8., margin * (y2-y1))
    bounds = (max(0, int(np.floor(x1-pad))), max(0, int(np.floor(y1-pad))),
              min(width, int(np.ceil(x2+pad))), min(height, int(np.ceil(y2+pad))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Box does not overlap image")
    return bounds


def _numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def restore_detection(detection: Mapping[str, Any], origin: Sequence[float], frame_id: int) -> dict[str, Any]:
    """Translate crop-original outputs; Ultralytics already reverses letterbox."""
    offset = np.asarray(origin, float)
    points = np.asarray(detection["keypoints"], float).copy()
    confidence = np.asarray(detection["keypoint_confidences"], float).copy()
    if offset.shape != (2,) or points.shape != (17, 2) or confidence.shape != (17,):
        raise ValueError("Expected COCO17 points/confidence and xy crop origin")
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (confidence > 0)
    valid &= np.any(np.abs(points) > 1e-9, axis=1)
    points[valid] += offset
    points[~valid] = 0.
    confidence[~valid] = 0.
    box = np.asarray(detection["bbox"], float) + np.tile(offset, 2)
    return {"frame": int(frame_id), "bbox": box.tolist(), "box_confidence": detection.get("box_confidence"),
            "keypoints": points.tolist(), "keypoint_confidences": confidence.tolist()}


def _detections(prediction: Any) -> list[dict[str, Any]]:
    keypoints, boxes = getattr(prediction, "keypoints", None), getattr(prediction, "boxes", None)
    if keypoints is None or boxes is None:
        return []
    xy, conf, bbox = _numpy(getattr(keypoints, "xy", None)), _numpy(getattr(keypoints, "conf", None)), _numpy(getattr(boxes, "xyxy", None))
    scores = _numpy(getattr(boxes, "conf", None))
    if xy is None or conf is None or bbox is None or xy.ndim != 3 or xy.shape[1:] != (17, 2):
        return []
    output = []
    for i in range(min(len(xy), len(bbox), len(conf))):
        if not np.isfinite(bbox[i]).all():
            continue
        output.append({"bbox": bbox[i].astype(float).tolist(), "box_confidence": float(scores[i]) if scores is not None else None,
                       "keypoints": xy[i].astype(float).tolist(), "keypoint_confidences": conf[i].astype(float).tolist()})
    return output


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    overlap = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    return intersection/max(1., union)


def select_anchored_detection(detections: Sequence[dict[str, Any]], anchor: Sequence[float], rivals: Sequence[Sequence[float]] = ()) -> tuple[dict[str, Any] | None, str]:
    """A bigger/higher-confidence rival never wins merely by detector confidence."""
    anchor = np.asarray(anchor, float)
    scale = max(1., anchor[3]-anchor[1])
    centre = (anchor[:2]+anchor[2:])/2
    area = max(1., float(np.prod(anchor[2:]-anchor[:2])))
    ranked = []
    for candidate in detections:
        box = np.asarray(candidate["bbox"], float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            continue
        distance = float(np.linalg.norm((box[:2]+box[2:])/2-centre)/scale)
        overlap, ratio = _iou(box, anchor), float(np.prod(box[2:]-box[:2]))/area
        if overlap < .20 or distance > .40 or not .25 <= ratio <= 4.0:
            continue
        # Rival tracks may include duplicate boxes for the same actor; highly
        # overlapping source anchors are not independent identity evidence.
        for rival in rivals:
            rival = np.asarray(rival, float)
            if _iou(rival, anchor) > .70:
                continue
            rival_distance = float(np.linalg.norm((box[:2]+box[2:])/2-(rival[:2]+rival[2:])/2)/scale)
            if _iou(box, rival) > overlap+.05 and rival_distance < distance:
                break
        else:
            ranked.append((overlap-.25*distance, candidate))
    ranked.sort(key=lambda item: -item[0])
    if not ranked:
        return None, "no_identity_consistent_crop_detection"
    if len(ranked) > 1 and ranked[0][0]-ranked[1][0] < .08:
        return None, "ambiguous_crop_identity"
    return ranked[0][1], "accepted"


def anchor_at_frame(observations: Mapping[int, Mapping[str, Any]], frame: int, max_gap: int = 5) -> tuple[list[float] | None, str]:
    if frame in observations:
        return list(observations[frame]["bbox"]), "observed"
    ids = sorted(observations)
    index = bisect_left(ids, frame)
    if index == 0 or index == len(ids):
        return None, "no_extrapolation"
    before, after = ids[index-1], ids[index]
    if after-before > max_gap:
        return None, "source_track_gap"
    weight = (frame-before)/(after-before)
    return ((1-weight)*np.asarray(observations[before]["bbox"])+weight*np.asarray(observations[after]["bbox"])).tolist(), "short_gap_interpolation"


def pose_sequence(observations: Sequence[Mapping[str, Any]], keypoint_conf: float = .25) -> dict[str, Any]:
    output = {key: [] for key in ("frames", "left_ankles", "right_ankles", "left_hips", "right_hips", "left_confidences", "right_confidences", "box_heights")}
    for item in observations:
        points, conf = np.asarray(item["keypoints"]), np.asarray(item["keypoint_confidences"])
        if points.shape != (17, 2) or conf.shape != (17,) or np.any(conf[REQUIRED] < keypoint_conf):
            continue
        if not np.isfinite(points[REQUIRED]).all() or any(np.all(np.abs(points[index]) < 1e-9) for index in REQUIRED):
            continue
        output["frames"].append(int(item["frame"]))
        for key, index in (("left_hips",11),("right_hips",12),("left_ankles",15),("right_ankles",16)):
            output[key].append(points[index].astype(float).tolist())
        output["left_confidences"].append(float(conf[15]))
        output["right_confidences"].append(float(conf[16]))
        output["box_heights"].append(float(item["bbox"][3]-item["bbox"][1]))
    return output


def _quality(observations: Sequence[Mapping[str, Any]], total_frames: int) -> dict[str, Any]:
    sequence = pose_sequence(observations)
    confidence = [float(item["keypoint_confidences"][j]) for item in observations for j in REQUIRED]
    return {"observation_count": len(observations), "valid_hip_ankle_frames": len(sequence["frames"]),
            "valid_hip_ankle_coverage": len(sequence["frames"])/max(1,total_frames),
            "mean_required_keypoint_confidence": float(np.mean(confidence)) if confidence else None,
            "denominator_frames": total_frames, "confidence_threshold": .25}


def refine_contact_track(video_path: str, pose_result: Mapping[str, Any], contact_result: Mapping[str, Any], pose_model: Any, *, imgsz: int = 384, device: str = "cpu") -> dict[str, Any]:
    """Refine an independently associated actor; unsupported contact abstains.

    Returns ``pose_frames``/``tracks``/``selected_pose_sequence`` in the existing
    pipeline shape, plus diagnostics. No old pose is mixed into refined output.
    """
    track_id = contact_result.get("contact_associated_track_id")
    contact = contact_result.get("contact_frame")
    output: dict[str, Any] = {"schema_version": 1, "method": "ball_associated_actor_cropped_pose_ablation", "clip_id": pose_result.get("clip_id"),
        "status": "abstain", "reason": None, "contact_status": contact_result.get("status"), "contact_frame": contact,
        "contact_associated_track_id": track_id, "selected_track_id": track_id, "video_path": str(video_path),
        "pose_frames": [], "tracks": [], "selected_pose_sequence": pose_sequence([]), "diagnostics": {},
        "human_verified": False, "accuracy_validated": False,
        "provenance": {"actor_source": "contact_result.contact_associated_track_id", "ignores_previous_selected_track_id": True,
            "frame_coordinates": "original_clip_zero_based", "pixel_coordinates": "original_full_image", "imgsz": imgsz, "device": str(device),
            "keypoint_model_output_not_ground_truth": True, "source_poses_not_modified": True, "confidence_is_not_accuracy": True}}
    if contact_result.get("status") not in {"supported", "tentative"} or track_id is None or contact is None:
        output["reason"] = "unsupported_contact_or_identity"
        return output
    if pose_result.get("clip_id") != contact_result.get("clip_id"):
        output["reason"] = "clip_id_mismatch"
        return output
    track = next((t for t in pose_result.get("tracks", []) if int(t["track_id"]) == int(track_id)), None)
    if track is None or not track.get("observations"):
        output["reason"] = "associated_track_missing"
        return output
    fps = float(pose_result.get("fps") or 30.)
    start = int(pose_result.get("shot_start_frame", 0))
    end = min(int(pose_result["shot_end_frame"]), int(contact)+max(2,round(fps*.4))+1)
    observations = {int(o["frame"]): o for o in track["observations"] if start <= int(o["frame"]) < end}
    if not start <= int(contact) < end:
        output["reason"] = "contact_outside_pose_shot"
        return output
    output.update({"fps": fps, "shot_start_frame": start, "shot_end_frame": end,
                   "source_shot_end_frame": int(pose_result["shot_end_frame"]), "source_frame_count": pose_result.get("source_frame_count"),
                   "frame_width": pose_result.get("frame_width"), "frame_height": pose_result.get("frame_height")})
    rival_by_frame: dict[int, list[list[float]]] = {}
    for other in pose_result.get("tracks", []):
        if int(other["track_id"]) == int(track_id):
            continue
        for observation in other.get("observations", []):
            rival_by_frame.setdefault(int(observation["frame"]), []).append(observation["bbox"])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        output["reason"] = "video_not_opened"
        return output
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    accepted, frame_diagnostics, batch = [], [], []

    def flush() -> None:
        if not batch:
            return
        predictions = list(pose_model.predict([item["crop"] for item in batch], imgsz=imgsz, conf=.10, device=device, verbose=False))
        if len(predictions) != len(batch):
            raise ValueError("Pose result count does not match crop batch")
        for task, prediction in zip(batch, predictions):
            detections = [restore_detection(d, task["bounds"][:2], task["frame"]) for d in _detections(prediction)]
            chosen, reason = select_anchored_detection(detections, task["anchor"], rival_by_frame.get(task["frame"], []))
            record = {"frame":task["frame"], "reason":reason, "anchor_kind":task["anchor_kind"], "crop_xyxy":list(task["bounds"]), "detection_count":len(detections)}
            if chosen is not None:
                chosen["track_id"] = int(track_id)
                chosen["pose_source"] = "cropped_medium_model"
                chosen["anchor_source"] = task["anchor_kind"]
                accepted.append(chosen)
            output["pose_frames"].append({"frame":task["frame"], "detections":[chosen] if chosen is not None else []})
            frame_diagnostics.append(record)
        batch.clear()

    try:
        for frame in range(start,end):
            ok, image = capture.read()
            if not ok:
                # Do not pretend subsequent reads preserve absolute indexing
                # after an unexpected decoder failure.
                for missing_frame in range(frame,end):
                    frame_diagnostics.append({"frame":missing_frame,"reason":"decode_failed"})
                    output["pose_frames"].append({"frame":missing_frame,"detections":[]})
                break
            anchor, anchor_kind = anchor_at_frame(observations,frame)
            if anchor is None:
                frame_diagnostics.append({"frame":frame,"reason":anchor_kind})
                output["pose_frames"].append({"frame":frame,"detections":[]})
                continue
            try:
                bounds = crop_bounds(anchor,image.shape[1],image.shape[0])
            except ValueError:
                frame_diagnostics.append({"frame":frame,"reason":"invalid_crop"})
                output["pose_frames"].append({"frame":frame,"detections":[]})
                continue
            x1,y1,x2,y2 = bounds
            batch.append({"frame":frame,"crop":image[y1:y2,x1:x2].copy(),"bounds":bounds,"anchor":anchor,"anchor_kind":anchor_kind})
            if len(batch) >= 8:
                flush()
        flush()
    finally:
        capture.release()
    accepted.sort(key=lambda item:item["frame"])
    output["pose_frames"].sort(key=lambda item:item["frame"])
    output["tracks"] = [{"track_id":int(track_id),"observations":accepted}]
    output["selected_pose_sequence"] = pose_sequence(accepted)
    original_quality = _quality(list(observations.values()),end-start)
    refined_quality = _quality(accepted,end-start)
    reasons = {}
    for record in frame_diagnostics:
        reasons[record["reason"]] = reasons.get(record["reason"],0)+1
    output["diagnostics"] = {"original_associated_track":original_quality,"refined_track":refined_quality,
        "valid_frame_delta":refined_quality["valid_hip_ankle_frames"]-original_quality["valid_hip_ankle_frames"],
        "reason_counts":reasons,"frames":sorted(frame_diagnostics,key=lambda item:item["frame"]),
        "accepted_frames":len(accepted), "input_contact_remains_tentative":contact_result.get("status")=="tentative",
        "improvement_claim":"Coverage/confidence changes are not measured accuracy or identity verification."}
    output["status"] = "refined" if len(output["selected_pose_sequence"]["frames"]) >= 3 else "abstain"
    output["reason"] = "coordinates_refined_not_accuracy_verified" if output["status"] == "refined" else "insufficient_refined_keypoints"
    return output
