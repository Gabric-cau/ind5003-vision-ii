"""Recover missing striking-foot evidence inside one measured camera segment."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import cv2
import numpy as np

from src.kick_contact import _camera_transforms
from src.runtime import file_sha256
from src.shoe_supervised_model import extract_evidence, predict_event


def failed_camera_links(camera):
    diagnostics = camera.get("camera_diagnostics", {})
    failures = diagnostics.get("failed_pairs")
    if failures is not None:
        return failures
    return [[row["frame"] - 1, row["frame"]] for row in camera.get("pairs", [])
            if row.get("accepted") is False]


def measure_segment_camera(video, inputs, pose):
    """Stream measured pair transforms, starting a new reference after failure."""
    source = inputs["source_mapping"]
    digest = file_sha256(video)
    if digest != source["clip_sha256"]:
        raise ValueError("Camera recovery video differs from the bound source")
    if (pose.get("source_mapping") != source
            or pose.get("inference_identity", {}).get("source_sha256") != digest):
        raise ValueError("Camera recovery pose belongs to another source")
    start, end = inputs["scene_interval_frames"]
    if not 0 <= start < end <= source["frame_count"]:
        raise ValueError("Camera scene is outside the source clock")
    fps = float(inputs["fps"])
    cap = cv2.VideoCapture(str(video))
    transforms, segment_ids, failures, inliers = {}, {}, [], []
    previous = None
    cumulative = np.eye(3)
    segment = 0
    try:
        if (not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps, atol=.01)
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != source["frame_count"]):
            raise ValueError("Camera recovery decoder clock differs from the source")
        for frame in range(end):
            ok, image = cap.read()
            if not ok or image.shape[:2] != (pose["frame_height"], pose["frame_width"]):
                raise ValueError("Camera recovery video decode or dimensions changed")
            if frame < start:
                continue
            expected = inputs["decoded_frame_hashes"].get(str(frame))
            if expected and hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != expected:
                raise ValueError("Camera recovery decoded pixels differ from pose evidence")
            if previous is not None:
                pair, diagnostics = _camera_transforms([previous, image], [frame - 1, frame], pose)
                if diagnostics["failed_pairs"]:
                    failures.append([frame - 1, frame])
                    segment += 1
                    cumulative = np.eye(3)
                else:
                    cumulative = cumulative @ pair[1]
                    inliers.append(diagnostics["median_inlier_ratio"])
            transforms[str(frame)] = cumulative.tolist()
            segment_ids[str(frame)] = segment
            previous = image
    finally:
        cap.release()
    return {"transforms": transforms, "segment_ids": segment_ids,
            "source_sha256": digest, "source_interval_frames": [start, end],
            "camera_diagnostics": {"failed_pairs": failures, "segment_count": segment + 1,
                "median_inlier_ratio": float(np.median(inliers)) if inliers else None},
            "method": "measured_background_affine_with_reference_reset_after_failed_link",
            "coordinates": "image_plane_of_each_contiguous_camera_segment",
            "interpolated_transforms_used": False, "cross_segment_motion_used": False}


def recover_missing_foot(inputs, records, model, prediction, *, video=None, pose=None, camera=None):
    """Keep every existing decision and landing; recover only an unavailable foot."""
    result = deepcopy(prediction)
    receipt = {"attempted": False, "used": False, "human_labels_used": False,
               "existing_foot_decision_preserved": prediction["striking_foot"] in ("left", "right"),
               "landing_output_preserved": True, "cross_segment_motion_used": False,
               "method": "automatic_contact_segment_frozen_foot_head"}
    if receipt["existing_foot_decision_preserved"]:
        return result, {**receipt, "reason": "existing_foot_decision"}, None
    old_failures = failed_camera_links(inputs["original_camera_motion"])
    if not old_failures:
        return result, {**receipt, "reason": "no_recorded_camera_failure"}, None
    receipt.update(attempted=True, original_failed_pairs=old_failures)
    if camera is None:
        if video is None or pose is None:
            return result, {**receipt, "reason": "source_for_camera_remeasurement_unavailable"}, None
        camera = measure_segment_camera(Path(video), inputs, pose)
    if camera.get("source_sha256") != inputs["source_mapping"]["clip_sha256"]:
        raise ValueError("Segment camera does not bind the shoe video")
    bracket = inputs.get("contact_bracket_frames")
    if not isinstance(bracket, list) or len(bracket) != 2:
        return result, {**receipt, "reason": "automatic_contact_unavailable"}, camera
    segments = camera["segment_ids"]
    ids = [segments.get(str(frame)) for frame in range(bracket[0], bracket[1] + 1)]
    if not ids or None in ids or len(set(ids)) != 1:
        return result, {**receipt, "reason": "contact_crosses_camera_failure"}, camera
    segment_id = ids[0]
    frames = sorted(int(frame) for frame, value in segments.items() if value == segment_id)
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("Camera segment is not contiguous")
    first, last = inputs["inference_interval_frames"]
    first, last = max(first, frames[0]), min(last, frames[-1])
    if not first <= bracket[0] <= bracket[1] <= last:
        return result, {**receipt, "reason": "contact_outside_saved_pose_segment"}, camera
    scoped = deepcopy(inputs)
    scoped["inference_interval_frames"] = [first, last]
    scoped["interval_frames"] = [max(first, inputs["interval_frames"][0]),
                                  min(last, inputs["interval_frames"][1])]
    scoped["original_camera_motion"] = {"transforms": {
        str(frame): camera["transforms"][str(frame)] for frame in range(first, last + 1)}}
    evidence = extract_evidence(scoped, records)
    candidate = predict_event(model, evidence, inputs["fps"])
    receipt.update(segment_id=segment_id, segment_frames=[first, last],
                   measured_failed_pairs=camera["camera_diagnostics"]["failed_pairs"],
                   supported_side_frames=evidence["supported_side_frames"],
                   contact_bracket_frames=bracket,
                   segment_foot_candidate=candidate["striking_foot"])
    if candidate["striking_foot"] not in ("left", "right"):
        return result, {**receipt, "reason": "segment_paired_pose_still_unavailable"}, camera
    result.update(striking_foot=candidate["striking_foot"],
                  right_score_uncalibrated=candidate["right_score_uncalibrated"])
    receipt.update(used=True, reason="recovered_within_measured_contact_segment")
    return result, receipt, camera
