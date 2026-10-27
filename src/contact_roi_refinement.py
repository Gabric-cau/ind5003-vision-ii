"""Reobserve automatically located pose gaps without creating interpolated poses."""
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np

from src.contact_tracking_recovery import recover_contact_evidence
from src.contact_video_features import geometric_features, rule_baseline
from src.runtime import file_sha256
from src.step_pipeline import _extract_result_detections
from src.pose_refinement import select_anchored_detection, restore_detection
from src.kick_contact import _xy_transform

REFINEMENT_SETTINGS = {"pose_gap_roi": True, "roi_pose_imgsz": 640, "roi_pose_conf": .1,
                       "roi_batch_size": 4, "roi_refinement_version": 1}


def refine_contact_record(record, evidence, video, pose_model, *, device="cpu"):
    if file_sha256(video) != record["video_sha256"] or evidence["video_sha256"] != record["video_sha256"]:
        raise ValueError("ROI source hash mismatch")
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    recovery = recover_contact_evidence(evidence, fps=record["fps"], frame_height=height,
        frame_width=width, frame_ids=record["frame_ids"])
    plans = recovery["pose_roi_requests"]
    added, receipts = [], []
    tracks = {t["track_id"]: t for t in recovery["tracks"]}
    try:
        for start in range(0, len(plans), REFINEMENT_SETTINGS["roi_batch_size"]):
            batch = plans[start:start + REFINEMENT_SETTINGS["roi_batch_size"]]
            images = []
            for plan in batch:
                cap.set(cv2.CAP_PROP_POS_FRAMES, plan["frame"])
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Cannot decode automatic ROI frame")
                x0, y0, x1, y1 = plan["bbox"]
                images.append(frame[y0:y1, x0:x1])
            results = pose_model.predict(images, imgsz=REFINEMENT_SETTINGS["roi_pose_imgsz"],
                conf=REFINEMENT_SETTINGS["roi_pose_conf"], device=device, verbose=False)
            if len(results) != len(batch):
                raise ValueError("Pose inference did not return every requested ROI")
            for plan, result in zip(batch, results):
                observations, _ = _extract_result_detections(result, plan["frame"])
                observations = [restore_detection(o, plan["bbox"][:2], plan["frame"])
                                for o in observations]
                # Interpolation is an identity/crop anchor, never an observation.
                anchors = [o for o in tracks[plan["track_id"]]["observations"]
                           if o["frame"] in plan["measured_anchor_frames"]]
                fraction = (plan["frame"] - anchors[0]["frame"]) / (anchors[-1]["frame"] - anchors[0]["frame"])
                stable_boxes = []
                for observation in anchors:
                    x0, y0, x1, y1 = observation["bbox"]
                    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                    stable_boxes.append(_xy_transform(corners,
                        np.asarray(evidence["transforms"][str(observation["frame"])], float)))
                stable_anchor = stable_boxes[0] * (1 - fraction) + stable_boxes[-1] * fraction
                source_anchor = _xy_transform(stable_anchor,
                    np.linalg.inv(np.asarray(evidence["transforms"][str(plan["frame"])], float)))
                anchor = np.r_[source_anchor.min(axis=0), source_anchor.max(axis=0)].tolist()
                selected, reason = select_anchored_detection(observations, anchor)
                receipt = {**plan, "detected_poses": len(observations), "identity_status": reason,
                           "accepted_neural_observation": selected is not None}
                if selected is not None:
                    observation = deepcopy(selected)
                    observation["observation_source"] = "new_neural_pose_on_automatic_gap_roi"
                    added.append(observation)
                receipts.append(receipt)
    finally:
        cap.release()
    augmented = deepcopy(evidence)
    next_id = max((t["track_id"] for t in augmented["tracks"]), default=-1) + 1
    augmented["tracks"] += [{"track_id": next_id + i, "observations": [o]} for i, o in enumerate(added)]
    recovered = recover_contact_evidence(augmented, fps=record["fps"], frame_height=height,
        frame_width=width, frame_ids=record["frame_ids"])
    result = deepcopy(record)
    diagnostics = {"requested_crops": len(plans), "accepted_neural_observations": len(added),
        "receipts": receipts, "recovery": recovered["recovery_diagnostics"],
        "human_labels_used": False, "interpolated_pose_used": False, "applied": False}
    baseline = record["baselines"]["existing_rule"]
    old_visible = record["diagnostics"]["actor"]["actor_visible_samples"]
    new_visible = recovered["recovery_diagnostics"]["actor_measured_samples"]
    if baseline["time_seconds"] is None and new_visible > old_visible:
        transforms = {int(k): np.asarray(v) for k, v in evidence["transforms"].items()}
        features, actor, proximity = geometric_features(record["frame_ids"], record["fps"], height,
            recovered["feature_tracks"], evidence["balls"], transforms,
            recovered["feature_candidates"], evidence["camera"])
        result["features"] = features
        result["diagnostics"]["actor"] = actor
        result["baselines"] = {"existing_rule": rule_baseline(recovered["feature_candidates"],
            evidence["camera"], record["window"]["end_frame_exclusive"]), "ankle_proximity": proximity}
        augmented["tracks"] = recovered["tracks"]
        augmented["candidate_contacts"] = recovered["candidates"]
        diagnostics["applied"] = True
    result["diagnostics"]["roi_refinement"] = diagnostics
    result["settings"].update(REFINEMENT_SETTINGS)
    result["provenance"]["code_sha256"].update({name: file_sha256(Path(__file__).parents[1] / name)
        for name in ("src/contact_roi_refinement.py", "src/contact_tracking_recovery.py", "src/pose_refinement.py")})
    result["runtime"]["roi_refinement_seconds"] = time.perf_counter() - started
    augmented["roi_refinement"] = diagnostics
    return result, augmented
