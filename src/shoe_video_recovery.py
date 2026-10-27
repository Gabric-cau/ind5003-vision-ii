"""Source-bound shoe fallback from measured automatic contact-stage actors."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.recover_shoe_evidence import build_bound
from scripts.run_fresh_pitch_experiment import _pose_input
from src.kick_contact import _camera_transforms
from src.runtime import file_sha256


def read(path):
    return json.loads(path.read_text())


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def contact_pose_context(source, original, features, evidence):
    """Use measured contact poses as source context without claiming a qualified goal scene."""
    if (original.get("source_mapping") != source
            or original.get("inference_identity", {}).get("source_sha256") != source["clip_sha256"]
            or original.get("reference_labels_used_for_prediction") is not False
            or original.get("new_neural_inference") is not True
            or original.get("source_frame_count") != source["frame_count"]
            or not np.isclose(original["fps"], source["fps"])):
        raise ValueError("Rejected pose context does not retain source provenance")
    if features.get("video_sha256") != source["clip_sha256"] or evidence.get("video_sha256") != source["clip_sha256"]:
        raise ValueError("Measured contact pose source mismatch")
    window = features.get("window", {})
    actor_id = features.get("diagnostics", {}).get("actor", {}).get("actor_id")
    actor = next((t for t in evidence.get("tracks", []) if t.get("track_id") == actor_id), None)
    if actor_id is None or actor is None:
        return None, "contact_actor_missing_from_measured_tracks"
    context = {key: copy.deepcopy(original[key]) for key in (
        "source_mapping", "inference_identity", "reference_labels_used_for_prediction", "new_neural_inference",
        "source_frame_count", "fps", "frame_width", "frame_height")}
    context.update(shot_start_frame=window.get("start_frame"), shot_end_frame=window.get("end_frame_exclusive"),
        selected_track_id=None, selected_track_qualified=False, tracks=copy.deepcopy(evidence["tracks"]),
        pose_evidence_source="trained_contact_measured_tracks", goal_scene_qualification_established=False)
    return context, None


def measured_bound(source, video, pose, contact, evidence, features):
    """Reuse measured boxes on the intersection of two automatic source windows."""
    digest, fps = source["clip_sha256"], source["fps"]
    if any(item.get("video_sha256") != digest for item in (contact, evidence, features)):
        raise ValueError("Recovery contact evidence belongs to another source video")
    if (pose.get("source_mapping") != source
            or pose.get("inference_identity", {}).get("source_sha256") != digest
            or pose.get("reference_labels_used_for_prediction") is not False
            or pose.get("new_neural_inference") is not True
            or pose.get("source_frame_count") != source["frame_count"]
            or not np.isclose(pose["fps"], fps) or not np.isclose(features["fps"], fps)):
        raise ValueError("Recovery pose must retain its measured source and frame clock")
    provenance = features.get("provenance", {})
    if (contact.get("manual_annotations_required") is not False
            or any(provenance.get(key) is not False for key in (
                "human_labels_used", "old_contact_used", "manual_camera_boundaries_used"))):
        raise ValueError("Recovery contact must be automatic and independent of manual labels")
    window = contact.get("automatic_window", {})
    if (window != features.get("window") or window.get("automatic") is not True
            or contact.get("diagnostics") != features.get("diagnostics")):
        raise ValueError("Contact prediction and measured feature evidence disagree")
    boundaries = [pose.get("shot_start_frame"), pose.get("shot_end_frame"),
                  window.get("start_frame"), window.get("end_frame_exclusive")]
    if (not all(integer(value) for value in boundaries)
            or not 0 <= boundaries[0] < boundaries[1] <= source["frame_count"]
            or not 0 <= boundaries[2] < boundaries[3] <= source["frame_count"]):
        raise ValueError("Recovery automatic windows must lie inside the source video")
    start, end = max(boundaries[0], boundaries[2]), min(boundaries[1], boundaries[3])
    point = contact.get("prediction", {}).get("frame")
    if point is None:
        return None, "automatic_contact_unavailable"
    if not integer(point) or point not in features.get("frame_ids", []):
        raise ValueError("Recovery contact frame is not on the measured feature clock")
    if not start < point - 1 < point + 1 < end:
        return None, "contact_outside_shared_automatic_shot"
    seconds = contact["prediction"].get("time_seconds")
    if not isinstance(seconds, (int, float)) or not np.isclose(seconds, point / fps):
        raise ValueError("Recovery contact time disagrees with its source frame")
    original, measured = copy.deepcopy(pose), copy.deepcopy(evidence)
    for payload in (original, measured):
        for track in payload["tracks"]:
            selected = []
            seen = set()
            for row in track["observations"]:
                frame = row.get("frame")
                if not integer(frame) or not 0 <= frame < source["frame_count"] or frame in seen:
                    raise ValueError("Recovery actor observations need unique source frame indices")
                seen.add(frame)
                if row.get("interpolated") is True or row.get("is_interpolated") is True:
                    raise ValueError("Interpolated poses cannot anchor recovered neural observations")
                if start <= frame < end:
                    selected.append(row)
            track["observations"] = selected
    mapping = {"shot_sha256": digest, "source_start_frame": 0,
               "source_end_frame_exclusive": end, "fps": fps,
               "width": pose["frame_width"], "height": pose["frame_height"]}
    # The shared helper's legacy key holds a clock mapping, not a new video export.
    bundle = {"source_mapping": source, "video_path": str(video),
              "old_selected_actor": pose.get("selected_track_id"), "lossless_shot": mapping}
    contact_only = pose.get("pose_evidence_source") == "trained_contact_measured_tracks"
    bound, reason = build_bound(bundle, contact, measured, None if contact_only else original)
    if bound is None:
        return None, reason
    bound["scene_interval_frames"] = [start, end]
    bound["interval_frames"][0] = start
    bound["inference_interval_frames"][0] = start
    bound["plan"] = [row for row in bound["plan"] if start <= row["frame"] < end]
    bound["clock_mapping"] = {"mapping_kind": "original_video_no_reencoding",
        "exported_intermediate_video": False, "source_frame_offset": 0,
        "source_sha256": digest, "evidence_video_sha256": digest,
        "automatic_pose_window": boundaries[:2], "automatic_contact_window": boundaries[2:],
        "shared_source_interval_frames": [start, end],
        "pose_evidence_source": "trained_contact_measured_tracks" if contact_only else "qualified_original_pose"}
    return bound, None


def measure_dense_camera(bound, pose):
    """Measure every camera link; failed links invalidate the following prefix."""
    start, end = bound["scene_interval_frames"]
    images, frames = [], []
    cap = cv2.VideoCapture(str(bound["video"]))
    try:
        if (not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - bound["fps"]) > .01
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != bound["source_mapping"]["frame_count"]):
            raise ValueError("Recovery decoder source clock differs from its manifest")
        for frame in range(end):
            ok, image = cap.read()
            if not ok or image.shape[:2] != (bound["height"], bound["width"]):
                raise ValueError("Cannot decode measured recovery shot")
            if frame >= start:
                images.append(image)
                frames.append(frame)
    finally:
        cap.release()
    matrices, diagnostics = _camera_transforms(images, frames, pose)
    if len(matrices) != len(frames):
        raise ValueError("Measured camera chain length differs from decoded source")
    failed = [pair[1] for pair in diagnostics["failed_pairs"]]
    first_failed = min(failed) if failed else float("inf")
    return {"transforms": {str(frame): matrix.tolist() if frame < first_failed else None
                           for frame, matrix in zip(frames, matrices)},
            "camera_diagnostics": diagnostics, "interpolated_transforms_used": False,
            "source_interval_frames": [start, end]}


def bind_recovered_event(analysis_dir, source, manifest_hash, threads=2):
    """Return an infer_event-compatible bound, with no human input at inference."""
    directory = Path(analysis_dir)
    cid = source["clip_id"]
    if not isinstance(cid, str) or Path(cid).name != cid or cid in (".", ".."):
        raise ValueError("Recovery clip ID must be a plain source identifier")
    paths = {"manifest": directory / "source/source_manifest.json",
        "video": directory / "source" / source["clip_filename"],
        "pose": directory / "pose" / (cid + ".json"),
        "contact_prediction": directory / "trained_contact/prediction.json",
        "contact_evidence": directory / "trained_contact/evidence.json",
        "contact_features": directory / "trained_contact/features.json"}
    receipt = {"method": "measured_trained_contact_actor_fallback", "used": False,
               "human_labels_used": False, "manual_roi_used": False,
               "interpolated_pose_used": False, "reencoded_video_used": False}
    if any(not path.is_file() for path in paths.values()):
        return None, {**receipt, "unavailable_reason": "recovery_source_artifact_unavailable"}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if hashes["video"] != source["clip_sha256"] or hashes["manifest"] != manifest_hash:
        raise ValueError("Recovery source video or manifest hash mismatch")
    manifest = read(paths["manifest"])
    if manifest.get("clips") != [source] or manifest.get("manual_annotations_used") is not False:
        raise ValueError("Recovery requires the same unannotated single-video manifest")
    pose, pose_receipt = _pose_input(paths["pose"], source)
    receipt.update(input_sha256=hashes, original_pose=pose_receipt)
    contact, evidence, features = (read(paths[name]) for name in (
        "contact_prediction", "contact_evidence", "contact_features"))
    if pose is None:
        if pose_receipt["status"] != "selected_person_not_qualified":
            return None, {**receipt, "unavailable_reason": pose_receipt["status"]}
        pose, reason = contact_pose_context(source, read(paths["pose"]), features, evidence)
        if pose is None:
            return None, {**receipt, "unavailable_reason": reason}
        receipt["selected_pose_evidence"] = "trained_contact_measured_tracks"
        receipt["goal_geometry_required_for_shoe_sampling"] = False
        receipt["original_goal_scene_qualified"] = False
    else:
        receipt["selected_pose_evidence"] = "qualified_original_pose"
    bound, reason = measured_bound(source, paths["video"], pose, contact, evidence, features)
    if bound is None:
        return None, {**receipt, "unavailable_reason": reason}
    cv2.setNumThreads(threads)
    bound["original_camera_motion"] = measure_dense_camera(bound, pose)
    if any(file_sha256(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Recovery source evidence changed during binding")
    receipt.update(used=True, clock_mapping=bound["clock_mapping"],
                   actor_association=bound["dense_actor_association"],
                   camera_diagnostics=bound["original_camera_motion"]["camera_diagnostics"])
    return bound, receipt
