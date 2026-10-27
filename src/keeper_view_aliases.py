"""Conservative cross-view aliases for the optional keeper recovery guard."""
from copy import deepcopy
import hashlib

import numpy as np

from src.person_appearance import similarity, torso_histogram


POLICY = {
    "minimum_shared_samples": 3, "minimum_shared_span_seconds": .04,
    "switch_neighborhood_seconds": .30, "minimum_bbox_iou": .60,
    "minimum_torso_keypoint_confidence": .50, "maximum_torso_error_body_heights": .15,
    "minimum_torso_color_similarity": .90, "separated_bbox_iou": .10,
    "separated_center_distance_body_heights": .75,
}


def pair_geometry(first, second):
    try:
        a, b = (np.asarray(item["bbox"], dtype=float) for item in (first, second))
        ka, kb = (np.asarray(item["keypoints"], dtype=float) for item in (first, second))
        ca, cb = (np.asarray(item["keypoint_confidences"], dtype=float) for item in (first, second))
    except (KeyError, TypeError, ValueError):
        return None
    js = [5, 6, 11, 12]
    if (a.shape != (4,) or b.shape != (4,) or ka.shape != (17, 2) or kb.shape != (17, 2) or
            ca.shape != (17,) or cb.shape != (17,) or
            any(not np.isfinite(x).all() for x in (a, b, ka[js], kb[js], ca[js], cb[js])) or
            min(*(a[2:]-a[:2]), *(b[2:]-b[:2])) <= 0):
        return None
    intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    iou = intersection/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    scale = max(a[3]-a[1], b[3]-b[1], 1.)
    return {"bbox_iou": float(iou), "torso_confidence": float(min(ca[js].min(), cb[js].min())),
            "torso_error_body_heights": float(np.linalg.norm(ka[js]-kb[js], axis=1).mean()/scale),
            "center_distance_body_heights": float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/scale)}


def source_view(observation):
    return observation.get("source_roi") or observation.get("inference_view") or "original_full_frame"


def pair_alias_evidence(first, second, switch_frames, fps, read_frame):
    a = {o["frame"]: o for o in first.get("observations", [])}
    b = {o["frame"]: o for o in second.get("observations", [])}
    shared = sorted(set(a) & set(b))
    report = {"source_ids": [first["track_id"], second["track_id"]], "accepted": False,
              "switch_frames": switch_frames, "shared_frame_count": len(shared),
              "separated_counterexample_frames": [], "support": [], "reason": None}
    geometric = []
    for frame in shared:
        stat = pair_geometry(a[frame], b[frame])
        if stat is None or stat["torso_confidence"] < POLICY["minimum_torso_keypoint_confidence"]:
            continue
        if (stat["bbox_iou"] < POLICY["separated_bbox_iou"] and
                stat["center_distance_body_heights"] > POLICY["separated_center_distance_body_heights"]):
            report["separated_counterexample_frames"].append(frame)
        if (source_view(a[frame]) != source_view(b[frame]) and
                stat["bbox_iou"] >= POLICY["minimum_bbox_iou"] and
                stat["torso_error_body_heights"] <= POLICY["maximum_torso_error_body_heights"] and
                any(abs(frame-switch)/fps <= POLICY["switch_neighborhood_seconds"] for switch in switch_frames)):
            geometric.append((frame, stat))
    if report["separated_counterexample_frames"]:
        report["reason"] = "simultaneously_separated_people_counterexample"
        return report
    for frame, stat in geometric:
        image = read_frame(frame)
        score = similarity(torso_histogram(image, a[frame]), torso_histogram(image, b[frame]))
        if score is not None and score >= POLICY["minimum_torso_color_similarity"]:
            report["support"].append({"frame": frame, **stat, "torso_color_similarity": score,
                "views": [source_view(a[frame]), source_view(b[frame])],
                "source_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    for switch in switch_frames:
        frames = [row["frame"] for row in report["support"]
                  if abs(row["frame"]-switch)/fps <= POLICY["switch_neighborhood_seconds"]]
        if (len(frames) >= POLICY["minimum_shared_samples"] and
                (frames[-1]-frames[0])/fps >= POLICY["minimum_shared_span_seconds"]-1e-9):
            report.update(accepted=True, reason="multiple_same_frame_cross_view_body_and_color_agreements")
            return report
    report["reason"] = "insufficient_local_cross_view_evidence"
    return report


def alias_guard_inputs(record, pose, video_path, *, frame_reader=None):
    """Canonicalize only positively supported aliases of the goal-line seed.

    This changes a derived guard view, never the saved neural tracks. It does
    not join non-overlapping fragments or allow transitive alias chains.
    """
    audit = {"policy": "source_bound_cross_view_keeper_alias_v1", "config": dict(POLICY),
             "identity_verified": False, "reference_labels_used": False,
             "applied": False, "pairs": [], "aliases": {}, "reason": None}
    selected = record.get("selected_candidate") or {}
    lineage = selected.get("source_track_id_by_frame")
    seed = (record.get("goal_identity_evidence") or {}).get("seed_track_id")
    tracks = {t["track_id"]: t for t in pose.get("tracks", [])}
    if not lineage or seed not in tracks or selected.get("qualifies") is not True:
        audit["reason"] = "missing_selected_frame_lineage_or_goal_seed"
        return record, pose, audit
    if (record.get("goal_identity_evidence") or {}).get("ambiguous_identity") is True:
        audit["reason"] = "ambiguous_goal_line_identity"
        return record, pose, audit
    sequence = sorted((int(frame), tid) for frame, tid in lineage.items())
    switches = {}
    for (_, previous), (frame, current) in zip(sequence, sequence[1:]):
        if current != previous and seed in (previous, current):
            other = current if previous == seed else previous
            switches.setdefault(other, []).append(frame)
    capture, images = None, {}

    def read_frame(frame):
        nonlocal capture
        if frame not in images:
            if frame_reader is not None:
                images[frame] = frame_reader(frame)
            else:
                import cv2
                if capture is None:
                    capture = cv2.VideoCapture(str(video_path))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame+1:
                    raise ValueError("Keeper alias source frame mapping failed")
                images[frame] = image
        return images[frame]

    aliases = {}
    try:
        for other, frames in sorted(switches.items()):
            if other == pose.get("selected_track_id"):
                audit["pairs"].append({"source_ids": [seed, other], "accepted": False,
                                        "reason": "selected_kicker_cannot_alias_keeper"})
                continue
            if other not in tracks:
                continue
            evidence = pair_alias_evidence(tracks[seed], tracks[other], frames, float(pose["fps"]), read_frame)
            audit["pairs"].append(evidence)
            if evidence["accepted"]:
                aliases[other] = seed
    finally:
        if capture is not None:
            capture.release()
    if not aliases:
        audit["reason"] = "no_supported_alias_of_goal_line_seed"
        return record, pose, audit
    onset = next((tid for frame, tid in sequence if frame >= record["kick_frame"]), None)
    if aliases.get(onset, onset) != seed:
        audit["reason"] = "selected_onset_not_bound_to_goal_line_seed"
        return record, pose, audit
    derived, view = deepcopy(record), deepcopy(pose)
    chosen = derived["selected_candidate"]
    chosen["native_source_track_ids"] = deepcopy(selected["source_track_ids"])
    chosen["native_source_track_id_by_frame"] = dict(lineage)
    chosen["source_track_ids"] = sorted({aliases.get(tid, tid) for tid in selected["source_track_ids"]})
    chosen["source_track_id_by_frame"] = {frame: aliases.get(tid, tid) for frame, tid in lineage.items()}
    grouped = {}
    for track in pose.get("tracks", []):
        tid = track["track_id"]
        for observation in track.get("observations", []):
            if lineage.get(str(observation["frame"])) == tid:
                row = {**deepcopy(observation), "native_source_track_id": tid}
                grouped.setdefault(aliases.get(tid, tid), []).append(row)
    view["tracks"] = [{"track_id": tid, "observations": rows} for tid, rows in grouped.items()]
    audit.update(applied=True, aliases={str(tid): value for tid, value in aliases.items()},
                 reason="guard_view_uses_exact_selected_observations_with_evidenced_cross_view_aliases")
    return derived, view, audit
