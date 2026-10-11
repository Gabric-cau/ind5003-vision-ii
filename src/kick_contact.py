"""Independent, conservative ball-contact evidence for an already selected shot.

This is an auxiliary timing probe, not ground truth or a replacement for pose
processing. No annotated/previous ``kick_frame`` is read. A sparse object pass
looks for a stationary ball approached by any tracked player's ankle. Only a
short candidate window is refined. Missing ball evidence means abstention, not
that a run-up should be discarded. All frames and pixels use original-video
coordinates; stabilized coordinates are additional internal evidence.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _xy_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return (np.column_stack((points, np.ones(len(points)))) @ matrix.T)[:, :2]


def _pose_samples(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    sequence = result.get("selected_pose_sequence") or {}
    output = []
    for index, frame in enumerate(sequence.get("frames") or []):
        ankles = []
        for side in ("left", "right"):
            points = sequence.get(f"{side}_ankles") or []
            confidences = sequence.get(f"{side}_confidences") or []
            if index < len(points) and index < len(confidences) and confidences[index] >= .25:
                point = np.asarray(points[index], dtype=float)
                if point.shape == (2,) and np.isfinite(point).all():
                    ankles.append(point.tolist())
        heights = sequence.get("box_heights") or []
        if ankles and index < len(heights) and heights[index] and heights[index] > 20:
            output.append({"frame": int(frame), "ankles": ankles, "height": float(heights[index])})
    return output


def _all_track_samples(result: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Do not inherit a possibly wrong selected identity from the pose pipeline."""
    output = {}
    for track in result.get("tracks") or []:
        samples = []
        for observation in track.get("observations") or []:
            points = observation.get("keypoints") or []
            confidences = observation.get("keypoint_confidences") or []
            box = observation.get("bbox") or []
            if len(box) != 4 or box[3]-box[1] <= 20:
                continue
            ankles = [points[index] for index in (15,16) if index < len(points) and index < len(confidences)
                      and confidences[index] >= .25 and np.isfinite(points[index]).all()]
            if ankles:
                samples.append({"frame":int(observation["frame"]),"ankles":ankles,"height":float(box[3]-box[1])})
        if samples:
            output[int(track["track_id"])] = samples
    if not output:
        output[int(result.get("selected_track_id") or 0)] = _pose_samples(result)
    return output


def _score_all_tracks(detections, tracks, transforms, fps, stride, contact_window_policy="legacy"):
    candidates = []
    for track_id, samples in tracks.items():
        ranked = score_contact_candidates(detections,samples,transforms,fps,stride,
                                          contact_window_policy=contact_window_policy)
        for item in ranked:
            item["contact_associated_track_id"] = track_id
        candidates.extend(ranked)
    ranked=sorted(candidates,key=lambda item:(-item["score"],item["contact_frame"]))
    return prefer_causal_first_departure(ranked) if contact_window_policy=="causal_first_departure" else ranked


def prefer_causal_first_departure(ranked):
    """Within the strongest actor, an earlier plausible shot cannot be replaced
    by that actor walking past another stationary ball-like mark afterwards.
    This is an explicit single-kick-per-shot prior, not a higher confidence score.
    """
    if not ranked:return ranked
    actor=ranked[0].get("contact_associated_track_id")
    eligible=[c for c in ranked if c.get("contact_associated_track_id")==actor and c["score"]>=.60
              and c.get("ankle_approach_supported") and c.get("ball_departure_frames")]
    if not eligible:return ranked
    first=min(eligible,key=lambda c:c["contact_frame"])
    return [first]+[c for c in ranked if c is not first]


def _background_mask(result: Mapping[str, Any], frame_id: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, np.uint8)
    mask[:int(height * .12)] = 0  # Scoreboard / changing spectators.
    for track in result.get("tracks") or []:
        near = [o for o in track.get("observations", []) if abs(int(o["frame"]) - frame_id) <= 2]
        if not near:
            continue
        x1, y1, x2, y2 = min(near, key=lambda o: abs(int(o["frame"]) - frame_id))["bbox"]
        pad = max(8, .15 * (y2 - y1))
        mask[max(0, int(y1-pad)):min(height, int(y2+pad)+1),
             max(0, int(x1-pad)):min(width, int(x2+pad)+1)] = 0
    return mask


def _camera_transforms(frames: Sequence[np.ndarray], frame_ids: Sequence[int], result: Mapping[str, Any]) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Robust frame-to-first-frame affine compensation, with explicit failures."""
    transforms = [np.eye(3)]
    failures, displacements, inlier_ratios = [], [], []
    if not frames:
        return [], {"failed_pairs": [], "reliable": False}
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for index in range(1, len(frames)):
        current = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
        mask = _background_mask(result, frame_ids[index-1], previous.shape)
        points = cv2.goodFeaturesToTrack(previous, 350, .02, 12, mask=mask)
        matrix = None
        ratio = 0.0
        if points is not None and len(points) >= 12:
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(25,25), maxLevel=3)
            if nxt is not None and ok is not None:
                keep = ok.ravel().astype(bool)
                old_xy, new_xy = points.reshape(-1,2)[keep], nxt.reshape(-1,2)[keep]
                if len(old_xy) >= 12:
                    affine, inliers = cv2.estimateAffinePartial2D(new_xy, old_xy, method=cv2.RANSAC, ransacReprojThreshold=2.5)
                    if affine is not None and inliers is not None:
                        ratio = float(np.mean(inliers))
                        scale = float(np.hypot(affine[0,0], affine[0,1]))
                        if ratio >= .55 and .85 <= scale <= 1.18:
                            matrix = np.vstack((affine, [0,0,1]))
                            displacements.append(float(np.median(np.linalg.norm(new_xy-old_xy, axis=1))))
        if matrix is None:
            failures.append([int(frame_ids[index-1]), int(frame_ids[index])])
            matrix = np.eye(3)
        transforms.append(transforms[-1] @ matrix)
        inlier_ratios.append(ratio)
        previous = current
    return transforms, {
        "method": "background_LK_partial_affine_RANSAC",
        "failed_pairs": failures,
        "reliable": len(failures) == 0,
        "median_inlier_ratio": float(np.median(inlier_ratios)) if inlier_ratios else None,
        "max_background_displacement_px_per_sample": max(displacements, default=0.0),
        "coordinates": "frame-to-first-sample affine, not calibrated ground-plane coordinates",
    }


def _read_frames(video_path: str, frame_ids: Sequence[int]) -> tuple[list[int], list[np.ndarray]]:
    capture = cv2.VideoCapture(video_path)
    loaded_ids, frames = [], []
    previous = None
    for frame_id in frame_ids:
        if previous is None or frame_id != previous + 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = capture.read()
        if ok:
            loaded_ids.append(int(frame_id))
            frames.append(frame)
        previous = frame_id
    capture.release()
    return loaded_ids, frames


def _detect(frames: Sequence[np.ndarray], frame_ids: Sequence[int], model: Any, imgsz: int, device: str) -> list[dict[str, Any]]:
    detections = []
    for start in range(0, len(frames), 8):
        predictions = model.predict(list(frames[start:start+8]), classes=[32], conf=.06, imgsz=imgsz, device=device, verbose=False)
        for frame_id, frame, prediction in zip(frame_ids[start:start+8], frames[start:start+8], predictions):
            if prediction.boxes is None:
                continue
            boxes = prediction.boxes.xyxy.cpu().numpy()
            scores = prediction.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                x1,y1,x2,y2 = map(float, box)
                width, height = x2-x1,y2-y1
                if min(width,height) < 2 or max(width,height) > frame.shape[0]*.10:
                    continue
                detections.append({"frame": int(frame_id), "xy": [(x1+x2)/2,(y1+y2)/2],
                                   "bbox": [x1,y1,x2,y2], "confidence": float(score), "diameter": (width+height)/2})
    return detections


def score_contact_candidates(detections: Sequence[Mapping[str, Any]], pose_samples: Sequence[Mapping[str, Any]],
                             transforms: Mapping[int, np.ndarray], fps: float, sample_stride: int = 6,
                             *, contact_window_policy: str = "legacy", stationary_seed_detection_ids=None) -> list[dict[str, Any]]:
    """Rank stationary-ball/ankle encounters without consulting prior kick time.

    ``transforms`` maps absolute frame -> image-to-reference homogeneous affine.
    Candidate timing uncertainty is explicit. Scores are heuristic, not a
    calibrated probability. A departure needs two geometrically consistent
    detections; ball disappearance alone never earns strong support.
    """
    if contact_window_policy not in {"legacy", "stationary_transition", "causal_first_departure"}:
        raise ValueError("Unknown contact_window_policy")
    if not detections or not pose_samples or not transforms:
        return []
    transform_frames = sorted(transforms)
    def transform_at(frame: int) -> np.ndarray:
        nearest = min(transform_frames, key=lambda value: abs(value-frame))
        return np.asarray(transforms[nearest], dtype=float)
    balls = []
    for detection in detections:
        if not np.isfinite(detection["xy"]).all():
            continue
        item = dict(detection)
        item["stable_xy"] = _xy_transform(np.array([detection["xy"]]), transform_at(int(detection["frame"])))[0]
        balls.append(item)
    # Greedy spatial seeds, temporal persistence distinguishes stationary balls
    # from isolated false detections. No assumption about ball motion direction.
    clusters: list[list[dict[str, Any]]] = []
    seed_balls = [ball for ball in balls if stationary_seed_detection_ids is None
                  or ball.get('detection_id') in stationary_seed_detection_ids]
    for ball in sorted(seed_balls, key=lambda item: item["frame"]):
        options = [(float(np.linalg.norm(ball["stable_xy"]-np.median([v["stable_xy"] for v in cluster],axis=0))),index)
                   for index,cluster in enumerate(clusters)]
        tolerance = max(5.0, float(ball.get("diameter", 5))*.75)
        if options and min(options)[0] <= tolerance:
            clusters[min(options)[1]].append(ball)
        else:
            clusters.append([ball])
    candidates = []
    for cluster in clusters:
        unique_frames = sorted({int(v["frame"]) for v in cluster})
        if len(unique_frames) < 3 or unique_frames[-1]-unique_frames[0] < max(2*sample_stride,.25*fps):
            continue
        center = np.median([v["stable_xy"] for v in cluster],axis=0)
        diameter = float(np.median([v.get("diameter",5) for v in cluster]))
        # Search only near the end of stationary evidence; far-away future
        # player movement cannot retroactively create a contact with this ball.
        # Experimental transition gate: a foot returning to the former spot
        # well AFTER the stationary ball disappeared must not win a global
        # closest-distance search. Keep the legacy policy for paired ablation.
        # This is a timing constraint, not evidence that disappearance is a kick.
        tolerance = (max(sample_stride + 1, int(round(.12 * fps)))
                     if contact_window_policy != "legacy" else int(.6 * fps))
        nearby = [p for p in pose_samples if unique_frames[0] <= int(p["frame"]) <= unique_frames[-1]+tolerance]
        distances = []
        for pose in nearby:
            xy = _xy_transform(np.asarray(pose["ankles"]), transform_at(int(pose["frame"])))
            distance = float(np.min(np.linalg.norm(xy-center,axis=1)))
            distances.append((distance/float(pose["height"]), int(pose["frame"]), distance, float(pose["height"])))
        if not distances:
            continue
        normalized_distance, contact_frame, pixel_distance, body_height = min(distances)
        if normalized_distance > .28:
            continue
        earlier = [v[0] for v in distances if contact_frame-int(.65*fps) <= v[1] <= contact_frame-int(.25*fps)]
        approached = bool(earlier and np.median(earlier) > normalized_distance+.10)
        past_contact = [b for b in balls if contact_frame <= b["frame"] <= contact_frame+.6*fps]
        moving = []
        for ball in past_contact:
            distance = float(np.linalg.norm(ball["stable_xy"]-center))
            if max(2.5*diameter,.10*body_height) < distance < 3*body_height:
                moving.append(ball)
        departure = None
        departure_observations = []
        for first in moving:
            for second in moving:
                gap = int(second["frame"])-int(first["frame"])
                if not 1 <= gap <= max(sample_stride*2, int(.2*fps)):
                    continue
                a,b = first["stable_xy"]-center,second["stable_xy"]-center
                if np.linalg.norm(b) <= np.linalg.norm(a)+max(3,diameter*.5):
                    continue
                cosine = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                if cosine >= .90:
                    departure = [int(first["frame"]),int(second["frame"])]
                    departure_observations = [first, second]
                    break
            if departure:
                break
        last_stationary = max((f for f in unique_frames if f <= contact_frame+sample_stride),default=unique_frames[-1])
        # Preserve the actual source detections used by this seed. Stabilized
        # centers support timing, but cannot identify the striking foot.
        def source_observation(ball):
            observation = {"frame": int(ball["frame"]), "xy": [float(v) for v in ball["xy"]],
                           "confidence": float(ball.get("confidence", 0)),
                           "diameter": float(ball.get("diameter", 5))}
            if ball.get("detection_id") is not None:
                observation["detection_id"] = ball["detection_id"]
            if ball.get("bbox") is not None:
                observation["bbox"] = [float(v) for v in ball["bbox"]]
            return observation
        stationary_observations, stationary_competitors = [], []
        for frame in unique_frames:
            same_frame = sorted([ball for ball in cluster if int(ball["frame"]) == frame],
                                key=lambda ball: (-float(ball.get("confidence", 0)),
                                                  str(ball.get("detection_id", "")), tuple(ball["xy"])))
            stationary_observations.append(source_observation(same_frame[0]))
            if len({(str(ball.get("detection_id", "")), tuple(ball["xy"])) for ball in same_frame}) > 1:
                stationary_competitors.append({"frame": frame,
                    "observations": [source_observation(ball) for ball in same_frame]})
        bracket = [last_stationary, departure[0]] if departure and last_stationary <= departure[0] else None
        stationary_after = sum(f > contact_frame+max(2*sample_stride,int(.3*fps)) for f in unique_frames)
        score = .30 + .20*max(0,1-normalized_distance/.28) + .10*min(1,len(unique_frames)/8) + .15*approached + .20*bool(departure)
        warnings = []
        if not departure:
            warnings.append("ball_departure_not_confirmed")
        if not approached:
            warnings.append("ankle_approach_not_clear")
            # A goalkeeper already standing beside a ball-like bottle/mark
            # must not beat the true kicker solely through false departures.
            score -= .25
        if stationary_after >= 2:
            score -= .25
            warnings.append("stationary_ball_persists_after_candidate")
        half_window = max(3,sample_stride,int(round(.10*fps)))
        # The closest ankle can be the planted support foot BEFORE contact.
        # Retain a wider forward neighborhood rather than claiming frame-exact
        # contact from the anatomical ankle point alone.
        interval_end = max(contact_frame+max(half_window,int(round(.25*fps))),departure[0] if departure else 0)
        candidates.append({"contact_frame": contact_frame, "contact_time_sec": contact_frame/fps,
                           "interval_frames": [max(unique_frames[0],contact_frame-half_window),interval_end],
                           "interval_kind": "heuristic_candidate_window_not_statistical_confidence_interval",
                           "score": round(float(np.clip(score,0,1)),4),
                           "ankle_ball_distance_px": pixel_distance,"ankle_ball_distance_body_heights": normalized_distance,
                           "stationary_ball_samples": len(unique_frames),"stationary_ball_first_frame": unique_frames[0],
                           "stationary_ball_last_frame": unique_frames[-1], "last_stationary_near_contact_frame": last_stationary,
                           "stationary_ball_reference_xy": center.tolist(),"ball_departure_frames": departure,
                           "stationary_ball_observations": stationary_observations,
                           "departure_ball_observations": [source_observation(ball) for ball in departure_observations],
                           "contact_bracket_frames": bracket,
                           "contact_bracket_status": "observed_stationary_to_departure_candidate" if bracket else "insufficient_or_conflicting_observations",
                           "stationary_observation_selection_policy": "highest_detector_confidence_per_seed_cluster_frame; ties_detection_id_then_source_xy",
                           "stationary_observation_competitors": stationary_competitors,
                           "stationary_identity_ambiguous_frames": [row["frame"] for row in stationary_competitors],
                           "contact_window_policy": contact_window_policy,
                           "ankle_approach_supported": approached,"warnings": warnings})
    return sorted(candidates,key=lambda item: (-item["score"],item["contact_frame"]))


def estimate_kick_contact(video_path: str, pose_result: Mapping[str, Any], object_model: Any,
                          sample_stride: int = 6, imgsz: int = 1280, device: str = "cpu",
                          *, contact_window_policy: str = "legacy") -> dict[str, Any]:
    """Scan the selected shot independently, refine one candidate, or abstain."""
    start, end = int(pose_result.get("shot_start_frame",0)),int(pose_result["shot_end_frame"])
    fps = float(pose_result.get("fps") or 30)
    track_samples = _all_track_samples(pose_result)
    ids, frames = _read_frames(video_path,list(range(start,end,max(1,sample_stride))))
    matrices, camera = _camera_transforms(frames,ids,pose_result)
    detections = _detect(frames,ids,object_model,imgsz,device)
    transforms = dict(zip(ids,matrices))
    candidates = _score_all_tracks(detections,track_samples,transforms,fps,sample_stride,contact_window_policy)
    sparse_count = len(frames)
    refine_count = 0
    if candidates and candidates[0]["score"] >= .40:
        candidate = candidates[0]["contact_frame"]
        refine_ids = list(range(max(start,candidate-int(.45*fps)),min(end,candidate+int(.45*fps)+1)))
        # Re-register the whole sparse-plus-refined timeline so camera transforms
        # have one reference frame, rather than mixing local coordinate systems.
        combined_ids = sorted(set(ids+refine_ids))
        ids, frames = _read_frames(video_path,combined_ids)
        matrices,camera = _camera_transforms(frames,ids,pose_result)
        new_ids = [f for f in refine_ids if f not in transforms]
        frame_map = dict(zip(ids,frames))
        detections += _detect([frame_map[f] for f in new_ids],new_ids,object_model,imgsz,device)
        transforms = dict(zip(ids,matrices))
        refine_count = len(new_ids)
        candidates = _score_all_tracks(detections,track_samples,transforms,fps,stride=2,
                                        contact_window_policy=contact_window_policy)
    selected = candidates[0] if candidates else None
    warnings = ["experimental_not_human_verified", "confidence_is_heuristic_not_calibrated"]
    if not camera["reliable"]:
        warnings.append("camera_compensation_incomplete")
    # Failures after the contact evidence cannot invalidate a preceding stable
    # segment. Failures before it do matter because transforms are cumulative.
    support_end = max(selected["interval_frames"]+[selected["ball_departure_frames"][-1] if selected and selected["ball_departure_frames"] else 0]) if selected else end
    local_camera_reliable = not any(pair[0] <= support_end for pair in camera["failed_pairs"])
    camera["reliable_through_contact_evidence"] = local_camera_reliable
    camera["contact_evidence_end_frame"] = support_end
    ambiguous_identity = bool(selected and any(c["contact_associated_track_id"] != selected["contact_associated_track_id"]
                              and c["score"] >= selected["score"]-.07 for c in candidates[1:]))
    if ambiguous_identity:
        warnings.append("competing_contact_identities")
    if selected and selected["score"] >= .75 and selected["ball_departure_frames"] and selected["ankle_approach_supported"] and local_camera_reliable and not ambiguous_identity:
        status = "supported"
    elif selected and selected["score"] >= .50:
        status = "tentative"
    else:
        status = "abstain"
        selected = None
        warnings.append("insufficient_independent_contact_evidence")
    return {"schema_version":1,"clip_id":pose_result.get("clip_id"),"method":"independent_stationary_ball_ankle_encounter_and_departure",
            "contact_window_policy":contact_window_policy,
            "status":status,"contact_frame":selected["contact_frame"] if selected else None,
            "contact_time_sec":selected["contact_time_sec"] if selected else None,
            "interval_frames":selected["interval_frames"] if selected else None,
            "confidence":selected["score"] if selected else None,
            "contact_associated_track_id":selected["contact_associated_track_id"] if selected else None,
            "shot_start_frame":start,"shot_end_frame_exclusive":end,"fps":fps,
            "sparse_frames":sparse_count,"refined_frames":refine_count,"object_images_processed":sparse_count+refine_count,
            "candidates":candidates,"ball_detections":sorted(detections,key=lambda item:item["frame"]),
            "camera_diagnostics":camera,"warnings":warnings,
            "human_verified":False,"uses_prior_kick_frame":False,"missing_ball_policy":"abstain_never_discard_runup"}
