"""Local source-pixel foot evidence. Never reads reviewer labels."""
import numpy as np

COCO_MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def undo_mirror(keypoints, confidence, width):
    """Undo pixel reflection AND anatomical channel reflection, once."""
    points = np.asarray(keypoints, float)[COCO_MIRROR].copy()
    points[:, 0] = width - 1 - points[:, 0]
    return points, np.asarray(confidence, float)[COCO_MIRROR].copy()


def transform_points(points, matrix):
    points = np.asarray(points, float)
    p = np.c_[points, np.ones(len(points))] @ np.asarray(matrix, float).T
    return p[:, :2] / p[:, 2:3]


def approach_candidates(samples, ball, goal, contact_start):
    """Explicitly uncalibrated two-dimensional approach heuristics."""
    rows = [s for s in samples if s["frame"] < contact_start-2]
    feet = [np.mean(np.asarray(s["keypoints"])[[15, 16]], axis=0)
            for s in rows if min(s["confidence"][15:17]) >= .4]
    if len(feet) < 3:
        return {"screen_x_candidate": "unknown", "goal_axis_candidate": "unknown", "reason": "too_few_precontact_ankles"}
    position = np.median(feet, axis=0)
    offset = position-np.asarray(ball)
    out = {"screen_x_candidate": "right" if offset[0] < -5 else "left" if offset[0] > 5 else "unknown",
           "screen_x_offset_px": float(offset[0]), "goal_axis_candidate": "unknown",
           "ground_calibrated": False, "contact_observed_by_this_rule": False,
           "scope": "Hypotheses from approach, not an anatomical guarantee"}
    if goal is None:
        return {**out,"reason":"missing_automatic_goal_geometry"}
    axis = np.asarray(goal,dtype=float)-np.asarray(ball,dtype=float)
    length = np.linalg.norm(axis)
    if length < 1:
        return {**out,"reason":"invalid_goal_axis"}
    axis /= length
    signed = float(axis[0]*offset[1]-axis[1]*offset[0])
    return {**out,"goal_axis_candidate": "right" if signed < -5 else "left" if signed > 5 else "unknown",
            "signed_projected_lateral_offset_px": signed}


def foot_evidence(samples, ball, contact_start, departure_frame):
    """Compare motion and ball proximity inside a measured departure bracket."""
    rows = sorted([s for s in samples if contact_start-4 <= s["frame"] <= departure_frame+3], key=lambda s: s["frame"])
    result = {"motion_candidate": "unknown", "contact_candidate": "unknown", "requires_review": True,
              "window": [contact_start-4, departure_frame+3], "observed_samples": len(rows)}
    if len(rows) < 4:
        return result
    stats = {}
    for label, index in (("left", 15), ("right", 16)):
        valid = [s for s in rows if min(s["confidence"][11],s["confidence"][12],s["confidence"][index]) >= .4]
        relative = {s["frame"]: np.asarray(s["keypoints"])[index]-np.mean(np.asarray(s["keypoints"])[[11,12]],axis=0) for s in valid}
        increments = [float(np.linalg.norm(relative[f+1]-p)) for f,p in relative.items() if f+1 in relative]
        near = [float(np.linalg.norm(np.asarray(s["keypoints"])[index]-ball)) for s in valid if contact_start-2 <= s["frame"] <= departure_frame]
        stats[label] = {"max_motion_px": max(increments,default=0), "median_motion_px": float(np.median(increments)) if increments else 0,
                        "min_ball_distance_px": min(near,default=None), "motion_pairs": len(increments), "contact_samples":len(near)}
    result["feet"] = stats
    left, right = stats["left"],stats["right"]
    lm,rm = left["max_motion_px"],right["max_motion_px"]
    result["motion_ratio"] = max(lm,rm)/max(min(lm,rm),1e-6)
    if min(left["motion_pairs"],right["motion_pairs"]) >= 3 and result["motion_ratio"] >= 1.25:
        result["motion_candidate"] = "left" if lm>rm else "right"
    ld,rd = left["min_ball_distance_px"],right["min_ball_distance_px"]
    if ld is not None and rd is not None:
        result["ball_distance_margin_px"] = abs(ld-rd)
        # A nearby ankle is still only a proxy for the shoe-ball contact.
        if min(ld,rd) <= 30 and abs(ld-rd) >= 6:
            result["contact_candidate"] = "left" if ld<rd else "right"
    return result


def fuse_evidence(normal, mirrored):
    """Abstain on disagreement. No majority or approach prior can override it."""
    labels = [d[k] for d in (normal,mirrored) for k in ("motion_candidate","contact_candidate") if d[k] != "unknown"]
    if len(labels) >= 3 and len(set(labels)) == 1:
        return {"candidate":labels[0],"status":"consistent_pose_and_contact_proxy_candidate","requires_review":True}
    return {"candidate":"unknown","status":"conflicting_or_insufficient_anatomical_evidence","requires_review":True,
            "component_labels":labels}


def followthrough_evidence(samples, forward_axis, departure_frame):
    """Foot extension along the shot axis after departure, not maximum run-up motion.

    Image-plane projection only. This is not proof of shoe-ball contact.
    """
    axis=np.asarray(forward_axis,dtype=float)
    norm=np.linalg.norm(axis)
    result={"candidate":"unknown","reason":"insufficient_visible_postdeparture_extension",
            "requires_review":True,"three_dimensional_anatomy_verified":False}
    if norm<1:return result
    axis/=norm
    observations=[]
    for s in samples:
        if not departure_frame <= s['frame'] <= departure_frame+6:continue
        p=np.asarray(s['keypoints']);c=np.asarray(s['confidence'])
        if min(c[[11,12,15,16]])<.4:continue
        hip=np.mean(p[[11,12]],axis=0)
        leg_scale=max(np.linalg.norm(p[15]-hip),np.linalg.norm(p[16]-hip),1)
        l=float(np.dot(p[15]-hip,axis));r=float(np.dot(p[16]-hip,axis))
        observations.append(dict(frame=s['frame'],left_projection_px=l,right_projection_px=r,
                                 normalized_right_minus_left=(r-l)/leg_scale))
    result['observations']=observations
    if len(observations)<3:return result
    # A sustained extension, not one pose outlier or a presumed dominant foot.
    median=float(np.median([r['normalized_right_minus_left'] for r in observations]))
    result['median_normalized_right_minus_left']=median
    agreement=sum(np.sign(r['normalized_right_minus_left'])==np.sign(median) for r in observations)/len(observations)
    result['sign_agreement']=agreement
    if abs(median)>=.15 and agreement>=.7:
        result.update(candidate='right' if median>0 else 'left',reason='sustained_projected_forward_extension')
    return result


def camera_scoped_approach(approach, camera_layout):
    """Experimental shortcut, gated to the reviewed goal-right camera layout.

    Predictive prior only; no reviewer labels and no automatic anatomy override.
    """
    if camera_layout != 'goal_right':
        return dict(candidate='unknown',method='out_of_scope_camera',requires_review=True)
    candidate=approach.get('goal_axis_candidate','unknown')
    if candidate!='unknown':
        return dict(candidate=candidate,method='projected_goal_axis_approach_prior',requires_review=True,
                    ground_calibrated=False,shoe_ball_contact_verified=False)
    candidate=approach.get('screen_x_candidate','unknown')
    return dict(candidate=candidate,method='screen_x_fallback_weaker_prior',requires_review=True,
                ground_calibrated=False,shoe_ball_contact_verified=False)
