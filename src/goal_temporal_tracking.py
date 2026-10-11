"""Experimental short-range tracking of an AUTOMATIC goal-frame proposal.

This carries image-plane geometry, never a ground-plane calibration or identity.
Shi-Tomasi + pyramidal LK API concepts follow the OpenCV optical-flow tutorial.
All masks, consistency gates and refusal rules below are project-written.
"""
import cv2
import numpy as np


def track_goal_pair(previous, current, corners, excluded_boxes=()):
    """Track one short within-shot step; return no geometry if checks fail.

    The caller must enforce shot boundaries and an independent-detection age cap.
    No missing point interpolation, fallback transform or identity claim is made.
    """
    result = {'status': 'abstain', 'human_verified': False, 'accuracy_validated': False}
    if previous.shape != current.shape or previous.ndim != 3:
        return {**result, 'reason': 'frame_shape_mismatch'}
    h, w = previous.shape[:2]
    c = np.asarray(corners, np.float32)
    if c.shape != (4, 2) or not np.isfinite(c).all():
        return {**result, 'reason': 'invalid_corners'}
    mask = np.zeros((h, w), np.uint8)
    thickness = max(8, round(h * .018))
    # Only the crossbar and upright neighborhoods, never the grass/goal interior.
    for a, b in ((0, 1), (1, 2), (3, 0)):
        cv2.line(mask, tuple(c[a].astype(int)), tuple(c[b].astype(int)), 255, thickness)
    for box in excluded_boxes:
        x0, y0, x1, y1 = np.asarray(box, int)
        cv2.rectangle(mask, (x0-4, y0-4), (x1+4, y1+4), 0, -1)
    old = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    new = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old, 240, .01, 5, mask=mask, blockSize=5)
    if p0 is None or len(p0) < 10:
        return {**result, 'reason': 'too_few_goal_features'}
    kw = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **kw)
    if p1 is None:
        return {**result, 'reason': 'forward_flow_failed'}
    back, s2, _ = cv2.calcOpticalFlowPyrLK(new, old, p1, None, **kw)
    if back is None:
        return {**result, 'reason': 'backward_flow_failed'}
    fb = np.linalg.norm(p0[:, 0] - back[:, 0], axis=1)
    good = (s1[:, 0] == 1) & (s2[:, 0] == 1) & (fb <= 1.)
    a, b = p0[good, 0], p1[good, 0]
    result.update(initial_features=len(p0), fb_consistent_features=len(a))
    if len(a) < 10:
        return {**result, 'reason': 'insufficient_consistent_features'}
    cv2.setRNGSeed(20261011)
    transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                   ransacReprojThreshold=2., maxIters=2000)
    if transform is None or inliers is None:
        return {**result, 'reason': 'transform_failed'}
    take = inliers[:, 0].astype(bool)
    points = a[take]
    ratio = float(take.mean())
    result.update(inliers=int(take.sum()), inlier_fraction=ratio)
    goal_range = np.maximum(np.ptp(c, axis=0), 1.)
    if len(points) < 10 or ratio < .7 or np.any(np.ptp(points, axis=0) / goal_range < [.55, .3]):
        return {**result, 'reason': 'insufficient_spatial_consensus'}
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    rotation = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    predicted = cv2.transform(c[None], transform)[0]
    if not .9 <= scale <= 1.1 or abs(rotation) > 4:
        return {**result, 'reason': 'implausible_short_step_transform'}
    if (predicted < 3).any() or (predicted[:, 0] >= w-3).any() or (predicted[:, 1] >= h-3).any():
        return {**result, 'reason': 'goal_outside_frame'}
    hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 170], np.uint8), np.array([179, 85, 255], np.uint8))
    distance = cv2.distanceTransform(255-white, cv2.DIST_L2, 3)
    support = []
    for i, j in ((0, 1), (1, 2), (3, 0)):
        p = np.rint(np.linspace(predicted[i], predicted[j], 100)).astype(int)
        support.append(float(np.mean(distance[p[:, 1], p[:, 0]] <= max(3, .005*h))))
    result.update(white_line_support=support, scale=scale, rotation_deg=rotation)
    if min(support) < .6:
        return {**result, 'reason': 'weak_current_frame_goal_line_support'}
    return {**result, 'status': 'tracked_candidate', 'reason': 'short_range_consistent_goal_features',
            'corners': predicted.tolist(), 'transform': transform.tolist(),
            'median_fb_error': float(np.median(fb[good])),
            'source': 'propagated_automatic_seed_not_new_independent_detection'}
