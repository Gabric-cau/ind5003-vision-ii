"""Experimental mask/flow primitives; no direction promotion or human labels.

OpenCV GrabCut and LK optical flow are used directly. Local Lab likelihood,
temporal gates, automatic pose seeds and QC thresholds are project heuristics.
"""
import cv2
import numpy as np


def pose_seed(observation, shape, offset=(0, 0)):
    mask = np.zeros(shape[:2], np.uint8)
    p = np.asarray(observation['keypoints'], float) - np.asarray(offset)
    c = np.asarray(observation['keypoint_confidences'])
    if np.all(c[[5, 6, 11, 12]] >= .5):
        poly = p[[5, 6, 12, 11]]
        mid = poly.mean(0)
        cv2.fillConvexPoly(mask, np.rint(mid + .65 * (poly-mid)).astype(np.int32), 1)
    return mask


def box_mask(box, shape, offset=(0, 0), pad=0):
    b = np.asarray(box) - np.tile(offset, 2)
    x0, y0, x1, y1 = np.rint(b + [-pad, -pad, pad, pad]).astype(int)
    m = np.zeros(shape[:2], np.uint8)
    m[max(0, y0):min(shape[0], y1+1), max(0, x0):min(shape[1], x1+1)] = 1
    return m


def train_lab_model(images, seeds, exclusions):
    fg, bg = [], []
    for img, seed, excluded in zip(images, seeds, exclusions):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grass = ((hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) &
                 (hsv[:, :, 1] > 35) & (excluded == 0))
        fg.append(lab[seed > 0]); bg.append(lab[grass])
    fg, bg = np.concatenate(fg), np.concatenate(bg)
    if len(fg) < 20 or len(bg) < 100:
        raise ValueError('Insufficient automatic foreground/background seeds')
    models = []
    for x in (fg, bg):
        # Robust center/diagonal dispersion avoids a single lawn outlier cluster.
        mu = np.median(x, axis=0)
        var = np.maximum((1.4826*np.median(np.abs(x-mu), axis=0))**2, 36.)
        models.append((mu, var))
    return models, dict(foreground_pixels=len(fg), background_pixels=len(bg),
                       means=[v[0].tolist() for v in models],
                       variances=[v[1].tolist() for v in models])


def color_score(image, models):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    ll = [-.5*np.sum((lab-mu)**2/var + np.log(var), axis=2) for mu, var in models]
    return ll[0]-ll[1]


def nearest_component(raw, prior, min_area=20, max_area=15000, max_distance=60):
    n, labels, stats, centers = cv2.connectedComponentsWithStats(raw.astype(np.uint8))
    py, px = np.nonzero(prior)
    if len(px) == 0:
        return np.zeros_like(raw, np.uint8), 'no_temporal_prior'
    target = np.array([px.mean(), py.mean()])
    candidates = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not min_area <= area <= max_area:
            continue
        dist = np.linalg.norm(centers[i]-target)
        overlap = np.count_nonzero((labels == i) & (prior > 0)) / area
        if dist <= max_distance:
            candidates.append((dist-30*overlap, i))
    if not candidates:
        return np.zeros_like(raw, np.uint8), 'no_component_in_continuity_gate'
    _, idx = min(candidates)
    return (labels == idx).astype(np.uint8), 'candidate_not_identity_verified'


def warp_mask(mask, backward_flow):
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    return cv2.remap(mask, xx+backward_flow[:, :, 0], yy+backward_flow[:, :, 1],
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)


def grabcut_mask(image, prior, score, reliable_seed=None):
    labels = np.zeros(image.shape[:2], np.uint8)
    broad = cv2.dilate(prior, np.ones((31, 31), np.uint8)) > 0
    labels[broad] = cv2.GC_PR_BGD
    labels[broad & (score > 0)] = cv2.GC_PR_FGD
    core = cv2.erode(prior, np.ones((3, 3), np.uint8)) > 0
    positives = core & (score > 1.0)
    if reliable_seed is not None:
        positives |= reliable_seed > 0
    if np.count_nonzero(positives) < 5:
        return np.zeros_like(prior), 'no_reliable_foreground_seed'
    labels[positives] = cv2.GC_FGD
    cv2.grabCut(image, labels, None, np.zeros((1, 65), float), np.zeros((1, 65), float),
                3, cv2.GC_INIT_WITH_MASK)
    raw = ((labels == 1) | (labels == 3)).astype(np.uint8)
    return nearest_component(raw, prior)


def lk_tracks(prev, curr, points):
    if points is None or len(points) == 0:
        return np.empty((0, 1, 2), np.float32), np.empty((0, 1, 2), np.float32)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, points, None, winSize=(21, 21),
                                            maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, nxt, None, winSize=(21, 21),
                                         maxLevel=3)
    ok = (status.ravel() == 1) & (st2.ravel() == 1)
    ok &= np.linalg.norm((back-points).reshape(-1, 2), axis=1) < 1.5
    return points[ok], nxt[ok]


def camera_affine(prev, curr, excluded):
    allowed = (1-cv2.dilate(excluded, np.ones((31, 31), np.uint8)))*255
    pts = cv2.goodFeaturesToTrack(prev, 200, .01, 8, mask=allowed)
    old, new = lk_tracks(prev, curr, pts)
    if len(new) < 12:
        return None, dict(status='insufficient_background_features', points=len(new))
    mat, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
    count = 0 if inliers is None else int(inliers.sum())
    if mat is None or count < 10 or count/len(new) < .5:
        return None, dict(status='background_model_rejected', points=len(new), inliers=count)
    return mat, dict(status='estimated_not_ground_truth', points=len(new), inliers=count,
                     matrix=mat.tolist())


def mask_info(mask, frame):
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return dict(frame=frame, pixels=0, centroid=None, bbox=None)
    return dict(frame=frame, pixels=len(x), centroid=[float(x.mean()), float(y.mean())],
                bbox=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)])
