"""Automatic, per-frame pitch-plane hypotheses with held-out line checks.

Goal PnP with assumed intrinsics supplies semantic search priors only. Four
automatic goal-area line intersections fit the ground homography; penalty-area
lines and goalpost feet are excluded from that fit. Passing these image checks
does not establish real-world metric accuracy or known camera intrinsics.
Dimensions: https://www.theifab.com/laws/latest/the-field-of-play/
"""
from itertools import product

import cv2
import numpy as np

from src.field_masks import build_field_mask
from src.goal_geometry import detect_goal_frame


GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
GOAL_AREA_HALF_WIDTH = GOAL_WIDTH / 2 + 5.5
PENALTY_AREA_HALF_WIDTH = GOAL_WIDTH / 2 + 16.5
PITCH_LINES = {
    "goal_line": ((-9.16, 0.), (9.16, 0.)),
    "goal_area_front": ((-9.16, 5.5), (9.16, 5.5)),
    "goal_area_left": ((-9.16, 0.), (-9.16, 5.5)),
    "goal_area_right": ((9.16, 0.), (9.16, 5.5)),
    "penalty_area_front": ((-20.16, 16.5), (20.16, 16.5)),
    "penalty_area_left": ((-20.16, 0.), (-20.16, 16.5)),
    "penalty_area_right": ((20.16, 0.), (20.16, 16.5)),
}
FIT_LINES = tuple(PITCH_LINES)[:4]
HELD_OUT_LINES = tuple(PITCH_LINES)[4:]


def _project(matrix, points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    valid = np.all(np.isfinite(homogeneous), axis=1) & (np.abs(homogeneous[:, 2]) > 1e-8)
    projected = np.full((len(points), 2), np.nan)
    projected[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
    return projected, valid


def _normal(points):
    a, b = np.asarray(points, float)
    direction = b - a
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return None
    normal = np.array([-direction[1], direction[0]]) / length
    return np.r_[normal, -normal @ a]


def _intersection(a, b):
    point = np.cross(a, b)
    return None if abs(point[2]) < 1e-6 else point[:2] / point[2]


def _field_lines(frame, field, corners):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 120), (179, 110, 255))
    white = cv2.bitwise_and(white, field.region)
    cv2.fillConvexPoly(white, np.rint(corners).astype(np.int32), 0)
    # Canny precedes ROI masking, avoiding an artificial field-mask boundary.
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 40, 100)
    edges = cv2.bitwise_and(edges, cv2.dilate(white, np.ones((3, 3), np.uint8)))
    h, w = frame.shape[:2]
    segments = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=35,
                              minLineLength=max(30, int(w * .047)), maxLineGap=int(w * .027))
    groups = []
    if segments is not None:
        segments = segments[:, 0].reshape(-1, 2, 2).astype(float)
        for segment in sorted(segments, key=lambda s: -np.linalg.norm(s[1] - s[0])):
            line = _normal(segment)
            match = next((g for g in groups
                          if abs(g["line"][:2] @ line[:2]) > np.cos(np.deg2rad(2))
                          and np.max(np.abs(np.c_[segment, np.ones(2)] @ g["line"])) < h * .012), None)
            if match is None:
                groups.append({"line": line, "points": segment.copy()})
            else:
                match["points"] = np.vstack((match["points"], segment))
                fit = cv2.fitLine(match["points"].astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
                match["line"] = _normal([fit[2:], fit[2:] + fit[:2]])
    for i, group in enumerate(groups):
        group["id"] = i
        group["length_px"] = float(np.max(np.linalg.norm(
            group["points"][:, None] - group["points"][None, :], axis=-1)))
    return groups, white


def _pnp_priors(corners, shape, focal_ratios):
    h, w = shape[:2]
    world = np.array([[-3.66, 0, 2.44], [3.66, 0, 2.44],
                      [3.66, 0, 0], [-3.66, 0, 0]], float)
    priors = []
    for ratio in focal_ratios:
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Focal ratios must be finite and positive")
        intrinsic = np.array([[w * ratio, 0, w / 2], [0, w * ratio, h / 2], [0, 0, 1.]], float)
        try:
            ok, rotation, translation = cv2.solvePnP(world, corners, intrinsic, None,
                                                     flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            continue
        if not ok:
            continue
        rotation_matrix = cv2.Rodrigues(rotation)[0]
        camera = (-rotation_matrix.T @ translation).ravel()
        predicted = cv2.projectPoints(world, rotation, translation, intrinsic, None)[0].reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((predicted - corners) ** 2, axis=1))))
        depths = (world @ rotation_matrix.T + translation.ravel())[:, 2]
        if not np.all(depths > 0) or not camera[1] < -1 or not .5 < camera[2] < 60 or error > h * .01:
            continue
        # PnP Y points into the goal; pitch Y points into the playing field.
        matrix = intrinsic @ np.c_[rotation_matrix[:, 0], -rotation_matrix[:, 1], translation.ravel()]
        priors.append({"pitch_to_image": matrix, "focal_width_ratio_assumption": float(ratio),
                       "goal_reprojection_rmse_px": error})
    return priors


def _match_lines(points, groups, shape, *, maximum_distance):
    h, w = shape[:2]
    points = np.asarray(points)
    reference = _normal(points[[0, -1]])
    if reference is None:
        return []
    inside = np.all(np.isfinite(points), axis=1) & (points[:, 0] > 0) & (points[:, 0] < w)
    inside &= (points[:, 1] > 0) & (points[:, 1] < h)
    if np.count_nonzero(inside) < 8:
        return []
    choices = []
    for group in groups:
        angle = float(np.rad2deg(np.arccos(np.clip(abs(group["line"][:2] @ reference[:2]), 0, 1))))
        distance = float(np.median(np.abs(np.c_[points[inside], np.ones(np.count_nonzero(inside))]
                                            @ group["line"])))
        if angle < 8 and distance < maximum_distance and group["length_px"] > w * .047:
            choices.append({"group": group, "angle_degrees": angle, "distance_px": distance,
                            "prior_cost": distance + angle * h / 240})
    return sorted(choices, key=lambda c: c["prior_cost"])[:2]


def _line_evidence(matrix, name, white_distance, field, radius):
    h, w = field.shape
    projected, valid = _project(matrix, np.linspace(*PITCH_LINES[name], 200))
    valid &= (projected[:, 0] >= 2) & (projected[:, 0] < w - 2)
    valid &= (projected[:, 1] >= 2) & (projected[:, 1] < h - 2)
    pixels = np.rint(projected[valid]).astype(int)
    if len(pixels):
        pixels = pixels[field[pixels[:, 1], pixels[:, 0]] > 0]
    if len(pixels) < 24:
        return {"visible_samples": len(pixels), "support_fraction": None, "median_distance_px": None}
    distances = white_distance[pixels[:, 1], pixels[:, 0]]
    return {"visible_samples": len(pixels), "support_fraction": float(np.mean(distances <= radius)),
            "median_distance_px": float(np.median(distances))}


def calibrate_pitch_frame(frame, *, frame_id=None, goal=None,
                          focal_ratios=(.7, 1.2, 1.8, 2.5, 3.5, 4.5), held_out_radius_px=6.):
    """Return an automatically checked ground-plane candidate or abstain.

    Selection uses goal/goal-area evidence; penalty-area evidence only accepts
    or rejects the selected hypothesis. It is never used to tune its geometry.
    A candidate applies to this exact frame, not an entire moving-camera clip.
    """
    field = build_field_mask(frame)
    h, w = frame.shape[:2]
    if frame_id is not None and (isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or frame_id < 0):
        raise ValueError("frame_id must be a nonnegative integer or None")
    if not np.isfinite(held_out_radius_px) or held_out_radius_px <= 0:
        raise ValueError("held_out_radius_px must be finite and positive")
    result = {"status": "abstain", "reason": None, "frame_id": None if frame_id is None else int(frame_id),
              "image_size": [w, h], "selected": None, "candidate": None,
              "metric_accuracy_verified": False, "camera_intrinsics_known": False,
              "human_verified": False, "ground_contact_only": True,
              "coordinate_convention": "x along image-left to image-right goal; y into playing field",
              "coordinate_unit": "m_hypothesis", "valid_for_other_frames": False,
              "dimension_source": "https://www.theifab.com/laws/latest/the-field-of-play/",
              "assumptions": ["regulation pitch markings", "planar playing surface",
                              "pinhole search priors with centered principal point and square pixels",
                              "unknown lens distortion and automatic line semantics"],
              "fit_line_names": list(FIT_LINES), "held_out_line_names": list(HELD_OUT_LINES)}
    if not field.valid:
        return dict(result, reason="insufficient_field_prior")
    goal = detect_goal_frame(frame) if goal is None else goal
    if goal.get("status") != "candidate_supported" or not goal.get("selected"):
        return dict(result, reason="insufficient_or_ambiguous_goal_geometry")
    corners = np.asarray(goal["selected"]["corners"], float)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return dict(result, reason="invalid_goal_corners")
    groups, white = _field_lines(frame, field, corners)
    priors = _pnp_priors(corners, frame.shape, focal_ratios)
    result["line_group_count"] = len(groups)
    result["pnp_prior_count"] = len(priors)
    if not priors:
        return dict(result, reason="no_plausible_camera_prior")
    baseline = _match_lines(np.linspace(corners[3], corners[2], 100), groups, frame.shape,
                            maximum_distance=h * .025)
    candidates = {}
    destination = np.array([[-9.16, 0], [9.16, 0], [-9.16, 5.5], [9.16, 5.5]], float)
    for prior in priors:
        choices = [baseline]
        for name in FIT_LINES[1:]:
            projected, valid = _project(prior["pitch_to_image"], np.linspace(*PITCH_LINES[name], 100))
            choices.append(_match_lines(projected, groups, frame.shape, maximum_distance=h * .05)
                           if np.all(valid) else [])
        if any(not choice for choice in choices):
            continue
        for matches in product(*choices):
            ids = tuple(match["group"]["id"] for match in matches)
            if len(set(ids)) < 4:
                continue
            goal_line, front, left, right = [match["group"]["line"] for match in matches]
            points = [_intersection(goal_line, left), _intersection(goal_line, right),
                      _intersection(front, left), _intersection(front, right)]
            if any(point is None for point in points):
                continue
            points = np.asarray(points)
            polygon = points[[0, 1, 3, 2]].astype(np.float32)
            if not np.all(np.isfinite(points)) or not cv2.isContourConvex(polygon):
                continue
            if abs(cv2.contourArea(polygon)) < .004 * h * w or np.max(np.abs(points)) > 4 * max(h, w):
                continue
            matrix, _ = cv2.findHomography(destination, points)
            if matrix is None or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-8:
                continue
            post_feet, valid = _project(matrix, [(-3.66, 0), (3.66, 0)])
            error = float(np.sqrt(np.mean(np.sum((post_feet - corners[[3, 2]]) ** 2, axis=1))))
            if not np.all(valid) or error > h * .025:
                continue
            cost = float(np.mean([match["prior_cost"] for match in matches]))
            candidate = {"line_group_ids": list(ids), "selection_cost_px": cost,
                         "fit_image_points": points.tolist(), "fit_pitch_points_m_hypothesis": destination.tolist(),
                         "pitch_to_image_hypothesis": matrix.tolist(),
                         "image_to_pitch_hypothesis": np.linalg.inv(matrix).tolist(),
                         "goal_foot_check_rmse_px": error,
                         "pnp_focal_width_ratio_assumption": prior["focal_width_ratio_assumption"],
                         "pnp_goal_reprojection_rmse_px": prior["goal_reprojection_rmse_px"]}
            if ids not in candidates or cost < candidates[ids]["selection_cost_px"]:
                candidates[ids] = candidate
    ranked = sorted(candidates.values(), key=lambda c: c["selection_cost_px"])
    result["distinct_fit_count"] = len(ranked)
    if not ranked:
        return dict(result, reason="no_consistent_goal_area_line_assignment")
    best = ranked[0]
    result["candidate"] = best
    result["alternative_fit_summaries"] = [{key: candidate[key] for key in
                                            ("line_group_ids", "selection_cost_px", "fit_image_points")}
                                           for candidate in ranked[1:5]]
    distance = cv2.distanceTransform(255 - white, cv2.DIST_L2, 3)
    best["held_out_checks"] = {name: _line_evidence(best["pitch_to_image_hypothesis"], name,
                                                   distance, field.region, held_out_radius_px)
                                for name in HELD_OUT_LINES}
    # Different Hough edges within 8 px are the same marking hypothesis.
    ambiguous = any(other["selection_cost_px"] <= best["selection_cost_px"] + h * .006
                    and np.max(np.linalg.norm(np.asarray(other["fit_image_points"])
                                               - best["fit_image_points"], axis=1)) > h * .012
                    for other in ranked[1:])
    result["ambiguous_line_assignment"] = bool(ambiguous)
    front_check = best["held_out_checks"]["penalty_area_front"]
    supported = front_check["support_fraction"] is not None and front_check["support_fraction"] >= .65
    if ambiguous:
        return dict(result, reason="ambiguous_goal_area_line_assignment")
    if best["goal_foot_check_rmse_px"] > h * .015:
        return dict(result, reason="goal_width_check_failed")
    if not supported:
        return dict(result, reason="held_out_penalty_front_not_supported")
    if any(check["support_fraction"] is not None and check["support_fraction"] < .35
           and check["median_distance_px"] > 2 * held_out_radius_px
           for check in best["held_out_checks"].values()):
        return dict(result, reason="contradictory_held_out_pitch_line")
    result.update(status="candidate_supported", reason="automatic_ground_geometry_checks_passed", selected=best)
    return result


def project_ground_contacts(calibration, points, *, frame_id, point_kind, on_ground):
    """Project explicitly ground-contact feet in the exact calibrated frame.

    The caller supplies contact decisions. Airborne feet, hips and footballs
    cannot be corrected with a pitch homography. Returned meters are hypotheses.
    """
    points = np.asarray(points, float).reshape(-1, 2)
    flags = list(on_ground)
    if len(flags) != len(points) or any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise ValueError("on_ground must contain one boolean per point")
    result = {"status": "abstain", "coordinates_m_hypothesis": [None] * len(points),
              "metric_accuracy_verified": False, "reason": None}
    if point_kind != "foot_ground_contact":
        return dict(result, reason="not_a_ground_contact_foot_point")
    if (calibration.get("frame_id") is None or isinstance(frame_id, bool)
            or not isinstance(frame_id, (int, np.integer)) or frame_id != calibration["frame_id"]):
        return dict(result, reason="calibration_frame_mismatch")
    if calibration.get("status") != "candidate_supported" or calibration.get("selected") is None:
        return dict(result, reason="calibration_not_supported")
    projected, valid = _project(calibration["selected"]["image_to_pitch_hypothesis"], points)
    w, h = calibration["image_size"]
    valid &= np.asarray(flags) & np.all(np.isfinite(points), axis=1)
    valid &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    valid &= (np.abs(projected[:, 0]) <= 20.16) & (projected[:, 1] >= 0) & (projected[:, 1] <= 16.5)
    result["coordinates_m_hypothesis"] = [point.tolist() if ok else None for point, ok in zip(projected, valid)]
    result["status"] = "candidate_supported" if np.any(valid) else "abstain"
    result["reason"] = "ground_contact_hypotheses_only" if np.any(valid) else "no_valid_ground_contact_in_checked_area"
    return result


def foot_contact_candidate(frame, observation, *, minimum_confidence=.5):
    """Find a visible sole boundary near the lower confident pose ankle.

    Image height and grass below a shoe are contact heuristics, not proof that
    a foot touches the plane. Preserve this distinction in exported trajectories.
    No box-bottom, hip or ankle coordinate is substituted when pixels fail.
    """
    result = {"status": "abstain", "reason": None, "sole_pixel": None,
              "ground_contact_verified": False, "human_verified": False,
              "method": "ankle_seeded_non_grass_shoe_boundary_and_grass_below"}
    points = np.asarray(observation.get("keypoints", []), float)
    confidences = np.asarray(observation.get("keypoint_confidences", []), float)
    box = np.asarray(observation.get("bbox", []), float)
    if points.shape != (17, 2) or confidences.shape != (17,) or box.shape != (4,):
        return dict(result, reason="incomplete_pose_observation")
    if not np.all(np.isfinite(box)) or box[3] <= box[1]:
        return dict(result, reason="invalid_person_box")
    height = box[3] - box[1]
    eligible = [index for index in (15, 16) if np.isfinite(confidences[index])
                and confidences[index] >= minimum_confidence and np.all(np.isfinite(points[index]))]
    if not eligible:
        return dict(result, reason="no_confident_ankle")
    ankle_index = max(eligible, key=lambda index: points[index, 1])
    ankle = points[ankle_index]
    if not box[3] - .16 * height <= ankle[1] <= box[3] + .03 * height:
        return dict(result, reason="ankle_not_near_visible_foot_band")
    h, w = frame.shape[:2]
    xa, xb = max(0, int(ankle[0] - .12 * height)), min(w, int(ankle[0] + .12 * height) + 1)
    ya, yb = max(0, int(ankle[1] - .04 * height)), min(h, int(ankle[1] + .14 * height) + 1)
    if xa >= xb or ya >= yb:
        return dict(result, reason="foot_roi_outside_frame")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35)
    foreground = (~grass[ya:yb, xa:xb]) & (hsv[ya:yb, xa:xb, 2] >= 65)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    matches = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 3:
            continue
        yy, xx = np.nonzero(labels == index)
        distance = np.min(np.linalg.norm(np.c_[xx + xa, yy + ya] - ankle, axis=1))
        if distance > max(3., .05 * height):
            continue
        bottom = yy >= np.quantile(yy, .90)
        pixel = np.array([np.median(xx[bottom]) + xa, np.median(yy[bottom]) + ya])
        if not ankle[1] - .02 * height <= pixel[1] <= box[3] + .08 * height:
            continue
        matches.append((distance, pixel, int(stats[index, cv2.CC_STAT_AREA])))
    if not matches:
        return dict(result, reason="no_visible_shoe_boundary")
    matches.sort(key=lambda match: match[0])
    _, sole, area = matches[0]
    rad = max(3, int(height * .06))
    x, y = np.rint(sole).astype(int)
    band = grass[min(h, y + 2):min(h, y + rad + 2), max(0, x - rad):min(w, x + rad + 1)]
    support = float(np.mean(band)) if band.size else 0.
    result.update(ankle_pixel=ankle.tolist(), ankle_confidence=float(confidences[ankle_index]),
                  foot="left" if ankle_index == 15 else "right", grass_below_fraction=support,
                  shoe_component_area_px=area)
    if support < .55:
        return dict(result, reason="insufficient_grass_below_sole")
    result.update(status="contact_candidate", reason="automatic_image_contact_heuristics_passed",
                  sole_pixel=sole.tolist())
    return result
