"""Goal-area and penalty-front fit, with independent penalty sides held out."""
from copy import deepcopy

import cv2
import numpy as np

from src.field_masks import build_field_mask
from src.goal_geometry import detect_goal_frame
from src.pitch_calibration import (FIT_LINES, PITCH_LINES, _field_lines, _intersection,
                                   _line_evidence, _match_lines, _project, calibrate_pitch_frame)


JOINT_FIT_LINES = (*FIT_LINES, "penalty_area_front")
JOINT_HELD_OUT_LINES = ("penalty_area_left", "penalty_area_right")
GROUND_POINTS = np.array([[-9.16, 0.], [9.16, 0.], [-9.16, 5.5], [9.16, 5.5],
                          [-9.16, 16.5], [9.16, 16.5]])


def _ambiguous_assignments(ranked, image_height):
    if len(ranked) < 2:
        return False
    probe = [[-9.16, 5.5], [9.16, 5.5], [0., 11.], [-20.16, 16.5], [20.16, 16.5]]
    best_points, best_valid = _project(ranked[0]["pitch_to_image_hypothesis"], probe)
    for other in ranked[1:]:
        if other["selection_cost_px"] > ranked[0]["selection_cost_px"] + image_height * .006:
            continue
        points, valid = _project(other["pitch_to_image_hypothesis"], probe)
        if (not np.all(best_valid & valid)
                or np.max(np.linalg.norm(points - best_points, axis=1)) > image_height * .012):
            return True
    return False


def calibrate_joint_pitch_frame(frame, *, frame_id=None, goal=None, held_out_radius_px=6.):
    """Select by fit evidence, then accept/reject with unused penalty-side paint.

    The two far fit points are intersections of the observed penalty-front
    line with the projected extensions of the two observed goal-area sides.
    They are geometric line correspondences, not manually selected corners.
    """
    goal = detect_goal_frame(frame) if goal is None else goal
    baseline = calibrate_pitch_frame(frame, frame_id=frame_id, goal=goal,
                                     held_out_radius_px=held_out_radius_px)
    result = deepcopy(baseline)
    result.update(status="abstain", reason=None, selected=None, candidate=None,
                  calibration_mode="joint_goal_area_penalty_front_with_held_out_penalty_sides",
                  fit_line_names=list(JOINT_FIT_LINES), held_out_line_names=list(JOINT_HELD_OUT_LINES),
                  baseline_comparison={"status": baseline["status"], "reason": baseline["reason"]},
                  held_out_used_for_hypothesis_selection=False)
    result.pop("alternative_fit_summaries", None)
    if baseline.get("candidate") is None:
        return dict(result, reason=baseline["reason"])
    field = build_field_mask(frame)
    corners = np.asarray(goal["selected"]["corners"], float)
    groups, white = _field_lines(frame, field, corners)
    groups_by_id = {group["id"]: group for group in groups}
    h, w = frame.shape[:2]
    bases = [baseline["candidate"], *baseline.get("alternative_fit_summaries", [])]
    result["association_search_scope"] = "retained_goal_area_assignments_and_top_two_prior_matched_penalty_front_lines"
    if baseline.get("distinct_fit_count", len(bases)) > len(bases):
        return dict(result, reason="unexplored_goal_area_assignments")
    candidates = {}
    for base in bases:
        ids = base["line_group_ids"]
        base_matrix, _ = cv2.findHomography(GROUND_POINTS[:4], np.asarray(base["fit_image_points"], float))
        if base_matrix is None:
            continue
        predicted, valid = _project(base_matrix, np.linspace(*PITCH_LINES["penalty_area_front"], 100))
        if not np.all(valid):
            continue
        matches = _match_lines(predicted, groups, frame.shape, maximum_distance=h * .08)
        for match in matches:
            front = match["group"]
            if front["id"] in ids:
                continue
            far = [_intersection(front["line"], groups_by_id[ids[index]]["line"]) for index in (2, 3)]
            if any(point is None for point in far):
                continue
            image_points = np.vstack([base["fit_image_points"], far])
            if not np.all(np.isfinite(image_points)) or np.max(np.abs(image_points)) > max(h, w) * 4:
                continue
            matrix, _ = cv2.findHomography(GROUND_POINTS, image_points)
            if matrix is None or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-8:
                continue
            fitted, valid = _project(matrix, GROUND_POINTS)
            if not np.all(valid):
                continue
            fit_error = float(np.sqrt(np.mean(np.sum((fitted - image_points) ** 2, axis=1))))
            post_feet, valid = _project(matrix, [(-3.66, 0.), (3.66, 0.)])
            if not np.all(valid):
                continue
            post_error = float(np.sqrt(np.mean(np.sum((post_feet - corners[[3, 2]]) ** 2, axis=1))))
            key = (*ids, front["id"])
            cost = (base["selection_cost_px"] * 4 + match["prior_cost"]) / 5 + fit_error
            candidate = {"line_group_ids": list(key), "selection_cost_px": float(cost),
                         "fit_image_points": image_points.tolist(),
                         "fit_pitch_points_m_hypothesis": GROUND_POINTS.tolist(),
                         "fit_reprojection_rmse_px": fit_error, "goal_foot_check_rmse_px": post_error,
                         "pitch_to_image_hypothesis": matrix.tolist(),
                         "image_to_pitch_hypothesis": np.linalg.inv(matrix).tolist(),
                         "fit_point_origin": "automatic_line_intersections_and_side_extensions"}
            if key not in candidates or cost < candidates[key]["selection_cost_px"]:
                candidates[key] = candidate
    ranked = sorted(candidates.values(), key=lambda item: item["selection_cost_px"])
    result["distinct_joint_fit_count"] = len(ranked)
    if not ranked:
        return dict(result, reason="no_consistent_joint_line_assignment")
    best = ranked[0]
    result["candidate"] = best
    result["alternative_fit_summaries"] = [{key: other[key] for key in
                    ("line_group_ids", "selection_cost_px", "fit_image_points")} for other in ranked[1:5]]
    result["ambiguous_line_assignment"] = _ambiguous_assignments(ranked, h)
    if result["ambiguous_line_assignment"]:
        return dict(result, reason="ambiguous_joint_line_assignment")
    if best["fit_reprojection_rmse_px"] > h * .012:
        return dict(result, reason="joint_fit_residual_too_large")
    if best["goal_foot_check_rmse_px"] > h * .015:
        return dict(result, reason="held_out_goal_width_check_failed")
    distance = cv2.distanceTransform(255 - white, cv2.DIST_L2, 3)
    checks = {name: _line_evidence(best["pitch_to_image_hypothesis"], name,
                                  distance, field.region, held_out_radius_px) for name in JOINT_HELD_OUT_LINES}
    for name, check in checks.items():
        points, valid = _project(best["pitch_to_image_hypothesis"], np.linspace(*PITCH_LINES[name], 200))
        valid &= (points[:, 0] >= 2) & (points[:, 0] < w - 2) & (points[:, 1] >= 2) & (points[:, 1] < h - 2)
        points = points[valid]
        if len(points):
            pixels = np.rint(points).astype(int)
            points = points[field.region[pixels[:, 1], pixels[:, 0]] > 0]
        check["visible_span_px"] = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.
        check["sufficient_visible_span"] = check["visible_span_px"] >= w * .04
    best["held_out_checks"] = checks
    visible = [check for check in checks.values() if check["sufficient_visible_span"] and check["support_fraction"] is not None]
    if any(check["support_fraction"] < .35 and check["median_distance_px"] > 2 * held_out_radius_px for check in visible):
        return dict(result, reason="contradictory_held_out_penalty_side")
    if not any(check["support_fraction"] >= .65 for check in visible):
        return dict(result, reason="independent_held_out_penalty_side_not_supported")
    result.update(status="candidate_supported", selected=best, reason="automatic_joint_geometry_checks_passed")
    return result
