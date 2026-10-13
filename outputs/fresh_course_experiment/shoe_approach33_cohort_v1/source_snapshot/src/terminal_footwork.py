"""Separate projected approach route from contact-anchored terminal rhythm.

This is an explicit, untrained change-point heuristic over *candidate* events,
not a foot-contact detector. No angle, ankle stiffness, intent, or success is
inferred. All intervals are half open and use source-video frame numbers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .penalty_phases import _dense_samples, _path_features, _safe, _smooth


@dataclass(frozen=True)
class TerminalConfig:
    search_seconds: float = 1.60
    strike_guard_seconds: float = 0.12
    max_tail_gap_seconds: float = 0.55
    min_intervals_each_side: int = 2
    max_baseline_intervals: int = 5
    rhythm_short_ratio: float = 0.78
    rhythm_long_ratio: float = 1.45
    min_rhythm_change_seconds: float = 0.06
    progress_short_ratio: float = 0.65
    min_progress_change_box_heights: float = 0.06
    min_sustained_fraction: float = 0.67
    min_split_gain: float = 0.25
    max_interpolation_gap_seconds: float = 0.15
    max_scale_ratio: float = 1.50


def _frames(events):
    values = [e.get("frame") if isinstance(e, Mapping) else e for e in events]
    if any(v is None or not np.isfinite(v) or int(v) != v or v < 0 for v in values):
        raise ValueError("Event frames must be finite nonnegative integers")
    values = [int(v) for v in values]
    if len(set(values)) != len(values):
        raise ValueError("Duplicate event frames cannot be counted as separate contacts")
    return sorted(values)


def _gain(before, after):
    all_values = np.r_[before, after]
    one = float(np.sum(np.abs(all_values - np.median(all_values))))
    two = float(np.sum(np.abs(before - np.median(before))) +
                np.sum(np.abs(after - np.median(after))))
    return max(0., 1 - two / one) if one > 1e-9 else 0.


def _contrast(before, after, cfg, *, progress=False):
    """A sustained median shift AND a two-segment absolute-error improvement."""
    if any(v is None or not np.isfinite(v) for v in [*before, *after]):
        return []
    a, b = float(np.median(before)), float(np.median(after))
    if a <= 1e-9:
        return []
    ratio = cfg.progress_short_ratio if progress else cfg.rhythm_short_ratio
    minimum = cfg.min_progress_change_box_heights if progress else cfg.min_rhythm_change_seconds
    candidates = []
    if b <= a * ratio and a - b >= minimum:
        fraction = float(np.mean(np.array(after) <= a * ratio))
        candidates.append(("lower_projected_progress" if progress else "rhythm_compression", fraction))
    if not progress and b >= a * cfg.rhythm_long_ratio and b - a >= minimum:
        candidates.append(("rhythm_lengthening", float(np.mean(np.array(after) >= a * cfg.rhythm_long_ratio))))
    gain = _gain(before, after)
    return [{"feature": name, "baseline_median": a, "terminal_median": b,
             "ratio": b / a, "sustained_fraction": fraction, "split_gain": gain,
             "units": "person_box_heights" if progress else "seconds"}
            for name, fraction in candidates if fraction + 1e-9 >= cfg.min_sustained_fraction and gain >= cfg.min_split_gain]


def analyze_terminal_footwork(frame_ids, fps, stabilized_hips, box_heights, *,
                             phase_result, contact, events, camera_status="unknown",
                             candidate_axis_reliable=True, config=None):
    """Find a variable-length terminal change; never prescribe a last-N count.

    Camera failure disables route/progress, but not translation-invariant event
    timing. A small-step interpretation is never made from timing alone. The
    contact guard is excluded before searching, not used to force a split.
    """
    cfg = config or TerminalConfig()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    fs = np.asarray(frame_ids)
    if fs.ndim != 1 or not np.all(np.isfinite(fs)) or not np.all(fs == np.floor(fs)):
        raise ValueError("Pose frame IDs must be integers")
    fs = fs.astype(int)
    if len(fs) and (fs[0] < 0 or np.any(np.diff(fs) <= 0)):
        raise ValueError("Pose frame IDs must be unique and increasing")
    heights = np.asarray(box_heights, float)
    if heights.shape != (len(fs),):
        raise ValueError("Expected one height per frame")
    if stabilized_hips is not None and np.asarray(stabilized_hips).shape != (len(fs), 2):
        raise ValueError("Expected Nx2 stabilized hip coordinates")
    if cfg.min_intervals_each_side < 2 or cfg.max_baseline_intervals < cfg.min_intervals_each_side:
        raise ValueError("Need at least two intervals on each side")
    event_frames = _frames(events)
    labels = {int(x["frame"]): x for x in phase_result.get("frame_labels", [])}
    onset = phase_result.get("approach_start_frame")
    kick = contact.get("contact_frame")
    warnings = []
    result = {
        "schema_version": "terminal-footwork-v1", "status": "abstained",
        "human_verified": False, "accuracy_validated": False, "review_required": True,
        "thresholds": asdict(cfg), "warnings": warnings,
        "route": {"status": "unavailable", "scope": None, "tags": [], "metrics": None},
        "terminal": {"status": "insufficient_evidence", "start_frame": None, "end_frame_exclusive": None,
                     "tags": [], "evidence": [], "search_candidates": [], "intervals": []},
        "counts": {"visible_candidates": len(event_frames), "nominal_approach_candidates": None,
                   "before_terminal_candidates": None, "terminal_adjustment_candidates": None,
                   "no_distinct_split_candidates": None, "unresolved_prestrike_candidates": None,
                   "contact_guard_candidates": None, "pose_gap_candidates": None},
        "events": [], "analysis_window": None,
        "ankle_control": {"status": "not_measurable", "reason": "COCO17 has ankles but no heel/toe or foot orientation; no stiffness, torque or control-quality measurement."},
        "limitations": ["Events are ankle-signal extrema, not confirmed landings.",
            "Projected hip displacement is not physical step length; timing is not verified cadence.",
            "No split detected does not prove no adjustment; abstention does not mean zero steps.",
            "No prescribed final three/five steps, no player-name rule, no use of outcome or user counts.",
            "The contact interval and parameter ranges are heuristic, not statistical confidence intervals."],
    }
    usable = (phase_result.get("kick_anchor", {}).get("usable") and onset is not None and kick is not None
              and contact.get("status") in ("supported", "tentative") and onset < kick)
    if not usable:
        warnings.append("approach_or_contact_unresolved")
        result["events"] = [{"frame": f, "role": "unresolved", "human_verified": False} for f in event_frames]
        return result
    window = contact.get("interval_frames") or [kick, kick]
    if len(window) != 2 or not window[0] <= kick <= window[1]:
        raise ValueError("Contact window must contain the nominal anchor")
    safe_end = max(onset, min(window[0], kick - max(1, round(cfg.strike_guard_seconds * fps))))
    search_start = max(onset, safe_end - round(cfg.search_seconds * fps))
    result["analysis_window"] = {"approach_start_frame": onset, "safe_end_frame_exclusive": safe_end,
        "nominal_contact_frame": kick, "contact_interval_inclusive": list(window),
        "search_start_frame": search_start, "scope": "visible_inferred_approach_not_full_unseen_runup"}
    support = lambda f: bool(labels.get(f, {}).get("pose_supported", False))
    safe_events = [f for f in event_frames if onset <= f < safe_end and support(f)]
    # Missing/invalid samples remain NaN; a route must never bridge such a gap.
    grid = np.arange(fs[0], fs[-1] + 1) if len(fs) else np.array([], int)
    path = None
    hvalid = np.isfinite(heights) & (heights > 0)
    scale_ratio = float(np.percentile(heights[hvalid], 90) / np.percentile(heights[hvalid], 10)) if np.any(hvalid) else None
    camera_usable = camera_status == "supported" and stabilized_hips is not None and scale_ratio is not None and scale_ratio <= cfg.max_scale_ratio
    if camera_usable:
        points = np.asarray(stabilized_hips, float)
        valid = hvalid & np.all(np.isfinite(points), axis=1) & np.array([support(int(f)) for f in fs])
        path = _smooth(_dense_samples(fs, points / np.median(heights[hvalid]), valid, grid,
                                     max(1, round(cfg.max_interpolation_gap_seconds * fps))), max(1, round(.06 * fps)))
        for i, f in enumerate(grid):
            if not support(int(f)):
                path[i] = np.nan
    else:
        warnings.append("route_and_progress_unavailable_camera_or_scale")
    def span_path(a, b):
        if path is None or not len(grid) or a < grid[0] or b > grid[-1] or b <= a:
            return None
        sample = path[a-grid[0]:b-grid[0]+1]
        return sample if np.all(np.isfinite(sample)) else None

    intervals = []
    for a, b in zip(safe_events[:-1], safe_events[1:]):
        valid = all(support(f) for f in range(a, b + 1))
        sample = span_path(a, b) if valid else None
        # Local Euclidean advancement avoids falsely shortening steps because a
        # curved approach turns away from one global projection axis.
        advance = float(np.linalg.norm(sample[-1] - sample[0])) if sample is not None else None
        intervals.append({"start_frame": a, "end_frame": b, "pose_supported": valid,
            "candidate_interval_seconds": (b-a)/fps if valid else None,
            "projected_hip_advance_box_heights": advance, "human_verified": False})
    terminal = result["terminal"]
    terminal["intervals"] = intervals
    minimum = cfg.min_intervals_each_side
    terminal_supported = (len(safe_events) > 0 and safe_end - safe_events[-1] <= cfg.max_tail_gap_seconds * fps
                          and all(support(f) for f in range(safe_events[-1], safe_end)))
    if not candidate_axis_reliable:
        warnings.append("candidate_axis_unreliable_for_terminal_split")
    if not terminal_supported:
        warnings.append("terminal_tail_not_observed")
    if len(intervals) < 2 * minimum:
        warnings.append("too_few_intervals_for_relative_change")
    can_compare = candidate_axis_reliable and terminal_supported and len(intervals) >= 2 * minimum
    comparisons = []
    for split in range(minimum, len(intervals) - minimum + 1) if can_compare else ():
        boundary = intervals[split]["start_frame"]
        if boundary < search_start:
            continue
        before = intervals[max(0, split-cfg.max_baseline_intervals):split]
        after = intervals[split:]
        if not all(i["pose_supported"] for i in before + after):
            continue
        evidence = _contrast([i["candidate_interval_seconds"] for i in before],
                             [i["candidate_interval_seconds"] for i in after], cfg)
        evidence += _contrast([i["projected_hip_advance_box_heights"] for i in before],
                              [i["projected_hip_advance_box_heights"] for i in after], cfg, progress=True)
        comparisons.append({"boundary_frame": boundary, "baseline_intervals": len(before),
            "terminal_intervals": len(after), "evidence": evidence,
            "ranking_score": max((e["split_gain"] for e in evidence), default=0.)})
    terminal["search_candidates"] = comparisons
    accepted = [c for c in comparisons if c["evidence"]]
    boundary = None
    if accepted:
        selected = max(accepted, key=lambda c: (c["ranking_score"], -c["boundary_frame"]))
        boundary = selected["boundary_frame"]
        terminal.update({"status": "candidate_change", "start_frame": boundary,
                         "end_frame_exclusive": safe_end, "evidence": selected["evidence"],
                         "tags": [e["feature"] for e in selected["evidence"]]})
        warnings.append("boundary_event_shared_by_before_and_after_intervals")
    elif comparisons:
        terminal["status"] = "no_distinct_change_detected"
    result["status"] = "provisional" if comparisons else "insufficient_terminal_evidence"
    if contact["status"] == "tentative":
        warnings.append("terminal_split_uses_tentative_contact")

    # Route is a separate output, not another mutually exclusive player class.
    route_end = boundary if boundary is not None else safe_end
    sample = span_path(onset, route_end-1)
    metrics = _path_features(sample, fps) if sample is not None else None
    route = result["route"]
    route.update({"start_frame": onset, "end_frame_exclusive": route_end,
                  "scope": "before_terminal_change" if boundary is not None else "visible_approach_without_confirmed_split",
                  "metrics": metrics})
    if metrics is not None and metrics["path_length_box_heights"] >= .35:
        ratio, deviation = metrics["path_to_chord_ratio"], metrics["max_cross_axis_deviation_box_heights"]
        route["status"] = "candidate_description"
        if ratio is not None and deviation is not None:
            if ratio >= 1.08 and deviation >= .12:
                route["tags"] = ["curved_image_plane"]
            elif ratio <= 1.08 and deviation < .12:
                route["tags"] = ["approximately_straight_image_plane"]
            else:
                route["tags"] = ["mixed_or_indeterminate_image_plane"]

    roles = []
    for f in event_frames:
        if f < onset:
            role = "pre_approach_context"
        elif f >= kick:
            role = "at_or_after_nominal_contact"
        elif f >= safe_end:
            role = "contact_guard"
        elif not support(f):
            role = "pose_gap"
        elif boundary is not None:
            role = "terminal_adjustment_candidate" if f >= boundary else "before_terminal"
        else:
            role = "no_distinct_split" if comparisons else "unresolved_prestrike"
        roles.append({"frame": f, "role": role, "human_verified": False,
                      "within_contact_interval": window[0] <= f <= window[1]})
    result["events"] = roles
    counts = result["counts"]
    counts["nominal_approach_candidates"] = sum(onset <= f < kick for f in event_frames)
    role_count = lambda name: sum(e["role"] == name for e in roles)
    counts.update({"before_terminal_candidates": role_count("before_terminal") if boundary is not None else None,
                   "terminal_adjustment_candidates": role_count("terminal_adjustment_candidate") if boundary is not None else None,
                   "no_distinct_split_candidates": role_count("no_distinct_split") if comparisons and boundary is None else None,
                   "unresolved_prestrike_candidates": role_count("unresolved_prestrike"),
                   "contact_guard_candidates": role_count("contact_guard"), "pose_gap_candidates": role_count("pose_gap")})
    if boundary is not None:
        counts["terminal_count_excluding_boundary_event"] = max(0, counts["terminal_adjustment_candidates"] - 1)
    result["partition_sum"] = sum(role_count(n) for n in ("before_terminal", "terminal_adjustment_candidate",
        "no_distinct_split", "unresolved_prestrike", "contact_guard", "pose_gap"))
    assert result["partition_sum"] == counts["nominal_approach_candidates"]
    return _safe(result)
