"""Attempt-level style-aware reporting of unverified step candidates.

Classification changes count presentation, never peak detection or total count.
Absence of a terminal change is not sufficient evidence for continuity. A short
approach can have enough evidence for regular rhythm without enough intervals
to fit the upstream before/after change-point model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median


@dataclass(frozen=True)
class StepStyleConfig:
    min_continuous_intervals: int = 2
    min_pose_supported_fraction: float = .90
    min_pose_observed_fraction: float = .80
    max_relative_interval_mad: float = .20
    max_interval_to_median_ratio: float = 1.80
    max_relative_interval_range: float = 1.0


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _range(values):
    good = [v for v in values if v is not None]
    return [min(good), max(good)] if good else None


def analyze_style_aware_steps(result, config=None):
    cfg = config or StepStyleConfig()
    if cfg.min_continuous_intervals < 2:
        raise ValueError("Continuity requires at least two observed candidate intervals")
    f = result["footwork_result"]
    terminal, window = f["terminal"], f["analysis_window"]
    phase = result["phase_result"]
    counts = f["counts"]
    nominal = counts["nominal_approach_candidates"]
    reasons, cautions = [], []
    style = "undetermined"
    evidence = {}
    sensitivity = f.get("sensitivity", {})
    comparison_variants = [v for group in ("event_prominence", "change_rules", "contact_anchor")
                           for v in sensitivity.get(group, {}).values()]
    conflicting_change = any(v["status"] == "candidate_change" for v in comparison_variants)
    interval_rows = terminal.get("intervals", [])
    intervals = [r["candidate_interval_seconds"] for r in interval_rows]
    intervals_ok = bool(intervals) and all(r["pose_supported"] and _finite_number(r["candidate_interval_seconds"])
        and r["candidate_interval_seconds"] > 0 for r in interval_rows)
    if window:
        a, b = window["approach_start_frame"], window["safe_end_frame_exclusive"]
        labels = {x["frame"]: x for x in phase["frame_labels"]}
        span = max(1, b-a)
        support_fraction = sum(bool(labels.get(i, {}).get("pose_supported")) for i in range(a, b)) / span
        observed_fraction = sum(bool(labels.get(i, {}).get("pose_observed")) for i in range(a, b)) / span
        evidence.update({"pose_supported_fraction": support_fraction, "pose_observed_fraction": observed_fraction,
                         "candidate_intervals": len(intervals), "scope_start_frame": a, "scope_end_frame_exclusive": b})
    else:
        support_fraction = observed_fraction = 0.
    if intervals_ok:
        m = median(intervals)
        relative_mad = median(abs(x-m) for x in intervals) / m
        max_ratio = max(intervals) / m
        relative_range = (max(intervals)-min(intervals)) / m
        evidence.update({"median_candidate_interval_seconds": m, "relative_interval_mad": relative_mad,
                         "max_interval_to_median_ratio": max_ratio, "relative_interval_range": relative_range})
    else:
        relative_mad = max_ratio = relative_range = None
    pause_evidence = any(t["tag"] == "pause_or_stutter_image_plane_candidate"
                         for t in phase.get("style_tags", []))
    evidence["upstream_pause_candidate"] = pause_evidence
    evidence["parameter_variant_has_terminal_change"] = conflicting_change

    if nominal is None or window is None:
        reasons.append("approach_or_contact_unresolved")
    elif terminal["status"] == "candidate_change":
        style = "terminal_adjustment_candidate"
        reasons.append("sustained_terminal_change_in_default_analysis")
    else:
        # All conditions are positive checks. Missing metadata never passes.
        if len(intervals) < cfg.min_continuous_intervals:
            reasons.append("too_few_intervals_for_continuity")
        if not intervals_ok:
            reasons.append("candidate_intervals_missing_or_pose_gap")
        if support_fraction < cfg.min_pose_supported_fraction or observed_fraction < cfg.min_pose_observed_fraction:
            reasons.append("continuity_pose_coverage_insufficient")
        if relative_mad is not None and (relative_mad > cfg.max_relative_interval_mad + 1e-9
            or max_ratio > cfg.max_interval_to_median_ratio + 1e-9
            or relative_range > cfg.max_relative_interval_range + 1e-9):
            reasons.append("candidate_rhythm_not_regular_enough")
        if pause_evidence:
            reasons.append("pause_evidence_prevents_continuous_label")
        if conflicting_change:
            reasons.append("parameter_variants_disagree_about_terminal_change")
        expected_groups = (("event_prominence", {"sensitive", "central", "strict"}),
                           ("change_rules", {"permissive", "central", "conservative"}),
                           ("contact_anchor", {"early", "nominal", "late"}))
        if any(not keys.issubset(sensitivity.get(group, {})) for group, keys in expected_groups):
            reasons.append("continuity_sensitivity_checks_incomplete")
        for code in ("terminal_tail_not_observed", "candidate_axis_unreliable_for_terminal_split"):
            if code in f["warnings"]:
                reasons.append(code)
        if not reasons:
            style = "continuous_approach_candidate"
            reasons.append("regular_candidate_rhythm_with_pose_and_tail_support_no_change_evidence")
            if terminal["status"] == "insufficient_evidence":
                cautions.append("short_approach_regular_rhythm_not_a_proven_negative_change_test")
    if not sensitivity.get("status_and_tags_agree", False):
        cautions.append("terminal_parameter_sensitive")
    if result["contact"]["status"] == "tentative":
        cautions.append("tentative_contact_anchor")
    if "approach_may_be_left_censored" in phase.get("warnings", []):
        cautions.append("visible_approach_only_start_may_be_missing")
    if counts.get("contact_guard_candidates"):
        cautions.append("contact_guard_candidates_remain_in_nominal_total")
    if f.get("route", {}).get("status") != "candidate_description":
        cautions.append("route_unresolved_style_describes_rhythm_not_true_field_path")

    # Event-threshold and contact-boundary variation is kept separate from style.
    # Apply the same half-open f < boundary convention to each variant.
    count_grid = {}
    if nominal is not None:
        start = window["approach_start_frame"]
        for name in ("sensitive", "central", "strict"):
            if name not in result["event_proposal"].get("variants", {}):
                continue
            events = result["event_proposal"]["variants"][name]
            count_grid[name] = {label: sum(start <= e["frame"] < anchor for e in events)
                for label, anchor in (("early", window["contact_interval_inclusive"][0]),
                                      ("nominal", window["nominal_contact_frame"]),
                                      ("late", window["contact_interval_inclusive"][1]))}
    nominal_variant_counts = [g["nominal"] for g in count_grid.values()]
    range_counts = [v for g in count_grid.values() for v in g.values()]
    if len(set(nominal_variant_counts)) > 1:
        cautions.append("count_changes_with_peak_threshold")
    partition = {"continuous_approach": None, "before_terminal": None, "terminal_adjustment": None,
                 "type_unresolved": None, "contact_guard": counts.get("contact_guard_candidates"),
                 "pose_gap": counts.get("pose_gap_candidates")}
    policy = "whole_approach_type_unresolved"
    if nominal is not None:
        unguarded = nominal - (partition["contact_guard"] or 0) - (partition["pose_gap"] or 0)
        if style == "terminal_adjustment_candidate":
            policy = "split_at_terminal_boundary"
            partition.update(before_terminal=counts["before_terminal_candidates"], terminal_adjustment=counts["terminal_adjustment_candidates"])
        elif style == "continuous_approach_candidate":
            policy = "whole_continuous_approach"
            partition["continuous_approach"] = unguarded
        else:
            partition["type_unresolved"] = unguarded
        if sum(x or 0 for x in partition.values()) != nominal:
            raise ValueError("Style-aware partition does not conserve the original candidate count")
    onset = window["approach_start_frame"] if window else None
    before_approach = sum(e["frame"] < onset for e in result["event_proposal"].get("events", [])) if onset is not None else None
    return {"schema_version": "style-aware-steps-v1", "attempt_style": style,
        "classification_scope": "this_attempt_visible_inferred_approach_not_player_identity",
        "candidate_step_count": nominal, "verified_step_count": None, "counts_are_verified_contacts": False,
        "count_policy": policy, "count_partition": partition,
        "before_approach_context_candidates": before_approach,
        "terminal_start_frame": terminal["start_frame"] if style == "terminal_adjustment_candidate" else None,
        "terminal_boundary_excluded_count": counts.get("terminal_count_excluding_boundary_event") if style == "terminal_adjustment_candidate" else None,
        "count_sensitivity": {"grid": count_grid, "threshold_only_range": _range(nominal_variant_counts),
            "threshold_and_contact_range": _range(range_counts), "boundary_convention": "start <= frame < contact_boundary",
            "is_confidence_interval": False},
        "evidence": evidence, "reasons": reasons, "cautions": cautions,
        "thresholds": asdict(cfg), "human_verified": False, "accuracy_validated": False, "review_required": True,
        "limitations": ["Classification does not alter event detection thresholds, remove events, or cap high counts.",
            "Continuous means evidence of relatively regular observed candidate rhythm, not proof of no adjustment.",
            "Nominal total includes separately reported contact guard candidates; not all are confirmed steps.",
            "Unknown type retains its count when possible; type unknown is not extraction failure or zero steps.",
            "All style thresholds are unvalidated engineering choices; ankle control and success causality are not measured."]}
