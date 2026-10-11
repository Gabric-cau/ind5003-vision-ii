"""Descriptive EDA of the canonical candidate bundle, with provenance checks."""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "outputs/final_analysis/features_39.csv"
REQUIRED_COLUMNS = {
    "clip_id", "match_id", "event_id", "kicking_foot_candidate",
    "step_count_candidate", "step_count_verified", "partial_runup",
    "runup_ground_plane_validated", "goalkeeper_direction_candidate",
    "goalkeeper_height_candidate", "goalkeeper_height_proxy_candidate",
    "ball_rule_outcome", "reference_outcome", "ball_final_position_verified",
    "source_sha256", "review_required", "ball_unknown_reason", "camera_interval_count",
}
NUMERIC_COLUMNS = (
    "step_count_candidate", "step_count_verified", "camera_interval_count",
    "step_count_sensitivity_min", "step_count_sensitivity_max", "ball_trajectory_point_count",
)
BOOLEAN_COLUMNS = (
    "partial_runup", "runup_ground_plane_validated", "ball_final_position_verified",
    "review_required", "ball_reference_source_verified",
)
CANDIDATES = {
    "kicking_foot_candidate": ("left", "right"),
    "goalkeeper_pre_movement_candidate": ("static", "moderate_jitter", "high_active"),
    "goalkeeper_direction_candidate": ("image_left", "image_right", "center"),
    "goalkeeper_height_candidate": ("low", "mid", "high"),
    "goalkeeper_height_proxy_candidate": ("low", "mid", "high"),
    "ball_rule_outcome": ("goal_likely", "no_goal_likely"),
    "reference_outcome": ("goal", "no_goal"),
}
LABELS = {
    "kicking_foot_candidate": "Kicking foot candidate",
    "step_count_candidate": "Step count candidate",
    "goalkeeper_pre_movement_candidate": "Legacy keeper pre-movement candidate",
    "goalkeeper_direction_candidate": "Keeper direction candidate",
    "goalkeeper_height_candidate": "Strict keeper height candidate",
    "goalkeeper_height_proxy_candidate": "Silhouette height proxy",
    "ball_rule_outcome": "Ball rule decision",
    "ball_trajectory_point_count": "Ball trajectory candidates",
    "reference_outcome": "External outcome reference",
    "camera_interval_count": "Camera interval candidates",
}


def _boolean(value):
    if pd.isna(value):
        return pd.NA
    word = str(value).strip().lower()
    if word in {"true", "1"}:
        return True
    if word in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_analysis(features_csv=DEFAULT_FEATURES, evaluation_json=None, provenance_json=None, manifest_json=None):
    """Read the frozen bundle; reject legacy tables and mismatched sidecars."""
    path = Path(features_csv)
    evaluation_path = Path(evaluation_json) if evaluation_json else path.with_name("evaluation.json")
    provenance_path = Path(provenance_json) if provenance_json else path.with_name("provenance.json")
    manifest_path = Path(manifest_json) if manifest_json else path.with_name("manifest.json")
    provenance, evaluation = _json(provenance_path), _json(evaluation_path)
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != "vision2-final-analysis-v1":
        raise ValueError("Expected vision2-final-analysis-v1 manifest")
    if manifest.get("evaluation_json_sha256") != hashlib.sha256(evaluation_path.read_bytes()).hexdigest():
        raise ValueError("Evaluation SHA-256 does not match manifest")
    if provenance.get("schema_version") != 1 or provenance.get("artifact_type") != "final_analysis_candidate_bundle":
        raise ValueError("Expected final_analysis_candidate_bundle provenance schema 1")
    if evaluation.get("schema_version") != 1 or evaluation.get("total_events") != 39:
        raise ValueError("Expected a schema-1 evaluation of exactly 39 events")
    if evaluation.get("held_out_test") is not False or evaluation.get("accuracy_validated") is not False:
        raise ValueError("This EDA accepts the explicitly unvalidated development bundle only")
    table = provenance.get("table", {})
    if table.get("path") != path.name:
        raise ValueError("Feature filename does not match provenance.table.path")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if table.get("sha256") != digest or manifest.get("features_csv_sha256") != digest:
        raise ValueError("Feature table SHA-256 does not match provenance")
    df = pd.read_csv(path, keep_default_na=False, na_values=["", "null", "None", "NaN", "nan"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Not a canonical feature table; missing columns: {sorted(missing)}")
    if len(df) != 39 or df["clip_id"].isna().any() or not df["clip_id"].is_unique:
        raise ValueError("Expected exactly 39 nonempty, unique clip_id values")
    canonical = provenance.get("canonical_clip_ids", [])
    if len(canonical) != 39 or len(set(canonical)) != 39 or set(canonical) != set(df["clip_id"]):
        raise ValueError("Feature IDs differ from the canonical provenance IDs")
    if manifest.get("clip_ids") != canonical:
        raise ValueError("Manifest and provenance canonical IDs differ")
    if df["match_id"].isna().any() or df["event_id"].isna().any() or not df["event_id"].is_unique:
        raise ValueError("Every event requires a match_id and unique event_id")
    if not df["source_sha256"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}").all():
        raise ValueError("Every row requires a recorded source SHA-256")
    for column in NUMERIC_COLUMNS:
        if column not in df:
            continue
        numbers = pd.to_numeric(df[column], errors="raise")
        present = numbers.dropna().to_numpy(dtype=float)
        if not np.isfinite(present).all() or (present < 0).any() or (present % 1 != 0).any():
            raise ValueError(f"{column} requires finite, nonnegative integer counts or missing values")
        df[column] = numbers
    for column in BOOLEAN_COLUMNS:
        if column in df:
            df[column] = df[column].map(_boolean).astype("boolean")
    for column, allowed in CANDIDATES.items():
        if column not in df:
            continue
        values = df[column].dropna()
        if not values.isin((*allowed, "unknown")).all():
            raise ValueError(f"Unrecognized {column} category")
    if {"step_count_sensitivity_min", "step_count_sensitivity_max"}.issubset(df.columns):
        low, high = df["step_count_sensitivity_min"], df["step_count_sensitivity_max"]
        if (low.notna() != high.notna()).any() or (low > high).any():
            raise ValueError("Step sensitivity intervals require paired, ordered endpoints")
    return df, evaluation, provenance


def candidate_mask(df, column):
    if column in CANDIDATES:
        return df[column].isin(CANDIDATES[column])
    if column in {"camera_interval_count", "ball_trajectory_point_count"}:
        return df[column].notna() & (df[column] > 0)
    return df[column].notna()


def summarize_analysis(df, evaluation):
    """Describe candidates and covered reference agreement; never fill missing counts."""
    total = len(df)
    columns = [column for column in LABELS if column in df]
    coverage = {column: int(candidate_mask(df, column).sum()) for column in columns}
    by_match = {}
    for match, group in df.groupby("match_id", sort=True):
        by_match[str(match)] = {
            "events": len(group),
            "available": {column: int(candidate_mask(group, column).sum()) for column in columns},
        }
    reference_available = candidate_mask(df, "reference_outcome")
    decided = candidate_mask(df, "ball_rule_outcome")
    comparable = reference_available & decided
    comparison = df.loc[comparable, ["clip_id", "ball_rule_outcome", "reference_outcome"]].copy()
    comparison["mapped_rule"] = comparison["ball_rule_outcome"].map({"goal_likely": "goal", "no_goal_likely": "no_goal"})
    counts = pd.crosstab(comparison["reference_outcome"], comparison["mapped_rule"]).reindex(
        index=["goal", "no_goal"], columns=["goal", "no_goal"], fill_value=0
    )
    matching = int((comparison["mapped_rule"] == comparison["reference_outcome"]).sum())
    covered = len(comparison)
    partial = df["partial_runup"].fillna(False)
    steps = df["step_count_candidate"]
    sensitivity = {}
    if {"step_count_sensitivity_min", "step_count_sensitivity_max"}.issubset(df.columns):
        width = df["step_count_sensitivity_max"] - df["step_count_sensitivity_min"]
        sensitivity = {"available": int(width.notna().sum()), "zero_width": int((width == 0).sum()),
                       "median_width": float(width.dropna().median()) if width.notna().any() else None}
    unknown_reasons = df.loc[~decided, "ball_unknown_reason"].fillna("reason_not_recorded").astype(str).value_counts()
    return {
        "schema_version": 1, "total_events": total, "scope": "development_records_not_held_out",
        "accuracy_validated": False,
        "reference_sources_independently_verified": evaluation.get("reference_sources_independently_verified") is True,
        "candidate_coverage": coverage, "coverage_by_match": by_match,
        "validated_coverage": {
            "step_count_verified": int(df["step_count_verified"].notna().sum()),
            "runup_ground_plane_validated": int(df["runup_ground_plane_validated"].fillna(False).sum()),
            "ball_final_position_verified": int(df["ball_final_position_verified"].fillna(False).sum()),
        },
        "steps": {"available": int(steps.notna().sum()), "missing": int(steps.isna().sum()),
                  "distribution": {str(int(k)): int(v) for k, v in steps.dropna().value_counts().sort_index().items()},
                  "known_partial_events": int(partial.sum()),
                  "partial_flag_missing": int(df["partial_runup"].isna().sum()),
                  "known_partial_ids": df.loc[partial, "clip_id"].tolist(), "parameter_sensitivity": sensitivity,
                  "group_effect_estimate": None,
                  "group_effect_status": "insufficient_independent_labels_and_partial_events"},
        "ball_reference_comparison": {
            "reference_available": int(reference_available.sum()), "rule_decided": int(decided.sum()),
            "covered": covered, "abstentions": int((~decided).sum()),
            "coverage": covered / total if total else None, "matching": matching,
            "covered_reference_agreement": matching / covered if covered else None,
            "status": "descriptive_reference_comparison" if covered else "insufficient_comparable_events",
            "confusion_matrix": {str(ref): {str(pred): int(counts.loc[ref, pred]) for pred in counts.columns} for ref in counts.index},
            "conflict_ids": comparison.loc[comparison["mapped_rule"] != comparison["reference_outcome"], "clip_id"].tolist(),
            "unknown_reasons": {str(k): int(v) for k, v in unknown_reasons.items()},
            "held_out_accuracy": None,
        },
    }


def make_figures(df, summary):
    import matplotlib.pyplot as plt

    figures = {}
    columns = list(summary["candidate_coverage"])
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    values = [summary["candidate_coverage"][column] for column in columns]
    axes[0].barh([LABELS[column] for column in columns], values, color="#257c7b")
    axes[0].set(xlim=(0, 42), xlabel="Events with available candidate/reference (of 39)")
    for i, value in enumerate(values):
        axes[0].text(value + 0.4, i, str(value), va="center")
    matches = list(summary["coverage_by_match"])
    proportions = np.array([[summary["coverage_by_match"][match]["available"][column] /
                             summary["coverage_by_match"][match]["events"] for column in columns] for match in matches])
    axes[1].imshow(proportions, vmin=0, vmax=1, cmap="Greens", aspect="auto")
    axes[1].set_xticks(range(len(columns)), [LABELS[column] for column in columns], rotation=75, ha="right")
    axes[1].set_yticks(range(len(matches)), matches)
    for i, match in enumerate(matches):
        for j, column in enumerate(columns):
            count = summary["coverage_by_match"][match]["available"][column]
            axes[1].text(j, i, f"{count}/{summary['coverage_by_match'][match]['events']}",
                         ha="center", va="center", fontsize=8, color="white" if proportions[i, j] > 0.55 else "black")
    figure.suptitle("Q1: Candidate availability by match | Development records, not accuracy", fontsize=13)
    figures["01_candidate_coverage"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    partial = df["partial_runup"].fillna(False)
    counts = sorted(df["step_count_candidate"].dropna().unique())
    for flag, color, label in [(False, "#257c7b", "No known partial-runup flag (completeness unverified)"),
                               (True, "#b54d59", "Known partial run-up")]:
        frequency = df.loc[partial == flag, "step_count_candidate"].value_counts()
        heights = [int(frequency.get(value, 0)) for value in counts]
        bottom = [int(df.loc[~partial, "step_count_candidate"].value_counts().get(value, 0)) for value in counts] if flag else None
        axes[0].bar(counts, heights, bottom=bottom, color=color, label=label)
    axes[0].set(xlabel="Visible step-count candidate", ylabel="Events")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.19), frameon=False)
    axes[0].set_title(f"{summary['steps']['available']} available; {summary['steps']['missing']} missing")
    if {"step_count_sensitivity_min", "step_count_sensitivity_max"}.issubset(df.columns):
        valid = df["step_count_sensitivity_min"].notna() & df["step_count_sensitivity_max"].notna()
        subset = df.loc[valid].sort_values(["match_id", "event_id"])
        positions = np.arange(len(subset))
        axes[1].vlines(positions, subset["step_count_sensitivity_min"], subset["step_count_sensitivity_max"], color="#257c7b")
        axes[1].scatter(positions, subset["step_count_candidate"], s=14, color="#333333")
        axes[1].set(xlabel="Event ordered by match and event ID", ylabel="Candidate count under fixed parameter variants")
        axes[1].set_title("Parameter ranges are not confidence intervals")
    else:
        axes[1].text(0.5, 0.5, "Insufficient parameter-sensitivity data", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    figure.suptitle("Q2: Step-count candidates and censoring | Development records, unverified landings", fontsize=13)
    figures["02_step_candidates"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    ball = summary["ball_reference_comparison"]
    matrix = np.array([[ball["confusion_matrix"][ref][pred] for pred in ("goal", "no_goal")] for ref in ("goal", "no_goal")])
    axes[0].imshow(matrix, cmap="Greens", vmin=0, vmax=max(1, int(matrix.max())))
    axes[0].set_xticks([0, 1], ["goal_likely", "no_goal_likely"])
    axes[0].set_yticks([0, 1], ["goal", "no_goal"])
    axes[0].set(xlabel="Automatic rule candidate", ylabel="External recorded reference")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center")
    axes[0].set_title(f"{ball['covered']} comparable; {ball['abstentions']} rule abstentions")
    reasons = ball["unknown_reasons"]
    if reasons:
        names = [textwrap.fill(key.replace("_", " "), width=32) for key in reasons]
        axes[1].barh(names, list(reasons.values()), color="#b54d59")
        axes[1].set(xlabel="Events", title="Rule abstention reasons")
        axes[1].tick_params(axis="y", labelsize=8)
    else:
        axes[1].text(0.5, 0.5, "No rule abstentions recorded", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    reference_scope = "reference source independently checked" if summary["reference_sources_independently_verified"] else "reference source unverified"
    figure.suptitle("Q3: Rule/reference comparison | Development records, " + reference_scope, fontsize=13)
    figures["03_ball_reference_comparison"] = figure
    return figures


def run_eda(features_csv=DEFAULT_FEATURES, output_dir=None, evaluation_json=None, provenance_json=None, manifest_json=None):
    df, evaluation, provenance = load_analysis(features_csv, evaluation_json, provenance_json, manifest_json)
    summary = summarize_analysis(df, evaluation)
    summary["input_table_sha256"] = provenance["table"]["sha256"]
    directory = Path(output_dir) if output_dir else Path(features_csv).parent / "eda"
    directory.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, figure in make_figures(df, summary).items():
        figure.savefig(directory / (name + ".png"), dpi=180)
        plt.close(figure)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(df), "output_dir": str(directory), "accuracy_validated": False}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    args = parser.parse_args()
    run_eda(args.features_csv, args.output_dir, args.evaluation_json, args.provenance_json, args.manifest_json)


if __name__ == "__main__":
    main()
