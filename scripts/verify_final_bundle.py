"""Verify artifact consistency; this is not a visual-accuracy check."""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eda_analysis import load_analysis, summarize_analysis
from src.final_results import compare_ball_references, read_csv, sha256


ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path, bundle: Path, require_eda: bool = False) -> dict:
    table = bundle / "features_39.csv"
    df, evaluation, provenance = load_analysis(table)
    for item in provenance["inputs"]:
        if item["path"].startswith("external/"):
            raise ValueError("External inputs cannot be resolved in this checkout; consolidate repository-relative inputs for portable verification")
        source = root / item["path"]
        if not source.is_file() or sha256(source) != item["sha256"]:
            raise ValueError(f"Input artifact missing or changed since consolidation: {item['path']}")
    rows = json.loads((bundle / "features_39.json").read_text(encoding="utf-8"))["rows"]
    if [r["clip_id"] for r in rows] != list(df["clip_id"]):
        raise ValueError("CSV and JSON event identities differ")
    for column in df.columns:
        from_json = [r.get(column) for r in rows]
        from_csv = df[column].tolist()
        for a, b in zip(from_json, from_csv):
            if a is None:
                if not pd.isna(b):
                    raise ValueError(f"CSV/JSON missing value mismatch: {column}")
            elif str(a) != str(b):
                if not (isinstance(a, (int, float)) and not isinstance(a, bool)
                        and math.isclose(a, float(b), rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError(f"CSV/JSON value mismatch: {column}")
    comparison = compare_ball_references(rows)
    if comparison != evaluation["ball_rule_reference_comparison"]:
        raise ValueError("Saved evaluation differs from the event-level comparison")
    camera_table = bundle / "camera_intervals_candidates.csv"
    if evaluation.get("camera_interval_diagnostics", {}).get("available"):
        camera_inputs = [item for item in provenance["inputs"]
                         if Path(item["path"]).name == "camera_timings_candidates.csv"]
        if len(camera_inputs) != 1 or not camera_table.is_file():
            raise ValueError("Missing or ambiguous camera interval source")
        if read_csv(root / camera_inputs[0]["path"]) != read_csv(camera_table):
            raise ValueError("Consolidated camera intervals differ from their source")
    summary = summarize_analysis(df, evaluation)
    eda_path = bundle / "eda/summary.json"
    eda_checked = False
    if eda_path.is_file():
        saved_eda = json.loads(eda_path.read_text(encoding="utf-8"))
        if saved_eda.get("input_table_sha256") != sha256(table):
            raise ValueError("EDA summary belongs to a different feature-table version")
        for key, value in summary.items():
            if saved_eda.get(key) != value:
                raise ValueError(f"Saved EDA summary differs from recomputation: {key}")
        eda_checked = True
    elif require_eda:
        raise FileNotFoundError(eda_path)
    return {"events": len(df), "input_artifacts_checked": len(provenance["inputs"]),
            "csv_json_consistent": True, "evaluation_recomputed": True, "eda_recomputed": eda_checked,
            "raw_source_verification": provenance["raw_source_verification"],
            "visual_accuracy_verified": False, "independent_holdout": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/final_analysis")
    parser.add_argument("--require-eda", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.bundle, args.require_eda), indent=2))


if __name__ == "__main__":
    main()
