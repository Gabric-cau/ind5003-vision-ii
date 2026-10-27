"""Replay fixed-point stance support with unchanged thresholds on saved poses."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.evaluate_shoe_landings import load_frozen_predictions, match_landings
from scripts.integrate_human_reference_2026_10_22 import single_landing_summary
from scripts.replay_shoe_landings import _validate, sha, write
from src.shoe_landing_events import LandingConfig
from src.shoe_pivot_support import propose_pivot_landings


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs/fresh_course_experiment"
OUT = EXP / "pivot_support_development_2026-10-22"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dense = EXP / "shoe_approach33_cohort_v1"
    validated, _ = _validate(dense, EXP / "shoe_approach_analysis_v2")
    frozen = load_frozen_predictions(EXP / "shoe_landing_replay_2026-10-18_v1/prediction_freeze.json")
    refs_path = EXP / "human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json"
    refs = {row["clip_id"]: row for row in json.loads(refs_path.read_text())["events"]}
    diagnostics = {"baseline": [], "pivot": []}
    rows, reasons = [], Counter()
    for source, entry, old, _, inputs, records in validated:
        cid = source["clip_id"]
        baseline_path = EXP / "shoe_landing_replay_2026-10-18_v1/events" / cid / "landing_result.json"
        baseline = json.loads(baseline_path.read_text())
        reasons.update(item["reason"] for item in baseline.get("diagnostics", []))
        result = (propose_pivot_landings(records, inputs["fps"], inputs["interval_frames"],
                                       inputs["original_camera_motion"], inputs["decoded_frame_hashes"])
                  if inputs else {**baseline, "baseline_candidate_count": None, "pivot_added_count": 0})
        if result["baseline_candidate_count"] != baseline["candidate_count"]:
            raise ValueError("The baseline is not reproduced from the saved poses")
        if frozen[cid]["candidate_count"] != baseline["candidate_count"]:
            raise ValueError("The baseline freeze and current prediction differ")
        retained = {(item["frame"], item["foot"]) for item in (result["landings"] or [])}
        if not {(item["frame"], item["foot"]) for item in (baseline["landings"] or [])} <= retained:
            raise ValueError("The supplement discarded or moved a frozen baseline landing")
        result.update(clip_id=cid, source_sha256=source["clip_sha256"], frame_count=source["frame_count"],
                      fps=source["fps"], baseline_prediction_sha256=sha(baseline_path),
                      human_verified=False, accuracy_validated=False, held_out_test=False)
        if refs[cid]["source_sha256"] != source["clip_sha256"] or refs[cid]["observable_window"] != result["observable_window"]:
            raise ValueError("Reference and replay source/window mismatch")
        write(OUT / "events" / cid / "landing_result.json", result)
        row = dict(clip_id=cid, event_id=source["event_id"], baseline_count=baseline["candidate_count"],
                   pivot_count=result["candidate_count"], added_count=result["pivot_added_count"], status=result["status"])
        for name, prediction in (("baseline", baseline), ("pivot", result)):
            diagnostic = {"clip_id": cid}
            if prediction["landings"] is not None:
                review = refs[cid]["reviews"][0]
                expected = review["verified_landings"]
                matching = match_landings(prediction["landings"], expected, .08 * source["fps"])
                diagnostic.update(comparison=matching, reviewed_visible_count=len(expected),
                                  visible_candidate_count=len(prediction["landings"]),
                                  count_absolute_error=abs(len(prediction["landings"]) - len(expected)))
            diagnostics[name].append(diagnostic)
        rows.append(row)
    code = {}
    for name in ("src/shoe_landing_events.py", "src/shoe_pivot_support.py", "scripts/run_pivot_support_development_2026_10_22.py"):
        target = OUT / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        code[name] = sha(target)
    summary = {
        "date": "2026-10-22", "status": "development_rule_comparison", "events": 39,
        "new_neural_inference_calls": 0, "thresholds_changed": False, "settings": asdict(LandingConfig()),
        "reference_labels_informed_problem_selection": True, "prediction_function_reads_reference_labels": False,
        "held_out_test": False, "single_human_reviewer": True, "original_predictions_modified": False,
        "method_change": "Allow the same fixed heel or toe landmark to support the full paired stance; retain original whole-shoe candidates.",
        "source_code_sha256": code, "reference_sha256": sha(refs_path),
        "dense_protocol_sha256": sha(dense / "protocol.json"), "dense_summary_sha256": sha(dense / "summary.json"),
        "baseline": single_landing_summary({"individual_review_diagnostics": diagnostics["baseline"]}),
        "pivot": single_landing_summary({"individual_review_diagnostics": diagnostics["pivot"]}),
        "additional_candidates": sum(row["added_count"] for row in rows),
        "baseline_rejection_diagnostic_counts": dict(reasons), "rows": rows,
        "limitations": ["All comparisons use the same development videos and one supplied reviewer.",
                        "A low-motion heel/toe point is still not proof of physical contact.",
                        "Confidence, mirror, camera, actor and anatomical-channel guards are unchanged.",
                        "The seven missing input windows and five channel-ambiguous cases remain unavailable.",
                        "Reference list completeness is not separately human-confirmed for every window."]}
    write(OUT / "summary.json", summary)
    write(OUT / "per_event_diagnostics.json", diagnostics)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for offset, method, label, color in ((-.18, "baseline", "Whole shoe", "#527d9f"),
                                        (.18, "pivot", "Fixed heel/toe supplement", "#bd644e")):
        values = [summary[method][key] * 100 for key in ("precision_against_supplied_list", "recall_against_supplied_list")]
        bars = axes[0].bar([offset, 1 + offset], values, width=.36, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set(xticks=[0, 1], xticklabels=["List precision", "List recall"], ylim=(0, 80), ylabel="Percent")
    axes[0].legend(fontsize=8)
    values = [summary[key]["visible_list_count_mae"] for key in ("baseline", "pivot")]
    bars = axes[1].bar(["Whole shoe", "Fixed heel/toe"], values, color=["#527d9f", "#bd644e"])
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].set(ylabel="Within-window count MAE", ylim=(0, 6))
    fig.suptitle("Same saved poses and thresholds: fixed-point support reduces undercount")
    fig.text(.05, .012, "27 comparable windows; 12 unavailable unchanged. One reviewer, development comparison, no new inference.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(OUT / "pivot_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
