"""Build review documents from frozen results; never run neural extraction."""

from collections import Counter
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.evaluate_shoe_landings import validate_prediction
from scripts.render_report_draft import parse_report, write_docx, write_pdf
from src.final_results import sha256
from src.fresh_eda import load_bundle


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/fresh_course_experiment"
OUTPUT = EXPERIMENT / "submission_review_2026-10-22"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--date", default="2026-10-22")
    parser.add_argument("--human-evaluation-dir", type=Path, default=EXPERIMENT / "human_evaluation_2026-10-22")
    args = parser.parse_args()
    output = args.output
    bundle = EXPERIMENT / "final_bundle_v3_shoe_fusion"
    rows, _, main_summary = load_bundle(bundle)
    replay = EXPERIMENT / "shoe_landing_replay_2026-10-18_v1"
    summary = read(replay / "summary.json")
    frozen = read(replay / "prediction_freeze.json")
    predictions = {}
    for item in frozen["predictions"]:
        path = replay / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Changed frozen prediction: {item['clip_id']}")
        prediction = validate_prediction(read(path))
        if prediction["clip_id"] != item["clip_id"]:
            raise ValueError("Frozen prediction identity mismatch")
        predictions[item["clip_id"]] = prediction
    if set(predictions) != {row["clip_id"] for row in rows}:
        raise ValueError("Main bundle and landing replay cohorts differ")
    if sum(p["candidate_count"] or 0 for p in predictions.values()) != summary["proposal_total"]:
        raise ValueError("Replay proposal total differs from frozen predictions")
    for row in summary["rows"]:
        if predictions[row["clip_id"]]["candidate_count"] != row["new_visible_transition_proposals"]:
            raise ValueError("Replay summary differs from a frozen prediction")
    evaluation = read(replay / "evaluation_pending.json")
    if evaluation["prediction_freeze_sha256"] != sha256(replay / "prediction_freeze.json"):
        raise ValueError("Evaluation belongs to a different prediction freeze")
    if evaluation["reference_sha256"] != sha256(replay / "human_reference_template.json"):
        raise ValueError("Saved evaluation reference has changed")
    supplement = EXPERIMENT / "shoe_approach_foot_supplement_v2"
    foot_summary = read(supplement / "summary.json")
    for row in foot_summary["rows"]:
        if sha256(supplement / "results" / f"{row['clip_id']}.json") != row["result_sha256"]:
            raise ValueError("Changed foot supplement")
    foot_counts = dict(Counter(row["candidate"] for row in foot_summary["rows"]))

    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), gridspec_kw={"height_ratios": [2, 1]})
    known = [row for row in summary["rows"] if row["new_visible_transition_proposals"] is not None]
    x = [row["event_id"] for row in known]
    axes[0].bar([value - .19 for value in x], [row["old_visible_transition_proposals"] for row in known],
                width=.38, label="13 Oct: truncated support", color="#527d9f")
    axes[0].bar([value + .19 for value in x], [row["new_visible_transition_proposals"] for row in known],
                width=.38, label="18 Oct: complete support", color="#bd644e")
    unknown = [row["event_id"] for row in summary["rows"] if row["new_visible_transition_proposals"] is None]
    axes[0].scatter(unknown, [-.35] * len(unknown), marker="x", color="#666666", label="Unknown window")
    axes[0].set(xlabel="Canonical event ID", ylabel="Landing candidates", ylim=(-.7, 7), xlim=(0, 40))
    axes[0].set_title("Same saved pose inputs: 12 candidates in 7 events to 22 in 12")
    axes[0].legend(fontsize=8, loc="upper right")
    labels = ["Candidate events", "No proposal", "Channel ambiguity", "No input window"]
    values = [summary["status_counts"][key] for key in (
        "paired_visible_transition_proposals", "no_paired_visible_transition_proposal",
        "temporal_anatomical_channel_ambiguity", "unavailable")]
    bars = axes[1].barh(labels[::-1], values[::-1], color=["#929292", "#c99c49", "#527d9f", "#bd644e"])
    axes[1].bar_label(bars, padding=3)
    axes[1].set(xlabel="Events in the repaired 39-event replay", xlim=(0, 18))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    figure_note = ("Saved-result replay only. See the separate single-reviewer comparison; coverage is not accuracy."
                   if args.human_evaluation_dir else
                   "Saved-result replay only. Independent human reviews: 0. Candidate coverage is not landing accuracy.")
    fig.text(.1, .012, figure_note, fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.savefig(output / "landing_replay_comparison.png", dpi=180)
    plt.close(fig)

    source = ROOT / "docs/REPORT_FRESH_REVIEW_DRAFT.md"
    blocks = parse_report(source)
    pdf = output / f"IND5003_Vision_II_Review_{args.date}.pdf"
    docx = output / f"IND5003_Vision_II_Review_{args.date}.docx"
    pages = write_pdf(source, pdf, blocks, Path("/System/Library/Fonts/Supplemental"))
    write_docx(source, docx, blocks)
    receipt = {
        "date": args.date, "status": "review_draft", "submitted": False,
        "main_bundle_events": len(rows), "main_bundle_artifact_hashes_checked": 4,
        "main_summary_recomputed": True, "main_candidate_availability": main_summary["candidate_availability"],
        "replay_prediction_hashes_checked": len(predictions), "replay_proposal_total": summary["proposal_total"],
        "replay_events_with_proposals": summary["events_with_stance_transition_proposals"],
        "foot_supplement_result_hashes_checked": len(foot_summary["rows"]), "supplement_foot_counts": foot_counts,
        "human_reviewed_events": evaluation["reviewed_events"], "new_neural_inference_calls": 0,
        "raw_video_extraction_reproduced": False, "held_out_accuracy_established": False,
        "pdf_pages": pages, "body_font": "Times New Roman", "body_font_pt": 11,
        "source_sha256": sha256(source), "pdf_sha256": sha256(pdf), "docx_sha256": sha256(docx),
        "figure_sha256": sha256(output / "landing_replay_comparison.png"),
        "unknown_combined_outcomes": [row["clip_id"] for row in rows if row["combined_rule_outcome"] == "unknown"],
    }
    if args.human_evaluation_dir:
        human = read(args.human_evaluation_dir / "integration_summary.json")
        reference = args.human_evaluation_dir / "reference_source/penalty_annotations_39_human_v040.json"
        if human["reference_sha256"] != sha256(reference) or human["prediction_freeze_sha256"] != sha256(replay / "prediction_freeze.json"):
            raise ValueError("Integrated human evaluation does not bind this reference and replay")
        receipt["human_reviewed_events"] = human["reviewed_events"]
        receipt["human_reviewer_count"] = human["reviewer_count"]
        receipt["human_reference_sha256"] = sha256(reference)
        receipt["human_evaluation_summary_sha256"] = sha256(args.human_evaluation_dir / "integration_summary.json")
        receipt["second_reviewer_consensus"] = False
    pivot_path = EXPERIMENT / "pivot_support_development_2026-10-22/summary.json"
    if pivot_path.exists():
        pivot = read(pivot_path)
        receipt["pivot_development_summary_sha256"] = sha256(pivot_path)
        receipt["pivot_development_candidates"] = pivot["pivot"]["predicted_landings_in_comparable_windows"]
        receipt["pivot_development_single_list_count_mae"] = pivot["pivot"]["visible_list_count_mae"]
    (output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
