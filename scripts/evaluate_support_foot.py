"""Fixed biomechanical support-foot control on already frozen OOF shoe predictions."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_shoe_supervised import read, write, sha, foot_metrics


def support_foot_candidate(prediction, fps):
    """A final observed plant close to contact proposes the opposite striking foot."""
    window = prediction.get("automatic_window")
    if not window or prediction.get("landings") is None:
        return {"striking_foot": "unknown", "reason": "no_automatic_window_or_landings"}
    # The automatic window ends just before the estimated contact neighborhood.
    contact = window["end_frame"] + 1
    eligible = [r for r in prediction["landings"]
                if contact - .25 * fps <= r["frame"] < contact and r["foot"] in ("left", "right")]
    if not eligible:
        return {"striking_foot": "unknown", "reason": "no_final_plant_candidate"}
    last_frame = max(r["frame"] for r in eligible)
    terminal = [r for r in eligible if r["frame"] == last_frame]
    if len({r["foot"] for r in terminal}) != 1:
        return {"striking_foot": "unknown", "reason": "simultaneous_opposite_foot_candidates"}
    support = terminal[0]
    return {"striking_foot": "right" if support["foot"] == "left" else "left",
            "support_candidate": support, "contact_frame_proxy": contact,
            "interpretation": "Opposite of the last automatic landing within 0.25 seconds before automatic contact"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Control output already exists")
    reference = ROOT / "outputs/fresh_course_experiment/human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json"
    rows = read(reference)["events"]
    events = [{"clip_id": r["clip_id"], "foot": r["reviews"][0]["striking_foot"], "fps": r["fps"]} for r in rows]
    predictions = {r["clip_id"]: r for r in read(args.predictions)["events"]}
    control = [support_foot_candidate(predictions[e["clip_id"]], e["fps"]) for e in events]
    combined = [p if p["striking_foot"] != "unknown" else predictions[e["clip_id"]] for e, p in zip(events, control)]
    report = {"schema_version": "fixed-last-support-foot-control-v1", "reference_sha256": sha(reference),
        "predictions_sha256": sha(args.predictions), "human_labels_used_for_routing": False,
        "independent_test": False, "scope": "previously inspected grouped development predictions",
        "selection_window_seconds_before_contact": .25, "threshold_search": False,
        "control": foot_metrics(events, control), "control_then_original_head_if_unavailable": foot_metrics(events, combined),
        "rows": [{"clip_id": e["clip_id"], "reference": e["foot"], **p} for e, p in zip(events, control)],
        "limitations": ["A predicted landing can be spurious or occur after actual contact due to timing error.",
                        "The last predicted foot plant is not necessarily the true final support plant."]}
    write(args.output, report)
    print({k: report[k] for k in ("control", "control_then_original_head_if_unavailable")})


if __name__ == "__main__":
    main()
