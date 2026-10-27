"""Inventory exact observability limits without filling missing features."""
from collections import Counter
import csv
import json
from pathlib import Path

from src.fresh_eda import load_bundle


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "outputs/fresh_course_experiment"
OUT = EXP / "feature_gap_audit_2026-10-22"


def main():
    _, structured, _ = load_bundle(EXP / "final_bundle_v3_shoe_fusion")
    refs_path = EXP / "human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json"
    refs = {row["clip_id"]: row for row in json.loads(refs_path.read_text())["events"]}
    rows, points, height_reasons, endpoint_reasons, playback_reasons = [], [], Counter(), Counter(), Counter()
    for event in structured["events"]:
        cid, components = event["clip_id"], event["components"]
        pitch, camera, height, endpoint = (components[name] for name in ("pitch", "camera", "height", "endpoint"))
        window = refs[cid]["observable_window"]
        pre = [point for point in pitch.get("sparse_ground_contact_candidates", []) if point["phase"] == "pre_kick_candidate"]
        approach = [point for point in pre if window and window["start_frame"] <= point["frame_id"] <= window["end_frame"]]
        for point in pre:
            points.append(dict(clip_id=cid, frame=point["frame_id"], source_frame=point["source_frame_id"],
                               x_hypothesis=point["coordinates_m_hypothesis"][0], y_hypothesis=point["coordinates_m_hypothesis"][1],
                               in_frozen_approach_window=point in approach, physical_accuracy_verified=False))
        frames = sorted({point["frame_id"] for point in approach})
        height_reason = height.get("reason") or height.get("status") or "unspecified"
        endpoint_reason = endpoint.get("reason") or endpoint.get("status") or "unspecified"
        height_reasons[height_reason] += 1
        endpoint_reasons[endpoint_reason] += 1
        segments = camera.get("segments", [])
        for segment in segments:
            evidence = segment.get("playback_evidence", {})
            playback_reasons[evidence.get("reason") or evidence.get("status") or "unspecified"] += 1
        rows.append(dict(clip_id=cid, ground_pre_contact_points=len(pre),
                         ground_points_within_frozen_approach=len(approach),
                         largest_approach_sample_gap_frames=max((b - a for a, b in zip(frames, frames[1:])), default=None),
                         full_continuous_metric_trajectory_verified=False,
                         strict_height_reason=height_reason, ball_endpoint_reason=endpoint_reason,
                         camera_intervals=len(segments),
                         available_raw_clip=(ROOT / "data/raw_clips" / f"{cid}.mp4").is_file()))
    OUT.mkdir(exist_ok=True)
    for name, records in (("event_gaps.csv", rows), ("pre_contact_ground_hypotheses.csv", points)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = dict(events=39, source="hash-verified frozen feature bundle and unchanged human reference",
                   saved_pre_contact_ground_points=len(points),
                   saved_ground_points_within_frozen_approach=sum(row["ground_points_within_frozen_approach"] for row in rows),
                   events_with_ground_points_within_frozen_approach=sum(row["ground_points_within_frozen_approach"] > 0 for row in rows),
                   available_raw_clips=[row["clip_id"] for row in rows if row["available_raw_clip"]],
                   height_failure_reasons=dict(height_reasons), endpoint_failure_reasons=dict(endpoint_reasons),
                   playback_failure_reasons=dict(playback_reasons),
                   required_inputs={
                       "continuous_top_down_trajectory": "Full source frames spanning approach; per-frame camera geometry with held-out calibration checks and continuously associated grounded body/foot points. Sparse saved sole hypotheses cannot establish continuous metric motion.",
                       "strict_keeper_height": "Uninterrupted preparation-to-landing keeper view, reliable body/ground reference, and calibrated geometry. Human low/middle categories do not supply physical heights or missing pixels.",
                       "final_ball_position": "Source footage through verified play end, visible ball identity after occlusion/cuts, and consistent coordinates. Last tracked sample cannot establish event-final position.",
                       "slow_motion": "Matched original-speed and replay actions, retained role identity, sufficient contiguous pose variation and an independently supported real-time clock. Container FPS alone does not establish playback speed.",
                       "missing_landing_inputs": "Seven events need valid source-bound approach/contact intervals and pose extraction; five require resolving anatomical-channel ambiguity before new counts can be accepted.",
                   }, new_neural_inference_calls=0, missing_features_filled=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
