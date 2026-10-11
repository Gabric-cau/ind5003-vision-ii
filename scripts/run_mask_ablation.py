"""Compare HSV priors on frozen pose detections, without treating AI labels as truth."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.field_masks import annotate_track_field_support, build_field_mask, background_feature_mask, white_ball_proposals
from src.kicker_analysis import KickerFeatureExtractor
from src.step_pipeline import _analyse_tracks, _track_pose_sequence, _call_step_counter, _estimate_runup_start, _json_safe


def recount(source, tracks, selected, summaries):
    if selected is None:
        return {"step_count": None, "reason": "no_pose_track"}
    seq = _track_pose_sequence(tracks[selected], .15)
    kick = summaries[selected]["kick_detection"].get("kick_frame")
    if kick is None:
        return {"step_count": None, "reason": "no_contact_estimate"}
    extractor = KickerFeatureExtractor()
    foot = extractor.classify_kicking_foot_from_motion(
        frames=seq["frames"], left_ankles=seq["left_ankles"], right_ankles=seq["right_ankles"],
        left_hips=seq["left_hips"], right_hips=seq["right_hips"], kick_frame=kick, window=3)
    start = _estimate_runup_start(seq, kick_frame=kick, fps=source["fps"])
    result, warnings = _call_step_counter(extractor, seq, fps=source["fps"], kick_frame=kick,
                                        runup_start_frame=start["runup_start_frame"],
                                        kicking_foot=foot.get("kicking_foot", "unknown"))
    return {"step_count": result.get("step_count"), "kick_frame": kick,
            "step_frames": result.get("step_frames"), "warnings": result.get("warnings", []) + warnings,
            "runup_start": start, "human_verified": False}


def export_panel(source, output_dir):
    cap = cv2.VideoCapture(source["video_path"])
    frame_id = max(0, int(source.get("kick_frame") or source.get("shot_end_frame", 2) // 2) - 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Failed to decode QA frame")
    field = build_field_mask(frame)
    boxes = []
    for track in source["tracks"]:
        obs = min(track["observations"], key=lambda o: abs(o["frame"] - frame_id), default=None)
        if obs is not None and abs(obs["frame"] - frame_id) <= 1:
            boxes.append(obs["bbox"])
    bg, stats = background_feature_mask(frame, boxes)
    white, proposals = white_ball_proposals(frame, field)
    original = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(original, (x1, y1), (x2, y2), (50, 200, 255), 2)
    views = [original, cv2.cvtColor(field.region, cv2.COLOR_GRAY2BGR),
             cv2.bitwise_and(frame, frame, mask=bg), cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)]
    titles = [f"Original | frame {frame_id}", f"HSV field | usable={field.valid}",
              "Background feature mask (people excluded)", f"White candidates: {len(proposals)} (NOT labels)"]
    panels = []
    for view, title in zip(views, titles):
        tile = cv2.resize(view, (640, 360))
        tile = cv2.copyMakeBorder(tile, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(tile)
    sheet = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    path = output_dir / f"{source['clip_id']}.png"
    success, encoded = cv2.imencode(".png", sheet)
    if not success:
        raise ValueError("Image encoding failed")
    path.write_bytes(encoded.tobytes())
    return {"frame": frame_id, "field_valid": field.valid, "white_proposal_count": len(proposals),
            "background": stats, "path": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "outputs/step_diagnostics.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mask_optimization_2026-10-08")
    parser.add_argument("--clip-id", action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = args.output_dir / "qa"
    images.mkdir(exist_ok=True)
    data = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    if args.clip_id:
        data = [row for row in data if row["clip_id"] in args.clip_id]
    rows, details = [], []
    warnings = Counter()
    extractor = KickerFeatureExtractor()
    for index, source in enumerate(data, 1):
        print(f"[{index}/{len(data)}] {source['clip_id']}", flush=True)
        tracks = {int(t["track_id"]): t for t in source["tracks"]}
        kwargs = dict(shot_frame_count=source["shot_end_frame"], frame_height=source["frame_height"],
                      keypoint_conf=.15, extractor=extractor)
        base, base_summaries, _ = _analyse_tracks(tracks, **kwargs)
        mask_stats = annotate_track_field_support(source["video_path"], tracks, samples_per_track=3)
        masked, mask_summaries, qualified = _analyse_tracks(tracks, use_field_mask=True, **kwargs)
        result = recount(source, tracks, masked, mask_summaries)
        panel = export_panel(source, images)
        warnings.update(source.get("qa_warnings", []))
        row = {"clip_id": source["clip_id"], "baseline_selected_track": base,
               "mask_selected_track": masked, "selection_changed": base != masked,
               "baseline_step_count": source.get("step_count"), "mask_step_count": result["step_count"],
               "mask_qualified_candidate": qualified, **mask_stats,
               "human_verified": False, "accuracy_evaluated": False}
        rows.append(row)
        details.append({**row, "baseline_candidate_scores": list(base_summaries.values()),
                        "mask_candidate_scores": list(mask_summaries.values()),
                        "recount": result, "qa": panel})
    summary = {"clips": len(rows), "baseline_nonempty_counts": sum(r["baseline_step_count"] is not None for r in rows),
               "mask_nonempty_counts": sum(r["mask_step_count"] is not None for r in rows),
               "selection_changes": sum(r["selection_changed"] for r in rows),
               "sampled_frames": sum(r["sampled_frames"] for r in rows),
               "decoded_frames": sum(r["decoded_frames"] for r in rows),
               "usable_masks": sum(r["usable_masks"] for r in rows),
               "baseline_warning_counts": dict(warnings.most_common()),
               "human_verified_reference_rows": 0, "accuracy": None, "mae": None,
               "interpretation": "Mask usability and output changes are diagnostics, not correctness or improvement.",
               "pose_cache_sha256": hashlib.sha256(args.diagnostics.read_bytes()).hexdigest()}
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["clip_id"])
        writer.writeheader()
        writer.writerows(rows)
    for name, value in [("summary.json", summary), ("diagnostics.json", details)]:
        (args.output_dir / name).write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
