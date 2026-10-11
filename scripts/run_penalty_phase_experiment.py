"""Versioned, non-destructive movement-phase experiment from cached detections.

Run independent contact probes first. This script never imports human-like
historical labels as truth and never replaces the project's old result table.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from src.motion_stabilization import apply_transforms, estimate_background_motion
from src.penalty_phases import classify_penalty_phases
from src.phase_step_events import propose_step_events

RUN = ROOT / "outputs/phase_optimization_2026-10-09"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence(track):
    observations = sorted(track.get("observations", []), key=lambda x: x["frame"])
    seq = {k: [] for k in ("frames", "hips", "left_ankles", "right_ankles", "heights", "confidences")}
    for item in observations:
        kp, conf = np.asarray(item["keypoints"], float), np.asarray(item["keypoint_confidences"], float)
        if kp.shape != (17, 2) or conf.shape != (17,):
            continue
        seq["frames"].append(item["frame"])
        seq["hips"].append(((kp[11] + kp[12]) / 2).tolist())
        seq["left_ankles"].append(kp[15].tolist())
        seq["right_ankles"].append(kp[16].tolist())
        seq["heights"].append(item["bbox"][3] - item["bbox"][1])
        c = [min(conf[11], conf[12]), conf[15], conf[16]]
        if not np.isfinite(kp[[11, 12, 15, 16]]).all():
            c = [0., 0., 0.]
        seq["confidences"].append([float(x) for x in c])
    return seq


def classify(seq, contact, motion, events, *, stabilized=True):
    anchor = contact.get("contact_frame") if contact["status"] in ("supported", "tentative") else None
    score = contact.get("confidence", 0) if anchor is not None else 0
    use_stabilized = stabilized and motion["status"] == "supported"
    return classify_penalty_phases(seq["frames"], contact["fps"], seq["hips"], seq["left_ankles"],
        seq["right_ankles"], seq["heights"], confidences=np.asarray(seq["confidences"]).reshape(-1, 3),
        stabilized_hips=apply_transforms(seq["hips"], seq["frames"], motion["transforms"]) if use_stabilized and seq["frames"] else None,
        camera_motion_status=motion["status"] if stabilized else "unknown",
        kick_frame=anchor, kick_confidence=score, step_events=events)


def counts_for(events, phase, contact_window=None):
    labels = {r["frame"]: r["phase"] for r in phase["frame_labels"]}
    counts = Counter(labels.get(e["frame"], "unknown") for e in events)
    onset, kick = phase["approach_start_frame"], phase["kick_anchor"]["frame"]
    early, late = contact_window if contact_window else (kick, kick)
    return {"by_phase": dict(counts), "all_visible_candidate_events": len(events),
            "approach_candidates_including_precontact_strike_window":
                sum(onset <= e["frame"] < kick for e in events) if onset is not None and kick is not None else None,
            "preparation_candidates": counts.get("preparation_adjustment", 0) if onset is not None else None,
            "contact_window_ambiguous_candidates": sum(early <= e["frame"] <= late for e in events) if early is not None else None,
            "approach_count_with_early_contact_boundary": sum(onset <= e["frame"] < early for e in events) if onset is not None and early is not None else None,
            "approach_count_with_late_contact_boundary": sum(onset <= e["frame"] <= late for e in events) if onset is not None and late is not None else None,
            "unknown_candidates": counts.get("unknown", 0)}


def render_sheet(result, observations, destination):
    phase = result["phase_result"]
    target_frames = [s["start_frame"] for s in phase["segments"] if s["phase"] != "unknown"]
    target_frames += [e["frame"] for e in phase["events"]]
    if result["contact"]["contact_frame"] is not None:
        target_frames.append(result["contact"]["contact_frame"])
    target_frames = sorted(set(target_frames))
    if not target_frames:
        target_frames = [result["shot_start_frame"], result["shot_end_frame_exclusive"] - 1]
    # A bounded overview, not an assertion that all frames were manually checked.
    if len(target_frames) > 18:
        target_frames = [target_frames[i] for i in np.linspace(0, len(target_frames) - 1, 18).astype(int)]
    by_frame = {o["frame"]: o for o in observations}
    labels = {x["frame"]: x["phase"] for x in phase["frame_labels"]}
    cap = cv2.VideoCapture(result["video_path"])
    panels = []
    for fid in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        ratio = 480 / frame.shape[1]
        frame = cv2.resize(frame, (480, 270))
        o = by_frame.get(fid)
        if o:
            box = np.round(np.asarray(o["bbox"]) * ratio).astype(int)
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 225, 255), 2)
            for k in (11, 12, 15, 16):
                if o["keypoint_confidences"][k] >= .35:
                    cv2.circle(frame, tuple(np.round(np.array(o["keypoints"][k]) * ratio).astype(int)), 3, (80, 50, 255), -1)
        frame = cv2.copyMakeBorder(frame, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
        label = labels.get(fid, "unknown")
        cv2.putText(frame, f"f{fid} {fid/result['fps']:.2f}s {label}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        panels.append(frame)
    cap.release()
    if panels:
        if len(panels) % 2:
            panels.append(np.zeros_like(panels[0]))
        sheet = np.vstack([np.hstack(panels[i:i+2]) for i in range(0, len(panels), 2)])
        ok, data = cv2.imencode(".jpg", sheet)
        if not ok:
            raise ValueError("QA encoding failed")
        destination.write_bytes(data.tobytes())


def process(path, args):
    old = load(path)
    baseline = old
    cid = old["clip_id"]
    cp = args.contact_dir / f"{cid}.json"
    override_pose = args.rescue_dir / "poses" / f"{cid}.json"
    override_contact = args.rescue_dir / "contacts" / f"{cid}.json"
    rescue_used = override_pose.exists() and override_contact.exists()
    if rescue_used:
        old = load(override_pose)
        cp = override_contact
    if not cp.exists():
        raise ValueError(f"Contact evidence missing: {cp}")
    contact = load(cp)
    track_id = contact.get("contact_associated_track_id")
    track = next((t for t in old["tracks"] if t["track_id"] == track_id), {"observations": []})
    source = "cached_full_frame_yolo11n_pose_contact_associated_track"
    refined_path = args.refined_dir / f"{cid}.json" if args.refined_dir else None
    if refined_path and refined_path.exists():
        refined = load(refined_path)
        if refined.get("status") == "refined" and refined.get("contact_associated_track_id") == track_id:
            candidate = next((t for t in refined["tracks"] if t["track_id"] == track_id), None)
            if candidate:
                track = candidate
                source = "yolo11m_pose_contact_associated_crop_ablation"
    start, end = old["shot_start_frame"], old["shot_end_frame"]
    motion_path = args.output_dir.parent / "camera_motion" / f"{cid}.json"
    camera_hash = digest(ROOT / "src/motion_stabilization.py")
    cached_motion = load(motion_path) if motion_path.exists() else {}
    if (not args.recompute_camera and cached_motion.get("source_sha256") == camera_hash
            and cached_motion.get("shot_interval") == [start, end]):
        motion = load(motion_path)
    else:
        boxes = {p["frame"]: [d["bbox"] for d in p["detections"]] for p in old["pose_frames"]}
        motion = estimate_background_motion(old["video_path"], start, end, boxes)
        motion["source_sha256"] = camera_hash
        motion["shot_interval"] = [start, end]
        save(motion_path, motion)
    seq = sequence(track)
    # Build no count at all when identity/contact association abstains.
    proposal = propose_step_events(seq["frames"], old["fps"], seq["hips"], seq["left_ankles"],
        seq["right_ankles"], seq["heights"], seq["confidences"], kick_frame=contact.get("contact_frame"),
        event_end_frame=(contact.get("interval_frames") or [None, None])[1]) if seq["frames"] else {"events": [], "variants": {}, "warnings": ["identity_or_contact_unresolved"]}
    phase = classify(seq, contact, motion, proposal["events"])
    unstabilized = classify(seq, contact, motion, proposal["events"], stabilized=False)
    for event in phase["events"]:
        bounds = contact.get("interval_frames")
        event["within_contact_uncertainty_window"] = bool(bounds and bounds[0] <= event["frame"] <= bounds[1])
        if event["within_contact_uncertainty_window"]:
            event["warnings"].append("support_foot_or_kicking_swing_cannot_be_separated_here")
    variants = {name: counts_for(events, phase, contact.get("interval_frames")) for name, events in proposal["variants"].items()}
    old_events = [{"frame": f, "foot": "unknown"} for f in baseline.get("step_frames", [])]
    old_same_identity = not rescue_used and baseline["selected_track_id"] == track_id
    legacy_attributed = counts_for(old_events, phase) if old_same_identity else None
    anchors = {key: contact.get(key) for key in ("status", "contact_frame", "contact_time_sec", "interval_frames", "confidence", "contact_associated_track_id", "warnings")}
    result = {"schema_version": "penalty-phase-experiment-v1", "clip_id": cid,
        "video_path": old["video_path"], "fps": old["fps"], "source_frame_count": old["source_frame_count"],
        "shot_start_frame": start, "shot_end_frame_exclusive": end,
        "human_verified": False, "accuracy_validated": False, "review_required": True,
        "pose_source": source, "contact": anchors, "phase_result": phase,
        "event_proposal": proposal, "count_sensitivity": variants,
        "camera_motion": {k: v for k, v in motion.items() if k not in ("transforms", "pairs")},
        "baseline": {"selected_track_id": baseline["selected_track_id"], "step_count": baseline["step_count"],
                     "step_frames": baseline["step_frames"], "kick_frame": baseline["kick_frame"],
                     "runup_start_frame": baseline["runup_start_frame"], "same_identity_as_contact_association": old_same_identity,
                     "legacy_events_relabelled_only_if_same_identity": legacy_attributed},
        "shot_rescue_used": rescue_used,
        "ablations": {"without_camera_compensation": {"status": unstabilized["status"],
            "approach_start_frame": unstabilized["approach_start_frame"], "style_tags": unstabilized["style_tags"],
            "phase_event_counts": unstabilized["phase_event_counts"]}},
        "review_overlay": [{"frame": o["frame"], "bbox": o["bbox"],
                            "ankles": [o["keypoints"][15], o["keypoints"][16]],
                            "ankle_confidences": [o["keypoint_confidences"][15], o["keypoint_confidences"][16]]}
                           for o in track["observations"]],
        "provenance": {"input_pose_sha256": digest(path), "input_contact_sha256": digest(cp),
            "code_sha256": {name: digest(ROOT / name) for name in (
                "src/motion_stabilization.py", "src/penalty_phases.py", "src/phase_step_events.py",
                "scripts/run_penalty_phase_experiment.py")},
            "annotation_source": "algorithmic_candidates_not_human_labels"}}
    if refined_path and refined_path.exists() and source.startswith("yolo11m"):
        result["provenance"]["refined_pose_sha256"] = digest(refined_path)
    if rescue_used:
        result["provenance"]["override_pose_sha256"] = digest(override_pose)
        result["provenance"]["override_pose_path"] = str(override_pose)
        result["provenance"]["override_contact_path"] = str(override_contact)
    if cid == "col_eng_p03_cuadrado_col11":
        result["user_reference"] = {"approximate_steps": 5, "acceptable_interval_this_clip_only": [4, 6],
            "exact_ground_truth": False, "event_frames_verified": False,
            "source": "User said: 5步吧，4-6步我觉得都正常"}
    save(args.output_dir / f"{cid}.json", result)
    render_sheet(result, track["observations"], args.output_dir / f"{cid}_qa.jpg")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", type=Path, default=ROOT / "outputs/shot_repair_2026-10-09/all_clips_final")
    parser.add_argument("--contact-dir", type=Path, default=RUN / "contact_probe")
    parser.add_argument("--output-dir", type=Path, default=RUN / "classified")
    parser.add_argument("--refined-dir", type=Path)
    parser.add_argument("--rescue-dir", type=Path, default=RUN / "shot_rescue")
    parser.add_argument("--clip-id", action="append")
    parser.add_argument("--recompute-camera", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.pose_dir.glob("*.json") if p.stem.startswith(("col_eng_p", "cro_den_p", "esp_rus_p", "rus_cro_p")))
    if args.clip_id:
        paths = [p for p in paths if p.stem in args.clip_id]
        if len(paths) != len(set(args.clip_id)):
            parser.error("Unknown clip ID")
    for i, path in enumerate(paths, 1):
        result = process(path, args)
        print(f"{i}/{len(paths)} {path.stem}: {result['contact']['status']} / {result['phase_result']['status']} {result['count_sensitivity']}", flush=True)
    results = [load(p) for p in args.output_dir.glob("*.json") if p.stem.startswith(("col_eng_p", "cro_den_p", "esp_rus_p", "rus_cro_p"))]
    summary = {"clips": len(results), "human_verified": False, "accuracy_validated": False,
        "contact_status": dict(Counter(r["contact"]["status"] for r in results)),
        "phase_status": dict(Counter(r["phase_result"]["status"] for r in results)),
        "camera_status": dict(Counter(r["camera_motion"]["status"] for r in results)),
        "changed_associated_identity": [r["clip_id"] for r in results if r["contact"]["contact_associated_track_id"] is not None and not r["baseline"]["same_identity_as_contact_association"]],
        "rows": [{"clip_id": r["clip_id"], "contact": r["contact"], "baseline": r["baseline"],
            "phase_status": r["phase_result"]["status"], "approach_start_frame": r["phase_result"]["approach_start_frame"],
            "style_tags": r["phase_result"]["style_tags"], "count_sensitivity": r["count_sensitivity"],
            "camera_status": r["camera_motion"]["status"], "pose_source": r["pose_source"]} for r in results],
        "interpretation": "Candidate segmentation and computational completeness, not verified count or classification accuracy."}
    save(args.output_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
