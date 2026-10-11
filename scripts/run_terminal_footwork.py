"""Reanalyse frozen pose/contact evidence without replacing historical results."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.motion_stabilization import apply_transforms
from src.terminal_footwork import TerminalConfig, analyze_terminal_footwork
from scripts.run_penalty_phase_experiment import sequence

UPSTREAM = ROOT / "outputs/phase_optimization_2026-10-09"
OUTPUT = ROOT / "outputs/terminal_footwork_2026-10-09"
PREFIXES = ("col_eng_p", "cro_den_p", "esp_rus_p", "rus_cro_p")


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def brief(result):
    return {"status": result["terminal"]["status"], "start_frame": result["terminal"]["start_frame"],
            "tags": result["terminal"]["tags"], "counts": result["counts"], "warnings": result["warnings"]}


def analyse(path, upstream):
    result = load(path)
    cid = result["clip_id"]
    for name, expected in result["provenance"]["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input code changed: {name}; regenerate explicitly, not silently")
    pose_path = ROOT / "outputs/shot_repair_2026-10-09/all_clips_final" / f"{cid}.json"
    expected = result["provenance"]["input_pose_sha256"]
    if sha(pose_path) != expected:
        raise ValueError(f"Original pose hash mismatch: {cid}")
    if result.get("shot_rescue_used"):
        pose_path = Path(result["provenance"]["override_pose_path"])
        expected = result["provenance"]["override_pose_sha256"]
    if result["pose_source"].startswith("yolo11m"):
        pose_path = upstream / "refined_pose" / f"{cid}.json"
        expected = result["provenance"]["refined_pose_sha256"]
    if sha(pose_path) != expected:
        raise ValueError(f"Selected pose hash mismatch: {cid}")
    pose = load(pose_path)
    tid = result["contact"]["contact_associated_track_id"]
    track = next((t for t in pose["tracks"] if t["track_id"] == tid), {"observations": []})
    seq = sequence(track)
    camera_path = upstream / "camera_motion" / f"{cid}.json"
    motion = load(camera_path)
    if motion["source_sha256"] != sha(ROOT / "src/motion_stabilization.py") or motion["shot_interval"] != [result["shot_start_frame"], result["shot_end_frame_exclusive"]]:
        raise ValueError(f"Camera evidence mismatch: {cid}")
    stabilized = apply_transforms(seq["hips"], seq["frames"], motion["transforms"]) if seq["frames"] and motion["status"] == "supported" else None
    kwargs = dict(frame_ids=seq["frames"], fps=result["fps"], stabilized_hips=stabilized,
                  box_heights=seq["heights"], phase_result=result["phase_result"], contact=result["contact"],
                  camera_status=motion["status"],
                  candidate_axis_reliable=result["event_proposal"].get("axis_energy_fraction", 0) >= .65)
    variants = result["event_proposal"].get("variants", {})
    central_events = variants.get("central", [])
    new = analyze_terminal_footwork(**kwargs, events=central_events)
    event_sensitivity = {name: brief(analyze_terminal_footwork(**kwargs, events=variants.get(name, [])))
                         for name in ("sensitive", "central", "strict")}
    cfg = TerminalConfig()
    profiles = {"permissive": replace(cfg, rhythm_short_ratio=.85, rhythm_long_ratio=1.3,
                    min_rhythm_change_seconds=.04, progress_short_ratio=.75, min_progress_change_box_heights=.04,
                    min_split_gain=.15),
                "central": cfg,
                "conservative": replace(cfg, rhythm_short_ratio=.70, rhythm_long_ratio=1.65,
                    min_rhythm_change_seconds=.08, progress_short_ratio=.55, min_progress_change_box_heights=.08,
                    min_split_gain=.35)}
    rule_sensitivity = {name: {**brief(analyze_terminal_footwork(**kwargs, events=central_events, config=config)),
                              "thresholds": config.__dict__} for name, config in profiles.items()}
    contacts = {}
    window = result["contact"].get("interval_frames")
    if window and kwargs["phase_result"]["kick_anchor"]["usable"]:
        for name, frame in (("early", window[0]), ("nominal", result["contact"]["contact_frame"]), ("late", window[1])):
            contact = {**result["contact"], "contact_frame": frame}
            contacts[name] = brief(analyze_terminal_footwork(**{**kwargs, "contact": contact}, events=central_events))
    full = [*event_sensitivity.values(), *rule_sensitivity.values(), *contacts.values()]
    signatures = {(r["status"], tuple(r["tags"])) for r in full}
    starts = [r["start_frame"] for r in full if r["start_frame"] is not None]
    new["sensitivity"] = {"event_prominence": event_sensitivity, "change_rules": rule_sensitivity,
        "contact_anchor": contacts, "status_and_tags_agree": len(signatures) == 1,
        "candidate_start_range_frames": [min(starts), max(starts)] if starts else None,
        "meaning": "Fixed parameter/anchor variants, not accuracy or a confidence interval; default is never selected by user count."}
    if len(signatures) != 1:
        new["warnings"].append("terminal_classification_parameter_sensitive")
    if starts and max(starts) - min(starts) > round(.15 * result["fps"]):
        new["warnings"].append("terminal_boundary_parameter_sensitive")
    result["schema_version"] = "penalty-phase-experiment-v2-terminal-footwork"
    result["footwork_result"] = new
    result["terminal_provenance"] = {"source_result_path": str(path), "source_result_sha256": sha(path),
        "selected_pose_path": str(pose_path), "selected_pose_sha256": sha(pose_path),
        "camera_path": str(camera_path), "camera_sha256": sha(camera_path),
        "code_sha256": {name: sha(ROOT / name) for name in ("src/terminal_footwork.py", "scripts/run_terminal_footwork.py")},
        "mode": "cached_inference_reanalysis_no_retraining", "human_labels_used": False}
    return result


def summarize(results):
    return {"clips": len(results), "human_verified": False, "accuracy_validated": False,
        "terminal_status": dict(Counter(r["footwork_result"]["terminal"]["status"] for r in results)),
        "terminal_evidence_breakdown": {
            "upstream_approach_or_contact_unresolved": sum(r["footwork_result"]["status"] == "abstained" for r in results),
            "upstream_usable_but_terminal_comparison_insufficient": sum(r["footwork_result"]["status"] == "insufficient_terminal_evidence" for r in results),
            "terminal_comparison_available": sum(r["footwork_result"]["status"] == "provisional" for r in results)},
        "route_status": dict(Counter(r["footwork_result"]["route"]["status"] for r in results)),
        "parameter_sensitive_clips": sum(not r["footwork_result"]["sensitivity"]["status_and_tags_agree"] for r in results),
        "nominal_counts_preserved": all(r["footwork_result"]["counts"]["nominal_approach_candidates"] ==
            r["count_sensitivity"].get("central", {}).get("approach_candidates_including_precontact_strike_window") for r in results),
        "rows": [{"clip_id": r["clip_id"], "contact_status": r["contact"]["status"],
                  "route": r["footwork_result"]["route"], "terminal": brief(r["footwork_result"]),
                  "parameter_sensitive": not r["footwork_result"]["sensitivity"]["status_and_tags_agree"]} for r in results],
        "interpretation": "Separate projected route, terminal rhythm changes, and contact ambiguity; not verified ankle control or accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--clip-id", action="append")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.upstream_dir.resolve():
        parser.error("Do not overwrite frozen input results")
    for folder in ("classified", "classified_refined"):
        paths = [p for p in sorted((args.upstream_dir / folder).glob("*.json")) if p.stem.startswith(PREFIXES)]
        if args.clip_id:
            paths = [p for p in paths if p.stem in args.clip_id]
            if len(paths) != len(set(args.clip_id)):
                parser.error("Unknown clip ID")
        if not paths:
            parser.error(f"No frozen inputs: {folder}")
        for path in paths:
            result = analyse(path, args.upstream_dir)
            save(args.output_dir / folder / path.name, result)
            f = result["footwork_result"]
            print(folder, path.stem, f["terminal"]["status"], f["terminal"]["start_frame"], f["terminal"]["tags"], flush=True)
        results = [load(p) for p in sorted((args.output_dir / folder).glob("*.json")) if p.stem.startswith(PREFIXES)]
        save(args.output_dir / folder / "summary.json", summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
