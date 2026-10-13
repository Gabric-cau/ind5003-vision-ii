"""Bind V2 and guarded shoe-region producers, then replay conservative selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.final_results import sha256
from src.fresh_results import bind_stage, read_json, write_json
from src.shoe_pose33_fusion import SCHEMA, artifact, select_candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clips-dir", "pose-dir", "ball-dir", "foot-v2-dir", "foot-neural-dir", "shoe-dir", "model", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a new fusion output directory")
    manifest_path = args.clips_dir / "source_manifest.json"
    sources = read_json(manifest_path)["clips"]
    if len(sources) != 39 or len({row["clip_id"] for row in sources}) != 39:
        raise ValueError("Fusion requires the exact 39-event source cohort")
    cohort_protocol = read_json(args.shoe_dir / "protocol.json")
    if sha256(args.model) != cohort_protocol["model_sha256"]:
        raise ValueError("Fusion model bytes differ from guarded shoe cohort")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results").mkdir()
    snapshots = ("src/shoe_pose33_fusion.py", "scripts/replay_fresh_foot_pose33_fusion.py",
                 "src/fresh_results.py", "docs/SHOE_POSE33_FUSION_PROTOCOL_2026-10-13.md")
    hashes = {}
    for name in snapshots:
        dest = args.output_dir / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        hashes[name] = sha256(dest)
    manifest_hash = sha256(manifest_path)
    write_json(args.output_dir / "protocol.json", dict(schema_version=SCHEMA,
        source_manifest_sha256=manifest_hash, code_sha256=hashes,
        shoe_cohort_protocol=artifact(args.shoe_dir / "protocol.json"),
        shoe_cohort_summary=artifact(args.shoe_dir / "summary.json"),
        actual_model_sha256=sha256(args.model), new_neural_inference_by_fusion=False,
        selection_protocol_written_before_fusion_results=True, development_selection_rule=True, held_out_test=False,
        development_data_previously_inspected=True, reference_labels_read=False,
        human_verified=False, accuracy_validated=False))
    rows = []
    for source in sources:
        cid = source["clip_id"]
        if sha256(args.clips_dir / source["clip_filename"]) != source["clip_sha256"]:
            raise ValueError(f"Actual fusion source video mismatch: {cid}")
        pose_path = args.pose_dir / f"{cid}.json"
        old_path = args.foot_v2_dir / "results" / f"{cid}.json"
        old = read_json(old_path)
        event_dir = args.shoe_dir / "events" / cid
        bindings = dict(v2_result=artifact(old_path),
            original_neural_result=artifact(args.foot_neural_dir / "results" / f"{cid}.json"),
            pose=artifact(pose_path), source_manifest=artifact(manifest_path),
            shoe_cohort_protocol=artifact(args.shoe_dir / "protocol.json"),
            shoe_cohort_summary=artifact(args.shoe_dir / "summary.json"))
        region, swing = "unknown", "unknown"
        frames = []
        if (event_dir / "result.json").is_file():
            result = read_json(event_dir / "result.json")
            protocol = read_json(event_dir / "protocol.json")
            region = result["source_coordinate_shoe_contact_region_candidate"]
            swing = result["source_coordinate_shoe_swing_candidate"]
            frames = result["joint_near_shoe_and_paired_pose_frames"]
            bindings.update(shoe_result=artifact(event_dir / "result.json"),
                shoe_protocol=artifact(event_dir / "protocol.json"),
                shoe_poses33=artifact(event_dir / "poses33.json"),
                shoe_roi_plan=artifact(event_dir / "roi_plan.json"),
                roi_cache=artifact(args.foot_neural_dir / "poses" / f"{cid}.json"),
                contact=artifact(args.ball_dir / "contacts" / f"{cid}.json"),
                shoe_code_snapshot={name: artifact(event_dir / "source_snapshot" / name)
                                    for name in protocol["code_sha256"]})
        else:
            bindings["shoe_unavailable"] = artifact(event_dir / "unavailable.json")
        candidate, reason = select_candidate(old["kicking_foot_candidate"], region)
        data = dict(schema_version=SCHEMA, clip_id=cid, source_mapping=source,
            kicking_foot_candidate=candidate, status="completed_conservative_foot_fusion", reason=reason,
            new_neural_inference=False, human_verified=False, accuracy_validated=False,
            held_out_test=False, reference_labels_used_for_prediction=False, shoe_ball_contact_verified=False,
            inference_identity=dict(source_sha256=source["clip_sha256"], pose_sha256=sha256(pose_path),
                source_manifest_sha256=manifest_hash, code_sha256=hashes),
            temporal_evidence=dict(schema_version=SCHEMA, v2_candidate=old["kicking_foot_candidate"],
                shoe_region_candidate=region, shoe_swing_diagnostic_candidate=swing,
                selection_reason=reason, joint_shoe_region_frames=frames, producer_bindings=bindings,
                shoe_ball_contact_verified=False, anatomical_side_verified=False,
                narrow_strike_window_diagnostic_used_as_vote=False))
        bind_stage("foot", data, source, sha256(pose_path), manifest_hash)
        write_json(args.output_dir / "results" / f"{cid}.json", data)
        rows.append(dict(clip_id=cid, candidate=candidate, v2_candidate=old["kicking_foot_candidate"],
                         shoe_region_candidate=region, shoe_swing_diagnostic_candidate=swing, reason=reason))
    summary = dict(events=39, rows=rows, candidate_counts=dict(Counter(row["candidate"] for row in rows)),
        v2_candidate_counts=dict(Counter(row["v2_candidate"] for row in rows)),
        shoe_region_candidate_counts=dict(Counter(row["shoe_region_candidate"] for row in rows)),
        selection_reason_counts=dict(Counter(row["reason"] for row in rows)),
        fallback_events=[row["clip_id"] for row in rows if row["reason"] == "shoe_region_fallback_for_v2_unknown"],
        known_overlap_events=[row["clip_id"] for row in rows if row["v2_candidate"] != "unknown"
                              and row["shoe_region_candidate"] != "unknown"],
        conflict_events=[row["clip_id"] for row in rows if row["reason"] == "v2_and_shoe_region_conflict"],
        new_neural_inference_by_fusion=False, reference_labels_read=False, human_verified=False,
        accuracy_validated=False, held_out_test=False,
        interpretation="Development candidate availability; source evidence and anatomical/contact accuracy remain unverified")
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return summary


if __name__ == "__main__":
    main()
