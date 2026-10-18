"""Replay landing rules on the frozen 39-event original-course pose cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shoe_landing_events import LandingConfig, propose_shoe_landings


METHOD_VERSION = "shoe-landing-maximal-stance-replay-v1"
MANIFEST = ROOT / "data/course_context_clips_2026-10-12/source_manifest.json"
MANIFEST_SHA256 = "1affeb149a6fffea0a4b1814ec7b10be0c75ec8c0f2e6428cc920e62d325a38c"
CODE_FILES = ("scripts/replay_shoe_landings.py", "src/shoe_landing_events.py")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _course_sources():
    _require(sha(MANIFEST) == MANIFEST_SHA256, "Frozen original-course manifest hash mismatch")
    sources = read(MANIFEST)["clips"]
    _require(len(sources) == 39 and len({s["clip_id"] for s in sources}) == 39,
             "The original course cohort must contain exactly 39 events")
    return {source["clip_id"]: source for source in sources}


def _indexed_rows(document, ids, label):
    rows = document.get("rows")
    _require(document.get("events") == 39 and isinstance(rows, list) and len(rows) == 39,
             f"{label} must contain the complete 39-event course cohort")
    _require(all(isinstance(row, dict) for row in rows), f"Invalid {label} row")
    _require(len({row.get("clip_id") for row in rows}) == 39
             and {row.get("clip_id") for row in rows} == ids,
             f"{label} has missing, duplicate or external course events")
    return {row["clip_id"]: row for row in rows}


def _snapshot_hashes(directory, protocol):
    hashes = protocol.get("code_sha256")
    _require(isinstance(hashes, dict) and hashes, "Missing frozen producer code hashes")
    for name, digest in hashes.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "Invalid frozen source snapshot path")
        snapshot = directory / "source_snapshot" / relative
        _require(snapshot.is_file() and sha(snapshot) == digest,
                 f"Frozen producer source snapshot hash mismatch: {name}")


def _flags(document, label):
    _require(document.get("original_course_videos_only") is True
             and document.get("additional_video_used") is False
             and document.get("human_verified") is False
             and document.get("held_out_test") is False,
             f"{label} lacks original-course development scope")


def _support(result):
    count, landings = result.get("candidate_count"), result.get("landings")
    if count is None:
        _require(landings is None, "Unknown landing count must retain null landings")
        return None, None
    _require(type(count) is int and count >= 0 and isinstance(landings, list)
             and count == len(landings), "Landing count and event list disagree")
    frames, seconds = [], []
    for event in landings:
        stance = event.get("stance_frames")
        _require(isinstance(stance, list) and len(stance) >= 3
                 and all(type(frame) is int and frame >= 0 for frame in stance)
                 and stance == sorted(set(stance)), "Invalid frozen stance support")
        frames.append(len(stance))
        seconds.append((stance[-1] - stance[0]) / result["fps"])
    return frames, seconds


def _validate(dense_dir, baseline_dir):
    sources = _course_sources()
    ids = set(sources)
    dense_protocol = read(dense_dir / "protocol.json")
    dense_summary = read(dense_dir / "summary.json")
    baseline_protocol = read(baseline_dir / "protocol.json")
    baseline_summary = read(baseline_dir / "summary.json")
    for document, label in ((dense_protocol, "Dense protocol"), (baseline_protocol, "Baseline protocol"),
                            (dense_summary, "Dense summary"), (baseline_summary, "Baseline summary")):
        _flags(document, label)
    _require(dense_protocol.get("events") == 39 and dense_protocol.get("pilot") is False
             and dense_protocol.get("plan_only") is False
             and dense_protocol.get("reference_labels_read") is False
             and dense_protocol.get("source_manifest_sha256") == MANIFEST_SHA256
             and dense_summary.get("source_manifest_sha256") == MANIFEST_SHA256
             and len(dense_protocol.get("selected_clip_ids", [])) == 39
             and set(dense_protocol["selected_clip_ids"]) == ids,
             "Replay requires the full frozen original-course inference cohort")
    _require(baseline_protocol.get("dense_protocol_sha256") == sha(dense_dir / "protocol.json")
             and baseline_protocol.get("dense_summary_sha256") == sha(dense_dir / "summary.json")
             and baseline_protocol.get("reference_labels_read") is False,
             "Baseline analysis does not bind the supplied dense cohort")
    _require(baseline_protocol.get("landing_settings") == asdict(LandingConfig()),
             "Replay must preserve the frozen landing thresholds")
    _snapshot_hashes(dense_dir, dense_protocol)
    _snapshot_hashes(baseline_dir, baseline_protocol)
    dense_rows = _indexed_rows(dense_summary, ids, "Dense summary")
    baseline_rows = _indexed_rows(baseline_summary, ids, "Baseline summary")
    validated = []
    for cid, source in sorted(sources.items(), key=lambda item: item[1]["event_id"]):
        entry, baseline_row = dense_rows[cid], baseline_rows[cid]
        path = baseline_dir / "events" / cid / "landing_result.json"
        _require(sha(path) == baseline_row.get("result_sha256"),
                 f"Baseline prediction hash mismatch: {cid}")
        old = read(path)
        _require(old.get("clip_id") == cid and old.get("source_mapping") == source
                 and old.get("source_sha256") == entry.get("source_sha256") == source["clip_sha256"]
                 and old.get("fps") == source["fps"]
                 and old.get("frame_count") == source["frame_count"]
                 and old.get("human_verified") is False
                 and old.get("accuracy_validated") is False
                 and old.get("held_out_test") is False
                 and old.get("reference_labels_used_for_prediction") is False
                 and old.get("full_approach_visible") is False
                 and old.get("count_complete") is False
                 and old.get("total_run_up_steps") is None,
                 f"Baseline metadata does not match the frozen course source: {cid}")
        _require(entry.get("event_id") == baseline_row.get("event_id") == source["event_id"]
                 and baseline_row.get("baseline_ankle_count") == old.get("baseline_ankle_count")
                 and baseline_row.get("baseline_foot") == old.get("baseline_foot")
                 and baseline_row.get("dense_foot") == old.get("striking_foot")
                 and baseline_row.get("visible_transition_proposals") == old.get("candidate_count")
                 and baseline_row.get("status") == old.get("status"),
                 f"Baseline summary disagrees with its prediction: {cid}")
        _require(old.get("baseline_foot") in ("left", "right", "unknown")
                 and old.get("striking_foot") in ("left", "right", "unknown"),
                 f"Invalid inherited foot result: {cid}")
        _support(old)
        inputs, records = None, None
        if entry.get("status") == "fresh_dense_pose_candidate":
            event_dir = dense_dir / "events" / cid
            inputs_path, poses_path = event_dir / "inputs.json", event_dir / "poses33.json"
            _require(sha(inputs_path) == entry.get("inputs_sha256")
                     and sha(poses_path) == entry.get("poses33_sha256"),
                     f"Dense input or pose hash mismatch: {cid}")
            inputs, records = read(inputs_path), read(poses_path)
            interval = inputs.get("interval_frames")
            _require(inputs.get("source_mapping") == source and inputs.get("fps") == source["fps"]
                     and inputs.get("source_manifest_sha256") == MANIFEST_SHA256
                     and inputs.get("poses33_sha256") == entry["poses33_sha256"]
                     and inputs.get("model_sha256") == dense_protocol.get("model", {}).get("sha256")
                     and inputs.get("code_sha256") == dense_protocol["code_sha256"]
                     and inputs.get("reference_labels_read") is False
                     and inputs.get("new_neural_inference") is True
                     and inputs.get("human_verified") is False
                     and inputs.get("coordinate_space") == "original_source_image_pixels"
                     and inputs.get("interval_endpoints") == "inclusive"
                     and isinstance(interval, list) and len(interval) == 2
                     and all(type(frame) is int for frame in interval)
                     and 0 <= interval[0] <= interval[1] < source["frame_count"]
                     and interval == entry.get("interval_frames")
                     and old.get("observable_window") == dict(start_frame=interval[0], end_frame=interval[1])
                     and old.get("dense_inputs_sha256") == entry["inputs_sha256"]
                     and old.get("dense_poses_sha256") == entry["poses33_sha256"],
                     f"Dense predictions lack exact source-only input bindings: {cid}")
        else:
            _require(entry.get("status") == "unavailable" and old.get("status") == "unavailable"
                     and old.get("candidate_count") is None and old.get("observable_window") is None
                     and entry.get("reason") == old.get("reason"),
                     f"Unavailable source event cannot be fabricated or filled: {cid}")
        validated.append((source, entry, old, baseline_row["result_sha256"], inputs, records))
    return validated, dense_summary


def replay(args):
    dense_dir, baseline_dir, output_dir = map(Path, (args.dense_dir, args.baseline_analysis_dir, args.output_dir))
    if output_dir.exists():
        raise FileExistsError("Replay output already exists; choose a new directory")
    _require(not any(output_dir.resolve().is_relative_to(directory.resolve())
                     or directory.resolve().is_relative_to(output_dir.resolve())
                     for directory in (dense_dir, baseline_dir)),
             "Replay output must be separate from frozen inputs")
    validated, dense_summary = _validate(dense_dir, baseline_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    code_hashes = {}
    for name in CODE_FILES:
        target = output_dir / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        code_hashes[name] = sha(target)
    protocol = dict(schema_version=METHOD_VERSION, method_version=METHOD_VERSION,
        source_manifest_sha256=MANIFEST_SHA256,
        dense_protocol_sha256=sha(dense_dir / "protocol.json"),
        dense_summary_sha256=sha(dense_dir / "summary.json"),
        baseline_protocol_sha256=sha(baseline_dir / "protocol.json"),
        baseline_summary_sha256=sha(baseline_dir / "summary.json"), code_sha256=code_hashes,
        landing_settings=asdict(LandingConfig()), landing_thresholds_changed=False,
        method_change="retain maximal supported stance runs before matching normal and mirror proposals",
        new_neural_inference=False, actual_new_neural_calls=0,
        original_dense_inference_calls=dense_summary["actual_inference_calls"],
        reference_labels_read=False, human_verified=False, held_out_test=False,
        accuracy_validated=False, original_course_videos_only=True, additional_video_used=False,
        foot_results_recomputed=False, foot_results_source="frozen baseline analysis, inherited unchanged",
        original_source_pixels_redecoded=False, main_feature_table_modified=False,
        scope="Complete 39-event course cohort; deterministic replay of saved source-coordinate poses only")
    write(output_dir / "protocol.json", protocol)
    rows = []
    for source, entry, old, old_hash, inputs, records in validated:
        cid = source["clip_id"]
        if inputs is None:
            result = deepcopy(old)
            result.update(dense_foot=deepcopy(old.get("dense_foot")), paired_unique_pose_frames=[])
        else:
            landing = propose_shoe_landings(records, inputs["fps"], inputs["interval_frames"],
                inputs["original_camera_motion"], inputs["decoded_frame_hashes"])
            result = {**deepcopy(old), **landing, "striking_foot": old["striking_foot"],
                      "dense_foot": deepcopy(old["dense_foot"]),
                      "diagnostic_counts": dict(Counter(item["reason"] for item in landing["diagnostics"]))}
        result.update(replay_method_version=METHOD_VERSION, baseline_analysis_result_sha256=old_hash,
                      new_neural_inference=False, actual_new_neural_calls=0,
                      foot_results_recomputed=False, foot_result_inherited=True,
                      foot_results_source="frozen baseline analysis, inherited unchanged",
                      human_verified=False, accuracy_validated=False, held_out_test=False)
        old_frames, old_seconds = _support(old)
        new_frames, new_seconds = _support(result)
        comparison = dict(old_visible_transition_proposals=old["candidate_count"],
            new_visible_transition_proposals=result["candidate_count"],
            old_stance_frame_counts=old_frames, new_stance_frame_counts=new_frames,
            old_stance_seconds=old_seconds, new_stance_seconds=new_seconds)
        result["baseline_landing_comparison"] = comparison
        target = output_dir / "events" / cid / "landing_result.json"
        write(target, result)
        rows.append(dict(clip_id=cid, event_id=source["event_id"], status=result["status"],
            baseline_status=old["status"], baseline_ankle_count=old["baseline_ankle_count"],
            visible_transition_proposals=result["candidate_count"],
            **comparison, baseline_foot=old["baseline_foot"], dense_foot=old["striking_foot"],
            foot_result_inherited=True, paired_unique_pose_frames=len(result.get("paired_unique_pose_frames", [])),
            baseline_result_sha256=old_hash, result_sha256=sha(target)))
    with (output_dir / "comparison_39.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                         for key, value in row.items()} for row in rows)
    summary = dict(events=39, rows=rows, method_version=METHOD_VERSION,
        status_counts=dict(Counter(row["status"] for row in rows)),
        events_with_stance_transition_proposals=sum((row["visible_transition_proposals"] or 0) > 0 for row in rows),
        proposal_count_available=sum(row["visible_transition_proposals"] is not None for row in rows),
        proposal_total=sum(row["visible_transition_proposals"] or 0 for row in rows),
        baseline_proposal_total=sum(row["old_visible_transition_proposals"] or 0 for row in rows),
        changed_count_events=[row["clip_id"] for row in rows
                              if row["old_visible_transition_proposals"] != row["visible_transition_proposals"]],
        changed_stance_support_events=[row["clip_id"] for row in rows
                                      if row["old_stance_frame_counts"] != row["new_stance_frame_counts"]
                                      or row["old_stance_seconds"] != row["new_stance_seconds"]],
        paired_unique_pose_frame_total=sum(row["paired_unique_pose_frames"] for row in rows),
        dense_foot_counts=dict(Counter(row["dense_foot"] for row in rows)),
        foot_results_recomputed=False, foot_results_inherited_unchanged=True,
        actual_new_neural_calls=0, original_dense_inference_calls=dense_summary["actual_inference_calls"],
        independently_reviewed_events=0, accuracy=None, step_count_mae=None,
        human_verified=False, accuracy_validated=False, held_out_test=False,
        original_course_videos_only=True, additional_video_used=False,
        landing_thresholds_changed=False, reference_labels_read=False,
        interpretation="Rule-support repair on frozen poses; no measured accuracy gain or complete step counts")
    write(output_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dense-dir", "baseline-analysis-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
