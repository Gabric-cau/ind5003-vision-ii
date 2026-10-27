"""Rebuild summaries from verified completed inference without running models."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_penalty_video import summarize
from scripts.evaluate_offline_video_batch import load_completed_batch, read, sha, source_path
from scripts.run_offline_video_batch import batch_summary, now, write_atomic


CODE_FILES = (
    "scripts/replay_offline_video_summaries.py", "scripts/analyze_penalty_video.py",
    "src/fresh_results.py", "scripts/evaluate_offline_video_batch.py",
    "scripts/run_offline_video_batch.py",
)


def code_hashes():
    return {name: sha(ROOT / name) for name in CODE_FILES}


def verified_event(directory, receipt, analysis):
    pipeline = source_path(directory, receipt["analysis_path"]).parent
    manifest_path = source_path(pipeline, "source/source_manifest.json")
    manifest = read(manifest_path)
    if manifest.get("manual_annotations_used") is not False or manifest.get("clips") != [analysis["source"]]:
        raise ValueError("Source manifest differs from the completed prediction")
    progress_path = source_path(pipeline, "progress.json")
    receipts = read(progress_path)["stages"]
    if receipts != analysis.get("stage_receipts") or any(r["exit_code"] != 0 for r in receipts):
        raise ValueError("Stage receipts differ from successful inference")
    proofs = {manifest_path: sha(manifest_path), progress_path: sha(progress_path)}
    artifact_bindings = analysis.get("artifact_bindings")
    if not artifact_bindings:
        raise ValueError("Completed prediction lacks reusable stage hashes")
    stage_directories = set()
    for binding in artifact_bindings.values():
        relative = Path(binding["path"])
        path = source_path(pipeline, relative)
        if len(relative.parts) < 2 or relative.parts[0] in ("source", "analysis.json"):
            raise ValueError("Invalid reusable stage directory")
        if sha(path) != binding["sha256"]:
            raise ValueError("Stage artifact changed after completed inference")
        proofs[path] = binding["sha256"]
        stage_directories.add(relative.parts[0])
    contact_path = source_path(pipeline, "trained_contact/prediction.json")
    if read(contact_path) != analysis.get("trained_contact"):
        raise ValueError("Trained contact differs from the completed prediction")
    proofs[contact_path] = sha(contact_path)
    stage_directories.add("trained_contact")
    video = source_path(pipeline, "source/" + analysis["source"]["clip_filename"])
    proofs[video] = analysis["source"]["clip_sha256"]
    return {"pipeline": pipeline, "proofs": proofs, "stage_directories": stage_directories,
            "receipts": receipts, "manifest": manifest_path, "video": video}


def recheck(proofs):
    if any(sha(path) != digest for path, digest in proofs.items()):
        raise ValueError("Original inference evidence changed during summary replay")


def replay(source_batch, output, replacement_batch=None):
    started = time.perf_counter()
    source_batch, output = Path(source_batch).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new summary replay directory")
    if output.is_relative_to(source_batch):
        raise ValueError("Summary replay cannot write inside its source batch")
    replacement_batch = Path(replacement_batch).resolve() if replacement_batch is not None else None
    if replacement_batch is not None and output.is_relative_to(replacement_batch):
        raise ValueError("Summary replay cannot write inside its replacement batch")
    predictions, samples, source_binding = load_completed_batch(source_batch)
    original_summary = read(source_batch / "summary.json")
    original_manifest = read(source_batch / "batch_manifest.json")
    events = {r["core_id"]: r for r in original_summary["events"]}
    plans = {cid: verified_event(source_batch, events[cid], analysis)
             for cid, analysis in predictions.items()}
    source_batches = {"base": {"directory": str(source_batch), **source_binding,
        "schema_version": original_manifest["bindings"].get("schema_version"),
        "created_at_utc": original_manifest.get("created_at_utc"),
        "bindings": copy.deepcopy(original_manifest["bindings"])}}
    selected_sources = {cid: "base" for cid in predictions}
    replaced_ids = []
    if replacement_batch is not None:
        replacement_predictions, replacement_samples, replacement_binding = load_completed_batch(replacement_batch)
        if not replacement_samples or not set(replacement_samples).issubset(samples):
            raise ValueError("Replacement events must be a nonempty subset of the base batch")
        if any(row["video_sha256"] != samples[cid]["video_sha256"]
               for cid, row in replacement_samples.items()):
            raise ValueError("Replacement event video hash differs from the base batch")
        replacement_manifest = read(replacement_batch / "batch_manifest.json")
        replacement_events = {r["core_id"]: r for r in read(replacement_batch / "summary.json")["events"]}
        replacement_plans = {cid: verified_event(replacement_batch, replacement_events[cid], analysis)
                             for cid, analysis in replacement_predictions.items()}
        source_batches["replacement"] = {"directory": str(replacement_batch), **replacement_binding,
            "schema_version": replacement_manifest["bindings"].get("schema_version"),
            "created_at_utc": replacement_manifest.get("created_at_utc"),
            "bindings": copy.deepcopy(replacement_manifest["bindings"])}
        predictions.update(replacement_predictions)
        events.update(replacement_events)
        plans.update(replacement_plans)
        replaced_ids = sorted(replacement_predictions)
        selected_sources.update({cid: "replacement" for cid in replaced_ids})
    event_sources = {cid: {"source_batch_key": key, "source_batch": source_batches[key]["directory"],
        "source_batch_sha256": source_batches[key]["batch_manifest_sha256"],
        "source_summary_sha256": source_batches[key]["summary_sha256"],
        "source_binding_schema_version": source_batches[key]["schema_version"],
        "source_created_at_utc": source_batches[key]["created_at_utc"],
        "original_analysis_path": events[cid]["analysis_path"],
        "original_analysis_sha256": events[cid]["analysis_sha256"],
        "original_analysis_schema_version": predictions[cid].get("schema_version"),
        "video_sha256": samples[cid]["video_sha256"]} for cid, key in selected_sources.items()}
    current_code = code_hashes()
    provenance = {"postprocess_only": True, "new_neural_inference": False,
        "manual_reference_labels_read": False, "source_batch": str(source_batch),
        "source_batch_sha256": source_binding["batch_manifest_sha256"],
        "source_summary_sha256": source_binding["summary_sha256"],
        "source_predictions": source_binding["events"], "replay_code_sha256": current_code,
        "source_batches": source_batches, "event_inference_sources": event_sources,
        "replacement_core_ids": replaced_ids, "reused_base_core_ids": sorted(set(samples) - set(replaced_ids)),
        "replacement_selection_policy": "explicit_replacement_batch_membership_without_reference_labels",
        "new_neural_inference_during_replay": False,
        "stage_storage": "relative_symlinks_to_verified_original_inference",
        "source_video_storage": "independent_verified_copy",
        "runtime_policy": "original_inference_and_current_summary_replay_recorded_separately"}
    output.mkdir(parents=True)
    manifest_path = output / "batch_manifest.json"
    bindings = copy.deepcopy(original_manifest["bindings"])
    if replacement_batch is not None:
        bindings = {"schema_version": "offline-video-integrated-batch-bindings-v1",
            "samples": copy.deepcopy(original_manifest["bindings"]["samples"]),
            "manual_labels_read_at_inference": False,
            "inference_source_policy": "per_event_original_completed_batch_bindings",
            "event_inference_sources": event_sources,
            "source_batches": source_batches}
    write_atomic(manifest_path, {"created_at_utc": now(),
        "bindings": bindings, "summary_replay": provenance})
    manifest_hash = sha(manifest_path)
    rows = []
    write_atomic(output / "summary.json", {**batch_summary(rows, len(samples), manifest_hash), **provenance})
    for cid in sorted(predictions):
        event_started, started_utc = time.perf_counter(), now()
        original, receipt, plan = predictions[cid], events[cid], plans[cid]
        pipeline = output / "events" / cid / "attempt_001" / "pipeline"
        source = pipeline / "source"
        source.mkdir(parents=True)
        for path in (plan["manifest"], plan["video"]):
            destination = source / path.name
            shutil.copy2(path, destination)
            if sha(destination) != plan["proofs"][path]:
                raise ValueError("Copied inference source changed")
        for name in sorted(plan["stage_directories"]):
            target = source_path(plan["pipeline"], name)
            if not target.is_dir():
                raise ValueError("Expected a reusable stage directory")
            (pipeline / name).symlink_to(os.path.relpath(target, pipeline), target_is_directory=True)
        shutil.copy2(plan["pipeline"] / "progress.json", pipeline / "progress.json")
        result = summarize(original["source"], pipeline, plan["receipts"])
        if result.get("all_stages_executed_successfully") is not True:
            raise ValueError("Rebuilt summary does not retain successful original processing")
        recheck(plan["proofs"])
        if code_hashes() != current_code:
            raise ValueError("Summary implementation changed during replay")
        elapsed = time.perf_counter() - event_started
        result.update(postprocess_only=True, new_neural_inference=False,
            end_to_end_seconds=original["end_to_end_seconds"],
            end_to_end_seconds_basis="original_inference_execution",
            original_inference_runtime_seconds=original["end_to_end_seconds"],
            original_inference_receipt_wall_seconds=receipt.get("wall_seconds"),
            summary_replay_seconds=elapsed, summary_replay={**provenance,
                **event_sources[cid],
                "original_analysis_path": receipt["analysis_path"],
                "original_analysis_sha256": receipt["analysis_sha256"]})
        write_atomic(pipeline / "analysis.json", result)
        new_receipt = {"schema_version": "offline-video-summary-replay-event-v1", "core_id": cid,
            "clip_id": receipt.get("clip_id"), "video_sha256": receipt["video_sha256"],
            "batch_bindings_sha256": manifest_hash, "attempt": 1, "status": "success", "exit_code": 0,
            "analysis_path": str((pipeline / "analysis.json").relative_to(output)),
            "analysis_sha256": sha(pipeline / "analysis.json"), "started_at_utc": started_utc,
            "finished_at_utc": now(), "wall_seconds": time.perf_counter() - event_started,
            "postprocess_only": True, "new_neural_inference": False,
            "original_inference_runtime_seconds": original["end_to_end_seconds"],
            "inference_source": event_sources[cid],
            "original_inference_receipt": receipt, "all_teacher_outputs_established": False,
            "accuracy_validated": False}
        write_atomic(pipeline.parent / "receipt.json", new_receipt)
        write_atomic(pipeline.parent.parent / "receipt.json", new_receipt)
        rows.append(new_receipt)
        write_atomic(output / "summary.json", {**batch_summary(rows, len(samples), manifest_hash), **provenance,
            "original_inference_runtime_seconds": sum(r["original_inference_runtime_seconds"] for r in rows),
            "summary_replay_seconds": time.perf_counter() - started})
        print(json.dumps({"core_id": cid, "status": "success", "postprocess_only": True,
                          "summary_replay_seconds": elapsed}), flush=True)
    load_completed_batch(output)
    for binding in source_batches.values():
        directory = Path(binding["directory"])
        if (sha(directory / "batch_manifest.json") != binding["batch_manifest_sha256"]
                or sha(directory / "summary.json") != binding["summary_sha256"]):
            raise ValueError("Original completed batch changed during replay")
    return read(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--replacement-batch", type=Path,
                        help="Completed same-video subset whose inference replaces base events")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.source_batch, args.output, args.replacement_batch)
    print(json.dumps({key: result[key] for key in ("complete", "successful_events",
        "new_neural_inference", "original_inference_runtime_seconds", "summary_replay_seconds")}))


if __name__ == "__main__":
    main()
