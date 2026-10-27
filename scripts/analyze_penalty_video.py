"""Run the empirical Vision II modules on one new, unannotated penalty clip."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime import file_sha256
from src.fresh_results import STAGE_PATHS, bind_stage, compact_stage, feature_row, combine_outcomes


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def stage_environment(threads):
    cache = ROOT / ".runtime-cache"
    return {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads), "MPLCONFIGDIR": str(cache / "matplotlib"),
            "YOLO_CONFIG_DIR": str(cache / "ultralytics"), "TORCH_HOME": str(cache / "torch")}


def prepare_source(video, folder):
    if video.suffix.lower() != ".mp4":
        raise ValueError("This pipeline requires an MP4 video; convert or remux the source to .mp4 before running")
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open input video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps > 0 or not frames > 0 or not width > 0 or not height > 0:
            raise ValueError("Video metadata is invalid")
    finally:
        cap.release()
    folder.mkdir(parents=True)
    name = "unseen_penalty" + video.suffix.lower()
    shutil.copy2(video, folder / name)
    digest = file_sha256(video)
    if file_sha256(folder / name) != digest:
        raise ValueError("Copied video differs from input")
    source = dict(clip_id="unseen_penalty", match_id="unassigned", event_id=1,
        kicker="unknown", team="unknown", penalty_number=None,
        slice_start_sec=0., slice_end_sec=frames / fps,
        source_filename=name, source_video_sha256=digest, fps=fps,
        source_start_frame=0, source_end_frame_exclusive=frames, frame_count=frames,
        original_slice_start_frame_in_clip=0, original_slice_end_frame_in_clip=frames,
        semantic_event_start_verified=False, semantic_event_end_verified=False,
        context_is_not_event_boundary_ground_truth=True, clip_sha256=digest,
        clip_bytes=video.stat().st_size, clip_filename=name)
    write_json(folder / "source_manifest.json", {
        "schema_version": "unannotated-single-penalty-v1", "clips": [source],
        "manual_annotations_used": False, "event_scope": "one penalty attempt supplied as a video clip"})
    return source


def stage_commands(source_dir, output, pose_model, ball_model, device,
                   ball_specialist=None, specialist_mode="full", specialist_imgsz=1280,
                   batch_size=4):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    shared = ["--clips-dir", source_dir, "--clips", "unseen_penalty"]
    producers = ["--pose-dir", output / "pose", "--ball-dir", output / "ball"]
    specialist = (["--ball-specialist", ball_specialist, "--specialist-mode", specialist_mode,
                   "--specialist-imgsz", specialist_imgsz] if ball_specialist is not None else [])
    return [
        ("pose", "scripts.run_fresh_pose_experiment", shared + ["--output-dir", output / "pose",
            "--model", pose_model, "--device", device, "--imgsz", "1280", "--batch-size", str(batch_size)]),
        ("ball", "scripts.run_fresh_ball_experiment", shared + ["--pose-dir", output / "pose",
            "--output-dir", output / "ball", "--model", ball_model, "--device", device,
            "--batch-size", str(batch_size)] + specialist),
        ("foot", "scripts.run_fresh_foot_experiment", shared + ["--pose-input", output / "pose",
            "--contact-json-dir", output / "ball" / "contacts", "--output-dir", output / "foot",
            "--model", pose_model, "--device", device, "--batch-size", str(batch_size)]),
        ("phase", "scripts.run_fresh_phase_experiment", shared + producers + ["--output-dir", output / "phase"]),
        ("pitch", "scripts.run_fresh_pitch_experiment", shared + producers + ["--output-dir", output / "pitch"]),
        ("camera", "scripts.run_fresh_camera_experiment_v2", shared + producers + ["--output-dir", output / "camera",
            "--model", pose_model, "--device", device]),
        ("height", "scripts.run_fresh_keeper_height_experiment", shared + producers + ["--output-dir", output / "height",
            "--camera-dir", output / "camera"]),
        ("endpoint", "scripts.run_fresh_endpoint_experiment", shared + producers + ["--output-dir", output / "endpoint",
            "--camera-dir", output / "camera"]),
    ]


def contact_stage_command(video, model, pose_model, ball_model, output, device, threads):
    metadata = json.loads((Path(model) / "deployment_metadata.json").read_text())
    maximum = metadata["visual_settings"]["max_shot_sec"]
    arguments = ["--video", video, "--model", model, "--pose-model", pose_model,
                 "--ball-model", ball_model, "--output", output / "trained_contact",
                 "--device", device, "--threads", threads]
    if maximum == 60.:
        module = "scripts.predict_video_contact_fullshot"
        arguments += ["--single-output", "--event-id", "unseen_penalty"]
    elif maximum == 12.:
        module = "scripts.predict_video_contact"
    else:
        raise ValueError("Unsupported contact window protocol; supply a bound 12s or 60s model")
    return "trained_contact", module, arguments


def run_stage(name, module, arguments, output, env, python=None):
    command = [str(python or sys.executable), "-m", module, *map(str, arguments)]
    started = time.perf_counter()
    log = output / "logs" / (name + ".log")
    log.parent.mkdir(exist_ok=True)
    print("Running " + name, flush=True)
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {"stage": name, "command": command, "exit_code": result.returncode,
               "seconds": time.perf_counter() - started, "log": str(log.relative_to(output))}
    if result.returncode:
        receipt["error_tail"] = log.read_text()[-3000:]
    print(json.dumps(receipt), flush=True)
    return receipt


def endpoint_output(endpoint, fps):
    point, frame = endpoint.get("last_observed_position"), endpoint.get("last_observed_frame")
    identity = endpoint.get("observed_ball_identity_candidate") is True
    last = None
    if identity and point is not None and frame is not None:
        observation = next((p for p in reversed(endpoint.get("trajectory", []))
                            if p.get("frame") == frame and [p.get("x"), p.get("y")] == point), {})
        last = {"xy": point, "frame": frame, "seconds": frame / fps,
                "coordinate_system": "source_image_pixels",
                "detection_id": observation.get("detection_id"), "identity_status": "automatic_candidate"}
    return {"status": endpoint.get("status", "unavailable"),
            "event_final_position": endpoint.get("final_position"),
            "event_end_observed": endpoint.get("final_position_verified") is True,
            "last_observed": last, "observation_end_reason": endpoint.get("reason"),
            "boundary_reason": endpoint.get("boundary_reason"),
            "stationary_end_candidate": endpoint.get("stationary_end_candidate"),
            "physical_ball_rest_established": endpoint.get("physical_ball_rest_established") is True,
            "last_observation_is_not_event_final_position": True}


def summarize(source, output, receipts):
    manifest_hash = file_sha256(output / "source" / "source_manifest.json")
    pose_path = output / "pose" / (source["clip_id"] + ".json")
    pose_hash = file_sha256(pose_path) if pose_path.exists() else None
    stages, bindings = {}, {}
    failed = {r["stage"] for r in receipts if r["exit_code"]}
    for name, template in STAGE_PATHS.items():
        if name == "scoreboard":
            continue
        path = output / name / template.format(clip_id=source["clip_id"])
        if not path.exists() or name in failed:
            stages[name] = {}
            continue
        data = json.loads(path.read_text())
        if name == "pitch" and "human_verified" not in data:
            # The older calibration producer uses separate geometry-verification flags.
            data = {**data, "human_verified": False}
        bind_stage(name, data, source, pose_hash, manifest_hash)
        stages[name] = data
        bindings[name] = {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
    row = feature_row(source, stages, {"reference_outcome": None, "reference_supported": False})
    contact_path = output / "trained_contact" / "prediction.json"
    contact = json.loads(contact_path.read_text()) if contact_path.exists() and "trained_contact" not in failed else None
    if contact is not None and contact.get("video_sha256") != source["clip_sha256"]:
        raise ValueError("Trained contact belongs to another source video")
    auxiliary = {}
    auxiliary_names = {"trained_shoes", "trained_keeper", "scoreboard_transition", "trained_ball", "trained_camera"}
    for name in sorted(auxiliary_names):
        path = output / name / "prediction.json"
        if path.exists() and name not in failed:
            data = json.loads(path.read_text())
            if (data.get("source_sha256") or data.get("video_sha256")) != source["clip_sha256"]:
                raise ValueError(name + ": prediction is bound to another video")
            auxiliary[name] = data
            bindings[name] = {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
    ball = stages.get("ball", {})
    runup_points = [point for point in stages.get("pitch", {}).get("sparse_ground_contact_candidates", [])
                    if point.get("phase") == "pre_kick_candidate"]
    fields = {
        "runup_top_down_trajectory": {"points": runup_points, "phase_filter": "pre_kick_candidate",
            "coordinate_unit": "m_hypothesis", "complete_trajectory_verified": False,
            "status": "sparse_projected_candidates_only"},
        "kicking_foot": row["kicking_foot_candidate"],
        "visible_runup_steps": row["steps_phase_candidate"],
        "keeper_pre_kick_movement": row["keeper_pre_movement_candidate"],
        "keeper_dive_direction": row["keeper_direction_image_candidate"],
        "keeper_dive_height": {"value": "unknown", "status": "unknown",
            "definition": "human_qualitative_image_plane", "supported_classes": ["low", "middle"],
            "high_supported": False, "physical_height_metres": None,
            "strict_mask_pose_category": row["keeper_strict_height_candidate"],
            "pose_phase_category": row.get("keeper_pose_ordinal_height_candidate", "unknown")},
        "ball_trajectory": ball.get("trajectory", []),
        "ball_final_position": endpoint_output(stages.get("endpoint", {}), source.get("fps", 1.)),
        "goal_outcome": row["combined_rule_outcome"],
    }
    selected_sources = {}
    shoes = auxiliary.get("trained_shoes")
    if shoes:
        fields["step_count_evidence"] = {"window": shoes.get("automatic_window"),
            "scope": "observed_automatic_approach_window", "complete_runup_verified": False,
            "total_runup_steps": None}
        if shoes.get("striking_foot") in ("left", "right"):
            fields["kicking_foot"] = shoes["striking_foot"]
            selected_sources["kicking_foot"] = "trained_shoe_head"
        if shoes.get("candidate_count") is not None:
            fields["visible_runup_steps"] = shoes["candidate_count"]
            selected_sources["visible_runup_steps"] = "trained_landings_in_automatic_observed_window"
    keeper = auxiliary.get("trained_keeper")
    if keeper:
        fields["keeper_learned_diagnostics"] = keeper
        # The wrapper retains supported rules and can reject an invalid old actor.
        fields["keeper_dive_direction"] = keeper["prediction"]["dive_direction"]
        selected_sources["keeper_dive_direction"] = keeper["prediction"]["direction_method"]
        value = keeper["prediction"].get("height_model_candidate", "unknown")
        fields["keeper_dive_height"].update(value=value,
            status="candidate" if value in ("low", "middle") else "unknown")
        selected_sources["keeper_dive_height"] = "learned_human_ordinal_low_middle_head"
    scoreboard = auxiliary.get("scoreboard_transition")
    if scoreboard:
        fields["goal_outcome"], selected_sources["goal_outcome"] = combine_outcomes(
            row["ball_rule_outcome"], scoreboard["decision"]["rule_outcome"])
    final_row = {"clip_id": source["clip_id"], "source_sha256": source["clip_sha256"],
        "kicking_foot": fields["kicking_foot"], "visible_runup_steps": fields["visible_runup_steps"],
        "keeper_pre_kick_movement": fields["keeper_pre_kick_movement"],
        "keeper_dive_direction": fields["keeper_dive_direction"],
        "keeper_dive_height": fields["keeper_dive_height"]["value"],
        "goal_outcome": fields["goal_outcome"], "runup_projected_candidate_points": len(runup_points),
        "ball_trajectory_observations": len(fields["ball_trajectory"])}
    successful = {r["stage"] for r in receipts if r["exit_code"] == 0}
    required_stages = (set(STAGE_PATHS) - {"scoreboard"}) | {"trained_contact"}
    complete = (not failed and required_stages <= successful
        and all(stages.get(k) for k in required_stages - {"trained_contact"}) and contact is not None
        and (successful & auxiliary_names) <= set(auxiliary))
    return {"schema_version": "offline-penalty-analysis-v2", "input_sha256": source["clip_sha256"],
        "manual_annotations_used": False, "source": source, "required_outputs": fields,
        "camera_intervals": stages.get("camera", {}).get("segments", []),
        "camera_teacher_category_intervals": stages.get("camera", {}).get("segments", []),
        "camera_spatial_intervals": auxiliary.get("trained_camera", {}).get("segments", []),
        "camera_interval_sources": {"teacher_categories": "geometric_camera_rules_with_unknowns",
                                    "spatial_classes": "learned_spatial_head_not_four_teacher_classes"},
        "trained_contact": contact, "auxiliary_models": auxiliary, "feature_table": row,
        "feature_table_provenance": {"role": "legacy_rule_candidates_before_auxiliary_selection",
                                     "auxiliary_overrides_applied": False,
                                     "selected_scalar_output": "final_feature_table"},
        "final_feature_table": final_row,
        "final_feature_table_provenance": {"source": "required_outputs_after_auxiliary_selection",
                                           "field_selection_sources": "selected_output_sources",
                                           "structured_evidence": "required_outputs"},
        "selected_output_sources": selected_sources,
        "components": {k: compact_stage(k, v) for k, v in stages.items()},
        "artifact_bindings": bindings, "stage_receipts": receipts,
        "all_stages_executed_successfully": bool(complete), "accuracy_validated": False,
        "all_teacher_outputs_established": False,
        "scope": "One penalty clip; automatic candidates retain missing evidence and unknown values",
        "limitations": ["Training improvement does not validate every downstream feature",
                        "Classical pose/ball/phase stages retain their own source-bound contact estimate; the trained contact is reported separately",
                        "Sparse ground projections are not a complete top-down route",
                        "Last visible ball position is not necessarily the final resting position",
                        "No shootout ordinal or human outcome is supplied at inference"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--ball-model", type=Path, required=True)
    parser.add_argument("--ball-specialist", type=Path,
                        help="Optional single-class ball detector supplementing the COCO model")
    parser.add_argument("--specialist-mode", choices=("full", "tiles"), default="full")
    parser.add_argument("--specialist-imgsz", type=int, default=1280)
    parser.add_argument("--contact-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Visual-stage batch size; use 1 on lower-memory GPUs")
    parser.add_argument("--shoe-model", type=Path)
    parser.add_argument("--pose33-model", type=Path)
    parser.add_argument("--shoe-python", type=Path)
    parser.add_argument("--keeper-model", type=Path)
    parser.add_argument("--ball-category-model", type=Path)
    parser.add_argument("--ball-height-model", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--camera-model", type=Path)
    parser.add_argument("--summarize-existing", action="store_true",
                        help="Summarize completed stage artifacts without rerunning inference")
    args = parser.parse_args()
    if args.threads < 1 or args.specialist_imgsz < 1 or args.batch_size < 1:
        parser.error("threads, specialist-imgsz and batch-size must be positive")
    if bool(args.shoe_model) != bool(args.pose33_model):
        parser.error("shoe-model and pose33-model must be supplied together")
    if (args.ball_category_model or args.camera_model) and not args.encoder:
        parser.error("Category and camera heads require their frozen encoder")
    if args.ball_height_model and not args.ball_category_model:
        parser.error("ball-height-model requires ball-category-model")
    args.output = args.output.resolve()
    if args.summarize_existing:
        if (args.output / "analysis.json").exists():
            raise FileExistsError("Analysis already exists")
        source = json.loads((args.output / "source" / "source_manifest.json").read_text())["clips"][0]
        receipts = json.loads((args.output / "progress.json").read_text())["stages"]
        result = summarize(source, args.output, receipts)
        result["summed_stage_runtime_seconds"] = sum(r["seconds"] for r in receipts)
        result["summarized_after_stage_execution"] = True
        write_json(args.output / "analysis.json", result)
        print(json.dumps({"output": str(args.output / "analysis.json"),
                          "success": result["all_stages_executed_successfully"]}))
        return
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    for name in ("video", "pose_model", "ball_model", "contact_model"):
        setattr(args, name, getattr(args, name).resolve())
    started = time.perf_counter()
    source = prepare_source(args.video, args.output / "source")
    env = stage_environment(args.threads)
    commands = [contact_stage_command(args.video, args.contact_model, args.pose_model,
                                     args.ball_model, args.output, args.device, args.threads)]
    commands += stage_commands(args.output / "source", args.output, args.pose_model, args.ball_model, args.device,
                               args.ball_specialist.resolve() if args.ball_specialist else None,
                               args.specialist_mode, args.specialist_imgsz, args.batch_size)
    if args.shoe_model:
        commands.append(("trained_shoes", "scripts.predict_video_shoes", ["--analysis-dir", args.output,
            "--model", args.shoe_model.resolve(), "--pose33-model", args.pose33_model.resolve(),
            "--output", args.output / "trained_shoes", "--threads", args.threads]))
    if args.keeper_model:
        commands.append(("trained_keeper", "scripts.predict_video_keeper", ["--analysis-dir", args.output,
            "--model", args.keeper_model.resolve(), "--output", args.output / "trained_keeper"]))
    if args.ball_category_model:
        arguments = ["predict", "--video", args.video, "--model", args.ball_category_model.resolve(),
            "--backbone", args.encoder.resolve(), "--candidate", "first_frame_difference",
            "--output", args.output / "trained_ball/prediction.json", "--threads", args.threads]
        if args.ball_height_model:
            arguments += ["--height-model", args.ball_height_model.resolve()]
        commands.append(("trained_ball", "scripts.train_external_ball_v2", arguments))
    if args.camera_model:
        commands.append(("trained_camera", "scripts.train_external_camera_classifier", ["predict", "--video", args.video,
            "--model", args.camera_model.resolve(), "--backbone", args.encoder.resolve(),
            "--output", args.output / "trained_camera/prediction.json", "--threads", args.threads]))
    commands.append(("scoreboard_transition", "scripts.predict_video_scoreboard", ["--analysis-dir", args.output,
        "--output", args.output / "scoreboard_transition"]))
    receipts = []
    for name, module, arguments in commands:
        receipts.append(run_stage(name, module, arguments, args.output, env,
                                 python=args.shoe_python if name == "trained_shoes" else None))
        write_json(args.output / "progress.json", {"completed": len(receipts), "total": len(commands), "stages": receipts})
    result = summarize(source, args.output, receipts)
    result["end_to_end_seconds"] = time.perf_counter() - started
    write_json(args.output / "analysis.json", result)
    print(json.dumps({"output": str(args.output / "analysis.json"),
        "seconds": result["end_to_end_seconds"], "success": result["all_stages_executed_successfully"]}), flush=True)
    if not result["all_stages_executed_successfully"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
