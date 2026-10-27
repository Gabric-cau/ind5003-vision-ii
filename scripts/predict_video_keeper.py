"""Apply the experimental keeper heads to source-bound automatic video analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fresh_results import bind_stage
from src.keeper_supervised_model import predict_keeper
from src.runtime import file_sha256


def hybrid_direction(rule_direction, learned_direction):
    """Retain existing categorical decisions; fill only the rule's abstentions."""
    allowed = {"left", "center", "right"}
    if rule_direction in allowed:
        return rule_direction, "existing_supported_rule"
    if learned_direction in allowed:
        return learned_direction, "learned_missing_rule_fallback"
    return "unknown", "insufficient_evidence"


def supported_rule(keeper):
    if keeper.get("status") != "ok" or (keeper.get("selected_candidate") or {}).get("qualifies") is not True:
        return "unknown"
    return {"image_left": "left", "center": "center", "image_right": "right"}.get(
        keeper.get("dive_direction"), "unknown")


def predict_analysis(analysis_dir, model_path, *, recover_from_contact=False):
    started = time.perf_counter()
    directory, model_path = Path(analysis_dir).resolve(), Path(model_path).resolve()
    model = json.loads(model_path.read_text())
    if model.get("schema_version") != "keeper-supervised-v1":
        raise ValueError("Unsupported keeper model schema")
    if model["evidence_contract"]["feature_code_sha256"] != file_sha256(ROOT / "src/keeper_supervised_model.py"):
        raise ValueError("Keeper feature code differs from training")
    manifest_path = directory / "source/source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manual_annotations_used") is not False or len(manifest["clips"]) != 1:
        raise ValueError("Expected one automatically analysed video without manual annotations")
    source = manifest["clips"][0]
    cid = source["clip_id"]
    if not isinstance(cid, str) or Path(cid).name != cid or cid in (".", ".."):
        raise ValueError("Invalid analysis clip ID")
    video_path = (directory / "source" / source["clip_filename"]).resolve()
    if not video_path.is_relative_to(directory / "source"):
        raise ValueError("Video must be inside the analysis source directory")
    if file_sha256(video_path) != source["clip_sha256"]:
        raise ValueError("Analysis source video hash mismatch")
    pose_path = directory / "pose" / f"{cid}.json"
    ball_path = directory / "ball/results" / f"{cid}.json"
    pose, ball = (json.loads(path.read_text()) for path in (pose_path, ball_path))
    pose_hash, manifest_hash = file_sha256(pose_path), file_sha256(manifest_path)
    bind_stage("pose", pose, source, pose_hash, manifest_hash)
    bind_stage("ball", ball, source, pose_hash, manifest_hash)
    if ball.get("reference_labels_used_for_prediction") is not False:
        raise ValueError("Ball/keeper evidence lacks an explicit label-free provenance declaration")
    keeper = ball.get("keeper") or {}
    original_rule = supported_rule(keeper)
    raw = predict_keeper(keeper, model)
    recovery = {"requested": recover_from_contact, "applied": False}
    recovery_bindings = {}
    if recover_from_contact and raw["status"] == "abstain":
        from src.keeper_contact_recovery import (automatic_goal_geometry, bounded_contact,
                                                 guard_recovered_prediction, recover_keeper)

        contact_path = directory / "trained_contact/prediction.json"
        contact = json.loads(contact_path.read_text())
        frame, reason = bounded_contact(pose, contact, source)
        recovery_bindings["contact_prediction_sha256"] = file_sha256(contact_path)
        recovery_bindings["recovery_code_sha256"] = file_sha256(ROOT / "src/keeper_contact_recovery.py")
        geometry = {}
        cache_path = directory / "ball/detections" / f"{cid}.json"
        if frame is not None and cache_path.exists():
            cache = json.loads(cache_path.read_text())
            identity = cache.get("inference_identity") or {}
            if (identity.get("source_sha256") != source["clip_sha256"] or
                    identity.get("pose_sha256") != pose_hash or
                    identity.get("source_manifest_sha256") != manifest_hash):
                raise ValueError("Keeper recovery detection cache binding mismatch")
            geometry = {int(key): value for key, value in cache.get("goal_proposals", {}).items()}
            recovery_bindings["detection_cache_sha256"] = file_sha256(cache_path)
        needed = range(max(int(pose["shot_start_frame"]), frame-round(.5*pose["fps"])), frame) if frame is not None else []
        if frame is not None and any(index not in geometry for index in needed):
            geometry = automatic_goal_geometry(video_path, pose, frame)
            recovery["new_cpu_goal_geometry"] = True
        recovered = recover_keeper(pose, contact, source, video_path, geometry)
        recovered_raw = guard_recovered_prediction(recovered, predict_keeper(recovered, model))
        recovery.update(contact_status=reason, keeper=recovered, learned_prediction=recovered_raw)
        if recovered_raw["actual_dive_evidence_supported"] and (
                recovered_raw["status"] != "abstain" or supported_rule(recovered) != "unknown"):
            keeper, raw = recovered, recovered_raw
            recovery["applied"] = True
    rule_direction = original_rule if original_rule != "unknown" else supported_rule(keeper)
    selected, method = hybrid_direction(rule_direction, raw["direction"])
    if original_rule == "unknown" and recovery["applied"] and method != "insufficient_evidence":
        method = "automatic_contact_recovery_" + method
    selected_image = {"left": "image_left", "center": "center", "right": "image_right", "unknown": "unknown"}[selected]
    return {"schema_version": "experimental-video-keeper-prediction-v1", "video_sha256": source["clip_sha256"],
            "clip_id": cid, "manual_annotations_required": False, "manual_annotations_used": False,
            "raw_learned_prediction": raw, "existing_rule_direction": original_rule,
            "keeper_contact_recovery": recovery,
            "prediction": {"dive_direction": selected_image, "direction": selected,
                           "direction_method": method, "height_model_candidate": raw["height"],
                           "height_high_class_supported": False, "physical_height_metres": None},
            "direction_policy": "retain_supported_rule_and_fill_missing_with_learned_candidate",
            "height_policy": "separate_qualitative_low_middle_candidate_not_strict_height_replacement",
            "model_sha256": file_sha256(model_path),
            "source_bindings": {"manifest_sha256": manifest_hash, "pose_sha256": pose_hash,
                                "ball_result_sha256": file_sha256(ball_path), **recovery_bindings},
            "new_neural_inference_calls": 0, "head_runtime_seconds_including_hash_checks": time.perf_counter() - started,
            "promoted_to_default_pipeline": False, "accuracy_validated": False,
            "limitations": ["Grouped development evidence only; this video has no supplied correctness reference.",
                            "Height is a two-class image-plane candidate and cannot represent unseen high saves.",
                            "The raw learned direction was weaker than the old rule on their common development cases."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to deployment.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-from-contact", action="store_true",
                        help="Replay keeper evidence using a source-bound automatic contact when the ball branch abstains")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Prediction output already exists")
    result = predict_analysis(args.analysis_dir, args.model, recover_from_contact=args.recover_from_contact)
    args.output.mkdir(parents=True)
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output / "prediction.json"), "prediction": result["prediction"]}, indent=2))


if __name__ == "__main__":
    main()
