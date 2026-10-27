"""Source-bound keeper direction/height development evaluation and training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.keeper_supervised_model import (COORDINATES, HEADS, classification_metrics,
                                         extract_features, fit_models, predict_features)
from scripts.predict_video_keeper import hybrid_direction


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_events(bundle, reference):
    sources = bundle["events"]
    references = reference["events"]
    if len({e["clip_id"] for e in sources}) != len(sources) or len({e["clip_id"] for e in references}) != len(references):
        raise ValueError("Duplicate event IDs")
    by_id = {e["clip_id"]: e for e in references}
    if set(by_id) != {e["clip_id"] for e in sources}:
        raise ValueError("Evidence/reference cohorts differ")
    events, bindings = [], []
    match_hashes = {}
    for source in sources:
        cid = source["clip_id"]
        ref, mapping = by_id[cid], source["source_mapping"]
        if ref["source_sha256"] != mapping["clip_sha256"]:
            raise ValueError(f"Reference clip hash mismatch: {cid}")
        if ref["fps"] != mapping["fps"] or ref["frame_count"] != mapping["frame_count"]:
            raise ValueError(f"Reference clock mismatch: {cid}")
        if mapping.get("clip_id") != cid:
            raise ValueError("Source mapping event ID mismatch")
        for field in ("clip_sha256", "source_video_sha256"):
            value = mapping[field]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Malformed source hash")
        group = mapping["match_id"]
        if group in match_hashes and match_hashes[group] != mapping["source_video_sha256"]:
            raise ValueError("A match group contains inconsistent original video hashes")
        match_hashes[group] = mapping["source_video_sha256"]
        annotation = ref["_shot_save_annotation"]
        if annotation.get("direction_coordinate_system") != COORDINATES:
            raise ValueError("Reference direction coordinate mismatch")
        for field in ("save_direction", "save_height"):
            provenance = annotation["field_provenance"][field]
            if provenance.get("human_verified") is not True or provenance.get("status") != "human_annotated":
                raise ValueError("Expected explicit human keeper reference")
        if ref["save_direction"] not in HEADS["direction"]["classes"] or ref["save_height"] not in HEADS["height"]["classes"]:
            raise ValueError("Unsupported keeper reference class")
        keeper = source["components"]["ball"]["keeper"]
        rule = {"direction": {"image_left": "left", "image_right": "right", "center": "center"}.get(
            keeper.get("dive_direction"), "unknown"),
            "height": {"low": "low", "mid": "middle", "middle": "middle"}.get(keeper.get("dive_height"), "unknown")}
        events.append({"clip_id": cid, "match_group": group, "source_sha256": ref["source_sha256"],
                       "direction": ref["save_direction"], "height": ref["save_height"],
                       "features": extract_features(keeper), "frozen_rule": rule})
        bindings.append({"clip_id": cid, "match_group": group, "clip_sha256": mapping["clip_sha256"],
                         "source_video_sha256": mapping["source_video_sha256"], "label_clip_hash_matches": True,
                         "fps": mapping["fps"], "frame_count": mapping["frame_count"]})
    return events, bindings


def evaluate(events):
    predictions, majority, folds, models = {}, {}, [], {}
    for group in sorted({e["match_group"] for e in events}):
        model = fit_models(events, held_out_match=group)
        validation = [e for e in events if e["match_group"] == group]
        training = [e for e in events if e["match_group"] != group]
        modes = {head: sorted(Counter(e[head] for e in training).items(), key=lambda item: (-item[1], item[0]))[0][0]
                 for head in HEADS}
        for event in validation:
            predictions[event["clip_id"]] = predict_features(event["features"], model)
            majority[event["clip_id"]] = modes.copy()
        folds.append({"held_out_match": group, "validation_ids": [e["clip_id"] for e in validation],
                      "training_ids": model["training_ids"], "training_majority_labels": modes})
        models[group] = model
    comparison = {"learned_lomo": predictions, "frozen_rule": {e["clip_id"]: e["frozen_rule"] for e in events},
                  "training_fold_majority": majority}
    comparison["retain_rule_fill_missing"] = {e["clip_id"]: {
        "direction": hybrid_direction(e["frozen_rule"]["direction"], predictions[e["clip_id"]]["direction"])[0],
        "height": predictions[e["clip_id"]]["height"]} for e in events}
    metrics, common_metrics, common_ids = {}, {}, {}
    for head, spec in HEADS.items():
        metrics[head] = {name: classification_metrics([e[head] for e in events],
            [rows[e["clip_id"]][head] for e in events], spec["classes"]) for name, rows in comparison.items()}
        common = [e for e in events if all(rows[e["clip_id"]][head] in spec["classes"] for rows in comparison.values())]
        common_ids[head] = [e["clip_id"] for e in common]
        common_metrics[head] = {name: classification_metrics([e[head] for e in common],
            [rows[e["clip_id"]][head] for e in common], spec["classes"]) for name, rows in comparison.items()}
    return {"metrics": metrics, "common_coverage_metrics": common_metrics, "common_event_ids": common_ids,
            "folds": folds, "predictions": predictions, "fold_models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/fresh_course_experiment"
    parser.add_argument("--bundle", type=Path, default=base / "final_bundle_v3_shoe_fusion/features_39.json")
    parser.add_argument("--references", type=Path, default=base / "human_evaluation_2026-10-22/reference_source/penalty_annotations_39_human_v040.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/keeper_supervised_2026-10-26/model_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; choose a new directory to preserve prior runs")
    started = time.perf_counter()
    events, bindings = load_events(read(args.bundle), read(args.references))
    if len(events) != 39 or len({e["match_group"] for e in events}) != 4:
        raise ValueError("This experiment requires the original 39 events and four match groups")
    with threadpool_limits(limits=2):
        result = evaluate(events)
        deployment = fit_models(events)
    args.output.mkdir(parents=True)
    deployment["evidence_contract"] = {"bundle_sha256": sha(args.bundle), "reference_sha256": sha(args.references),
        "feature_code_sha256": sha(ROOT / "src/keeper_supervised_model.py"),
        "automatic_windows_only": True, "reference_labels_required_for_prediction": False,
        "tracking_identity_is_model_estimated": True, "height_is_qualitative_not_physical_measurement": True}
    write(args.output / "deployment.json", deployment)
    for group, model in result.pop("fold_models").items():
        write(args.output / f"fold_{group}.json", model)
    predictions = result.pop("predictions")
    write(args.output / "oof_predictions.json", {"events": [{"clip_id": e["clip_id"], "match_group": e["match_group"],
        "source_sha256": e["source_sha256"], "reference": {head: e[head] for head in HEADS},
        "features": e["features"], "frozen_rule": e["frozen_rule"], "learned": predictions[e["clip_id"]]} for e in events]})
    report = {"schema_version": "keeper-supervised-development-v1", "events": len(events),
        "evaluation": "four_fold_leave_one_match_out_previously_inspected_development", "held_out_test": False,
        "new_neural_inference_calls": 0, "original_outputs_modified": False,
        "config": deployment["config"], "source_bindings": bindings, "evidence_contract": deployment["evidence_contract"],
        "label_counts": {head: dict(Counter(e[head] for e in events)) for head in HEADS},
        "training_label_counts": {head: deployment["heads"][head]["label_counts"] for head in HEADS},
        "feature_extraction_and_training_seconds": time.perf_counter() - started,
        "deployment_model_sha256": sha(args.output / "deployment.json"),
        "dependencies": {name: version(name) for name in ("numpy", "scikit-learn")},
        "code_sha256": {p: sha(ROOT / p) for p in ("src/keeper_supervised_model.py", "scripts/train_keeper_supervised.py")},
        "limitations": ["Development validation on four previously inspected matches, not an untouched test.",
            "The retain-rule direction fallback was evaluated after seeing raw-head development scores; this is an exploratory deployment policy.",
            "Human categorical references have one reviewer and no precise save-phase frame or physical height annotation.",
            "Only low/middle height references exist; high saves are unsupported and must not be claimed as validated.",
            "Only automatic keeper tracks with complete camera stabilization pass the geometry gate; other events abstain.",
            "Partial post-kick trajectories can miss the final dive; tracking identity is not human verified.",
            "Source hashes and clocks match frozen metadata; original video files were not decoded or rehashed in this experiment.",
            "Fold majority uses training-match reference labels only. All-event scores retain missing-track events.",
            "Scores are uncalibrated. Raw-head hyperparameters were fixed before evaluation; no hyperparameter search was run."], **result}
    write(args.output / "report.json", report)
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"],
                      "common_coverage_metrics": report["common_coverage_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
