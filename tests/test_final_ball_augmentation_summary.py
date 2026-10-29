import json

import pytest

from scripts import summarize_final_ball_augmentation as review
from scripts.train_ball_frame_detector import score_frame, summarize


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def choices(values):
    groups = {}
    for group, (_, count, _) in review.GROUPS.items():
        groups[group] = {"comparisons": [{"model": model, "mode": "tiles",
            "deduplicated_frames": {"hits": values[model][group], "denominator": count}}
            for model in review.MODELS]}
    return groups


def test_historical_gate_alone_cannot_promote_regressing_candidate():
    data = {"initializer": {"core_development": 70, "arsenal_development": 65},
            "control_full": {"core_development": 69, "arsenal_development": 70},
            "augmented_mixed": {"core_development": 70, "arsenal_development": 65}}
    result = review.choose_candidate(choices(data))
    assert result["runner_historical_decision"]["selected_candidate"] == "control_full"
    assert result["selected_model"] == "initializer"
    assert any("Regresses" in reason for reason in result["reasons"]["control_full"])


def test_new_candidate_must_pass_gates_and_improve_and_ties_prefer_control():
    data = {"initializer": {"core_development": 65, "arsenal_development": 61},
            "control_full": {"core_development": 66, "arsenal_development": 62},
            "augmented_mixed": {"core_development": 66, "arsenal_development": 62}}
    assert review.choose_candidate(choices(data))["selected_model"] == "control_full"
    data["augmented_mixed"]["arsenal_development"] = 63
    assert review.choose_candidate(choices(data))["selected_model"] == "augmented_mixed"
    data["control_full"]["arsenal_development"] = 61
    data["augmented_mixed"]["arsenal_development"] = 61
    assert review.choose_candidate(choices(data))["selected_model"] == "initializer"


def complete_run(directory):
    directory.mkdir()
    initial = directory / "initial.pt"
    initial.write_bytes(b"initial fixture weights")
    source = ["old.jpg", "new.jpg"] * 221
    schedules = {key: [source] * 10 for key in ("source", *review.ARMS)}
    save(directory / "schedules.json", schedules)
    save(directory / "dataset/manifest.json", {"exports": {name: {"source_name": name} for name in source}})
    protocol = {"epochs_each": 10, "samples_per_epoch": 442,
        "expected_optimizer_steps_per_epoch": review.STEPS,
        "initial_weights": str(initial), "initial_sha256": review.sha(initial),
        "schedule_sha256": review.sha(directory / "schedules.json"),
        "dataset_manifest_sha256": review.sha(directory / "dataset/manifest.json"),
        "code_sha256": {"src/ball_specialist.py": "c" * 64}}
    save(directory / "protocol.json", protocol)
    report = {"protocol": protocol, "training_completed": True, "independent_test_evaluated": False,
        "overall_project_accuracy_measured": False, "candidates": {}, "results": {},
        "eligible_arms": list(review.ARMS), "selected_candidate": "control_full"}
    for arm in review.ARMS:
        weights = directory / arm / "last.pt"
        weights.parent.mkdir()
        weights.write_bytes(arm.encode())
        sampling = [{"epoch": index + 1, "samples": 442, "old": 221, "new": 221,
            "optimizer_steps": step, "source_sequence_sha256": review.sequence_sha(source),
            "variant_sequence_sha256": review.sequence_sha(source),
            "batch_checks": {"batches": 442, "remaining_targets": 442, "empty_after_framework_augmentation": 0},
            "photometric_calls_cumulative": 442 * (index + 1) if arm == "augmented_mixed" else 0}
            for index, step in enumerate(review.STEPS)]
        receipt = {"completed_epochs": review.EPOCHS, "optimizer_steps": 388,
            "initial_sha256": protocol["initial_sha256"], "weights": str(weights),
            "weights_sha256": review.sha(weights), "changed_parameter_names": ["weight"],
            "maximum_observed_parameter_delta": .01, "checkpoint_selection": "fixed_last_epoch_10",
            "current_round_scores_seen_before_checkpoint_freeze": False, "sampling": sampling,
            "photometric_transform_calls": 4420 if arm == "augmented_mixed" else 0, "training_seconds": 1.}
        report["candidates"][arm] = receipt
        save(directory / arm / "selected_candidate.json", receipt)
        save(directory / arm / "actual_sampling.json", sampling)
        folder = directory / arm / "fit"
        folder.mkdir()
        (folder / "results.csv").write_text("epoch,train/box_loss,train/cls_loss,train/dfl_loss\n" +
            "".join(f"{epoch},1,1,1\n" for epoch in review.EPOCHS))
    for group, (total, unique, _) in review.GROUPS.items():
        report["results"][group] = {}
        for model in review.MODELS:
            report["results"][group][model] = {}
            for mode in review.MODES:
                rows = []
                for index in range(total):
                    target = [1, 1, 5, 5]
                    candidates = [{"bbox": target, "confidence": .9}]
                    if model == "initializer" and index == 0:
                        candidates.insert(0, {"bbox": [20, 20, 30, 30], "confidence": .95})
                    rows.append({"image": f"image_{index}.png", "video_id": "video",
                        "annotation_id": str(index), "event_id": f"event_{index % 2}",
                        "reference_bbox": target, "deduplicated_primary": index < unique,
                        "candidates": candidates, **score_frame(target, candidates)})
                indexed = {(row["video_id"], row["annotation_id"]): row for row in rows}
                original, _, macro = review.event_metrics(indexed)
                prediction = {"weights_sha256": protocol["initial_sha256"] if model == "initializer" else report["candidates"][model]["weights_sha256"],
                    "mode": mode, "image_size": 1280, "confidence": .05, "nms_iou": .6, "class_id": 0,
                    "inference_code_sha256": "c" * 64, "prediction_uses_reference": False,
                    "seconds_including_model_load": 1.5, "rows": rows,
                    "metrics": {**{scope: summarize([row for row in rows if scope == "all_frames" or row["deduplicated_primary"]]) for scope in review.SCOPES},
                        "by_event": original, "macro_event_localization_recall_iou_0_5": macro}}
                save(directory / f"{group}_{model}_{mode}.json", prediction)
                report["results"][group][model][mode] = {key: value for key, value in prediction.items() if key != "rows"}
    save(directory / "report.json", report)
    return report


def test_complete_audit_recomputes_event_metrics_and_writes_review(tmp_path):
    directory = tmp_path / "run"
    complete_run(directory)
    result = review.write_review(directory, tmp_path / "review")
    assert result["release_selection"]["selected_model"] == "control_full"
    assert result["overall_project_accuracy_measured"] is False
    core = result["groups"]["core_development"]
    initial = next(row for row in core["comparisons"] if row["model"] == "initializer" and row["mode"] == "tiles")
    assert initial["deduplicated_frames"]["hits"] == 90
    assert initial["deduplicated_frames"]["oracle_candidate_hits"] == 91
    assert core["paired_changes_from_initializer"]["control_full"]["tiles"]["event_0"]["corrected"] == 1
    assert (tmp_path / "review/release_selection.json").is_file()
    assert (tmp_path / "review/event_metrics.csv").is_file()
    assert "no independent test or whole-project accuracy" in (tmp_path / "review/summary.md").read_text()


@pytest.mark.parametrize("mutation", ["incomplete", "checkpoint", "source_sequence", "optimizer", "event_metrics", "candidate_score"])
def test_incomplete_or_inconsistent_evidence_cannot_be_published(tmp_path, mutation):
    directory = tmp_path / "run"
    report = complete_run(directory)
    if mutation == "incomplete":
        report["training_completed"] = False
    elif mutation == "checkpoint":
        (directory / "control_full/last.pt").write_bytes(b"changed")
    elif mutation in {"source_sequence", "optimizer"}:
        receipt = report["candidates"]["control_full"]
        if mutation == "source_sequence":
            receipt["sampling"][0]["source_sequence_sha256"] = "0" * 64
        else:
            receipt["optimizer_steps"] = 387
        save(directory / "control_full/selected_candidate.json", receipt)
        save(directory / "control_full/actual_sampling.json", receipt["sampling"])
    else:
        path = directory / "core_development_initializer_tiles.json"
        prediction = json.loads(path.read_text())
        if mutation == "event_metrics":
            prediction["metrics"]["by_event"]["event_0"]["deduplicated_frames"]["top1_iou_0_5_hits"] = 0
        else:
            prediction["rows"][0]["any_candidate_iou"] = 0
        save(path, prediction)
        report["results"]["core_development"]["initializer"]["tiles"] = {key: value for key, value in prediction.items() if key != "rows"}
    save(directory / "report.json", report)
    with pytest.raises(ValueError):
        review.write_review(directory, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_running_job_is_rejected_without_writing_zero_result(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        review.write_review(directory, tmp_path / "review")
    assert not (tmp_path / "review").exists()
