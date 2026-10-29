import json
from pathlib import Path

import pytest
import yaml

from scripts import summarize_final_keeper_training as review
from scripts.prepare_keeper_detector_final import split_for


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def choice_scores(control, photo):
    scores = {}
    for arm, data in (("control", control), ("photometric", photo)):
        scores[arm] = {"rows": [{"split": group, "video_id": f"{group}_{index}", "top1_hit": hit}
            for group, clips in data.items() for index, clip in enumerate(clips) for hit in clip]}
    return scores


def test_gate_requires_each_group_not_to_regress_even_if_mean_improves():
    scores = choice_scores({"subject": [[True]], "external": [[False, False]]},
                           {"subject": [[True, False]], "external": [[True, True]]})
    decision = review.choose_candidate(scores, ["subject", "external"])
    assert decision["selected_model"] == "control"
    assert decision["photometric_strict_mean_improvement"] is True
    assert decision["photometric_regressing_groups"] == ["subject"]


def test_gate_uses_clip_macro_not_frame_micro_and_excludes_training():
    control = {"subject": [[True] * 9, [False]], "external": [[True]], "train": [[True]]}
    photo = {"subject": [[True] * 5 + [False] * 4, [True]], "external": [[True]], "train": [[False] * 100]}
    decision = review.choose_candidate(choice_scores(control, photo), ["subject", "external"])
    assert decision["selected_model"] == "photometric"
    assert decision["macro_clip_top1_by_group"]["control"]["subject"] == .5
    assert decision["macro_clip_top1_by_group"]["photometric"]["subject"] == pytest.approx(7 / 9)
    assert decision["training_diagnostics_used_for_selection"] is False


def test_gate_tie_keeps_control_and_refuses_training_or_missing_group():
    data = {"subject": [[True, False]], "external": [[True]]}
    scores = choice_scores(data, data)
    assert review.choose_candidate(scores, ["subject", "external"])["selected_model"] == "control"
    for groups in (["subject", "train"], ["subject"], ["subject", "absent"]):
        with pytest.raises(ValueError):
            review.choose_candidate(scores, groups)


def complete_run(directory):
    directory.mkdir()
    dataset = directory / "dataset"
    exports = []
    sources = [("05-04", i) for i in range(835)] + [("02-05", 0), ("02-05", 1), ("02-06", 0), ("match_17", 0), ("match_18", 0)]
    for video, index in sources:
        split = split_for(video)
        identifier = f"{video}_{index}"
        image, label = Path(f"images/{split}/{identifier}.jpg"), Path(f"labels/{split}/{identifier}.txt")
        for path in (dataset / image, dataset / label):
            path.parent.mkdir(parents=True, exist_ok=True)
        (dataset / image).write_bytes(identifier.encode())
        (dataset / label).write_text("0 0.2 0.2 0.2 0.2\n")
        exports.append({"id": identifier, "video_id": video, "split": split, "event_id": video,
            "subject_group": "subject_05" if split == "train" else "subject_02" if video.startswith("02-") else None,
            "match_id": "world_cup" if video.startswith("match") else None,
            "image": str(image), "label": str(label), "width": 100, "height": 100,
            "image_sha256": review.sha256(dataset / image), "label_sha256": review.sha256(dataset / label),
            "class_id": 0, "class_name": "active_keeper", "bbox": [10, 10, 30, 30],
            "existing_pose_reference": index == 0})
    manifest_path = dataset / "manifest.json"
    save(manifest_path, {"schema": "active_keeper_detector_final/v1", "source_pixels_verified": True,
        "all_partitions_are_development": True, "unknown_frames_are_negative_examples": False, "exports": exports})
    (dataset / "dataset.yaml").write_text(yaml.safe_dump({"path": str(dataset), "train": "images/train", "val": "images/train", "names": {0: "active_keeper"}}))
    initial = directory / "initial.pt"
    initial.write_bytes(b"initial keeper model fixture")
    schedules = review.schedules_for([Path(row["image"]).name for row in exports if row["split"] == "train"])
    save(directory / "schedules.json", schedules)
    evaluation = [row for row in exports if row["split"] != "train" or row["existing_pose_reference"]]
    protocol = {"epochs": 12, "training_frames": 835, "expected_optimizer_steps_per_epoch": review.STEPS,
        "initial_weights": str(initial), "initial_weights_sha256": review.sha256(initial),
        "manifest": str(manifest_path), "manifest_sha256": review.sha256(manifest_path),
        "dataset_yaml_sha256": review.sha256(dataset / "dataset.yaml"),
        "schedule_sha256": review.sha256(directory / "schedules.json"),
        "checkpoint_selection": "fixed_last_epoch_12", "all_candidates_frozen_before_development_evaluation": True,
        "validation_yaml_uses_train_only": True, "human_boxes_used_for_inference": False,
        "independent_test": False, "whole_pipeline_accuracy_measured": False,
        "evaluation_frame_ids": [row["id"] for row in evaluation],
        "evaluation_counts": dict(review.Counter(row["split"] for row in evaluation))}
    save(directory / "protocol.json", protocol)
    report = {"protocol": protocol, "training_completed": True, "default_pipeline_changed": False,
              "independent_test": False, "full_video_role_tracking_measured": False, "candidates": {}, "evaluation": {}}
    for arm in review.ARMS:
        weights = directory / arm / "fit/weights/last.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(arm.encode())
        sampling = [{"epoch": index + 1, "samples": 835, "unique_images": 835, "optimizer_steps": step,
            "sequence_sha256": review.sequence_digest(schedules[index]),
            "target_counts": {"frames": 835, "targets": 835, "fallback_frames": 0}}
            for index, step in enumerate(review.STEPS)]
        candidate = {"initial_weights": str(initial), "initial_sha256": protocol["initial_weights_sha256"],
            "weights": str(weights), "weights_sha256": review.sha256(weights),
            "initial_parameter_digests": {"weight": "a" * 64}, "completed_epochs": review.EPOCH_NUMBERS,
            "optimizer_steps": 838, "changed_parameters": ["weight"], "maximum_observed_parameter_delta": .01,
            "checkpoint_selection": "fixed_last_epoch_12", "development_scored_before_checkpoint_freeze": False,
            "promoted_to_default": False, "photometric_enabled": arm == "photometric",
            "sampling": sampling, "training_seconds": 1.}
        report["candidates"][arm] = candidate
        save(directory / arm / "candidate.json", candidate)
        save(directory / arm / "actual_sampling.json", sampling)
        save(directory / arm / "photometric_receipt.json", {"batches_augmented": 10020 if arm == "photometric" else 0})
    save(directory / "frozen_candidates.json", report["candidates"])
    for model in review.MODELS:
        rows = []
        for sample in evaluation:
            detections = [{"bbox": sample["bbox"], "confidence": .8}]
            if model == review.GENERIC or model == "control" and sample["id"] == "02-05_0":
                detections.insert(0, {"bbox": [60, 60, 80, 80], "confidence": .9})
            if model == "photometric" and sample["split"] == "train":
                detections = []
            rows.append({"id": sample["id"], "detections": detections})
        prediction = {"rows": rows, "seconds": 1., "reference_labels_used_for_prediction": False,
                      "mode": "full_original_image", "confidence": .1, "image_size": 1280}
        score = review.score_predictions(rows, evaluation, specialized=model != review.GENERIC)
        save(directory / f"{model}_predictions.json", prediction)
        save(directory / f"{model}_scores.json", score)
        report["evaluation"][model] = {"scores": {key: value for key, value in score.items() if key != "rows"},
            "seconds": 1., "weights_sha256": protocol["initial_weights_sha256"] if model == review.GENERIC else report["candidates"][model]["weights_sha256"]}
    save(directory / "report.json", report)
    return report


def test_completed_run_recomputes_scores_and_writes_development_only_review(tmp_path):
    directory = tmp_path / "run"
    complete_run(directory)
    summary = review.write_review(directory, tmp_path / "review")
    assert summary["selection"]["selected_model"] == "photometric"
    assert summary["selection"]["default_pipeline_changed"] is False
    assert len(summary["development_comparisons"]) == 6
    assert all(row["split"] != "train" for row in summary["development_comparisons"])
    generic = [row for row in summary["development_comparisons"] if row["model"] == review.GENERIC]
    assert all(row["top1_hits"] is None and row["macro_clip_top1_recall_iou_0_5"] is None for row in generic)
    assert all(row["any_candidate_hits"] == row["frames"] for row in generic)
    assert summary["diagnostics_not_used_for_selection"]["photometric"]["training_reference_frames"]["top1_hits"] == 0
    assert (tmp_path / "review/selection.json").is_file()
    assert (tmp_path / "review/comparisons.csv").is_file()
    assert "no independent test or whole-project accuracy" in (tmp_path / "review/summary.md").read_text()


@pytest.mark.parametrize("mutation", ["incomplete", "checkpoint", "initial_parameters", "optimizer", "sample_order", "lost_target", "stored_score", "raw_prediction", "bad_box", "missing_prediction", "manifest"])
def test_inconsistent_or_incomplete_evidence_is_rejected_without_output(tmp_path, mutation):
    directory = tmp_path / "run"
    report = complete_run(directory)
    candidate = report["candidates"]["control"]
    if mutation == "incomplete":
        report["training_completed"] = False
    elif mutation == "checkpoint":
        Path(candidate["weights"]).write_bytes(b"different checkpoint")
    elif mutation == "initial_parameters":
        candidate["initial_parameter_digests"]["weight"] = "b" * 64
    elif mutation == "optimizer":
        candidate["sampling"][3]["optimizer_steps"] -= 1
    elif mutation == "sample_order":
        candidate["sampling"][0]["sequence_sha256"] = "0" * 64
    elif mutation == "lost_target":
        candidate["sampling"][0]["target_counts"]["targets"] -= 1
    elif mutation == "stored_score":
        path = directory / "control_scores.json"
        value = review.read_json(path)
        value["by_split"]["development_subject_02"]["top1_hits"] += 1
        save(path, value)
    elif mutation in {"raw_prediction", "bad_box", "missing_prediction"}:
        path = directory / "control_predictions.json"
        value = review.read_json(path)
        if mutation == "raw_prediction":
            value["rows"][0]["detections"] = []
        elif mutation == "bad_box":
            value["rows"][0]["detections"][0]["bbox"][0] = float("nan")
        else:
            value["rows"].pop()
        save(path, value)
    elif mutation == "manifest":
        path = Path(report["protocol"]["manifest"])
        path.write_text(path.read_text() + "\n")
    save(directory / "control/candidate.json", candidate)
    save(directory / "control/actual_sampling.json", candidate["sampling"])
    save(directory / "frozen_candidates.json", report["candidates"])
    save(directory / "report.json", report)
    with pytest.raises(ValueError):
        review.write_review(directory, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_missing_final_report_refuses_a_partial_summary(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        review.write_review(directory, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_missing_final_model_score_refuses_a_partial_summary(tmp_path):
    directory = tmp_path / "run"
    complete_run(directory)
    (directory / "photometric_scores.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        review.write_review(directory, tmp_path / "review")
    assert not (tmp_path / "review").exists()


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), True, -.1, 1.1])
def test_invalid_confidence_is_not_silently_ranked(bad):
    sample = {"id": "frame", "width": 100, "height": 100}
    prediction = {"reference_labels_used_for_prediction": False, "mode": "full_original_image",
        "image_size": 1280, "confidence": .1, "seconds": 1.,
        "rows": [{"id": "frame", "detections": [{"bbox": [1, 1, 5, 5], "confidence": bad}]}]}
    with pytest.raises(ValueError, match="confidence"):
        review.verify_predictions(prediction, [sample])
