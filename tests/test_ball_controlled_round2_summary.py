import json
from pathlib import Path

import pytest

from scripts import summarize_ball_controlled_round2 as review
from scripts.train_ball_frame_detector import score_frame, summarize


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def prediction(model, mode, model_hash):
    hits = {"previous_refit": {0}, "new_round1": {2}, review.ARMS[0]: {0, 2},
            review.ARMS[1]: {0}, review.ARMS[2]: {2}}[model]
    rows = []
    for index in range(3):
        reference = [4, 4, 12, 12]
        candidates = [{"bbox": reference if index in hits else [20, 20, 30, 30], "confidence": .9}]
        if index not in hits and index != 2:
            candidates.append({"bbox": reference, "confidence": .6})
        rows.append({"image": f"frame_{index}.png", "video_id": "clip", "annotation_id": f"frame_{index}",
                     "event_id": f"event_{index // 2}", "reference_bbox": reference,
                     "deduplicated_primary": index != 1, "candidates": candidates, **score_frame(reference, candidates)})
    return {"mode": mode, "weights_sha256": model_hash, "prediction_uses_reference": False,
            "inference_code_sha256": "f" * 64, "seconds_including_model_load": 1.25, "rows": rows,
            "metrics": {scope: summarize([r for r in rows if scope == "all_frames" or r["deduplicated_primary"]])
                        for scope in review.SCOPES}}


def complete_run(tmp_path):
    baseline = tmp_path / "round1"
    directory = tmp_path / "round2"
    baseline.mkdir()
    directory.mkdir()
    hashes = {model: letter * 64 for model, letter in zip(review.MODELS, "abcde")}
    old = {"training_completed": True, "detector_weights_updated": True,
           "protocol": {"device": "mps", "evaluation_image_size": 1280, "evaluation_batch": 1,
                        "confidence": .05, "nms_iou": .6, "evaluation_modes": ["full", "tiles"],
                        "previous_weights_sha256": hashes["previous_refit"], "new_match_evaluation_frames": 3,
                        "regression_frames": 3}, "candidate": {"weights_sha256": hashes["new_round1"]}, "results": {}}
    for old_group in review.GROUPS.values():
        old["results"][old_group] = {}
        for model in review.BASELINES:
            old["results"][old_group][model] = {}
            for mode in review.MODES:
                result = prediction(model, mode, hashes[model])
                save(baseline / f"{old_group}_{model}_{mode}.json", result)
                old["results"][old_group][model][mode] = {k: v for k, v in result.items() if k != "rows"}
    save(baseline / "report.json", old)
    protocol = {"device": "mps", "epochs_each": 2, "expected_optimizer_steps": 114, "samples_per_epoch": 4,
                "arms": {arm: {} for arm in review.ARMS}, "round1_report_sha256": review.sha(baseline / "report.json"),
                "round1_weights_sha256": hashes["new_round1"], "previous_weights_sha256": hashes["previous_refit"],
                "evaluation": {"scope": "exposed_development_retests", "independent_test": False,
                               "all_arms_frozen_before_scoring": True, "reference_used_in_inference": False,
                               "imgsz": 1280, "batch": 1, "confidence": .05, "nms_iou": .6, "modes": ["full", "tiles"]},
                "limitations": ["Exposed development data, not independent tests."]}
    report = {"training_completed": True, "independent_test_evaluated": False, "protocol": protocol,
              "candidates": {}, "baseline_results": old["results"], "results": {}, "eligible_arms": [], "selected_candidate": None}
    sampling = [{"epoch": epoch, "samples": 4, "old": 2, "new": 2, "optimizer_steps": 50 if epoch == 1 else 114}
                for epoch in (1, 2)]
    for arm in review.ARMS:
        initial = hashes["previous_refit"] if arm == review.ARMS[2] else hashes["new_round1"]
        report["candidates"][arm] = {"completed_epochs": [1, 2], "optimizer_steps": 114, "changed_parameter_names": ["weight"],
            "maximum_observed_parameter_delta": .01, "round2_scores_seen_before_checkpoint_freeze": False,
            "checkpoint_selection": "fixed_last_epoch_2", "initial_sha256": initial, "weights_sha256": hashes[arm],
            "training_seconds": 2., "sampling": sampling}
        folder = directory / arm / "fit"
        folder.mkdir(parents=True)
        (folder / "results.csv").write_text("epoch,train/box_loss,train/cls_loss,train/dfl_loss\n1,2,3,1\n2,1,2,.5\n")
    for group in review.GROUPS:
        report["results"][group] = {}
        for arm in review.ARMS:
            report["results"][group][arm] = {}
            for mode in review.MODES:
                result = prediction(arm, mode, hashes[arm])
                if arm == review.ARMS[0]:
                    result["rows"].reverse()
                save(directory / f"{group}_{arm}_{mode}.json", result)
                report["results"][group][arm][mode] = {k: v for k, v in result.items() if k != "rows"}
    save(directory / "report.json", report)
    return directory, baseline


def test_summary_pairs_by_event_identity_and_distinguishes_candidate_oracle(tmp_path):
    directory, baseline = complete_run(tmp_path)
    result, _ = review.build_summary(directory, baseline)
    assert result["evaluation_scope"] == "exposed_development_retests"
    assert result["overall_project_accuracy_measured"] is False
    assert result["independent_test_evaluated"] is False
    for group in result["groups"].values():
        assert (group["all_frames"], group["deduplicated_frames"], group["event_count"]) == (3, 2, 2)
        previous = next(r for r in group["comparisons"] if r["model"] == "previous_refit" and r["mode"] == "tiles")
        assert previous["all_frames"]["oracle_candidate_hits"] == 2
        assert previous["deduplicated_frames"]["oracle_candidate_hits"] == 1
        assert previous["all_frames"]["candidate_ranking_gap"] == 1
        arm = group["paired_changes"]["new_round1"][review.ARMS[0]]["tiles"]["deduplicated_frames"]
        assert arm["event_0"]["corrected"] == 1
        assert arm["event_1"]["both_correct"] == 1
        regressed = group["paired_changes"]["new_round1"][review.ARMS[1]]["tiles"]["deduplicated_frames"]
        assert regressed["event_1"]["regressed"] == 1


@pytest.mark.parametrize("mutation", ["missing_result", "partial_epoch", "different_sampling", "wrong_initial", "bad_selection"])
def test_incomplete_or_changed_controlled_protocol_is_rejected(tmp_path, mutation):
    directory, baseline = complete_run(tmp_path)
    report = json.loads((directory / "report.json").read_text())
    if mutation == "missing_result":
        (directory / "arsenal_development_A_round1_natural_tiles.json").unlink()
    elif mutation == "partial_epoch":
        report["candidates"][review.ARMS[0]]["completed_epochs"] = [1]
    elif mutation == "different_sampling":
        report["candidates"][review.ARMS[2]]["sampling"][0]["old"] = 1
    elif mutation == "wrong_initial":
        report["candidates"][review.ARMS[0]]["initial_sha256"] = "0" * 64
    else:
        report["selected_candidate"] = review.ARMS[0]
    save(directory / "report.json", report)
    with pytest.raises((ValueError, FileNotFoundError)):
        review.build_summary(directory, baseline)


def test_oracle_recomputed_from_candidates_not_trusted_from_stored_metrics(tmp_path):
    directory, baseline = complete_run(tmp_path)
    prediction_path = directory / "arsenal_development_A_round1_natural_tiles.json"
    data = json.loads(prediction_path.read_text())
    data["rows"][0]["any_candidate_iou"] = 0
    save(prediction_path, data)
    with pytest.raises(ValueError, match="Candidate-derived"):
        review.build_summary(directory, baseline)


def test_mps_and_reference_binding_must_match(tmp_path):
    directory, baseline = complete_run(tmp_path)
    report = json.loads((directory / "report.json").read_text())
    report["protocol"]["evaluation"]["batch"] = 2
    save(directory / "report.json", report)
    with pytest.raises(ValueError, match="inference settings differ"):
        review.build_summary(directory, baseline)


def test_cpu_writer_makes_summary_csv_plots_and_optional_contact_sheets(tmp_path):
    from PIL import Image
    directory, baseline = complete_run(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        Image.new("RGB", (64, 48), "green").save(images / f"frame_{index}.png")
    output = tmp_path / "review"
    result = review.write_review(directory, baseline, output, {group: images for group in review.GROUPS})
    assert result["evaluation_scope"] == "exposed_development_retests"
    assert (output / "summary.json").is_file()
    assert (output / "comparisons.csv").is_file()
    assert (output / "paired_event_changes.csv").is_file()
    assert (output / "recall_vs_oracle.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "training_loss.png").is_file()
    assert len(list(output.glob("*_contact_sheet.png"))) == 6
    assert "not independent tests or overall project accuracy" in (output / "summary.md").read_text()
    with pytest.raises(FileExistsError):
        review.write_review(directory, baseline, output)


def test_running_job_is_not_reported_as_zero_results(tmp_path):
    running = tmp_path / "running"
    running.mkdir()
    output = tmp_path / "review"
    with pytest.raises(FileNotFoundError, match="incomplete"):
        review.write_review(running, running, output)
    assert not output.exists()
