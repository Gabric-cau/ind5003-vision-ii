import json

import pytest

from scripts import summarize_external_ball_round as summary
from scripts.train_ball_frame_detector import score_frame, summarize


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def completed_round(tmp_path):
    directory = tmp_path / "round"
    (directory / "fit").mkdir(parents=True)
    (directory / "fit/results.csv").write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss\n"
        "1,2.0,3.0,1.5,0\n2,1.5,2.0,1.0,0\n", encoding="utf-8")
    hashes = dict(zip(summary.MODELS, ("a" * 64, "b" * 64, "c" * 64)))
    report = {"training_completed": True, "detector_weights_updated": True,
              "protocol": {"epochs": 2, "evaluation_modes": list(summary.MODES),
                           "initial_weights_sha256": hashes["pretrained"],
                           "previous_weights_sha256": hashes["previous_refit"],
                           "new_match_evaluation_frames": 3, "regression_frames": 3,
                           "training_frames": 5, "limitations": ["Synthetic positive-frame test data"]},
              "candidate": {"completed_epochs": [1, 2], "weights_sha256": hashes["new_round1"],
                            "training_seconds": 2.5}, "results": {}}
    for group in summary.GROUPS:
        report["results"][group] = {}
        for model in summary.MODELS:
            report["results"][group][model] = {}
            for mode in summary.MODES:
                rows = []
                for index in range(3):
                    hit = (model == "previous_refit" and index == 0) or (model == "new_round1" and index > 0)
                    reference = [0, 0, 10, 10]
                    candidates = [{"bbox": reference, "confidence": .9}] if hit else []
                    rows.append({"video_id": group + "_clip", "annotation_id": f"frame_{index}",
                                 "image": f"frame_{index}.jpg", "event_id": f"event_{1 if index < 2 else 2}",
                                 "reference_bbox": reference, "deduplicated_primary": index != 1,
                                 "candidates": candidates, **score_frame(reference, candidates)})
                if model == "new_round1":
                    rows.reverse()
                prediction = {"mode": mode, "weights_sha256": hashes[model],
                              "seconds_including_model_load": 1.25, "rows": rows,
                              "metrics": {"all_frames": summarize(rows),
                                          "deduplicated_frames": summarize([r for r in rows if r["deduplicated_primary"]])}}
                write_json(directory / f"{group}_{model}_{mode}.json", prediction)
                report["results"][group][model][mode] = {key: value for key, value in prediction.items() if key != "rows"}
    write_json(directory / "report.json", report)
    return directory


def test_summary_reads_counts_and_pairs_reordered_rows_without_inflating_dedup(tmp_path):
    result = summary.build_summary(completed_round(tmp_path))
    for group in result["groups"].values():
        assert (group["all_frames"], group["deduplicated_frames"], group["event_count"]) == (3, 2, 2)
        current = next(row for row in group["comparisons"] if row["model"] == "new_round1" and row["mode"] == "full")
        assert current["all_frames"] == {"hits": 2, "denominator": 3, "recall": 2 / 3, "prediction_available_frames": 2}
        assert current["deduplicated_frames"] == {"hits": 1, "denominator": 2, "recall": .5, "prediction_available_frames": 1}
        paired = group["paired_previous_to_new"]["full"]
        assert paired["all_frames"]["event_1"]["corrected"] == 1
        assert paired["all_frames"]["event_1"]["regressed"] == 1
        assert paired["deduplicated_frames"]["event_1"]["corrected"] == 0
        assert paired["deduplicated_frames"]["event_1"]["regressed"] == 1
        assert paired["deduplicated_frames"]["event_2"]["corrected"] == 1
    assert result["overall_project_accuracy_measured"] is False


def test_missing_final_report_is_incomplete_not_zero_and_creates_no_output(tmp_path):
    directory = tmp_path / "running"
    directory.mkdir()
    output = tmp_path / "summary"
    with pytest.raises(FileNotFoundError, match="incomplete"):
        summary.write_summary(directory, output)
    assert not output.exists()


def test_missing_prediction_is_rejected(tmp_path):
    directory = completed_round(tmp_path)
    (directory / "arsenal_new_match_new_round1_tiles.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        summary.build_summary(directory)


def test_changed_reference_cannot_be_compared_as_the_same_frame(tmp_path):
    directory = completed_round(tmp_path)
    path = directory / "arsenal_new_match_new_round1_full.json"
    prediction = json.loads(path.read_text())
    prediction["rows"][0]["reference_bbox"] = [1, 1, 10, 10]
    write_json(path, prediction)
    with pytest.raises(ValueError, match="Frame binding differs"):
        summary.build_summary(directory)


def test_stale_metrics_in_prediction_or_report_are_rejected(tmp_path):
    directory = completed_round(tmp_path)
    path = directory / "arsenal_new_match_new_round1_full.json"
    prediction = json.loads(path.read_text())
    prediction["rows"][0].update(candidates=[], prediction_available=False, top1_iou=0., any_candidate_iou=0.)
    write_json(path, prediction)
    with pytest.raises(ValueError, match="metrics disagree"):
        summary.build_summary(directory)


def test_partial_training_csv_is_not_plotted_as_a_complete_round(tmp_path):
    directory = completed_round(tmp_path)
    (directory / "fit/results.csv").write_text("epoch,train/box_loss,train/cls_loss,train/dfl_loss\n1,2,3,1\n")
    with pytest.raises(ValueError, match="incomplete"):
        summary.build_summary(directory)


def test_summary_writes_json_markdown_and_cpu_loss_png(tmp_path):
    directory = completed_round(tmp_path)
    output = tmp_path / "summary"
    summary.write_summary(directory, output)
    assert (output / "training_loss.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    document = json.loads((output / "summary.json").read_text())
    assert document["training"]["losses"][1]["train/box_loss"] == 1.5
    text = (output / "summary.md").read_text()
    assert "ARS New Match" in text and "Old Core Development Retest" in text
    assert "not overall project accuracy" in text
    assert "validation/mAP placeholders" in text
    assert "no per-event timing" in text
