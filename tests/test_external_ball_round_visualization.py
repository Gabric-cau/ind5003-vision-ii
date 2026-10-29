import json

from PIL import Image
import pytest

from scripts import visualize_external_ball_round as visual
from scripts.summarize_external_ball_round import indexed_rows
from scripts.train_ball_frame_detector import score_frame, summarize


def row(event, frame, hit, primary=True):
    reference = [20, 20, 30, 30]
    candidates = [{"bbox": reference, "confidence": .9}] if hit else []
    return {"video_id": event + "_clip", "annotation_id": f"frame_{frame}", "event_id": event,
            "image": f"{event}_{frame}.png", "reference_bbox": reference,
            "deduplicated_primary": primary, "candidates": candidates, **score_frame(reference, candidates)}


def test_selection_is_event_balanced_deterministic_and_deduplicated():
    identities = [("event_a", number) for number in range(5)] + [("event_b", 0), ("event_c", 0)]
    previous = [row(event, frame, False) for event, frame in identities]
    current = [row(event, frame, True) for event, frame in reversed(identities)]
    previous.append(row("event_0_duplicate", 0, False, primary=False))
    current.append(row("event_0_duplicate", 0, True, primary=False))
    selected, counts = visual.select_examples(indexed_rows(previous), indexed_rows(current))
    assert [item["event_id"] for item in selected] == ["event_a", "event_b", "event_c"]
    assert counts["corrected"] == {"eligible_frames": 7, "eligible_events": 3, "selected_frames": 3}
    assert counts["regressed"]["selected_frames"] == 0


def test_top1_uses_highest_confidence_and_checks_recorded_iou():
    item = row("event", 0, True)
    item["candidates"].insert(0, {"bbox": [100, 100, 110, 110], "confidence": .1})
    assert visual.top_candidate(item)["bbox"] == item["reference_bbox"]
    item["top1_iou"] = .2
    with pytest.raises(ValueError, match="highest-confidence"):
        visual.top_candidate(item)


def test_duplicate_or_unsafe_image_names_are_rejected(tmp_path):
    for folder in ("one", "two"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "frame.png").write_bytes(b"not decoded")
    with pytest.raises(ValueError, match="exactly one"):
        visual.image_paths(tmp_path, [{"image": "frame.png"}])
    with pytest.raises(ValueError, match="plain image filename"):
        visual.image_paths(tmp_path, [{"image": "../frame.png"}])


def completed_visual_round(tmp_path):
    directory = tmp_path / "round"
    directory.mkdir()
    roots = {}
    report = {"training_completed": True, "detector_weights_updated": True,
              "protocol": {"epochs": 1, "previous_weights_sha256": "a" * 64,
                           "new_match_evaluation_frames": 3, "regression_frames": 3},
              "candidate": {"completed_epochs": [1], "weights_sha256": "b" * 64}, "results": {}}
    for group in visual.GROUPS:
        root = tmp_path / group
        root.mkdir()
        roots[group] = root
        report["results"][group] = {}
        previous = [row("event_1", 0, False), row("event_2", 0, True), row("event_3", 0, False)]
        current = [row("event_1", 0, True), row("event_2", 0, False), row("event_3", 0, False)]
        for item in previous:
            Image.new("RGB", (320, 180), "#b0b0b0").save(root / item["image"])
        for model, rows, digest in (("previous_refit", previous, "a" * 64), ("new_round1", current, "b" * 64)):
            prediction = {"mode": "tiles", "weights_sha256": digest, "rows": rows, "prediction_uses_reference": False,
                          "metrics": {"all_frames": summarize(rows), "deduplicated_frames": summarize(rows)}}
            (directory / f"{group}_{model}_tiles.json").write_text(json.dumps(prediction))
            report["results"][group][model] = {"tiles": {key: value for key, value in prediction.items() if key != "rows"}}
    (directory / "report.json").write_text(json.dumps(report))
    return directory, roots


def test_cpu_contact_sheets_and_selection_are_written_without_modifying_inputs(tmp_path):
    directory, roots = completed_visual_round(tmp_path)
    inputs = {path: path.read_bytes() for path in directory.rglob("*.json")}
    output = tmp_path / "review"
    selection = visual.visualize(directory, roots, output)
    assert "already observed" in selection["diagnosis"]
    assert "no expected image hash" in selection["image_identity"]
    for group, data in selection["groups"].items():
        assert len(data["selected"]) == 3
        assert {item["category"] for item in data["selected"]} == set(visual.CATEGORIES)
        assert all(len(item["image_sha256"]) == 64 and item["width"] == 320 for item in data["selected"])
        assert (output / data["contact_sheet"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((output / "selection.json").read_text()) == selection
    assert all(path.read_bytes() == content for path, content in inputs.items())
    with pytest.raises(FileExistsError, match="preserved"):
        visual.visualize(directory, roots, output)


def test_missing_final_report_creates_no_review(tmp_path):
    directory = tmp_path / "running"
    directory.mkdir()
    output = tmp_path / "review"
    with pytest.raises(FileNotFoundError, match="incomplete"):
        visual.visualize(directory, {}, output)
    assert not output.exists()


def test_shared_missing_frame_is_rejected_even_when_old_new_still_pair(tmp_path):
    directory, roots = completed_visual_round(tmp_path)
    for model in ("previous_refit", "new_round1"):
        path = directory / f"arsenal_new_match_{model}_tiles.json"
        prediction = json.loads(path.read_text())
        prediction["rows"].pop(0)
        path.write_text(json.dumps(prediction))
    with pytest.raises(ValueError, match="frame count"):
        visual.prepare_selection(directory, roots)


def test_changed_row_scores_are_rejected_when_metrics_are_stale(tmp_path):
    directory, roots = completed_visual_round(tmp_path)
    path = directory / "arsenal_new_match_previous_refit_tiles.json"
    prediction = json.loads(path.read_text())
    prediction["rows"][0] = row("event_1", 0, True)
    path.write_text(json.dumps(prediction))
    with pytest.raises(ValueError, match="reported metrics"):
        visual.prepare_selection(directory, roots)
