import copy
import json
from pathlib import Path
import shutil

import pytest

from scripts import replay_offline_video_summaries as replay


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def original(tmp_path, monkeypatch):
    batch = tmp_path / "original"
    pipeline = batch / "events/core_01/attempt_001/pipeline"
    source_dir = pipeline / "source"
    source_dir.mkdir(parents=True)
    video = source_dir / "unseen_penalty.mp4"
    video.write_bytes(b"unchanged source video")
    source = {"clip_id": "unseen_penalty", "clip_filename": video.name,
              "clip_sha256": replay.sha(video)}
    write(source_dir / "source_manifest.json", {"clips": [source], "manual_annotations_used": False})
    stage = pipeline / "pose/unseen_penalty.json"
    write(stage, {"measurement": [1, 2, 3]})
    contact = {"prediction": {"time_seconds": 1.2}}
    write(pipeline / "trained_contact/prediction.json", contact)
    receipts = [{"stage": "pose", "exit_code": 0, "seconds": 45.}]
    write(pipeline / "progress.json", {"stages": receipts})
    analysis = {"input_sha256": source["clip_sha256"], "all_stages_executed_successfully": True,
        "manual_annotations_used": False, "required_outputs": {"keeper_dive_height": "unknown"},
        "source": source, "stage_receipts": receipts, "trained_contact": contact,
        "end_to_end_seconds": 123.4,
        "artifact_bindings": {"pose": {"path": "pose/unseen_penalty.json", "sha256": replay.sha(stage)}}}
    write(pipeline / "analysis.json", analysis)
    manifest = {"bindings": {"samples": [{"core_id": "core_01", "video_sha256": source["clip_sha256"]}],
        "source_code_sha256": {"original.py": "original-code"}, "execution": {"device": "cuda:0"}}}
    write(batch / "batch_manifest.json", manifest)
    digest = replay.sha(batch / "batch_manifest.json")
    receipt = {"core_id": "core_01", "status": "success", "batch_bindings_sha256": digest,
        "video_sha256": source["clip_sha256"], "analysis_path": str((pipeline / "analysis.json").relative_to(batch)),
        "analysis_sha256": replay.sha(pipeline / "analysis.json"), "wall_seconds": 124.}
    write(batch / "summary.json", {"complete": True, "failed_events": 0, "expected_events": 1,
        "batch_bindings_sha256": digest, "events": [receipt]})

    def summarize(source, folder, receipts):
        assert (folder / "pose").is_symlink()
        assert replay.read(folder / "pose/unseen_penalty.json") == {"measurement": [1, 2, 3]}
        result = copy.deepcopy(analysis)
        result["required_outputs"]["keeper_dive_height"] = {"value": "low", "status": "candidate"}
        return result

    monkeypatch.setattr(replay, "summarize", summarize)
    return batch, pipeline


def test_replay_preserves_source_and_runtime_but_rebinds_new_summary(original, tmp_path):
    batch, old_pipeline = original
    hashes = {p: replay.sha(p) for p in batch.rglob("*") if p.is_file()}
    output = tmp_path / "rebuilt"
    result = replay.replay(batch, output)
    predictions, samples, binding = replay.load_completed_batch(output)
    analysis = predictions["core_01"]
    assert result["complete"] is True
    assert result["new_neural_inference"] is False
    assert analysis["required_outputs"]["keeper_dive_height"]["value"] == "low"
    assert analysis["end_to_end_seconds"] == analysis["original_inference_runtime_seconds"] == 123.4
    assert analysis["end_to_end_seconds_basis"] == "original_inference_execution"
    assert analysis["summary_replay_seconds"] >= 0
    pipeline = output / result["events"][0]["analysis_path"]
    pipeline = pipeline.parent
    assert not (pipeline / "source").is_symlink()
    assert not (pipeline / "source/unseen_penalty.mp4").is_symlink()
    assert not Path((pipeline / "pose").readlink()).is_absolute()
    assert (pipeline / "pose").resolve() == old_pipeline / "pose"
    assert replay.read(output / "batch_manifest.json")["bindings"] == replay.read(batch / "batch_manifest.json")["bindings"]
    assert result["source_batch_sha256"] == replay.sha(batch / "batch_manifest.json")
    assert binding["events"][0]["analysis_sha256"] != result["events"][0]["original_inference_receipt"]["analysis_sha256"]
    assert all(replay.sha(p) == digest for p, digest in hashes.items())


@pytest.mark.parametrize("artifact", ["analysis.json", "pose/unseen_penalty.json", "trained_contact/prediction.json",
                                      "progress.json", "source/unseen_penalty.mp4"])
def test_replay_rejects_changed_evidence_before_creating_output(original, tmp_path, artifact):
    batch, pipeline = original
    path = pipeline / artifact
    path.write_bytes(path.read_bytes() + b" ")
    if artifact in ("trained_contact/prediction.json", "progress.json"):
        write(path, {"changed": True})
    output = tmp_path / "rejected"
    with pytest.raises((ValueError, KeyError), match="changed|differs|stages"):
        replay.replay(batch, output)
    assert not output.exists()


def test_replay_refuses_existing_or_nested_destination(original, tmp_path):
    batch, _ = original
    with pytest.raises(FileExistsError):
        replay.replay(batch, batch)
    with pytest.raises(ValueError, match="inside"):
        replay.replay(batch, batch / "new")


def test_changed_stage_during_replay_does_not_get_success_receipt(original, tmp_path, monkeypatch):
    batch, pipeline = original
    original_summary = replay.summarize

    def changed(source, folder, receipts):
        result = original_summary(source, folder, receipts)
        write(pipeline / "pose/unseen_penalty.json", {"measurement": [4]})
        return result

    monkeypatch.setattr(replay, "summarize", changed)
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="during summary replay"):
        replay.replay(batch, output)
    assert replay.read(output / "summary.json")["complete"] is False
    assert not (output / "events/core_01/receipt.json").exists()


def rebind_batch(batch):
    manifest = replay.read(batch / "batch_manifest.json")
    summary = replay.read(batch / "summary.json")
    summary["batch_bindings_sha256"] = replay.sha(batch / "batch_manifest.json")
    summary["expected_events"] = len(manifest["bindings"]["samples"])
    for receipt in summary["events"]:
        receipt["batch_bindings_sha256"] = summary["batch_bindings_sha256"]
        receipt["analysis_sha256"] = replay.sha(batch / receipt["analysis_path"])
    write(batch / "summary.json", summary)


def replacement_copy(original, tmp_path):
    batch, _ = original
    replacement = tmp_path / "replacement"
    shutil.copytree(batch, replacement)
    pipeline = replacement / "events/core_01/attempt_001/pipeline"
    stage = pipeline / "pose/unseen_penalty.json"
    write(stage, {"measurement": [9, 8, 7]})
    analysis = replay.read(pipeline / "analysis.json")
    analysis["artifact_bindings"]["pose"]["sha256"] = replay.sha(stage)
    analysis["end_to_end_seconds"] = 321.
    analysis["schema_version"] = "updated-analysis-v2"
    write(pipeline / "analysis.json", analysis)
    manifest = replay.read(replacement / "batch_manifest.json")
    manifest["bindings"]["source_code_sha256"] = {"original.py": "updated-code"}
    manifest["bindings"]["schema_version"] = "new-inference-bindings-v2"
    write(replacement / "batch_manifest.json", manifest)
    rebind_batch(replacement)
    return replacement, pipeline


def test_replacement_uses_its_stages_and_runtime_and_leaves_other_event_unchanged(original, tmp_path, monkeypatch):
    batch, base_pipeline = original
    replacement, replacement_pipeline = replacement_copy(original, tmp_path)
    second_pipeline = batch / "events/core_02/attempt_001/pipeline"
    shutil.copytree(base_pipeline, second_pipeline)
    manifest = replay.read(batch / "batch_manifest.json")
    manifest["bindings"]["samples"].append({**manifest["bindings"]["samples"][0], "core_id": "core_02"})
    write(batch / "batch_manifest.json", manifest)
    summary = replay.read(batch / "summary.json")
    summary["events"].append({**summary["events"][0], "core_id": "core_02",
        "analysis_path": "events/core_02/attempt_001/pipeline/analysis.json"})
    write(batch / "summary.json", summary)
    rebind_batch(batch)
    hashes = {p: replay.sha(p) for folder in (batch, replacement) for p in folder.rglob("*") if p.is_file()}

    def summarize(source, folder, receipts):
        original_pipeline = (folder / "pose").resolve().parent
        result = replay.read(original_pipeline / "analysis.json")
        result["selected_measurement"] = replay.read(folder / "pose/unseen_penalty.json")["measurement"]
        return result

    monkeypatch.setattr(replay, "summarize", summarize)
    output = tmp_path / "combined"
    result = replay.replay(batch, output, replacement)
    predictions, _, _ = replay.load_completed_batch(output)
    assert result["replacement_core_ids"] == ["core_01"]
    assert result["reused_base_core_ids"] == ["core_02"]
    assert result["new_neural_inference"] is False
    assert predictions["core_01"]["selected_measurement"] == [9, 8, 7]
    assert predictions["core_02"]["selected_measurement"] == [1, 2, 3]
    assert predictions["core_01"]["end_to_end_seconds"] == 321.
    assert predictions["core_02"]["end_to_end_seconds"] == 123.4
    assert result["original_inference_runtime_seconds"] == 444.4
    assert predictions["core_01"]["summary_replay"]["source_batch_sha256"] == replay.sha(replacement / "batch_manifest.json")
    assert predictions["core_02"]["summary_replay"]["source_batch_sha256"] == replay.sha(batch / "batch_manifest.json")
    assert (output / "events/core_01/attempt_001/pipeline/pose").resolve() == replacement_pipeline / "pose"
    assert (output / "events/core_02/attempt_001/pipeline/pose").resolve() == second_pipeline / "pose"
    assert all(replay.sha(p) == digest for p, digest in hashes.items())


@pytest.mark.parametrize("problem", ["different_video", "unexpected_event", "corrupt_stage"])
def test_replacement_rejects_wrong_identity_or_evidence_before_output(original, tmp_path, problem):
    batch, _ = original
    replacement, pipeline = replacement_copy(original, tmp_path)
    if problem == "corrupt_stage":
        write(pipeline / "pose/unseen_penalty.json", {"changed": True})
    else:
        manifest = replay.read(replacement / "batch_manifest.json")
        summary = replay.read(replacement / "summary.json")
        if problem == "unexpected_event":
            manifest["bindings"]["samples"][0]["core_id"] = "core_99"
            summary["events"][0]["core_id"] = "core_99"
        else:
            video = pipeline / "source/unseen_penalty.mp4"
            video.write_bytes(b"different actual video")
            digest = replay.sha(video)
            manifest["bindings"]["samples"][0]["video_sha256"] = digest
            summary["events"][0]["video_sha256"] = digest
            analysis = replay.read(pipeline / "analysis.json")
            analysis["input_sha256"] = analysis["source"]["clip_sha256"] = digest
            write(pipeline / "analysis.json", analysis)
            write(pipeline / "source/source_manifest.json", {"clips": [analysis["source"]], "manual_annotations_used": False})
        write(replacement / "batch_manifest.json", manifest)
        write(replacement / "summary.json", summary)
        rebind_batch(replacement)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="hash|subset|changed"):
        replay.replay(batch, output, replacement)
    assert not output.exists()
