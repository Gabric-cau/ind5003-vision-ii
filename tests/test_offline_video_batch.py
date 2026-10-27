import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import run_offline_video_batch as batch


@pytest.fixture
def setup(tmp_path, monkeypatch):
    samples = []
    for index in (1, 2):
        video = tmp_path / f"core_{index:02d}.mp4"
        video.write_bytes(f"unique source {index}".encode())
        samples.append({"core_id": video.stem, "video_path": str(video), "video_sha256": batch.sha(video)})
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"reference_labels_included": False, "samples": samples}))
    weight = tmp_path / "weight.bin"
    weight.write_bytes(b"frozen weights")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({key: str(weight) for key in batch.REQUIRED_CONFIG}))
    args = SimpleNamespace(inputs=inputs, config=config, output=tmp_path / "batch", python=sys.executable,
                           device="cpu", threads=2, resume=False)
    state = {"calls": [], "fail_first": False, "code_hash": "fixed"}
    monkeypatch.setattr(batch, "code_bindings", lambda: {"scripts/analyze_penalty_video.py": state["code_hash"]})

    def process(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        video = Path(command[command.index("--video") + 1])
        state["calls"].append(video.stem)
        if state["fail_first"] and video.stem == "core_01":
            return SimpleNamespace(returncode=2)
        output.mkdir(parents=True)
        (output / "analysis.json").write_text(json.dumps({
            "input_sha256": batch.sha(video), "manual_annotations_used": False,
            "all_stages_executed_successfully": True, "required_outputs": {"goal_outcome": "unknown"},
            "all_teacher_outputs_established": False, "accuracy_validated": False}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, "run", process)
    return args, state, weight


def test_failed_video_does_not_stop_batch_and_resume_preserves_success(setup):
    args, state, _ = setup
    state["fail_first"] = True
    first = batch.run(args)
    assert first["complete"] is False
    assert first["successful_events"] == 1
    assert state["calls"] == ["core_01", "core_02"]
    prior = (args.output / first["events"][1]["analysis_path"]).read_bytes()
    args.resume = True
    state["fail_first"] = False
    second = batch.run(args)
    assert second["complete"] is True
    assert state["calls"] == ["core_01", "core_02", "core_01"]
    assert second["events"][0]["attempt"] == 2
    assert (args.output / first["events"][0]["log_path"]).exists()
    assert (args.output / second["events"][1]["analysis_path"]).read_bytes() == prior
    assert second["accuracy_evaluated"] is False


@pytest.mark.parametrize("changed", ["config", "weight", "code"])
def test_resume_rejects_changed_experiment_before_running_a_child(setup, changed):
    args, state, weight = setup
    batch.run(args)
    args.resume = True
    if changed == "config":
        args.config.write_text(args.config.read_text() + "\n")
    elif changed == "weight":
        weight.write_bytes(b"replacement weights")
    else:
        state["code_hash"] = "changed"
    with pytest.raises(ValueError, match="Resume refused"):
        batch.run(args)
    assert state["calls"] == ["core_01", "core_02"]


def test_resume_detects_changed_saved_prediction(setup):
    args, state, _ = setup
    result = batch.run(args)
    analysis = args.output / result["events"][0]["analysis_path"]
    analysis.write_text(analysis.read_text() + "\n")
    args.resume = True
    with pytest.raises(ValueError, match="changed after successful"):
        batch.run(args)
    assert len(state["calls"]) == 2


def test_labels_cannot_enter_batch_inputs(setup):
    args, state, _ = setup
    document = batch.read(args.inputs)
    document["samples"][0]["reference_contact_seconds"] = 1.8
    args.inputs.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="annotations"):
        batch.run(args)
    assert state["calls"] == []
    assert not args.output.exists()


def test_completed_process_with_wrong_video_output_is_a_failed_event(setup, monkeypatch):
    args, state, _ = setup
    original = batch.verify_analysis
    def check(path, digest):
        document = batch.read(path)
        document["input_sha256"] = "0" * 64
        path.write_text(json.dumps(document))
        return original(path, digest)
    monkeypatch.setattr(batch, "verify_analysis", check)
    result = batch.run(args)
    assert result["failed_events"] == 2
    assert result["complete"] is False
    assert len(state["calls"]) == 2


def test_config_preserves_virtual_environment_interpreter_symlink(setup):
    args, _, _ = setup
    executable = args.config.parent / "isolated" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    config = batch.read(args.config)
    config["shoe_python"] = "isolated/bin/python"
    args.config.write_text(json.dumps(config))
    loaded = batch.load_config(args.config)
    assert loaded["shoe_python"] == str(executable.absolute())
    assert loaded["shoe_python"] != str(executable.resolve())
