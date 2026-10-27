import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from scripts.package_offline_video_deployment import (
    INFERENCE_REPORT_SETTINGS, package, relative_config, sha, verify_model_contracts,
)


def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    detector = tmp_path / "detectors"

    def file(relative, content=b"frozen-test-weight"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return sha(path)

    def doc(relative, value):
        file(relative, json.dumps(value).encode())

    shoe_hash = file("src/shoe_supervised_model.py", b"# frozen shoe feature\n")
    keeper_hash = file("src/keeper_supervised_model.py", b"# frozen keeper feature\n")
    contact_code = file("src/contact_video_features.py", b"# frozen contact feature\n")
    file("scripts/analyze_penalty_video.py", b"# entry point\n")
    batch_source = Path(__file__).resolve().parents[1] / "scripts/run_offline_video_batch.py"
    file("scripts/run_offline_video_batch.py", batch_source.read_bytes())
    file("inputs.json", b"private-input-manifest")
    file("lab_config.json", b"private-lab-paths")
    file("src/.env", b"must-not-copy")
    file("models/features.npz", b"not-needed-at-inference")
    detector.mkdir()
    for name in ("yolo11m.pt", "yolo11m-pose.pt"):
        (detector / name).write_bytes(name.encode())
    doc("models/contact/deployment_metadata.json", {
        "weights_sha256": file("models/contact/deployment_weights.npz"),
        "visual_model_sha256": {p.name: sha(p) for p in detector.iterdir()},
        "extractor_code_sha256": {"src/contact_video_features.py": contact_code}})
    task = file("models/pose_landmarker_heavy.task")
    doc("models/shoes/deployment_metadata.json", {"model_sha256": file("models/shoes/deployment.joblib"),
        "evidence_contract": {"feature_code_sha256": shoe_hash, "pose_model_sha256": [task]}})
    doc("models/keeper.json", {"evidence_contract": {"feature_code_sha256": keeper_hash}})
    encoder = file("models/encoder.pt")
    settings = {"frames": 64, "width": 512, "height": 288, "max_shot_seconds": 20.0,
                "scene_cut_threshold": 30.0, "window_policy": "automatic_first_shot",
                "temporal_bins": 4, "spatial_pool": [2, 2], "horizontal_flip": False}
    doc("models/ball/report.json", {"portable_encoder_sha256": encoder, "settings": settings, "candidates": {
        "first_frame_difference": {"weights_sha256": file("models/ball/first_frame_difference_weights.npz")}}})
    for name in ("height", "camera"):
        doc(f"models/{name}/report.json", {"portable_encoder_sha256": encoder,
            "settings": {key: settings[key] for key in INFERENCE_REPORT_SETTINGS[name]},
            "weights_sha256": file(f"models/{name}/{name}_weights.npz")})
    return root, detector


def build(tmp_path):
    frozen, detectors = snapshot(tmp_path)
    output, archive = tmp_path / "portable_bundle", tmp_path / "portable.zip"
    result = package(frozen, detectors, output, archive)
    return frozen, output, archive, result


def test_relative_model_config_preserves_environment_executable():
    config = relative_config()
    assert config["shoe_python"] == ".venv-pose33/bin/python"
    assert all(not Path(v).is_absolute() for k, v in config.items() if k != "schema_version")


def test_package_contains_bound_artifacts_not_inputs_accounts_or_caches(tmp_path):
    frozen, output, archive, result = build(tmp_path)
    assert verify_model_contracts(output)["all_passed"]
    assert result["clean_install_tested"] is False
    assert result["zip_sha256"] == sha(archive)
    assert not (output / "inputs.json").exists()
    assert not (output / "lab_config.json").exists()
    assert not (output / "src/.env").exists()
    assert not (output / "models/features.npz").exists()
    assert not (output / ".venv-pose33").exists()
    assert (output / "install.py").is_file()
    assert sha(output / "src/shoe_supervised_model.py") == sha(frozen / "src/shoe_supervised_model.py")
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert all(name.startswith("portable_bundle/") and ".." not in Path(name).parts for name in zipped.namelist())
        assert "portable_bundle/bundle_config.json" in zipped.namelist()
    result = subprocess.run([sys.executable, str(output / "verify_bundle.py")], capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["valid"] is True


def test_verifier_rejects_changed_artifact(tmp_path):
    _, output, _, _ = build(tmp_path)
    (output / "models/yolo11m.pt").write_bytes(b"changed")
    result = subprocess.run([sys.executable, str(output / "verify_bundle.py")], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["failures"][0]["path"] == "models/yolo11m.pt"


def test_report_export_removes_private_training_fields_and_preserves_originals(tmp_path):
    frozen, detectors = snapshot(tmp_path)
    original_reports = {}
    private = "private-review-message-and-local-path"
    for model in ("ball", "height", "camera"):
        path = frozen / "models" / model / "report.json"
        report = json.loads(path.read_text())
        report.update(admitted_cores=[{"raw_user_message": private, "reviewer_id": private}],
                      samples=[{"raw_human_label": private}], local_path=private)
        report["settings"]["source_path"] = private
        if model == "ball":
            report["candidates"]["first_frame_difference"]["predictions"] = [private]
            report["candidates"]["unused_candidate"] = {"raw_human_label": private}
        path.write_text(json.dumps(report))
        original_reports[model] = path.read_bytes()
    output = tmp_path / "bundle"
    package(frozen, detectors, output, tmp_path / "bundle.zip")
    assert verify_model_contracts(output)["all_passed"]
    for model, original in original_reports.items():
        source = frozen / "models" / model / "report.json"
        exported_text = (output / "models" / model / "report.json").read_text()
        report = json.loads(exported_text)
        assert source.read_bytes() == original
        assert private not in exported_text
        assert report["schema_version"] == "portable-inference-report-v1"
        assert report["original_report_sha256"] == sha(source)
        assert set(report["settings"]) == set(INFERENCE_REPORT_SETTINGS[model])
        expected = {"schema_version", "original_report_sha256", "portable_encoder_sha256", "settings"}
        expected.add("candidates" if model == "ball" else "weights_sha256")
        assert set(report) == expected
        if model == "ball":
            assert set(report["candidates"]) == {"first_frame_difference"}
            assert set(report["candidates"]["first_frame_difference"]) == {"weights_sha256"}
    ball = json.loads((output / "models/ball/report.json").read_text())
    height = json.loads((output / "models/height/report.json").read_text())
    assert ball["settings"] == height["settings"]


def test_packaging_rejects_changed_frozen_feature_code(tmp_path):
    frozen, detectors = snapshot(tmp_path)
    (frozen / "src/contact_video_features.py").write_text("# incompatible\n")
    with pytest.raises(ValueError, match="binding mismatch"):
        package(frozen, detectors, tmp_path / "bundle", tmp_path / "bundle.zip")


def test_packaging_rejects_existing_output(tmp_path):
    frozen, detectors = snapshot(tmp_path)
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        package(frozen, detectors, output, tmp_path / "bundle.zip")


def test_packaging_rejects_source_symlink(tmp_path):
    frozen, detectors = snapshot(tmp_path)
    (frozen / "src/linked.py").symlink_to(frozen / "src/contact_video_features.py")
    with pytest.raises(ValueError, match="regular source"):
        package(frozen, detectors, tmp_path / "bundle", tmp_path / "bundle.zip")


def test_run_wrapper_preserves_shoe_venv_symlink_and_uses_main_interpreter(tmp_path):
    _, output, _, _ = build(tmp_path)
    executable = output / ".venv-pose33/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(Path(sys.executable).resolve())
    video = tmp_path / "test video.mp4"
    video.write_bytes(b"source-test-fixture")
    result = subprocess.run([sys.executable, str(output / "run_video.py"), "--video", str(video),
                             "--output", str(tmp_path / "run"), "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["command"]
    assert command[0] == sys.executable
    assert command[command.index("--batch-size") + 1] == "1"
    assert command[command.index("--shoe-python") + 1] == str(executable)
    assert str(executable.resolve()) != str(executable)
    result = subprocess.run([sys.executable, str(output / "run_video.py"), "--video", str(video),
                             "--output", str(tmp_path / "run"), "--batch-size", "2", "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["command"]
    assert command[command.index("--batch-size") + 1] == "2"


def test_manifest_builder_creates_label_free_inputs_and_rejects_duplicates(tmp_path):
    import cv2
    import numpy as np

    _, output, _, _ = build(tmp_path)
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 32))
    assert writer.isOpened()
    for value in range(3):
        writer.write(np.full((32, 32, 3), value * 50, dtype=np.uint8))
    writer.release()
    manifest = tmp_path / "my_inputs.json"
    result = subprocess.run([sys.executable, str(output / "prepare_video_inputs.py"), "--videos", str(video),
                             "--output", str(manifest)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(manifest.read_text())
    assert data["reference_labels_included"] is False and len(data["samples"]) == 1
    assert data["samples"][0]["video_sha256"] == sha(video)
    from scripts.run_offline_video_batch import load_inputs
    assert load_inputs(manifest)[0]["frame_count"] == 3
    duplicate = tmp_path / "duplicate.avi"
    shutil.copyfile(video, duplicate)
    result = subprocess.run([sys.executable, str(output / "prepare_video_inputs.py"), "--videos", str(video), str(duplicate),
                             "--output", str(tmp_path / "duplicates.json")], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "duplicates.json").exists()


def test_snapshot_label_and_linux_dependency_limits_are_recorded(tmp_path):
    frozen, detectors = snapshot(tmp_path)
    output = tmp_path / "bundle_v3"
    package(frozen, detectors, output, tmp_path / "v3.zip", snapshot_label="deployment_v3")
    manifest = json.loads((output / "bundle_manifest.json").read_text())
    readme = (output / "README.md").read_text()
    assert manifest["frozen_snapshot"] == "deployment_v3"
    assert "`deployment_v3`" in readme
    assert "`libGL.so.1`" in readme and "`libglib-2.0.so.0`" in readme
    assert "did not install" in readme
    assert "py -3.11 install.py --torch cu118" in readme
    assert "torch==2.7.0+cu118" in (output / "requirements-torch-cu118.txt").read_text()
    assert r'.\.venv\Scripts\python.exe' in readme
    assert r'C:\videos\penalty.mp4' in readme
    assert not any(ord(character) < 32 and character not in "\n\t" for character in readme)
    assert manifest["clean_install_tested"] is False
