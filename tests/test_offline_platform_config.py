import json
from types import SimpleNamespace

import pytest

from scripts import run_offline_video_batch as batch


@pytest.mark.parametrize("platform,configured,expected", [
    ("win32", ".venv-pose33/bin/python", ".venv-pose33/Scripts/python.exe"),
    ("linux", ".venv-pose33/bin/python", ".venv-pose33/bin/python"),
    ("win32", "custom/python.exe", "custom/python.exe"),
])
def test_load_config_selects_platform_interpreter_without_modifying_config(tmp_path, monkeypatch, platform, configured, expected):
    model = tmp_path / "model"
    model.write_text("test artifact")
    executable = tmp_path / expected
    executable.parent.mkdir(parents=True)
    executable.write_text("test interpreter")
    path = tmp_path / "bundle_config.json"
    path.write_text(json.dumps({"pose_model": "model", "ball_model": "model", "contact_model": "model",
                                "shoe_python": configured}))
    original = path.read_bytes()
    monkeypatch.setattr(batch, "sys", SimpleNamespace(platform=platform))
    assert batch.load_config(path)["shoe_python"] == str(executable.absolute())
    assert path.read_bytes() == original
