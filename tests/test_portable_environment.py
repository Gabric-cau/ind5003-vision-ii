import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import bootstrap_portable_environment as installer


def fake_runtime(monkeypatch, platform="linux", version=(3, 11)):
    executable = r"C:\Python311\python.exe" if platform == "win32" else "base-python"
    if platform == "win32":
        monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setattr(installer, "sys", SimpleNamespace(platform=platform, version_info=version,
                                                         executable=executable, stderr=sys.stderr))


@pytest.mark.parametrize("platform,variant", [("win32", "cu118"), ("win32", "cu128"), ("linux", "cpu"), ("darwin", "cpu")])
def test_install_creates_two_environments_and_checks_them_separately(tmp_path, monkeypatch, platform, variant):
    fake_runtime(monkeypatch, platform)
    monkeypatch.setenv("PYTHONPATH", "unrelated-packages")
    monkeypatch.setenv("PIP_TARGET", "unrelated-target")
    monkeypatch.setenv("PIP_CACHE_DIR", "unrelated-windows-store-cache")
    for name in ("MPLCONFIGDIR", "YOLO_CONFIG_DIR", "TORCH_HOME"):
        monkeypatch.setenv(name, "unrelated-user-cache")
    for name in ("main", "shoes"):
        (tmp_path / f"requirements-{name}.txt").write_text("numpy==1.0\n")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert "PYTHONPATH" not in kwargs["env"] and "PIP_TARGET" not in kwargs["env"]
        assert kwargs["env"]["PIP_CACHE_DIR"] == str(tmp_path / ".pip-cache")
        for name, folder in (("MPLCONFIGDIR", "matplotlib"), ("YOLO_CONFIG_DIR", "ultralytics"),
                             ("TORCH_HOME", "torch")):
            assert kwargs["env"][name] == str(tmp_path / ".runtime-cache" / folder)
        if command[1:3] == ["-m", "venv"]:
            directory = Path(command[3])
            executable = installer.environment_python(tmp_path, directory.name)
            executable.parent.mkdir(parents=True)
            executable.write_text("fake interpreter")
            (directory / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
        payload = {"imports_passed": True, "versions": {"numpy": "1.0"}}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(installer.subprocess, "run", run)
    receipt = installer.install(tmp_path, variant)
    assert receipt["all_environment_checks_passed"] is True
    assert receipt["end_to_end_inference_tested"] is False
    assert receipt["accuracy_validated"] is False
    assert set(receipt["environments"]) == {"main", "shoes"}
    assert json.loads((tmp_path / "environment_check.json").read_text()) == receipt
    assert installer.os.environ["PIP_CACHE_DIR"] == "unrelated-windows-store-cache"
    assert all(installer.os.environ[name] == "unrelated-user-cache"
               for name in ("MPLCONFIGDIR", "YOLO_CONFIG_DIR", "TORCH_HOME"))
    pip_installs = [command for command, _ in calls if command[1:4] == ["-m", "pip", "install"]]
    torch_installs = [command for command in pip_installs if any("requirements-torch-" in arg for arg in command)]
    assert len(torch_installs) == 1
    assert torch_installs[0][0] == str(installer.environment_python(tmp_path, ".venv"))
    assert torch_installs[0][-1].endswith(f"requirements-torch-{variant}.txt")
    probes = [json.loads(command[-1]) for command, _ in calls if "-c" in command and command[2] == installer.CHECK_PROGRAM]
    assert probes[0]["torch"] == variant and probes[1]["torch"] is None
    assert "mediapipe" in probes[1]["imports"] and "torch" not in probes[1]["imports"]


def test_installer_rejects_wrong_python_before_running_commands(tmp_path, monkeypatch):
    fake_runtime(monkeypatch, version=(3, 14))
    with pytest.raises(RuntimeError, match="Python 3.11"):
        installer.install(tmp_path, "cpu")
    assert not list(tmp_path.iterdir())


def test_windows_child_path_excludes_inherited_conda_without_changing_parent(tmp_path, monkeypatch):
    fake_runtime(monkeypatch, "win32")
    inherited = r"C:\Users\reviewer\miniconda3\Library\bin;C:\unrelated\Scripts"
    monkeypatch.setenv("PATH", inherited)
    original = dict(installer.os.environ)
    env = installer.isolated_environment(tmp_path)
    assert env["PATH"].split(";") == [str(tmp_path / ".venv/Scripts"),
        str(tmp_path / ".venv-pose33/Scripts"), r"C:\Python311", r"C:\Windows\System32", r"C:\Windows"]
    assert "miniconda" not in env["PATH"] and "unrelated" not in env["PATH"]
    assert dict(installer.os.environ) == original


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_child_preserves_path(tmp_path, monkeypatch, platform):
    fake_runtime(monkeypatch, platform)
    inherited = "/custom/python/bin:/usr/local/bin:/usr/bin:/bin"
    monkeypatch.setenv("PATH", inherited)
    assert installer.isolated_environment(tmp_path)["PATH"] == inherited
    assert installer.os.environ["PATH"] == inherited


@pytest.mark.parametrize("system_root", [None, "Windows", r"C:\Windows;C:\unrelated"])
def test_windows_child_requires_valid_system_root(tmp_path, monkeypatch, system_root):
    fake_runtime(monkeypatch, "win32")
    if system_root is None:
        monkeypatch.delenv("SystemRoot")
    else:
        monkeypatch.setenv("SystemRoot", system_root)
    with pytest.raises(ValueError, match="SystemRoot"):
        installer.isolated_environment(tmp_path)


def test_check_only_never_creates_a_missing_environment(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(FileNotFoundError, match="Missing environment"):
        installer.install(tmp_path, "cpu", check_only=True)
    assert len(calls) == 1 and calls[0][-1].endswith("verify_bundle.py")
    assert not (tmp_path / "environment_check.json").exists()


def test_installer_rejects_shared_system_environment(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    executable = installer.environment_python(tmp_path, ".venv")
    executable.parent.mkdir(parents=True)
    executable.write_text("fake interpreter")
    (tmp_path / ".venv/pyvenv.cfg").write_text("include-system-site-packages = true\n")
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: calls.append(command))
    with pytest.raises(ValueError, match="must not inherit"):
        installer.install(tmp_path, "cpu")
    assert len(calls) == 1
    assert not (tmp_path / "environment_check.json").exists()


def test_failed_import_check_does_not_write_success_receipt(tmp_path, monkeypatch):
    fake_runtime(monkeypatch)
    executable = installer.environment_python(tmp_path, ".venv")
    executable.parent.mkdir(parents=True)
    executable.write_text("fake interpreter")
    (tmp_path / ".venv/pyvenv.cfg").write_text("include-system-site-packages = false\n")
    (tmp_path / "requirements-main.txt").write_text("numpy==1.0\n")

    def run(command, **kwargs):
        if "-c" in command and command[2] == installer.CHECK_PROGRAM:
            raise subprocess.CalledProcessError(1, command, stderr="ImportError: missing runtime library")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(installer.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        installer.install(tmp_path, "cpu", check_only=True)
    assert not (tmp_path / "environment_check.json").exists()
