"""Install and check the portable bundle's two isolated Python 3.11 environments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import ntpath
import os
from pathlib import Path
import subprocess
import sys


CHECK_PROGRAM = r'''
import contextlib
import importlib
from importlib.metadata import version
import json
import sys

request = json.loads(sys.argv[1])
if sys.version_info[:2] != (3, 11) or sys.prefix == sys.base_prefix:
    raise RuntimeError("Expected an isolated Python 3.11 environment")
versions = {name: version(name) for name in request["versions"]}
mismatches = {name: {"expected": expected, "installed": versions[name]}
              for name, expected in request["versions"].items() if versions[name] != expected}
if mismatches:
    raise RuntimeError("Dependency versions differ: " + json.dumps(mismatches))
with contextlib.redirect_stdout(sys.stderr):
    for name in request["imports"]:
        importlib.import_module(name)
result = {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
          "base_prefix": sys.base_prefix, "versions": versions, "imports_passed": True}
if request.get("torch"):
    import torch
    import torchvision
    expected_suffix = "+" + request["torch"]
    expected_torch = "2.7.0" + ("" if sys.platform == "darwin" else expected_suffix)
    expected_vision = "0.22.0" + ("" if sys.platform == "darwin" else expected_suffix)
    if str(torch.__version__) != expected_torch or str(torchvision.__version__) != expected_vision:
        raise RuntimeError("Unexpected PyTorch wheel variant or version")
    result["torch"] = {"version": str(torch.__version__), "torchvision": str(torchvision.__version__),
                       "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if request["torch"] != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the NVIDIA driver or install --torch cpu")
        result["torch"].update(gpu_name=torch.cuda.get_device_name(0),
                              capability=list(torch.cuda.get_device_capability(0)))
        value = torch.ones(1, device="cuda:0") + 1
        torch.cuda.synchronize()
        if value.item() != 2:
            raise RuntimeError("CUDA tensor check failed")
print(json.dumps(result))
'''


def environment_python(root, name, platform=None):
    platform = sys.platform if platform is None else platform
    return root / name / ("Scripts/python.exe" if platform == "win32" else "bin/python")


def pinned_versions(path):
    versions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("==")
        if not separator or not name or not value or ";" in value:
            raise ValueError("Expected direct exact dependency pins in " + str(path))
        versions[name] = value
    return versions


def isolated_environment(root):
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PIP_USER", "PIP_TARGET", "PIP_PREFIX"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    if sys.platform == "win32":
        system_root = next((value for key, value in env.items() if key.upper() == "SYSTEMROOT"), None)
        if not system_root or not ntpath.isabs(system_root) or ";" in system_root:
            raise ValueError("Windows installation requires an absolute SystemRoot environment value")
        base_python = ntpath.dirname(sys.executable)
        if not ntpath.isabs(base_python):
            raise ValueError("Windows installation requires an absolute Python executable path")
        # pip resolves every PATH entry while installing scripts; stale Conda junctions can fail that step.
        for key in list(env):
            if key.upper() == "PATH":
                del env[key]
        env["PATH"] = ";".join((str(environment_python(root, ".venv").parent),
                               str(environment_python(root, ".venv-pose33").parent), base_python,
                               ntpath.join(system_root, "System32"), system_root))
    return env


def install(root, torch_variant, check_only=False):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Run this installer with Python 3.11: py -3.11 install.py (Windows) or python3.11 install.py")
    if torch_variant != "cpu" and sys.platform not in ("win32", "linux"):
        raise ValueError("CUDA environments require Windows or Linux; use --torch cpu")
    root = root.absolute()
    env = isolated_environment(root)
    env["PIP_CACHE_DIR"] = str(root / ".pip-cache")

    def run(command, capture=False):
        display = [command[0], "-c", "environment check"] if "-c" in command else command
        print("Running: " + subprocess.list2cmdline([str(value) for value in display]), flush=True)
        try:
            return subprocess.run([str(value) for value in command], cwd=root, env=env, check=True,
                                  capture_output=capture, text=True, encoding="utf-8")
        except subprocess.CalledProcessError as error:
            if capture:
                print((error.stdout or "") + (error.stderr or ""), file=sys.stderr)
            raise

    run([sys.executable, root / "verify_bundle.py"])
    specifications = (
        ("main", ".venv", "requirements-main.txt", ["numpy", "scipy", "pandas", "yaml", "sklearn",
            "cv2", "PIL", "joblib", "matplotlib", "threadpoolctl", "ultralytics", "torch", "torchvision"]),
        ("shoes", ".venv-pose33", "requirements-shoes.txt", ["numpy", "scipy", "pandas", "yaml",
            "sklearn", "joblib", "cv2", "mediapipe", "PIL", "matplotlib", "threadpoolctl"]),
    )
    receipt = {"schema_version": "portable-environment-check-v1",
               "checked_at_utc": datetime.now(timezone.utc).isoformat(), "platform": sys.platform,
               "torch_variant": torch_variant, "installation_performed": not check_only,
               "end_to_end_inference_tested": False, "accuracy_validated": False, "environments": {}}
    for label, name, requirements, imports in specifications:
        directory = root / name
        python = environment_python(root, name)
        if not directory.exists():
            if check_only:
                raise FileNotFoundError("Missing environment: " + str(directory))
            run([sys.executable, "-m", "venv", directory])
        config = directory / "pyvenv.cfg"
        if not python.is_file() or not config.is_file():
            raise ValueError("Existing environment is incomplete: " + str(directory))
        settings = dict(line.strip().split("=", 1) for line in config.read_text().splitlines() if "=" in line)
        settings = {key.strip(): value.strip() for key, value in settings.items()}
        if settings.get("include-system-site-packages", "").lower() != "false":
            raise ValueError("Environment must not inherit system packages: " + str(directory))
        run([python, "-c", "import sys; assert sys.version_info[:2] == (3, 11) and sys.prefix != sys.base_prefix, 'Expected an isolated Python 3.11 environment'"])
        if not check_only:
            run([python, "-m", "pip", "install", "--upgrade", "pip"])
            if label == "main":
                run([python, "-m", "pip", "install", "-r", root / ("requirements-torch-" + torch_variant + ".txt")])
            run([python, "-m", "pip", "install", "-r", root / requirements])
        run([python, "-m", "pip", "check"])
        request = {"versions": pinned_versions(root / requirements), "imports": imports,
                   "torch": torch_variant if label == "main" else None}
        result = run([python, "-c", CHECK_PROGRAM, json.dumps(request)], capture=True)
        receipt["environments"][label] = json.loads(result.stdout)
    receipt["all_environment_checks_passed"] = True
    destination = root / "environment_check.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(destination), "all_environment_checks_passed": True,
                      "end_to_end_inference_tested": False}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", choices=("cpu", "cu118", "cu128"), default="cpu")
    parser.add_argument("--check-only", action="store_true", help="Verify existing environments without installing packages")
    args = parser.parse_args()
    install(Path(__file__).resolve().parent, args.torch, args.check_only)


if __name__ == "__main__":
    main()
