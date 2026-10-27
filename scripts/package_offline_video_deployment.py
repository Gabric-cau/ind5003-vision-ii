"""Package frozen source and bound inference weights without videos or environments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


MODEL_FILES = (
    "contact/deployment_metadata.json", "contact/deployment_weights.npz",
    "shoes/deployment_metadata.json", "shoes/deployment.joblib", "keeper.json", "encoder.pt",
    "ball/report.json", "ball/first_frame_difference_weights.npz",
    "height/report.json", "height/height_weights.npz", "camera/report.json", "camera/camera_weights.npz",
    "pose_landmarker_heavy.task",
)
INFERENCE_REPORT_SETTINGS = {
    "ball": ("frames", "width", "height", "max_shot_seconds", "scene_cut_threshold", "window_policy",
             "temporal_bins", "spatial_pool", "horizontal_flip"),
    "height": ("frames", "width", "height", "max_shot_seconds", "scene_cut_threshold", "window_policy",
               "temporal_bins", "spatial_pool", "horizontal_flip"),
    "camera": ("frames", "width", "height", "temporal_bins", "spatial_pool", "horizontal_flip"),
}
MAIN_REQUIREMENTS = """numpy==1.26.0
scipy==1.15.3
pandas==3.0.3
PyYAML==6.0.2
scikit-learn==1.9.0
opencv-python==4.11.0.86
Pillow==11.3.0
joblib==1.5.3
matplotlib==3.10.3
threadpoolctl==3.6.0
ultralytics==8.4.175
"""
SHOE_REQUIREMENTS = """numpy==2.4.6
scipy==1.15.3
pandas==3.0.3
PyYAML==6.0.2
scikit-learn==1.8.0
joblib==1.6.0
opencv-contrib-python==5.0.0.93
mediapipe==1.1.0
Pillow==11.3.0
matplotlib==3.10.3
threadpoolctl==3.6.0
"""

RUN_VIDEO = '''"""Run one unannotated video with the portable bundle's relative model config."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from scripts.run_offline_video_batch import load_config

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.batch_size) < 1 or not args.video.is_file():
        parser.error("Supply a real video and positive thread and batch counts")
    config = load_config(ROOT / "bundle_config.json")
    command = [sys.executable, "-m", "scripts.analyze_penalty_video", "--video", str(args.video.absolute()),
               "--output", str(args.output.absolute()), "--device", args.device, "--threads", str(args.threads),
               "--batch-size", str(args.batch_size)]
    for key, value in config.items():
        command.extend(["--" + key.replace("_", "-"), value])
    if args.dry_run:
        print(json.dumps({"command": command, "working_directory": str(ROOT)}))
        return
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)

if __name__ == "__main__":
    main()
'''

PREPARE_INPUTS = '''"""Create a source-hashed batch manifest without annotations or model answers."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new manifest path")
    samples, seen = [], set()
    for index, item in enumerate(args.videos, 1):
        path = item.resolve()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in seen:
            raise ValueError("Duplicate source video bytes in batch")
        seen.add(digest)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("Cannot open video: " + str(path))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not fps > 0 or min(frames, width, height) <= 0:
                raise ValueError("Invalid source video metadata")
        finally:
            capture.release()
        samples.append({"core_id": "video_" + str(index).zfill(3), "clip_id": path.stem,
                        "match_group": "unassigned", "video_path": str(path), "video_sha256": digest,
                        "fps": fps, "frame_count": frames, "width": width, "height": height,
                        "duration_seconds": frames / fps})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "unannotated-portable-batch-v1",
        "reference_labels_included": False, "samples": samples}, indent=2, allow_nan=False) + "\\n")
    print(json.dumps({"output": str(args.output), "videos": len(samples)}))

if __name__ == "__main__":
    main()
'''

VERIFY_BUNDLE = '''"""Verify every shipped artifact using only Python's standard library."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "bundle_manifest.json").read_text())
failures = []
for item in manifest["files"]:
    path = root / item["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        failures.append({"path": item["path"], "reason": "missing_or_unsafe_path"})
        continue
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
        failures.append({"path": item["path"], "reason": "content_changed"})
print(json.dumps({"checked_files": len(manifest["files"]), "failures": failures,
                  "clean_install_tested": False, "valid": not failures}, indent=2))
raise SystemExit(bool(failures))
'''

README = r"""# Portable Offline Penalty Video Analysis

Generated from the frozen `{snapshot_label}` snapshot on {date}. The code and model
weights are included. Source video files, standalone annotation datasets, account
configuration, virtual environments and dependency wheels are not included. After dependencies are installed, normal
inference uses local files and does not require a model download or login.

The ball, height and camera reports contain only inference settings and weight
hashes, plus the SHA-256 of each original report. They are not training reports
and do not contain the original per-event annotations or review messages.

This is a deployment package, not an accuracy report. A successful run can still
contain unknown, partial or incorrect candidates. Read the generated limitations.
The input is one already isolated penalty attempt starting with its shooting
view; this is not a full-match event detector. No test-video annotations are used.

## Model Sources

- Detection and pose weights: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
  with the [upstream license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- The heavy pose task: [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).
- `encoder.pt`: an exported frozen [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

These pretrained networks are upstream models. The fitted prediction heads are
project artifacts. Upstream terms still apply to the included models and code;
this package does not grant a blanket license over those components.

## Check the Download

Run from this extracted directory using Python 3.11. On Windows use
`py -3.11 verify_bundle.py`; on Linux or macOS:

```sh
python3.11 verify_bundle.py
```

`SHA256SUMS` also lists checksums. The build manifest records frozen source and
weight bindings. Do not edit the hashed feature implementations or replace model
weights independently: prediction rejects several incompatible combinations.

## Install Two Separate Python 3.11 Environments

Install 64-bit Python 3.11 first. Run the installer from the extracted bundle;
it creates `.venv` for the main pipeline and `.venv-pose33` for the shoe model.
For NVIDIA CUDA, install a compatible NVIDIA driver first. The installer checks
CUDA availability and a small GPU tensor operation; it does not install drivers.

Windows PowerShell or Command Prompt, using the Dell RTX 2060 driver 555.99:

```powershell
py -3.11 install.py --torch cu118
.\.venv\Scripts\python.exe run_video.py --video "C:\videos\penalty.mp4" --output runs\new_penalty --device cuda:0 --batch-size 1
```

Linux NVIDIA:

```sh
python3.11 install.py --torch cu128
.venv/bin/python run_video.py --video /path/to/penalty.mp4 --output runs/new_penalty --device cuda:0 --batch-size 1
```

`cu118` selects CUDA 11.8 PyTorch wheels and `cu128` selects CUDA 12.8 wheels.
Use `cu118` for the above Dell driver; use `cu128` for newer supported GPUs and
compatible drivers. Both use PyTorch 2.7.0 and torchvision 0.22.0, as listed in
the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).

CPU setup uses `py -3.11 install.py --torch cpu` on Windows or
`python3.11 install.py --torch cpu` on Linux/macOS, followed by `--device cpu`.
CUDA is not available on macOS. Batch size 1 reduces the memory used by the
unbound YOLO stages. The trained contact extractor retains its frozen settings.
If a GPU stage runs out of memory, use CPU with a new output directory.

The installer checks exact direct dependency versions, imports, `pip check`,
and the selected PyTorch wheel. It writes `environment_check.json` only after
both environments pass. Rerun with `--check-only` to validate them without
installing packages. That receipt is an environment check, not an end-to-end
video test or accuracy measurement. Packaging itself does not establish that a
clean install has passed; target-machine test receipts are supplied separately.
The previous Linux shoe dependency dry-run did not install the packages.

The requirements are direct pins, not a fully locked transitive dependency set.
Use this bundle's `requirements-main.txt`, `requirements-shoes.txt`, and selected
torch file, rather than the repository's general development requirements.
Main inference uses NumPy 1.26.0, scikit-learn 1.9.0 and PyTorch 2.7.0;
the serialized shoe model requires NumPy 2.4.6 and scikit-learn 1.8.0.
Keep these environments separate. Only `opencv-contrib-python` is installed in
the shoe environment; do not add `opencv-python` beside it.

The configured `.venv-pose33/bin/python` automatically maps to
`.venv-pose33/Scripts/python.exe` on Windows. No configuration editing is needed.
On Linux, retain the virtual-environment executable path instead of replacing
it with its resolved system-Python symlink target.

Linux OpenCV also requires operating-system shared libraries. Ubuntu 22.04
uses `libgl1` and `libglib2.0-0`, providing `libGL.so.1`, `libgthread-2.0.so.0`
and `libglib-2.0.so.0`. Install those with the OS package manager if missing.

Installation needs a package index. For an air-gapped target, download wheels
on a machine with the same OS/architecture/Python, copy them, and install into
the two environments with `--no-index --find-links`. This ZIP does not contain
Python, drivers or dependency wheels. Normal inference uses the bundled model
files and does not need a model download or login.

## One New Video

CPU:

```sh
.venv/bin/python run_video.py --video /path/to/penalty.mp4 --output runs/new_penalty --device cpu --threads 2
```

RTX 4090:

```sh
.venv/bin/python run_video.py --video /path/to/penalty.mp4 --output runs/new_penalty --device cuda:0 --threads 2
```

Use a new output directory for each run. `--dry-run` prints the assembled command
after checking configured files; it does not run models. `analysis.json` contains
the required output candidates, provenance, stage receipts and limitations.
`progress.json` and per-stage logs explain failed or incomplete stages.

## Batch

Create a label-free manifest from local videos once:

```sh
.venv/bin/python prepare_video_inputs.py --videos /path/to/clips/*.mp4 --output my_inputs.json
```

Then run the batch; change `cuda:0` to `cpu` for a local CPU run:

```sh
.venv/bin/python scripts/run_offline_video_batch.py --inputs my_inputs.json --config bundle_config.json --output runs/my_batch --device cuda:0 --threads 2
```

Repeat that same command with `--resume` after interruption. Resume verifies
input, model and source-code hashes, skips verified successful events, and keeps
attempt folders immutable. It does not silently overwrite an earlier attempt.
Changing inputs, code, model files, device or execution bindings requires a new
output directory. `summary.json` reports execution status; execution completion
must not be reported as prediction accuracy. Runtime reports from the ongoing
lab batch are supplied separately and are not fabricated in this package.
"""


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative_config():
    return {"schema_version": 1, "pose_model": "models/yolo11m-pose.pt", "ball_model": "models/yolo11m.pt",
            "contact_model": "models/contact", "shoe_model": "models/shoes",
            "pose33_model": "models/pose_landmarker_heavy.task", "shoe_python": ".venv-pose33/bin/python",
            "keeper_model": "models/keeper.json", "ball_category_model": "models/ball",
            "ball_height_model": "models/height", "encoder": "models/encoder.pt", "camera_model": "models/camera"}


def checked_copy(source, destination):
    if source.is_symlink() or not source.is_file():
        raise ValueError("Expected a regular source artifact: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha(source) != sha(destination):
        raise ValueError("Copied artifact differs from frozen source")


def export_inference_report(source, destination, model):
    if source.is_symlink() or not source.is_file():
        raise ValueError("Expected a regular source artifact: " + str(source))
    report = read(source)
    projected = {"schema_version": "portable-inference-report-v1",
                 "original_report_sha256": sha(source),
                 "portable_encoder_sha256": report["portable_encoder_sha256"],
                 "settings": {key: report["settings"][key] for key in INFERENCE_REPORT_SETTINGS[model]}}
    if model == "ball":
        projected["candidates"] = {"first_frame_difference": {
            "weights_sha256": report["candidates"]["first_frame_difference"]["weights_sha256"]}}
    else:
        projected["weights_sha256"] = report["weights_sha256"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(destination, projected)


def verify_model_contracts(root):
    checks = []

    def check(relative, expected):
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError("Frozen model/code binding mismatch: " + relative)
        checks.append(relative)

    contact = read(root / "models/contact/deployment_metadata.json")
    check("models/contact/deployment_weights.npz", contact["weights_sha256"])
    for name, digest in contact["visual_model_sha256"].items():
        check("models/" + name, digest)
    for name, digest in contact["extractor_code_sha256"].items():
        check(name, digest)
    shoe = read(root / "models/shoes/deployment_metadata.json")
    check("models/shoes/deployment.joblib", shoe["model_sha256"])
    check("src/shoe_supervised_model.py", shoe["evidence_contract"]["feature_code_sha256"])
    if sha(root / "models/pose_landmarker_heavy.task") not in shoe["evidence_contract"]["pose_model_sha256"]:
        raise ValueError("MediaPipe checkpoint differs from frozen shoe model")
    check("src/keeper_supervised_model.py", read(root / "models/keeper.json")["evidence_contract"]["feature_code_sha256"])
    ball = read(root / "models/ball/report.json")
    check("models/encoder.pt", ball["portable_encoder_sha256"])
    check("models/ball/first_frame_difference_weights.npz", ball["candidates"]["first_frame_difference"]["weights_sha256"])
    for name in ("height", "camera"):
        report = read(root / f"models/{name}/report.json")
        check("models/encoder.pt", report["portable_encoder_sha256"])
        check(f"models/{name}/{name}_weights.npz", report["weights_sha256"])
    return {"checked_bindings": checks, "all_passed": True}


def package(snapshot, detector_dir, output, archive, snapshot_label=None):
    snapshot, detector_dir, output, archive = map(Path, (snapshot, detector_dir, output, archive))
    snapshot_label = snapshot_label or snapshot.name
    if not snapshot_label.strip() or any(ord(char) < 32 for char in snapshot_label):
        raise ValueError("Snapshot label must be a nonempty single-line description")
    if output.exists() or archive.exists():
        raise FileExistsError("Preserve existing packages; choose new output paths")
    created = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True)
    for folder in ("src", "scripts"):
        files = sorted((snapshot / folder).rglob("*.py"))
        if not files:
            raise ValueError("Frozen code folder is missing: " + folder)
        for source in files:
            checked_copy(source, output / source.relative_to(snapshot))
    for relative in MODEL_FILES:
        model = Path(relative).parts[0]
        if relative == model + "/report.json" and model in INFERENCE_REPORT_SETTINGS:
            export_inference_report(snapshot / "models" / relative, output / "models" / relative, model)
        else:
            checked_copy(snapshot / "models" / relative, output / "models" / relative)
    for name in ("yolo11m.pt", "yolo11m-pose.pt"):
        checked_copy(detector_dir / name, output / "models" / name)
    model_checks = verify_model_contracts(output)
    write(output / "bundle_config.json", relative_config())
    text_files = {"README.md": README.format(date=created, snapshot_label=snapshot_label), "requirements-main.txt": MAIN_REQUIREMENTS,
        "requirements-shoes.txt": SHOE_REQUIREMENTS,
        "requirements-torch-cpu.txt": "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.7.0+cpu; sys_platform != 'darwin'\ntorchvision==0.22.0+cpu; sys_platform != 'darwin'\ntorch==2.7.0; sys_platform == 'darwin'\ntorchvision==0.22.0; sys_platform == 'darwin'\n",
        "requirements-torch-cu118.txt": "--extra-index-url https://download.pytorch.org/whl/cu118\ntorch==2.7.0+cu118\ntorchvision==0.22.0+cu118\n",
        "requirements-torch-cu128.txt": "--extra-index-url https://download.pytorch.org/whl/cu128\ntorch==2.7.0+cu128\ntorchvision==0.22.0+cu128\n",
        "run_video.py": RUN_VIDEO, "prepare_video_inputs.py": PREPARE_INPUTS, "verify_bundle.py": VERIFY_BUNDLE}
    for name, text in text_files.items():
        (output / name).write_text(text, encoding="utf-8")
    checked_copy(Path(__file__).with_name("bootstrap_portable_environment.py"), output / "install.py")
    files = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(output.rglob("*")) if p.is_file()]
    write(output / "bundle_manifest.json", {"schema_version": "offline-video-portable-bundle-v1",
        "created_at_utc": created, "frozen_snapshot": snapshot_label, "models_and_code_binding": model_checks,
        "files": files, "file_count_excluding_manifest_and_checksum_list": len(files),
        "payload_bytes": sum(row["bytes"] for row in files), "requires_existing_or_installed_python_environments": True,
        "python_version_observed": "3.11.15", "clean_install_tested": False,
        "new_video_accuracy_validated_by_packaging": False, "video_or_input_manifest_included": False,
        "accounts_credentials_environments_or_wheels_included": False,
        "shoe_opencv_observation": {"observed_cv2_runtime": "5.0.0", "observed_distribution": "opencv-contrib-python==5.0.0.93",
            "lab_also_had_system_opencv_distribution": True, "clean_environment_instruction": "install only contrib"}})
    sums = [f"{sha(p)}  {p.relative_to(output)}" for p in sorted(output.rglob("*")) if p.is_file()]
    (output / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
    receipt = {"bundle": str(output), "zip": str(archive), "zip_sha256": sha(archive),
               "zip_bytes": archive.stat().st_size, "created_at_utc": created,
               "model_bindings_passed": True, "clean_install_tested": False}
    write(archive.with_suffix(".receipt.json"), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--snapshot-label", help="Human-readable source snapshot label; defaults to folder name")
    args = parser.parse_args()
    print(json.dumps(package(args.snapshot, args.detector_dir, args.output, args.zip, args.snapshot_label), indent=2))


if __name__ == "__main__":
    main()
