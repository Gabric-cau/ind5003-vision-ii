"""Run unannotated penalty clips with immutable attempt folders and checked resume."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIG = {"pose_model", "ball_model", "contact_model"}
OPTIONAL_CONFIG = {"shoe_model", "pose33_model", "shoe_python", "keeper_model",
                   "ball_category_model", "ball_height_model", "encoder", "camera_model"}
SAMPLE_KEYS = {"core_id", "clip_id", "match_group", "video_path", "video_sha256",
               "fps", "frame_count", "width", "height", "duration_seconds"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def resolve_path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Configured paths must be nonempty strings")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_inputs(path):
    document = read(path)
    if document.get("reference_labels_included") is True:
        raise ValueError("Batch inference requires a label-free input manifest")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Input manifest must contain nonempty samples")
    output, ids, videos = [], set(), set()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) - SAMPLE_KEYS:
            raise ValueError("Unexpected input fields; annotations must not enter batch inference")
        cid = sample.get("core_id")
        if (not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cid)
                or cid in ids):
            raise ValueError("core_id must be a unique safe directory name")
        digest = sample.get("video_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in videos:
            raise ValueError("Each event needs a unique video SHA-256")
        video = resolve_path(sample.get("video_path"), path.parent)
        if not video.is_file() or sha(video) != digest:
            raise ValueError(f"Video hash mismatch or missing video: {cid}")
        ids.add(cid)
        videos.add(digest)
        output.append({**sample, "video_path": str(video)})
    return output


def load_config(path):
    document = read(path)
    if not isinstance(document, dict):
        raise ValueError("Bundle config must be a flat object")
    if document.get("schema_version", 1) != 1:
        raise ValueError("Unsupported bundle schema version")
    keys = set(document) - {"schema_version"}
    if not REQUIRED_CONFIG <= keys or keys - REQUIRED_CONFIG - OPTIONAL_CONFIG:
        raise ValueError("Bundle keys differ from supported analyzer model arguments")
    config = {}
    for key, value in document.items():
        if key == "schema_version" or value is None:
            continue
        if key == "shoe_python":
            # Resolving a venv executable symlink selects its base environment.
            executable = Path(value).expanduser()
            if sys.platform == "win32" and value == ".venv-pose33/bin/python":
                executable = Path(".venv-pose33/Scripts/python.exe")
            config[key] = str((path.parent / executable).absolute()
                              if not executable.is_absolute() else executable)
        else:
            config[key] = str(resolve_path(value, path.parent))
    if not REQUIRED_CONFIG <= set(config):
        raise ValueError("Required model paths cannot be null")
    if bool(config.get("shoe_model")) != bool(config.get("pose33_model")):
        raise ValueError("shoe_model and pose33_model must be supplied together")
    for key, value in config.items():
        if not Path(value).exists():
            raise ValueError(f"Missing configured artifact: {key}: {value}")
    return config


def artifact_bindings(config):
    bindings = {}
    for key, value in sorted(config.items()):
        path = Path(value)
        if path.is_file():
            bindings[key] = {"path": str(path), "sha256": sha(path)}
        else:
            files = sorted(p for p in path.rglob("*") if p.is_file())
            if not files:
                raise ValueError(f"Configured model directory is empty: {key}")
            bindings[key] = {"path": str(path), "files": {str(p.relative_to(path)): sha(p) for p in files}}
    return bindings


def code_bindings():
    files = sorted([*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").rglob("*.py")])
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def interpreter_path(value):
    candidate = shutil.which(str(value))
    if candidate is None:
        raise ValueError("Cannot find the requested Python interpreter")
    return str(Path(candidate).absolute())


def make_bindings(inputs_path, config_path, samples, config, python, device, threads):
    return {"schema_version": "offline-video-batch-bindings-v1",
            "inputs_sha256": sha(inputs_path), "config_sha256": sha(config_path),
            "samples": samples, "resolved_config": config, "artifacts": artifact_bindings(config),
            "source_code_sha256": code_bindings(),
            "execution": {"python": python, "python_binary_sha256": sha(python),
                          "device": device, "threads": threads},
            "manual_labels_read_at_inference": False}


def verify_analysis(path, video_sha256):
    if not path.is_file():
        raise ValueError("Analyzer returned without analysis.json")
    document = read(path)
    if document.get("input_sha256") != video_sha256:
        raise ValueError("Analysis belongs to another source video")
    if document.get("manual_annotations_used") is not False:
        raise ValueError("Analysis does not declare label-free inference")
    if document.get("all_stages_executed_successfully") is not True:
        raise ValueError("Analysis reports incomplete or failed processing stages")
    if not isinstance(document.get("required_outputs"), dict):
        raise ValueError("Analysis lacks required output fields")
    return document


def verify_success_receipt(output, sample, receipt, expected_bindings_sha):
    if (receipt.get("core_id") != sample["core_id"]
            or receipt.get("video_sha256") != sample["video_sha256"]
            or receipt.get("batch_bindings_sha256") != expected_bindings_sha):
        raise ValueError("Saved event receipt does not match this bound batch")
    relative = Path(receipt["analysis_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid saved analysis path")
    path = output / relative
    if sha(path) != receipt["analysis_sha256"]:
        raise ValueError("Saved analysis changed after successful execution")
    verify_analysis(path, sample["video_sha256"])


def command_for(sample, config, python, device, threads, pipeline_output):
    command = [python, "-m", "scripts.analyze_penalty_video", "--video", sample["video_path"],
               "--output", str(pipeline_output), "--device", device, "--threads", str(threads)]
    for key, value in sorted(config.items()):
        command.extend(["--" + key.replace("_", "-"), value])
    return command


def run_one(sample, config, execution, output, bindings_sha):
    event_dir = output / "events" / sample["core_id"]
    event_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (event_dir / f"attempt_{attempt:03d}").exists():
        attempt += 1
    attempt_dir = event_dir / f"attempt_{attempt:03d}"
    attempt_dir.mkdir()
    pipeline_output = attempt_dir / "pipeline"
    command = command_for(sample, config, execution["python"], execution["device"], execution["threads"], pipeline_output)
    started = time.perf_counter()
    receipt = {"schema_version": "offline-video-event-receipt-v1", "core_id": sample["core_id"],
        "clip_id": sample.get("clip_id"), "video_sha256": sample["video_sha256"],
        "batch_bindings_sha256": bindings_sha, "attempt": attempt, "started_at_utc": now(),
        "command": command, "log_path": str((attempt_dir / "run.log").relative_to(output)),
        "status": "running", "analysis_path": str((pipeline_output / "analysis.json").relative_to(output))}
    write_atomic(attempt_dir / "receipt.json", receipt)
    try:
        with (attempt_dir / "run.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, check=False)
        receipt["exit_code"] = process.returncode
        if process.returncode:
            raise ValueError(f"Analyzer exited with code {process.returncode}")
        analysis = verify_analysis(pipeline_output / "analysis.json", sample["video_sha256"])
        receipt.update(status="success", analysis_sha256=sha(pipeline_output / "analysis.json"),
                       all_teacher_outputs_established=analysis.get("all_teacher_outputs_established") is True,
                       accuracy_validated=analysis.get("accuracy_validated") is True)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        receipt.update(status="failed", error=str(error))
    except KeyboardInterrupt:
        receipt.update(status="interrupted", error="Batch interrupted during analyzer execution")
        raise
    finally:
        receipt.update(finished_at_utc=now(), wall_seconds=time.perf_counter() - started)
        write_atomic(attempt_dir / "receipt.json", receipt)
        write_atomic(event_dir / "receipt.json", receipt)
    return receipt


def batch_summary(rows, total, bindings_sha):
    return {"schema_version": "offline-video-batch-summary-v1", "updated_at_utc": now(),
            "batch_bindings_sha256": bindings_sha, "expected_events": total,
            "recorded_events": len(rows), "successful_events": sum(r["status"] == "success" for r in rows),
            "failed_events": sum(r["status"] != "success" for r in rows),
            "complete": len(rows) == total and all(r["status"] == "success" for r in rows),
            "accuracy_evaluated": False, "labels_read": False, "events": rows}


def run(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    inputs_path, config_path = args.inputs.resolve(), args.config.resolve()
    samples, config = load_inputs(inputs_path), load_config(config_path)
    python = interpreter_path(args.python)
    bindings = make_bindings(inputs_path, config_path, samples, config, python, args.device, args.threads)
    output = args.output.resolve()
    manifest_path = output / "batch_manifest.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError("Batch directory exists; use --resume only with unchanged inputs/config/code")
        if not manifest_path.is_file() or read(manifest_path)["bindings"] != bindings:
            raise ValueError("Resume refused: input, config, model artifact, interpreter or source code changed")
    else:
        output.mkdir(parents=True)
        write_atomic(manifest_path, {"created_at_utc": now(), "bindings": bindings})
    bindings_sha = sha(manifest_path)
    rows = []
    for index, sample in enumerate(samples, 1):
        if sha(sample["video_path"]) != sample["video_sha256"] or code_bindings() != bindings["source_code_sha256"]:
            raise ValueError("Input video or source code changed while batch was running")
        receipt_path = output / "events" / sample["core_id"] / "receipt.json"
        reused = False
        if args.resume and receipt_path.exists():
            receipt = read(receipt_path)
            if receipt.get("status") == "success":
                verify_success_receipt(output, sample, receipt, bindings_sha)
                reused = True
        if not reused:
            receipt = run_one(sample, config, bindings["execution"], output, bindings_sha)
        rows.append(receipt)
        write_atomic(output / "summary.json", batch_summary(rows, len(samples), bindings_sha))
        print(json.dumps({"event": sample["core_id"], "index": index, "total": len(samples),
                          "status": receipt["status"], "reused": reused,
                          "seconds": round(receipt.get("wall_seconds", 0), 3)}, ensure_ascii=False), flush=True)
    return batch_summary(rows, len(samples), bindings_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="Interpreter for analyze_penalty_video")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}), flush=True)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
