#!/usr/bin/env python3
"""Reproducible raw-video frame pilot; no labels or outcome references are read."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from src.pitch_calibration import PITCH_LINES, _project, calibrate_pitch_frame


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlay(frame, calibration):
    output = frame.copy()
    candidate = calibration.get("candidate")
    if candidate:
        matrix = candidate["pitch_to_image_hypothesis"]
        for name, ends in PITCH_LINES.items():
            points, valid = _project(matrix, np.linspace(*ends, 200))
            valid &= np.max(np.abs(points), axis=1) < 100000
            for run in np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1):
                if len(run) > 1:
                    color = (0, 220, 255) if name in calibration["fit_line_names"] else (255, 180, 0)
                    cv2.polylines(output, [np.rint(points[run]).astype(np.int32)], False, color, 2)
        for point in candidate["fit_image_points"]:
            if np.max(np.abs(point)) < 100000:
                cv2.circle(output, tuple(np.rint(point).astype(int)), 5, (255, 0, 255), -1)
    cv2.putText(output, calibration["status"] + ": " + calibration["reason"], (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--video", type=Path, help="Original full video or a context clip")
    sources.add_argument("--manifest", type=Path, help="Slicing manifest with source video filenames")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[0.])
    parser.add_argument("--offsets", type=float, nargs="+", default=[0., 1., 3.],
                        help="Seconds after each manifest start; independent frame candidates")
    parser.add_argument("--first-only", action="store_true", help="First penalty from each match")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    samples = []
    if args.video:
        for i, time in enumerate(args.times):
            samples.append((args.video.resolve(), f"sample_{i + 1:03d}", time))
        source_config = None
    else:
        if args.raw_dir is None:
            parser.error("--raw-dir is required with --manifest")
        with args.manifest.open() as handle:
            manifest = yaml.safe_load(handle)
        for match, record in manifest.items():
            penalties = record["penalties"][:1] if args.first_only else record["penalties"]
            for penalty in penalties:
                for i, offset in enumerate(args.offsets):
                    time = float(penalty["start_sec"]) + offset
                    if time < float(penalty["end_sec"]):
                        identity = f"{match}_{int(penalty['number']):02d}_{penalty['kicker']}_s{i:02d}"
                        samples.append(((args.raw_dir / record["video_filename"]).resolve(), identity, time))
        source_config = {"path": str(args.manifest.resolve()), "sha256": _hash(args.manifest)}
    if any(not np.isfinite(time) or time < 0 for _, _, time in samples):
        parser.error("Sample times must be finite and nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path): _hash(path) for path in dict.fromkeys(path for path, _, _ in samples)}
    records = []
    active_path, capture = None, None
    try:
        for path, identity, time in samples:
            if path != active_path:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(str(path))
                active_path = path
                if not capture.isOpened():
                    raise ValueError(f"Cannot decode video: {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid video fps: {path}")
            frame_id = int(round(time * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            record = {"sample_id": identity, "source_path": str(path),
                      "source_sha256": source_hashes[str(path)], "requested_time_sec": time,
                      "requested_frame_id": frame_id, "fps": fps}
            if not ok:
                record.update(status="abstain", reason="frame_decode_failed", calibration=None)
            else:
                decoded_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                calibration = calibrate_pitch_frame(frame, frame_id=decoded_frame)
                record.update(status=calibration["status"], reason=calibration["reason"],
                              decoded_frame_id=decoded_frame, decoded_time_sec=decoded_frame / fps,
                              decoded_pixels_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                              calibration=calibration)
                image_name = identity + ".jpg"
                if not cv2.imwrite(str(args.output_dir / image_name), _overlay(frame, calibration)):
                    raise OSError(f"Failed to write image: {image_name}")
                record["diagnostic_image"] = image_name
            records.append(record)
            print(identity, record["status"], record["reason"], flush=True)
    finally:
        if capture is not None:
            capture.release()
    output = {"schema_version": 1, "opencv_version": cv2.__version__, "manifest": source_config,
              "algorithm_sha256": _hash(ROOT / "src/pitch_calibration.py"),
              "pilot_script_sha256": _hash(Path(__file__)), "source_hashes": source_hashes,
              "sample_count": len(records),
              "supported_frame_count": sum(record["status"] == "candidate_supported" for record in records),
              "metric_accuracy_verified": False, "human_reference_used": False,
              "scope": "Per-frame automatic geometry checks; no clip-wide calibration or visual accuracy estimate",
              "records": records}
    output_path = args.output_dir / "pitch_calibration_pilot.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: output[key] for key in ("sample_count", "supported_frame_count", "metric_accuracy_verified")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
