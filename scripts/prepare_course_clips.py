"""Create independent, hashed context clips from the supplied course recordings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import yaml

from src.final_results import canonical_events, sha256
from src.runtime import resolve_ffmpeg


ROOT = Path(__file__).resolve().parents[1]


def plan_context_clips(manifest: dict, metadata: dict, before: float, after: float) -> list[dict]:
    if before < 0 or after < 0:
        raise ValueError("Context durations must be nonnegative")
    records = []
    for event in canonical_events(manifest):
        match = manifest[event["match_id"]]
        video = metadata[match["video_filename"]]
        fps = video["fps"]
        start_frame = max(0, round((event["slice_start_sec"] - before) * fps))
        end_frame = min(video["frame_count"], round((event["slice_end_sec"] + after) * fps))
        if end_frame <= start_frame:
            raise ValueError(f"Empty clip interval: {event['clip_id']}")
        records.append({**event, "source_filename": match["video_filename"],
                        "source_video_sha256": video["sha256"], "fps": fps,
                        "source_start_frame": start_frame, "source_end_frame_exclusive": end_frame,
                        "frame_count": end_frame - start_frame,
                        "original_slice_start_frame_in_clip": round(event["slice_start_sec"] * fps) - start_frame,
                        "original_slice_end_frame_in_clip": round(event["slice_end_sec"] * fps) - start_frame,
                        "semantic_event_start_verified": False, "semantic_event_end_verified": False,
                        "context_is_not_event_boundary_ground_truth": True})
    return records


def prepare(source_dir: Path, expected_hashes: Path, manifest_path: Path,
            output_dir: Path, ffmpeg: str, before: float, after: float) -> dict:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    hashes = json.loads(expected_hashes.read_text(encoding="utf-8"))
    metadata = {}
    for match in manifest.values():
        name = match["video_filename"]
        path = source_dir / name
        actual_hash = sha256(path)
        if hashes.get(name) != actual_hash:
            raise ValueError(f"Course recording does not match its recorded source reference: {name}")
        cap = cv2.VideoCapture(str(path))
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps <= 0 or count <= 0:
            raise ValueError(f"Unreadable course recording: {name}")
        metadata[name] = {"sha256": actual_hash, "bytes": path.stat().st_size,
                          "fps": fps, "frame_count": count, "width": width, "height": height}
    records = plan_context_clips(manifest, metadata, before, after)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty clip directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        target = output_dir / f"{record['clip_id']}.mp4"
        start_sec = record["source_start_frame"] / record["fps"]
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-ss", str(start_sec), "-i", str(source_dir / record["source_filename"]),
                   "-frames:v", str(record["frame_count"]), "-an", "-c:v", "libx264",
                   "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-threads", "2", "-movflags", "+faststart", str(target)]
        subprocess.run(command, check=True)
        cap = cv2.VideoCapture(str(target))
        decoded_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if decoded_count != record["frame_count"]:
            raise ValueError(f"Clip frame count mismatch: {record['clip_id']}")
        record["clip_sha256"] = sha256(target)
        record["clip_bytes"] = target.stat().st_size
        record["clip_filename"] = target.name
        print(f"Prepared {index + 1}/39: {target.name} ({decoded_count} frames)", flush=True)
    result = {"schema_version": 1, "dataset": "course_context_clips_2026-10-12",
              "manifest_sha256": sha256(manifest_path), "source_hash_reference_sha256": sha256(expected_hashes),
              "source_hash_reference_origin": "prior_course_download_record_not_new_external_verification",
              "sources": metadata, "context_before_sec": before, "context_after_sec": after,
              "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
              "encoding": {"codec": "libx264", "crf": 18, "resize": False, "audio_included": False},
              "old_experiment_caches_compatible": False, "human_verified": False, "clips": records}
    (output_dir / "source_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expected-hashes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/penalty_slicing_manifest.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--before", type=float, default=6.0)
    parser.add_argument("--after", type=float, default=8.0)
    args = parser.parse_args()
    prepare(args.source_dir, args.expected_hashes, args.manifest, args.output_dir,
            resolve_ffmpeg(args.ffmpeg, ROOT), args.before, args.after)


if __name__ == "__main__":
    main()
