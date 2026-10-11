"""Export a complete, unannotated H.264 playback copy; keep the original intact."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    if not decoded or fps <= 0:
        raise ValueError(f"No decodable video: {path}")
    return {"decoded_frames": decoded, "fps": fps, "duration_sec": decoded / fps,
            "width": width, "height": height,
            "codec": "".join(chr((codec >> (8 * i)) & 255) for i in range(4))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    original = inspect_video(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-n",
                    "-i", str(args.source), "-map", "0:v:0", "-map", "0:a?",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                    str(args.output)], check=True)
    converted = inspect_video(args.output)
    for key in ("decoded_frames", "fps", "width", "height"):
        if original[key] != converted[key]:
            raise ValueError(f"Playback copy changed {key}")
    report = {"source": str(args.source.resolve()), "playback_copy": str(args.output.resolve()),
              "original": original, "converted": converted,
              "full_duration_preserved": True, "annotations_added": False, "human_verified": False}
    args.output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
