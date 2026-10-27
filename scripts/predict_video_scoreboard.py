"""Read a visible score-icon change without manual shootout metadata."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.video_scoreboard_transition import analyze_video_transition
from src.runtime import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    start = time.perf_counter()
    source = json.loads((args.analysis_dir / "source/source_manifest.json").read_text())["clips"][0]
    video = args.analysis_dir / "source" / source["clip_filename"]
    contact = json.loads((args.analysis_dir / "trained_contact/prediction.json").read_text())
    if contact["video_sha256"] != source["clip_sha256"] or file_sha256(video) != source["clip_sha256"]:
        raise ValueError("Contact and video sources differ")
    result = analyze_video_transition(video, contact["prediction"]["time_seconds"])
    result["runtime_seconds"] = time.perf_counter() - start
    args.output.mkdir(parents=True)
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["decision"]))


if __name__ == "__main__":
    main()
