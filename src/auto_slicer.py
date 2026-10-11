import os
import yaml
import argparse
from typing import List, Tuple
from src.video_utils import VideoHandler, slice_video

class AutomatedPenaltySlicer:
    def slice_from_manifest(self, manifest_yaml: str, video_dir: str, output_dir: str) -> List[str]:
        with open(manifest_yaml, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        all_created = []
        for match_key, match_data in manifest.items():
            video_file = os.path.join(video_dir, match_data["video_filename"])
            if not os.path.exists(video_file):
                print(f"[!] Video not found: {video_file}, skipping...")
                continue
            for penalty in match_data.get("penalties", []):
                p_num = penalty["number"]
                kicker = penalty.get("kicker", f"kicker_{p_num}")
                start_t = penalty["start_sec"]
                end_t = penalty["end_sec"]
                out_name = f"{match_key}_p{p_num:02d}_{kicker}.mp4"
                out_path = os.path.join(output_dir, out_name)
                print(f"[*] Slicing {match_key} #{p_num} ({kicker}) -> {out_name}")
                slice_video(video_file, out_path, start_t, end_t)
                all_created.append(out_path)
        return all_created

def main():
    parser = argparse.ArgumentParser(description="Automated Video Slicer")
    parser.add_argument("--manifest", default="configs/penalty_slicing_manifest.yaml")
    parser.add_argument("--video_dir", default="data/full_videos")
    parser.add_argument("--output_dir", default="data/raw_clips")
    args = parser.parse_args()
    slicer = AutomatedPenaltySlicer()
    slicer.slice_from_manifest(args.manifest, args.video_dir, args.output_dir)

if __name__ == "__main__":
    main()