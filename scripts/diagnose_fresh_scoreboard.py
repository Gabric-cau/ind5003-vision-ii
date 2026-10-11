"""Save source-bound broadcast bands for a visual feasibility diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--sample-stride-seconds", type=float, default=1.)
    args = parser.parse_args()
    if args.sample_stride_seconds <= 0 or not np.isfinite(args.sample_stride_seconds):
        parser.error("Sampling stride must be positive and finite")
    if args.output_dir.resolve() == args.clips_dir.resolve():
        parser.error("Output must not overwrite source clips")
    manifest_path = args.clips_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = {row["clip_id"]: row for row in manifest["clips"]}
    if len(sources) != len(manifest["clips"]) or len(set(args.clips)) != len(args.clips):
        raise ValueError("Duplicate source or requested clip ID")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for cid in args.clips:
        source = sources[cid]
        video = args.clips_dir / source["clip_filename"]
        if sha256(video) != source["clip_sha256"]:
            raise ValueError("Source clip hash mismatch")
        stride = max(1, int(round(source["fps"] * args.sample_stride_seconds)))
        frames = sorted(set(range(0, source["frame_count"], stride)) | {source["frame_count"] - 1})
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), source["fps"]):
            capture.release()
            raise ValueError("Source decode/fps mismatch")
        cells = []
        observations = []
        try:
            for fid in frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ok, image = capture.read()
                if not ok or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1 != fid:
                    raise ValueError("Source decode/frame identity mismatch")
                height, width = image.shape[:2]
                # Preserve the entire width: scoreboards need not sit at the same corner.
                band_height = int(round(height * .25))
                top = image[:band_height]
                bottom = image[-band_height:]
                top_path = args.output_dir / f"{cid}_f{fid:06d}_top.png"
                bottom_path = args.output_dir / f"{cid}_f{fid:06d}_bottom.png"
                for target, pixels in ((top_path, top), (bottom_path, bottom)):
                    if not cv2.imwrite(str(target), pixels):
                        raise OSError("Cannot write source band")
                label = np.full((24, 640, 3), 245, np.uint8)
                cv2.putText(label, f"f{fid} | {fid / source['fps']:.2f}s | upper + lower broadcast bands",
                            (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (30, 30, 30), 1)
                cells.append(np.concatenate([label, cv2.resize(top, (640, 90)),
                                             cv2.resize(bottom, (640, 90))], axis=0))
                observations.append({"frame_id": fid, "source_frame_id": source["source_start_frame"] + fid,
                    "decoded_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "image_size": [width, height], "top_roi_xyxy": [0, 0, width, band_height],
                    "bottom_roi_xyxy": [0, height - band_height, width, height],
                    "top_path": top_path.name, "top_sha256": sha256(top_path),
                    "bottom_path": bottom_path.name, "bottom_sha256": sha256(bottom_path)})
        finally:
            capture.release()
        columns = 2
        while len(cells) % columns:
            cells.append(np.full_like(cells[0], 245))
        sheet = np.concatenate([np.concatenate(cells[start:start + columns], axis=1)
                                for start in range(0, len(cells), columns)], axis=0)
        sheet_path = args.output_dir / (cid + "_broadcast_bands.jpg")
        if not cv2.imwrite(str(sheet_path), sheet):
            raise OSError("Cannot write source band contact sheet")
        if sha256(video) != source["clip_sha256"]:
            raise ValueError("Source changed during diagnostic")
        records.append({"clip_id": cid, "event_id": source["event_id"], "source_sha256": source["clip_sha256"],
                        "sampled_frames": observations, "sheet_path": sheet_path.name,
                        "sheet_sha256": sha256(sheet_path)})
        print(cid, len(observations), "source-bound sampled frames", flush=True)
    result = {"schema_version": "broadcast-band-diagnostic-v1", "clips": records,
              "source_manifest_sha256": sha256(manifest_path), "code_sha256": sha256(Path(__file__)),
              "sampling_policy": "uniform time stride plus final frame; no outcome or reference inputs",
              "sample_stride_seconds": args.sample_stride_seconds,
              "outcome_labels_read": False, "reference_labels_used_for_prediction": False,
              "human_verified": False, "is_gold_annotation": False, "accuracy_validated": False,
              "scope": "Source-pixel feasibility extraction only; no OCR or outcome decisions"}
    (args.output_dir / "broadcast_band_sources.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
