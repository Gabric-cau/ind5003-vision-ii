"""Separate external foot labels from source-only video inputs; keep WC8 quarantined."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--remote-video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("External foot data directory already exists")
    metadata = args.dataset_root / "metadata/annotations.json"
    data = json.loads(metadata.read_text())
    inputs, labels, upload = [], [], []
    for row in data["clips"]:
        if row["dataset_id"] == "world_cup_2022" or row["filename"] == "match_09.mp4":
            continue
        features = row["human_features"]
        if features["provenance"] != "human_user_observation_in_chat":
            raise ValueError("External foot label lacks explicit human provenance")
        foot = {"左脚": "left", "右脚": "right"}.get(features["kicking_foot_label"])
        if foot is None:
            raise ValueError("Unknown external foot label")
        video = args.dataset_root / row["file"]
        digest = sha(video)
        if digest != row["media_check"]["sha256"]:
            raise ValueError("External video differs from its annotation source")
        cid = Path(row["filename"]).stem
        inputs.append({"clip_id": cid, "match_group": row["dataset_id"],
            "video_path": str(args.remote_video_root / row["filename"]), "video_sha256": digest,
            "fps": row["fps"], "frame_count": row["media_check"]["video_frames"],
            "width": row["width"], "height": row["height"]})
        labels.append({"clip_id": cid, "match_group": row["dataset_id"], "source_sha256": digest,
            "striking_foot": foot, "provenance": features["provenance"],
            "raw_user_observations": features["raw_user_observations"]})
        upload.append(str(video))
    if len(inputs) != 15 or sum(r["striking_foot"] == "left" for r in labels) != 3:
        raise ValueError("Expected 15 eligible external events including three human left-foot labels")
    args.output.mkdir(parents=True)
    for name, value in (("inputs.json", {"schema_version": "external-foot-video-inputs-v1", "samples": inputs,
        "manual_annotations_included": False, "world_cup_holdout_accessed": False}),
        ("labels.json", {"schema_version": "external-foot-human-targets-v1", "annotation_sha256": sha(metadata),
        "labels": labels, "landing_supervision_available": False}),
        ("upload.json", {"source_files": upload})):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print({"events": len(inputs), "left": 3, "right": 12, "total_video_bytes": sum(Path(p).stat().st_size for p in upload)})


if __name__ == "__main__":
    main()
