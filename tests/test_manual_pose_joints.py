import csv
import json

import cv2
import numpy as np
import pytest

from scripts.import_manual_pose_joints import CSV_FIELDS, JOINT_NAMES, prepare, sha256, sparse_frames


def save_sources(json_path, csv_path, document):
    json_path.write_text(json.dumps(document), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in document["annotations"]:
            writer.writerow({**{name: row[name] for name in ("video_id", "track_id", "frame", "timestamp_s", "joint_id", "joint_name", "status", "updated_at")},
                             "x_px": row["x"], "y_px": row["y"], "human_reviewed": "true",
                             "reviewer": document["reviewer"], "frame_note": "",
                             "source_sha256": document["clips"][row["video_id"]]["source_sha256"]})


@pytest.fixture
def bundle(tmp_path):
    videos, times, poses = (tmp_path / name for name in ("videos", "timestamps", "poses"))
    for path in (videos, times, poses):
        path.mkdir()
    video = videos / "05-04.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    assert writer.isOpened()
    for frame in range(5):
        writer.write(np.full((48, 64, 3), frame * 30, dtype=np.uint8))
    writer.release()
    # Deliberately nonuniform PTS demonstrate that frame/FPS must not replace the source timeline.
    with (times / "05-04.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "frame", "time_s"])
        writer.writeheader()
        for frame, pts in enumerate([0, .04, .081, .12, .16]):
            writer.writerow({"video_id": "05-04", "frame": frame, "time_s": pts})
    pose = poses / "05-04.json"
    pose.write_text('{"tracks": []}')
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fixture weights")
    document = {"schema": "vision2-manual-joints-v1",
                "coordinate_system": "original image pixels; top-left origin, x right, y down",
                "frame_numbering": "zero-based", "reviewer": "111", "frame_notes": {},
                "clips": {"05-04": {"track_id": 7, "source_sha256": sha256(video),
                                      "pose_json_sha256": sha256(pose), "model_sha256": sha256(weights)}},
                "annotations": []}
    for joint, status, xy in ((13, "visible_corrected", (20., 30.)), (14, "visible_confirmed", (24., 32.)), (16, "occluded", (None, None))):
        document["annotations"].append({"video_id": "05-04", "track_id": 7, "frame": 2,
                                        "timestamp_s": .081, "joint_id": joint, "joint_name": JOINT_NAMES[joint],
                                        "status": status, "x": xy[0], "y": xy[1], "human_reviewed": True,
                                        "updated_at": "2026-10-27T08:41:36.615Z"})
    json_path, csv_path = tmp_path / "source.json", tmp_path / "source.csv"
    save_sources(json_path, csv_path, document)
    return json_path, csv_path, document, dict(video_root=videos, timestamps_root=times, pose_json_root=poses, model_weights=weights)


def test_sparse_masks_bound_sources_and_exact_outputs(bundle, tmp_path):
    json_path, csv_path, document, arguments = bundle
    before = (sha256(json_path), sha256(csv_path))
    output = tmp_path / "output"
    result = prepare(json_path, csv_path, output, **arguments)
    assert before == (sha256(json_path), sha256(csv_path))
    assert result["reviewed_frames"] == 1
    assert result["joint_records"] == 3
    assert result["coordinate_targets"] == 2
    assert result["visibility_targets"] == 3
    assert result["unreviewed_joint_slots"] == 14
    assert result["exact_pose_output_files_verified"] == 1
    assert result["model_file_hash_verified"]
    target = json.loads((output / "frame_targets.json").read_text())["frames"][0]
    assert target["joint_xy_px"][13] == [20, 30]
    assert target["coordinate_loss_mask"][13]
    assert target["visibility_target"][16] is False
    assert target["visibility_loss_mask"][16]
    assert target["coordinate_loss_mask"][16] is False
    assert target["joint_xy_px"][16] == [None, None]
    assert target["visibility_target"][0] is None
    assert target["visibility_loss_mask"][0] is False
    assert target["timestamp_s"] == .081
    assert cv2.imread(str(output / target["image"])).shape[:2] == (48, 64)
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["independent_test"]
    assert manifest["videos"]["05-04"]["decoded_frame_count"] == 5
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare(json_path, csv_path, output, **arguments)


def test_missing_pose_output_is_pending_not_falsely_verified(bundle, tmp_path):
    json_path, csv_path, document, arguments = bundle
    arguments.pop("pose_json_root")
    arguments.pop("model_weights")
    result = prepare(json_path, csv_path, tmp_path / "pending", **arguments)
    assert result["exact_pose_output_files_verified"] == 0
    assert not result["model_file_hash_verified"]


@pytest.mark.parametrize("change,error", [
    (lambda d: d["annotations"][0].update(x=90), "outside original image"),
    (lambda d: d["annotations"][0].update(timestamp_s=.08), "differs from frame PTS"),
    (lambda d: d["annotations"][0].update(joint_name="wrong"), "COCO17 index"),
    (lambda d: d["annotations"][2].update(x=20), "unknown coordinates"),
    (lambda d: d["clips"]["05-04"].update(source_sha256="0" * 64), "Source video hash mismatch"),
    (lambda d: d["clips"]["05-04"].update(pose_json_sha256="0" * 64), "pose JSON mismatch"),
    (lambda d: d["clips"]["05-04"].update(model_sha256="0" * 64), "weights do not match"),
])
def test_rejects_misalignment_and_unknown_coordinates(bundle, tmp_path, change, error):
    json_path, csv_path, document, arguments = bundle
    change(document)
    save_sources(json_path, csv_path, document)
    with pytest.raises(ValueError, match=error):
        prepare(json_path, csv_path, tmp_path / "invalid", **arguments)
    assert not (tmp_path / "invalid").exists()


def test_csv_disagreement_rejected(bundle, tmp_path):
    json_path, csv_path, document, arguments = bundle
    csv_path.write_text(csv_path.read_text(encoding="utf-8-sig").replace("20.0", "21.0"), encoding="utf-8-sig")
    with pytest.raises(ValueError, match="JSON and CSV disagree: x"):
        prepare(json_path, csv_path, tmp_path / "badcsv", **arguments)
