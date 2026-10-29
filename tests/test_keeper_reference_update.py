import copy
import csv
import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from scripts import import_keeper_reference_update as importer


def joint(joint_id, status="visible_confirmed", origin="original_user_export", frame=0):
    visible = status in importer.VISIBLE
    record = {"video_id": "02-05", "track_id": "02-05_GK01", "frame": frame, "timestamp_s": frame / 25,
              "joint_id": joint_id, "joint_name": importer.JOINT_NAMES[joint_id], "status": status,
              "x": 45 if visible else None, "y": 25 if visible else None, "human_reviewed": True,
              "reference_source": "assistant_visual_proposal", "reviewer": "", "import_group": "fixture",
              "updated_at": "2026-10-28T00:00:00Z"}
    if origin != "original_user_export":
        record.update(annotation_origin=origin, confirmation_authority="fixture authority",
                      reference_proposal_sha256="b" * 64, reference_source_video_sha256="a" * 64,
                      independent_blinded_annotation=False)
    return record


def package_fixture():
    first = joint(5)
    bulk = joint(6, origin="user_message_bulk_reference_confirmation")
    occluded = joint(15, "occluded")
    ambiguous = joint(16, "left_right_ambiguous")
    raw = {"video_id": "02-05", "frame": 0, "timestamp_s": 0.0, "review_timestamp_s": 0.0,
           "keeper_id": "02-05_GK01", "bbox": [35, 15, 55, 40], "bbox_status": "visible_confirmed",
           "bbox_human_reviewed": True, "bbox_changed_by_human": False, "occluded": False,
           "out_of_frame": False, "proposal_wrong_person": False, "identity_resolution_pending": False,
           "proposal_was_interpolated": False, "all_required_stage_joints_reviewed": False,
           "box_reviewer": "fixture", "stage": "ready",
           "human_joints": {str(row["joint_id"]): row for row in (first, bulk, occluded, ambiguous)}}
    box_csv = {key: str(value) if value is not None else "" for key, value in raw.items()
               if key not in {"bbox", "human_joints"}}
    box_csv.update({key: str(value) for key, value in zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), raw["bbox"])})
    joints_csv = []
    for record in raw["human_joints"].values():
        row = {key: str(value) if value is not None else "" for key, value in record.items()}
        row.setdefault("annotation_origin", "original_user_export")
        row.setdefault("independent_blinded_annotation", "")
        joints_csv.append(row)
    document = {"video_id": "02-05", "source_sha256": "a" * 64, "proposal_sha256": "b" * 64,
                "source_dimensions": [96, 64], "frame_numbering": "zero-based original source",
                "coordinate_system": "original pixels; x right/y down; anatomical left/right", "frames": [raw]}
    return {"clips": {"02-05": document}, "boxes_csv": [box_csv], "joints_csv": joints_csv,
            "empty_references": [{"video_id": "02-05", "frame": "0", "joint_id": "11", "x": "", "y": ""}],
            "declared_conflicts": [{"video_id": "02-05", "frame": "0", "joint_id": "5"}],
            "member_sha256": {}, "declared_status": {"box_records": 1, "human_joint_records": 4},
            "declared_validation": {}}


def test_provenance_and_sparse_joint_masks_preserve_uncertainty():
    package = package_fixture()
    frames = importer.normalize_package(package)
    targets = importer.sparse_roi_targets(frames[0], [30, 10, 60, 45])
    assert targets["annotation_origin"][5] == "original_user_export"
    assert targets["annotation_origin"][6] == "user_message_bulk_reference_confirmation"
    assert targets["independent_blinded_annotation"][5] is None
    assert targets["independent_blinded_annotation"][6] is False
    assert targets["declared_box_point_conflict"][5] is True
    assert targets["coordinate_loss_mask"][5] is True
    assert targets["coordinate_loss_mask"][6] is True
    assert targets["joint_xy_roi_pixels"][5] == [15, 15]
    assert targets["coordinate_loss_mask"][15] is False
    assert targets["visibility_loss_mask"][15] is True and targets["visibility_target"][15] is False
    assert targets["visibility_loss_mask"][16] is False and targets["visibility_target"][16] is None
    assert targets["joint_xy_original_pixels"][11] == [None, None]
    assert targets["coordinate_loss_mask"][11] is False and targets["visibility_loss_mask"][11] is False


def test_crop_depends_only_on_bbox_and_masks_points_outside_actual_crop():
    frames = importer.normalize_package(package_fixture())
    bounds = importer.roi_bounds([35.1, 15.2, 54.8, 39.7], 96, 64, .25)
    assert bounds == [30, 9, 60, 46]
    targets = importer.sparse_roi_targets(frames[0], [0, 0, 20, 20])
    assert targets["joint_xy_original_pixels"][5] == [45, 25]
    assert targets["coordinate_loss_mask"][5] is False
    assert importer.roi_bounds([0, 0, 10, 10], 96, 64, .25) == [0, 0, 13, 13]


def test_stale_derived_csv_flag_is_preserved_without_discarding_reviewed_coordinates():
    package = package_fixture()
    package["clips"]["02-05"]["frames"][0]["all_required_stage_joints_reviewed"] = True
    frames = importer.normalize_package(package)
    assert frames[0]["derived_review_flag_csv_stale"] is True
    assert frames[0]["source_csv_record"]["all_required_stage_joints_reviewed"] == "False"
    assert frames[0]["source_json_record"]["all_required_stage_joints_reviewed"] is True
    assert frames[0]["joints"][0]["coordinate_loss_eligible"] is True


@pytest.mark.parametrize("mutation", ["independent", "unknown_coordinate", "csv_disagree", "empty_filled", "duplicate_joint"])
def test_rejects_fabricated_or_inconsistent_references(mutation):
    package = package_fixture()
    if mutation == "independent":
        package["joints_csv"][1]["independent_blinded_annotation"] = "True"
    elif mutation == "unknown_coordinate":
        package["clips"]["02-05"]["frames"][0]["human_joints"]["15"]["x"] = 4
        package["joints_csv"][2]["x"] = "4"
    elif mutation == "csv_disagree":
        package["boxes_csv"][0]["bbox_x1"] = "34"
    elif mutation == "empty_filled":
        package["empty_references"][0]["x"] = "1"
    else:
        package["joints_csv"].append(copy.deepcopy(package["joints_csv"][0]))
    with pytest.raises(ValueError):
        importer.normalize_package(package)


def test_group_identity_keeps_shared_shooter_and_source_events():
    pose = {clip: {"source_url": "fixture"} for clip in ("02-05", "02-06", "07-08")}
    first, second = [importer.identity(clip, pose, {}) for clip in ("02-05", "02-06")]
    assert first["subject_group"] == second["subject_group"]
    assert first["event_id"] != second["event_id"]
    assert first["split"] == second["split"] == "validation"
    assert importer.identity("07-08", pose, {})["split"] == "train"
    match = {"match_17": {"original_event_id": "worldcup_2022_final_kicker_01", "dataset_id": "world_cup_2022",
                           "source_url": "fixture", "original_match_id": "match_01", "source_start_frame_zero_based": 4571}}
    assert importer.identity("match_17", {}, match)["split"] == "test"
    assert importer.identity("match_17", {}, match)["event_id"] == "worldcup_2022_final_kicker_01"


def test_package_never_extracts_or_executes_provenance_code(tmp_path):
    package = tmp_path / "bad.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../provenance/execute.py", "raise Exception('must never execute')")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        importer.read_package(package)
    assert list(tmp_path.iterdir()) == [package]


def csv_bytes(rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def test_import_verifies_video_pts_exports_original_and_roi_without_mutation(tmp_path):
    package = package_fixture()
    pack = tmp_path / "pack"
    matches = tmp_path / "matches"
    (pack / "videos").mkdir(parents=True)
    (matches / "metadata").mkdir(parents=True)
    source = pack / "videos/02-05.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 64))
    assert writer.isOpened()
    writer.write(np.full((64, 96, 3), 80, dtype=np.uint8))
    writer.write(np.full((64, 96, 3), 100, dtype=np.uint8))
    writer.release()
    source_hash = importer.sha256(source)
    package["clips"]["02-05"]["source_sha256"] = source_hash
    package["clips"]["02-05"]["frames"][0]["human_joints"]["6"]["reference_source_video_sha256"] = source_hash
    (pack / "inventory.json").write_text(json.dumps({"02-05": {"source_url": "fixture"}}))
    (matches / "metadata/annotations.json").write_text(json.dumps({"clips": []}))
    archive_path = tmp_path / "review.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("human_reviewed/02-05.json", json.dumps(package["clips"]["02-05"]))
        for filename, value in [("human_boxes.csv", "boxes_csv"), ("human_joints.csv", "joints_csv"),
                                (importer.EMPTY_REFERENCE_FILE, "empty_references"), (importer.CONFLICT_FILE, "declared_conflicts")]:
            archive.writestr(filename, csv_bytes(package[value]))
        archive.writestr("current_status.json", json.dumps(package["declared_status"]))
        archive.writestr("validation.json", "{}")
        archive.writestr("provenance/do_not_run.py", "raise RuntimeError('never execute')")
    archive_hash = importer.sha256(archive_path)
    output = tmp_path / "imported"
    summary = importer.import_update(archive_path, output, pack, matches)
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["roi_exports"] == 1 and summary["empty_references_preserved"] == 1
    assert importer.sha256(archive_path) == archive_hash and importer.sha256(source) == source_hash
    assert manifest["full_frame_detection_training_allowed"] is False
    assert manifest["independent_ground_truth"] is False
    assert manifest["source_pixels_verified"] is True
    row = manifest["exports"][0]
    assert row["split"] == "validation"
    assert row["coordinate_loss_mask"][5] is True
    assert cv2.imread(str(output / row["image"])).shape[:2] == (64, 96)
    assert cv2.imread(str(output / row["roi_image"])).shape[:2] == (row["roi_height"], row["roi_width"])
    assert importer.sha256(output / row["image"]) == row["image_sha256"]
    assert importer.sha256(output / row["roi_image"]) == row["roi_image_sha256"]
    assert not (output / "provenance").exists()
    with pytest.raises(FileExistsError):
        importer.import_update(archive_path, output, pack, matches)
