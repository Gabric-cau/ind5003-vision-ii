import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import openpyxl
import pytest

from scripts import prepare_external_ball_frames_round1 as importer
from scripts.optimize_ball_detector import load_exports


def annotation(frame=3, local=2, visibility="\u6e05\u6670", box=True, note=None):
    return [None, None, 1, "fixture player", "0042", "fixture camera", frame, frame / 25, local,
            45, 40, *([40, 35, 50, 45] if box else [None] * 4), visibility, "manual", "after_contact", note]


def test_partial_shot_uses_effective_first_frame_and_retains_unknown():
    config = {"match_id": "lev_stu_20240817", "width": 1280, "height": 720,
              "split": "train", "first_col": 6, "local_col": 8}
    rows = importer.normalize_rows([importer.FRAME_HEADERS, annotation(4429, 158),
                                    annotation(4430, 159, "\u770b\u4e0d\u6e05", False)], config, "digest")
    summary = {importer.INDEX_SHEET: [[]] * 7 + [[1, "player", "0042", "view", 0, 0, 4429, 4551, 158]]}
    mappings = importer.shot_mappings(summary, config, rows)
    assert mappings[1, "0042"]["full_shot_first_original_frame"] == 4271
    assert rows[0]["review_frame_zero_based"] == 0
    assert rows[1]["review_frame_zero_based"] == 1
    assert rows[0]["eligible_box"] is True
    assert rows[1]["eligible_box"] is False and rows[1]["used_as_negative"] is False
    assert rows[0]["video_id"] == "lev_stu_20240817_p01_s0042"


@pytest.mark.parametrize("change", ["time", "box", "local"])
def test_rejects_inconsistent_geometry_and_frame_contract(change):
    config = {"match_id": "fixture", "width": 96, "height": 64, "split": "train", "first_col": 6, "local_col": 8}
    values = annotation()
    if change == "time":
        values[7] += 1
    elif change == "box":
        values[13] = 120
    else:
        values[8] = 8
    with pytest.raises(ValueError):
        rows = importer.normalize_rows([importer.FRAME_HEADERS, values], config, "digest")
        importer.shot_mappings({importer.INDEX_SHEET: [[]] * 7 + [[1, "p", "0042", "v", 0, 0, 3, 5, 2]]}, config, rows)


def test_partial_and_estimated_rows_do_not_become_hard_boxes():
    config = {"match_id": "fixture", "width": 96, "height": 64, "split": "train"}
    rows = importer.normalize_rows([importer.FRAME_HEADERS,
                                    annotation(3, 2, "\u90e8\u5206\u53ef\u89c1"),
                                    annotation(4, 3, note="\u63d2\u503c\u4f30\u8ba1"),
                                    annotation(5, 4, "\u53ef\u89c1\uff08\u8fd0\u52a8\u6a21\u7cca\uff09")], config, "digest")
    assert [row["eligible_box"] for row in rows] == [False, False, True]
    assert all(not row["used_as_negative"] for row in rows)


@pytest.mark.parametrize("filename", ["../match_10_P1_0042_review.mp4", "/match_10_P1_0042_review.mp4", "C:/match_10_P1_0042_review.mp4"])
def test_rejects_unsafe_review_archives(tmp_path, filename):
    archive = tmp_path / "review.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(filename, b"not a video")
    with pytest.raises(ValueError, match="Unsafe"):
        importer.extract_review_archive(archive, tmp_path / "extract", {(1, "0042"): {}})


def test_scene_comparison_ignores_annotation_colors_and_rejects_wrong_scene():
    rng = np.random.default_rng(7)
    gray = rng.integers(30, 150, (360, 640), dtype=np.uint8)
    clean = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    review = clean.copy()
    review[:75] = 0
    cv2.rectangle(review, (280, 260), (310, 290), (0, 255, 0), 2)
    cv2.circle(review, (180, 220), 20, (0, 160, 255), 2)
    quality = importer.scene_similarity(clean, review)
    importer.require_alignment(quality, "fixture")
    assert quality["trimmed_scene_mae"] < 0.1
    assert quality["mask_policy"].endswith("no_label_coordinates")
    with pytest.raises(ValueError, match="alignment rejected"):
        importer.require_alignment(importer.scene_similarity(np.full_like(clean, 255), review), "wrong source")


def write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 64))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_workbook(path, sheets):
    book = openpyxl.Workbook()
    book.remove(book.active)
    for title, rows in sheets.items():
        sheet = book.create_sheet(title)
        for row in rows:
            sheet.append(row)
    book.save(path)
    book.close()


def test_exports_clean_pixels_with_verified_offset_and_loader_compatibility(tmp_path, monkeypatch):
    configs = []
    source = tmp_path / "source"
    source.mkdir()
    (source / "clean_sources").mkdir()
    for index, split in enumerate(("train", "validation")):
        key = f"fixture{index}"
        config = {"key": key, "match_id": key, "folder": key, "frames": "frames.xlsx", "summary": "summary.xlsx",
                  "archive": "review.zip", "width": 96, "height": 64, "split": split, "expected_rows": 2,
                  "expected_boxes": 1, "first_col": 6, "local_col": 8}
        configs.append(config)
        folder = source / key
        folder.mkdir()
        rng = np.random.default_rng(index + 12)
        frames = [cv2.cvtColor(rng.integers(25, 155, (64, 96), dtype=np.uint8), cv2.COLOR_GRAY2BGR) for _ in range(8)]
        clean = source / "clean_sources" / f"{key}.mp4"
        write_video(clean, frames)
        cap = cv2.VideoCapture(str(clean))
        review_frames = []
        for frame_index in (4, 5, 6):
            frame = importer.read_frame(cap, frame_index)
            frame[:12] = 0
            cv2.rectangle(frame, (40, 35), (50, 45), (0, 255, 0), 1)
            review_frames.append(frame)
        cap.release()
        review = folder / "review.mp4"
        write_video(review, review_frames)
        with zipfile.ZipFile(folder / "review.zip", "w") as stream:
            stream.write(review, "match_10_P1_0042_\u5e26\u5e27\u53f7.mp4")
        write_workbook(folder / "frames.xlsx", {importer.FRAME_SHEET: [importer.FRAME_HEADERS,
                        annotation(), annotation(4, 3, "\u770b\u4e0d\u6e05", False)]})
        write_workbook(folder / "summary.xlsx", {importer.INDEX_SHEET: [[]] * 7 + [[1, "p", "0042", "v", 0, 0, 3, 5, 2]],
                                                  importer.SOURCE_SHEET: [["source_url", "fixture"]]})
    monkeypatch.setattr(importer, "SOURCE_CONFIGS", tuple(configs))
    output = tmp_path / "export"
    summary = importer.export_round(source, output, maximum_offset=1)
    assert summary["export_counts"] == {"train": 1, "validation": 1}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["clean_source_pixels_verified"] is True
    assert manifest["selection_uses_evaluation"] is False
    assert manifest["development_only"] is True and manifest["independent_ground_truth"] is False
    assert all(row["selected_frame_offset"] == 1 for row in manifest["exports"])
    assert {row["event_id"] for row in manifest["exports"]} == {"fixture0_p01", "fixture1_p01"}
    assert not (output / "review_alignment_sources").exists()
    assert len(list((output / "alignment_samples").glob("*.jpg"))) == 6
    records = load_exports(output / "manifest.json")
    assert len(records) == 2
    cap = cv2.VideoCapture(str(source / "clean_sources/fixture0.mp4"))
    expected = importer.read_frame(cap, 4)
    cap.release()
    actual = cv2.imread(str(output / manifest["exports"][0]["image"]))
    assert float(np.abs(expected.astype(float) - actual.astype(float)).mean()) < 3
    assert actual[:12].mean() > 30
    with pytest.raises(FileExistsError):
        importer.export_round(source, output)
