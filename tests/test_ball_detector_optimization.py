import pytest

from scripts.optimize_ball_detector import ball_class, tile_windows, summarize_groups, make_training_dataset
from scripts.train_ball_frame_detector import score_frame


def test_tiles_cover_all_edges_without_leaving_the_picture():
    windows = tile_windows(1920, 1080)
    assert (0, 0, 960, 960) in windows
    assert (960, 120, 1920, 1080) in windows
    assert all(0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080 for x1, y1, x2, y2 in windows)
    assert tile_windows(640, 480) == [(0, 0, 640, 480)]


def test_ball_class_handles_custom_class_zero_and_coco32():
    class Model:
        names = {0: "person", 32: "sports ball"}
    assert ball_class(Model()) == 32
    Model.names = {0: "ball"}
    assert ball_class(Model()) == 0
    Model.names = {0: "person"}
    with pytest.raises(ValueError):
        ball_class(Model())


def test_dedup_does_not_inflate_hit_count_or_drop_misses():
    hit = score_frame([0, 0, 10, 10], [{"bbox": [0, 0, 10, 10], "confidence": .9}])
    miss = score_frame([0, 0, 10, 10], [])
    rows = [{**hit, "video_id": "core_05", "deduplicated_primary": True},
            {**hit, "video_id": "core_05", "deduplicated_primary": False},
            {**miss, "video_id": "core_05", "deduplicated_primary": True}]
    result = summarize_groups(rows)
    assert result["all_frames"]["top1_iou_0_5_hits"] == 2
    assert result["deduplicated_frames"]["top1_localization_recall_iou_0_5"] == .5


def test_evaluation_split_cannot_silently_enter_training_validation(tmp_path):
    with pytest.raises(ValueError, match="Evaluation-only"):
        make_training_dataset([{"split": "test", "video_id": "core_05"}], tmp_path, 2)
