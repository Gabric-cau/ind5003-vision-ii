from copy import deepcopy

import numpy as np
import pytest

from scripts import train_external_camera_classifier as model


def metadata():
    clips, timeline = [], []
    for i, group in enumerate(["german_cup_2022", "coppa_italia_2025", "world_cup_2022"], 1):
        clips.append({"export_index": i, "dataset_id": group, "filename": f"match_{i:02d}.mp4",
                      "file": f"videos/match_{i:02d}.mp4", "original_event_id": str(i),
                      "duration_seconds": 10., "fps": 25., "media_check": {"sha256": str(i)}})
        timeline.append({"export_index": i, "original_shot_id": "shot" + str(i),
                         "video_file": clips[-1]["file"], "camera_label": next(iter(model.LABEL_MAP)),
                         "camera_annotation_source": "human_user_observation_in_chat",
                         "camera_label_start_seconds": 2., "camera_label_end_seconds_exclusive": 5.})
    return {"clips": clips, "shot_timeline": timeline}


def test_holdout_metadata_never_enters_pixels_or_training_manifest(tmp_path):
    rows, _ = model.select_intervals(metadata(), tmp_path)
    assert len(rows) == 2
    assert {row["match_group"] for row in rows} == model.ALLOWED_MATCHES


def test_partial_interval_respects_exact_human_label_window(tmp_path):
    data = metadata()
    data["shot_timeline"][0].update(segment_start_seconds=0., segment_end_seconds_exclusive=10.,
                                     camera_label_scope="partial_shot")
    rows, _ = model.select_intervals(data, tmp_path)
    assert rows[0]["start_seconds"] == 2 and rows[0]["end_seconds_exclusive"] == 5


def test_unlabelled_or_inexact_annotation_is_not_a_negative(tmp_path):
    data = metadata()
    extra = deepcopy(data["shot_timeline"][0])
    extra.update(original_shot_id="unknown", camera_label=None)
    data["shot_timeline"].append(extra)
    rows, excluded = model.select_intervals(data, tmp_path)
    assert len(rows) == 2 and excluded[0]["reason"] == "unlabelled_not_negative"
    extra.update(camera_label=next(iter(model.LABEL_MAP)), camera_label_start_seconds=None)
    _, excluded = model.select_intervals(data, tmp_path)
    assert excluded[0]["reason"] == "no_exact_label_interval"


def test_matches_events_and_videos_do_not_cross_camera_folds(tmp_path):
    rows, _ = model.select_intervals(metadata(), tmp_path)
    for _, training, validation in model.group_folds(rows):
        assert not set(training) & set(validation)
        assert not {rows[i]["match_group"] for i in training} & {rows[i]["match_group"] for i in validation}
    rows[1]["video_sha256"] = rows[0]["video_sha256"]
    with pytest.raises(ValueError, match="crosses"):
        model.group_folds(rows)


def test_camera_train_only_preprocessing_and_numeric_export_parity():
    rng = np.random.default_rng(38)
    features = rng.normal(size=(32, 32))
    training = list(range(18))
    labels = [model.CLASSES[i % 3] for i in training]
    fitted = model.fit_head(features, training, labels)
    altered = features.copy()
    altered[18:] = 1e9
    changed = model.fit_head(altered, training, labels)
    original, actual = model.export_head(*fitted), model.export_head(*changed)
    for key in original:
        np.testing.assert_array_equal(original[key], actual[key])
    x = fitted[1].transform(fitted[0].transform(features))
    np.testing.assert_allclose(model.predict_head(features, original)["probabilities"],
                               fitted[2].predict_proba(x), atol=1e-12)


def test_long_automatic_shots_are_bounded_without_gaps_or_boundary_claims():
    shots = [{"shot_start_frame": 0, "shot_end_frame": 17, "fps": 10., "shot_index": 0},
             {"shot_start_frame": 17, "shot_end_frame": 93, "fps": 10., "shot_index": 1}]
    windows = model.prediction_windows(shots, 2.)
    assert windows[0]["start_seconds"] == 0 and windows[-1]["end_seconds_exclusive"] == 9.3
    assert all(row["end_seconds_exclusive"] - row["start_seconds"] <= 2.000001 for row in windows)
    assert all(first["end_seconds_exclusive"] == second["start_seconds"]
               for first, second in zip(windows[:-1], windows[1:]))
    assert windows[0]["subdivided_automatic_shot"] is False
    assert all(row["subdivided_automatic_shot"] for row in windows[1:])


@pytest.mark.parametrize("seconds", [0, -1, float("nan")])
def test_invalid_prediction_window_limit_is_rejected(seconds):
    with pytest.raises(ValueError, match="positive"):
        model.prediction_windows([], seconds)


def test_sequential_decoding_reads_shared_video_once_and_keeps_frame_clock(monkeypatch):
    class Capture:
        reads = 0
        released = False

        def read(self):
            frame = np.full((10, 10, 3), self.reads, dtype=np.uint8)
            self.reads += 1
            return True, frame

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(model.cv2, "VideoCapture", lambda _: capture)
    windows = [{"start_seconds": 0., "end_seconds_exclusive": .8},
               {"start_seconds": .8, "end_seconds_exclusive": 1.6}]
    decoded = model.decode_windows("unused.mp4", windows, 10.)
    assert capture.reads == 16 and capture.released
    assert decoded[0][1] == list(range(8)) and decoded[1][1] == list(range(8, 16))
    assert len(decoded[0][0]) == len(decoded[1][0]) == 8
