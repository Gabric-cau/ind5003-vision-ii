from copy import deepcopy
import json

import pytest

from scripts.prepare_external_ball_training_2026_10_25 import build_manifest, candidate_segments


def fixture(tmp_path):
    camera = "\u70b9\u7403\u8005\u540e\u4fa7\u89c6\u89d2"
    document = {"clips": [], "shot_timeline": []}
    for index, dataset in enumerate(("german_cup_2022", "coppa_italia_2025", "world_cup_2022"), 1):
        clip = {
            "dataset_id": dataset, "original_event_id": f"event_{index}",
            "file": f"videos/match_{index:02}.mp4", "filename": f"match_{index:02}.mp4",
            "fps": 25, "duration_seconds": 3, "expected_video_frames": 75,
            "source_start_frame_zero_based": 100, "training_goal_label": 1,
            "human_features": {
                "provenance": "human_user_observation_in_chat",
                "direction_reference_frame": "behind_kicker_view",
                "ball_direction_label": "\u5de6", "ball_height_label": "\u4f4e",
            },
        }
        document["clips"].append(clip)
        document["shot_timeline"].append({
            "video_file": clip["file"], "included_in_confirmed_event_range": True,
            "camera_annotation_source": "human_user_observation_in_chat",
            "camera_label": camera, "camera_label_scope": "whole_shot",
            "camera_label_start_seconds": 0, "camera_label_end_seconds_exclusive": 3,
            "segment_start_seconds": 0, "segment_end_seconds_exclusive": 3,
            "original_shot_id": f"shot_{index}",
        })
    path = tmp_path / "metadata/annotations.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document), encoding="utf-8")
    return document, path


def test_match_split_excludes_holdout_targets_and_preserves_source(tmp_path):
    document, path = fixture(tmp_path)
    before = deepcopy(document)
    result = build_manifest(document, path)
    assert [sample["split"] for sample in result["samples"]] == ["train", "validation"]
    assert result["summary"]["heldout_events_locked"] == 1
    assert result["excluded_events"][0]["split"] == "heldout"
    assert "targets" not in result["excluded_events"][0]
    assert "selected_segments" not in result["excluded_events"][0]
    assert document == before


@pytest.mark.parametrize("field,value,reason", [
    ("direction_reference_frame", None, "direction_reference_unconfirmed"),
    ("ball_direction_label", "unknown", "unknown_or_unsupported_target_label"),
    ("ball_height_label", None, "unknown_or_unsupported_target_label"),
    ("provenance", "model_prediction", "human_event_label_provenance_missing"),
])
def test_unknown_and_nonhuman_labels_are_excluded(tmp_path, field, value, reason):
    document, path = fixture(tmp_path)
    document["clips"][0]["human_features"][field] = value
    result = build_manifest(document, path)
    assert all(sample["split"] != "train" for sample in result["samples"])
    assert result["excluded_events"][0]["reason"] == reason


def test_result_conflict_quarantines_the_entire_event(tmp_path):
    document, path = fixture(tmp_path)
    document["clips"][1]["outcome_conflict"] = True
    result = build_manifest(document, path)
    assert result["summary"]["validation_events"] == 0
    assert "quarantined" in result["excluded_events"][0]["reason"]


def test_partial_annotations_use_only_exact_annotated_range(tmp_path):
    document, _ = fixture(tmp_path)
    shot = document["shot_timeline"][0]
    shot.update(camera_label_scope="partial_shot", camera_label_start_seconds=1.2,
                camera_label_end_seconds_exclusive=2.4)
    selected = candidate_segments(document["clips"][0], document["shot_timeline"], "behind_kicker")
    assert [(s["frame_start_zero_based"], s["frame_end_exclusive"]) for s in selected] == [(30, 60)]
    assert selected[0]["source_frame_start_zero_based"] == 130
    shot["camera_label_start_seconds"] = None
    assert candidate_segments(document["clips"][0], document["shot_timeline"], "behind_kicker") == []


def test_crowd_unlabelled_occluded_and_outside_shots_are_not_supervised(tmp_path):
    document, _ = fixture(tmp_path)
    good = document["shot_timeline"][0]
    extras = []
    for update in ({"camera_label": None}, {"camera_label": "crowd"},
                   {"occlusion_label": True}, {"included_in_confirmed_event_range": False}):
        extra = {**good, **update, "original_shot_id": f"excluded_{len(extras)}"}
        extras.append(extra)
    selected = candidate_segments(document["clips"][0], [good, *extras], "behind_kicker")
    assert [segment["shot_id"] for segment in selected] == ["shot_1"]


def test_primary_action_does_not_select_a_later_replay(tmp_path):
    document, _ = fixture(tmp_path)
    shot = document["shot_timeline"][0]
    shot.update(replay_label=True, segment_start_seconds=1, camera_label_start_seconds=1)
    assert candidate_segments(document["clips"][0], [shot], "primary_action") == []
    assert candidate_segments(document["clips"][0], [shot], "behind_kicker")[0]["replay_label"] is True


def test_optional_replay_augmentation_is_train_only_and_event_weighted(tmp_path):
    document, path = fixture(tmp_path)
    for index in (0, 1):
        document["shot_timeline"].append({
            **document["shot_timeline"][index],
            "original_shot_id": f"replay_{index}",
            "segment_start_seconds": 1,
            "camera_label_start_seconds": 1,
            "replay_label": True,
        })
    result = build_manifest(document, path, augment_training_behind_kicker=True)
    train, validation = result["samples"]
    assert len(train["selected_segments"]) == 2
    assert sum(segment["event_segment_weight"] for segment in train["selected_segments"]) == 1
    assert len(validation["selected_segments"]) == 1
    assert validation["selected_segments"][0]["event_segment_weight"] == 1
    assert train["targets"] == {"ball_direction": "left", "ball_height_category": "low"}


def test_duplicate_event_ids_and_out_of_range_timing_fail(tmp_path):
    document, path = fixture(tmp_path)
    duplicate = deepcopy(document)
    duplicate["clips"].append(duplicate["clips"][0])
    with pytest.raises(ValueError, match="Duplicate event"):
        build_manifest(duplicate, path)
    document["shot_timeline"][0]["camera_label_end_seconds_exclusive"] = 4
    with pytest.raises(ValueError, match="Invalid annotated interval"):
        build_manifest(document, path)
