import pytest

from scripts.evaluate_local_video_vlm import api, sampling_plan, validate_prediction


def contact(frame=100, first=0, end=200):
    return {"prediction": {"frame": frame}, "automatic_window": {
        "start_frame": first, "end_frame_exclusive": end, "automatic": True}}


def test_sampling_uses_automatic_anchor_and_never_crosses_camera_cut():
    plan = sampling_plan(contact(frame=100, first=90, end=120), 300, 50)
    assert plan["first_frame"] == 90
    assert plan["last_frame"] == 119
    assert 100 in plan["sampled_frames"]
    assert plan["sampled_frames"] == sorted(set(plan["sampled_frames"]))
    assert all(90 <= frame < 120 for frame in plan["sampled_frames"])
    assert plan["pre_context_clipped"] and plan["post_context_clipped"]


def test_unbounded_contact_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        sampling_plan(contact(frame=200, end=200), 300, 50)


def test_nonlocal_endpoint_is_rejected_before_network_access():
    with pytest.raises(ValueError, match="local Ollama"):
        api("https://example.com", "/api/chat", {})


def test_output_requires_all_classes_and_rejects_invented_values():
    value = {"kicking_foot": "left", "goal_outcome": "unknown", "keeper_ordinal_height": "low",
             "keeper_dive_direction": "image_right", "visual_evidence": "Ball result is unresolved"}
    assert validate_prediction(value) == value
    with pytest.raises(Exception):
        validate_prediction({**value, "kicking_foot": "probably left"})


def test_unknown_is_not_counted_as_a_correct_or_covered_prediction():
    from scripts.evaluate_offline_video_batch import categorical_metrics
    metric = categorical_metrics([{"core_id": "one", "reference": "left", "prediction": "unknown"},
                                  {"core_id": "two", "reference": "right", "prediction": "right"}], ("left", "right"))
    assert metric["correct_events"] == 1
    assert metric["strict_accuracy_unknown_as_incorrect"] == .5
    assert metric["coverage_on_comparable_references"] == .5
