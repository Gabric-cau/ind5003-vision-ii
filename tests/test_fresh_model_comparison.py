import pytest
from scripts.compare_fresh_ball_models import compare_event


def predictions():
    identity = {"source_sha256": "source", "pose_sha256": "pose", "source_manifest_sha256": "manifest",
                "settings": {"imgsz": 1280}, "code_sha256": {"rules": "same"}}
    return [{"clip_id": "event", "inference_identity": {**identity, "model_sha256": model},
             "reference_labels_used_for_prediction": False} for model in ("nano", "medium")]


def test_detector_comparison_rejects_changed_temporal_rules():
    first, second = predictions()
    second["inference_identity"]["code_sha256"] = {"rules": "different"}
    with pytest.raises(ValueError, match="Uncontrolled"):
        compare_event(first, second)


def test_abstentions_are_preserved_in_paired_comparison():
    first, second = predictions()
    result = compare_event(first, second)
    assert result["contact_pair"] == "00"
    assert result["contact_frame_difference"] is None
    assert result["n"]["outcome"] == "unknown"
