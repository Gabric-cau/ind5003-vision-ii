from copy import deepcopy

import pytest

from src import contact_fullshot as full
from src.contact_temporal_model import LAGS_SECONDS, SMOOTHING_SECONDS, MINIMUM_SUPPORT_SAMPLES
from src.contact_video_features import SETTINGS


def fixture_record():
    record = {"settings": dict(SETTINGS), "feature_names": ["a"], "video_sha256": "a" * 64,
              "fps": 25., "frame_ids": [0, 2, 4], "window": {"start_frame": 0, "end_frame_exclusive": 6},
              "provenance": {"model_sha256": {"model.pt": "b" * 64}, "code_sha256": {"old.py": "c" * 64}},
              "diagnostics": {}}
    metadata = {"visual_settings": dict(SETTINGS), "feature_names": ["a"],
                "visual_model_sha256": record["provenance"]["model_sha256"],
                "extractor_code_sha256": record["provenance"]["code_sha256"],
                "lags_seconds": list(LAGS_SECONDS), "smoothing_seconds": SMOOTHING_SECONDS,
                "minimum_support_samples": MINIMUM_SUPPORT_SAMPLES}
    return record, metadata


def shot(end=6, capped=False):
    return {"shot_start_frame": 0, "shot_end_frame": end, "camera_shot_end_frame": end + (100 if capped else 0),
            "shot_end_reason": "duration_cap" if capped else "scene_cut"}


def test_reuse_requires_exact_sampled_frames_and_bound_source():
    record, metadata = fixture_record()
    sample = {"video_sha256": "a" * 64, "fps": 25.}
    evidence = {"video_sha256": "a" * 64}
    assert full.reusable_cache(record, evidence, sample, metadata, shot(), [0, 2, 4])
    assert not full.reusable_cache(record, evidence, sample, metadata, shot(8), [0, 2, 4, 6])
    with pytest.raises(ValueError, match="source mismatch"):
        full.reusable_cache(record, {"video_sha256": "d" * 64}, sample, metadata, shot(), [0, 2, 4])


def test_transfer_is_explicit_and_preserves_real_inference_settings():
    record, metadata = fixture_record()
    result, _ = full.stamp_protocol(record, {}, shot())
    before = deepcopy(result)
    with pytest.raises(ValueError, match="explicit experimental"):
        full.validate_fullshot_prediction(result, metadata)
    audit = full.validate_fullshot_prediction(result, metadata, allow_window_transfer=True)
    assert audit["mode"] == "experimental_frozen_head_window_transfer"
    assert result == before
    assert result["settings"]["max_shot_sec"] == 60
    assert record["settings"]["max_shot_sec"] == 12


def test_transfer_rejects_other_feature_changes():
    record, metadata = fixture_record()
    result, _ = full.stamp_protocol(record, {}, shot())
    result["settings"]["stride"] = 4
    with pytest.raises(ValueError, match="other than"):
        full.validate_fullshot_prediction(result, metadata, allow_window_transfer=True)


def test_matched_metadata_validates_new_protocol_exactly():
    record, metadata = fixture_record()
    result, _ = full.stamp_protocol(record, {}, shot(capped=True))
    metadata["visual_settings"] = deepcopy(result["settings"])
    metadata["extractor_code_sha256"] = deepcopy(result["provenance"]["code_sha256"])
    assert full.validate_fullshot_prediction(result, metadata)["mode"] == "matched_training_protocol"
    assert result["diagnostics"]["fullshot_protocol"]["duration_capped"]
    assert result["diagnostics"]["fullshot_protocol"]["contact_time_or_labels_used_to_select_window"] is False
