"""Explicit full-first-shot protocol around the frozen contact extractor."""
from copy import deepcopy
from pathlib import Path

from src.contact_temporal_model import validate_prediction_bindings
from src.contact_video_features import SETTINGS
from src.runtime import file_sha256
from src.step_pipeline import _scan_camera_shots

MAX_SHOT_SECONDS = 60.0
PROTOCOL_CODE = "src/contact_fullshot.py"
ROOT = Path(__file__).resolve().parents[1]


def fullshot_settings(settings):
    if not set(SETTINGS) <= set(settings):
        raise ValueError("Missing frozen extractor settings")
    return {**settings, "max_shot_sec": MAX_SHOT_SECONDS}


def scan_fullshot(video, settings):
    shot = _scan_camera_shots(str(video), MAX_SHOT_SECONDS, settings["cut_threshold"])[0]
    ids = list(range(shot["shot_start_frame"], shot["shot_end_frame"], settings["stride"]))
    return shot, ids


def stamp_protocol(record, evidence, shot, *, cache_source=None):
    result, visual = deepcopy(record), deepcopy(evidence)
    result["settings"] = fullshot_settings(result["settings"])
    result["provenance"]["code_sha256"][PROTOCOL_CODE] = file_sha256(ROOT / PROTOCOL_CODE)
    result["diagnostics"]["fullshot_protocol"] = {
        "version": 1, "selection": "first_automatic_camera_shot",
        "maximum_seconds": MAX_SHOT_SECONDS,
        "camera_shot_end_frame": shot["camera_shot_end_frame"],
        "analysis_end_frame_exclusive": shot["shot_end_frame"],
        "duration_capped": shot["shot_end_reason"] == "duration_cap",
        "shot_end_reason": shot["shot_end_reason"],
        "contact_time_or_labels_used_to_select_window": False,
        "cached_features_reused": cache_source is not None,
        "cache_source": cache_source,
    }
    visual["fullshot_protocol"] = result["diagnostics"]["fullshot_protocol"]
    return result, visual


def reusable_cache(record, evidence, sample, metadata, shot, frame_ids):
    """Reuse only an identical sampled window with all frozen bindings intact."""
    validate_prediction_bindings(record, metadata)
    if record["video_sha256"] != sample["video_sha256"] or evidence["video_sha256"] != sample["video_sha256"]:
        raise ValueError("Cached contact evidence source mismatch")
    if record["fps"] != sample["fps"]:
        raise ValueError("Cached contact evidence FPS mismatch")
    expected = {"start_frame": shot["shot_start_frame"], "end_frame_exclusive": shot["shot_end_frame"]}
    return record["frame_ids"] == frame_ids and all(record["window"][key] == value for key, value in expected.items())


def validate_fullshot_prediction(record, metadata, *, allow_window_transfer=False):
    if metadata["visual_settings"].get("max_shot_sec") == MAX_SHOT_SECONDS:
        validate_prediction_bindings(record, metadata)
        return {"mode": "matched_training_protocol", "changed_inference_settings": {}}
    if not allow_window_transfer:
        raise ValueError("Old short-window model requires explicit experimental transfer flag")
    if metadata["visual_settings"].get("max_shot_sec") != SETTINGS["max_shot_sec"]:
        raise ValueError("Only the frozen 12-second protocol supports this transfer audit")
    if record["settings"] != fullshot_settings(metadata["visual_settings"]):
        raise ValueError("Transfer changed settings other than the declared window cap")
    expected_codes = {**metadata["extractor_code_sha256"], PROTOCOL_CODE: file_sha256(ROOT / PROTOCOL_CODE)}
    if record["provenance"]["code_sha256"] != expected_codes:
        raise ValueError("Transfer changed feature code outside the declared protocol wrapper")
    # Validate every unchanged binding on a separate audit view, never alter evidence.
    audit_view = deepcopy(record)
    audit_view["settings"] = deepcopy(metadata["visual_settings"])
    audit_view["provenance"]["code_sha256"] = deepcopy(metadata["extractor_code_sha256"])
    validate_prediction_bindings(audit_view, metadata)
    return {"mode": "experimental_frozen_head_window_transfer",
            "changed_inference_settings": {"max_shot_sec": {"training": 12.0, "inference": MAX_SHOT_SECONDS}},
            "matched_training_protocol": False}
