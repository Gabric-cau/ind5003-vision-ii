from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.shoe_supervised_model import fit_models, landing_training_rows, predict_event
from scripts.train_shoe_supervised import landing_metrics


def event(cid, group, sign=1):
    rows = [{"frame": f, "foot": side, "x": np.full(100, f / 20)}
            for f in range(20) for side in ("left", "right")]
    return {"clip_id": cid, "match_group": group, "fps": 30,
            "landings": [{"frame": 10, "foot": "left"}],
            "foot": "right" if sign == 1 else "left",
            "evidence": {"landing_rows": rows, "foot_vector": np.full(57, sign)}}


def test_held_out_labels_cannot_change_model_or_prediction():
    events = [event("a", "train"), event("b", "train", -1), event("c", "validation")]
    first = fit_models(events, held_out_match="validation")
    modified = deepcopy(events)
    modified[2]["landings"] = [{"frame": 0, "foot": "right"}]
    modified[2]["foot"] = "left"
    second = fit_models(modified, held_out_match="validation")
    assert first["training_foot_ids"] == ["a", "b"]
    assert first["training_landing_ids"] == ["a", "b"]
    assert predict_event(first, events[2]["evidence"], 30) == predict_event(second, events[2]["evidence"], 30)


def test_anatomical_side_swap_reverses_foot_score():
    model = fit_models([event("a", "train"), event("b", "train", -1)])
    positive = predict_event(model, event("x", "test")["evidence"], 30)
    negative = predict_event(model, event("x", "test", -1)["evidence"], 30)
    assert positive["striking_foot"] == "right"
    assert negative["striking_foot"] == "left"
    assert np.isclose(positive["right_score_uncalibrated"] + negative["right_score_uncalibrated"], 1)


def test_missing_evidence_abstains_instead_of_counting_zero_steps():
    model = fit_models([event("a", "train")])
    result = predict_event(model, {"landing_rows": [], "foot_vector": None}, 30)
    assert result["landings"] is None
    assert result["candidate_count"] is None
    assert result["striking_foot"] == "unknown"


def test_landing_training_ignores_ambiguous_time_band_and_balances_event():
    e = event("a", "train")
    x, y, weights = landing_training_rows(e["evidence"], e["landings"], 30)
    assert np.isclose(weights[y == 0].sum(), 0.5)
    assert np.isclose(weights[y == 1].sum(), 0.5)
    assert len(x) < len(e["evidence"]["landing_rows"])
    assert int(y.sum()) == 3


def test_unavailable_event_remains_in_landing_recall_denominator():
    events = [event("available", "one"), event("missing", "two")]
    predictions = [{"landings": [{"frame": 10, "foot": "right"}]}, {"landings": None}]
    result = landing_metrics(events, predictions)
    assert result["events"] == 2
    assert result["output_events"] == 1
    assert result["recall_all_reference_events"] == 0.5
    assert result["side_accuracy_on_temporal_matches"] == 0.0


def test_video_adapter_keeps_missing_automatic_window_unavailable(tmp_path, monkeypatch):
    from scripts import predict_video_shoes as adapter
    source_dir = tmp_path / "analysis/source"
    source_dir.mkdir(parents=True)
    (source_dir / "source_manifest.json").write_text(json.dumps({
        "manual_annotations_used": False, "clips": [{"clip_id": "unseen_penalty",
        "clip_filename": "unseen_penalty.mp4", "clip_sha256": "a" * 64, "fps": 30}]}))
    checkpoint = tmp_path / "pose.task"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(adapter, "load_bound_model", lambda *args: ({}, {"model_sha256": "b" * 64}))
    monkeypatch.setattr(adapter, "bind_event", lambda *args: (None, {"unavailable_reason": "visible_approach_onset_unavailable"}))
    monkeypatch.setattr(adapter, "infer_event", lambda *args: pytest.fail("Missing window must not invoke pose inference"))
    result = adapter.run(SimpleNamespace(analysis_dir=tmp_path / "analysis", output=tmp_path / "result",
                                        pose33_model=checkpoint, model=tmp_path / "model", threads=2))
    assert result["status"] == "unavailable"
    assert result["candidate_count"] is None
    assert result["striking_foot"] == "unknown"
    assert result["manual_annotations_used"] is False
    assert result["labels_read_at_inference"] is False


def test_video_adapter_rejects_changed_model_before_deserialization(tmp_path):
    from scripts.predict_video_shoes import load_bound_model
    (tmp_path / "deployment.joblib").write_bytes(b"changed-not-a-model")
    (tmp_path / "deployment_metadata.json").write_text(json.dumps({"model_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="changed"):
        load_bound_model(tmp_path, tmp_path / "pose.task")
