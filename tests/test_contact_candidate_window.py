import numpy as np

from src import contact_candidate_window as window


def test_larger_probability_outside_automatic_interval_cannot_win(monkeypatch):
    record = {"frame_ids": [0, 10, 20, 30, 40], "time_seconds": [0., .2, .4, .6, .8],
              "diagnostics": {"candidate_window_recovery": {"used": True, "candidate_interval_frames": [20, 30]}}}
    monkeypatch.setattr(window, "build_temporal_features", lambda r: (None, np.ones(5, dtype=bool)))
    monkeypatch.setattr(window, "predict_probabilities", lambda *a: np.array([.99, .9, .7, .8, .95]))
    result = window.predict_candidate_contact(record, {})
    assert result["frame"] == 30
    assert result["time_seconds"] == .6
    assert result["eligible_frames"] == 2


def test_no_supported_candidate_interval_abstains(monkeypatch):
    record = {"frame_ids": [0, 10], "time_seconds": [0., .2],
              "diagnostics": {"candidate_window_recovery": {"used": True, "candidate_interval_frames": [20, 30]}}}
    monkeypatch.setattr(window, "build_temporal_features", lambda r: (None, np.ones(2, dtype=bool)))
    monkeypatch.setattr(window, "predict_probabilities", lambda *a: np.array([.9, .8]))
    assert window.predict_candidate_contact(record, {})["time_seconds"] is None
