from copy import deepcopy

import numpy as np
import pytest

from src import contact_actor_support as recovery


def fixture_record():
    return {"video_sha256": "a" * 64, "frame_ids": [0, 2, 4, 6], "fps": 25., "features": [[0]] * 4,
            "settings": {}, "provenance": {"code_sha256": {}},
            "diagnostics": {"actor": {"actor_id": 1}}, "baselines": {"existing_rule": {"time_seconds": 1}},
            "window": {"end_frame_exclusive": 8}}


def test_supported_actor_is_preserved(monkeypatch):
    record = fixture_record()
    monkeypatch.setattr(recovery, "build_temporal_features", lambda r: (None, np.array([False, True, True, True])))
    result, _ = recovery.recover_actor_support(record, {"video_sha256": "a" * 64}, 1080)
    assert result["features"] == record["features"]
    assert result["baselines"] == record["baselines"]
    assert not result["diagnostics"]["actor_support_recovery"]["used"]


def test_recovery_uses_candidate_interval_and_keeps_ranking(monkeypatch):
    record = fixture_record()
    candidates = [{"contact_associated_track_id": i, "interval_frames": [2, 6]} for i in (11, 1)]
    evidence = {"video_sha256": "a" * 64, "transforms": {"0": np.eye(3).tolist()},
                "tracks": [], "balls": [], "camera": {}, "candidate_contacts": candidates}
    def support(r):
        return None, np.array([False, True, True, True]) if r["features"][0][0] == 1 else np.zeros(4, bool)
    monkeypatch.setattr(recovery, "build_temporal_features", support)
    monkeypatch.setattr(recovery, "geometric_features", lambda ids, fps, height, tracks, balls, matrices, ordered, camera:
        ([[ordered[0]["contact_associated_track_id"]]] * 4, {"actor_id": ordered[0]["contact_associated_track_id"]}, {}))
    monkeypatch.setattr(recovery, "rule_baseline", lambda *a: {"time_seconds": None})
    original = deepcopy(evidence)
    result, output = recovery.recover_actor_support(record, evidence, 1080)
    assert result["diagnostics"]["actor_support_recovery"]["selected_candidate_rank"] == 1
    assert result["diagnostics"]["actor"]["actor_id"] == 1
    assert output["candidate_contacts"][0]["contact_associated_track_id"] == 1
    assert evidence == original


def test_recovery_rejects_source_mismatch():
    with pytest.raises(ValueError, match="source mismatch"):
        recovery.recover_actor_support(fixture_record(), {"video_sha256": "b" * 64}, 1080)
