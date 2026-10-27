from copy import deepcopy

import pytest

from scripts.recover_shoe_evidence import associate_measured_actor, build_bound


def fixtures():
    rows = [{"frame": i, "bbox": [10, 10, 40, 80]} for i in range(0, 20, 2)]
    source = {"clip_sha256": "a" * 64, "frame_count": 60}
    bundle = {"source_mapping": source, "video_path": "/unused/source.mp4", "old_selected_actor": 99,
              "lossless_shot": {"shot_sha256": "b" * 64, "source_start_frame": 20,
                                "source_end_frame_exclusive": 40, "fps": 30., "width": 100, "height": 100}}
    contact = {"video_sha256": "b" * 64, "prediction": {"frame": 12},
               "diagnostics": {"actor": {"actor_id": 2}},
               "automatic_window": {"end_frame_exclusive": 20}}
    evidence = {"video_sha256": "b" * 64, "tracks": [{"track_id": 2, "observations": rows}],
                "transforms": {str(i): [[1, 0, 0], [0, 1, 0], [0, 0, 1]] for i in range(0, 20, 2)}}
    return bundle, contact, evidence


def test_restores_original_clock_and_keeps_unmeasured_frames_missing():
    bundle, contact, evidence = fixtures()
    bound, reason = build_bound(bundle, contact, evidence)
    assert reason is None
    assert bound["contact_bracket_frames"] == [31, 33]
    assert bound["inference_interval_frames"] == [20, 39]
    assert bound["plan"][1]["anchor_bbox"] is None
    assert "20" in bound["original_camera_motion"]["transforms"]
    assert "0" not in bound["original_camera_motion"]["transforms"]


def test_wrong_shot_hash_rejected_before_using_tracks():
    bundle, contact, evidence = fixtures()
    evidence["video_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="verified shot"):
        build_bound(bundle, contact, evidence)


def test_interpolated_pose_not_promoted_to_measured_anchor():
    bundle, contact, evidence = fixtures()
    evidence["tracks"][0]["observations"][2]["interpolated"] = True
    with pytest.raises(ValueError, match="Interpolated"):
        build_bound(bundle, contact, evidence)


def test_association_requires_unique_agreement_not_old_selected_identity():
    bundle, contact, evidence = fixtures()
    actor = evidence["tracks"][0]
    old = {"track_id": 7, "observations": [{**r, "frame": r["frame"] + 20} for r in actor["observations"]]}
    selected, audit = associate_measured_actor([old], actor, 20)
    assert selected == 7
    assert audit["candidates"][0]["median_iou"] == 1.
    duplicate = {**deepcopy(old), "track_id": 8}
    assert associate_measured_actor([old, duplicate], actor, 20)[0] is None


def test_old_measured_frame_can_fill_gap_without_overwriting_new_measurement():
    bundle, contact, evidence = fixtures()
    actor = evidence["tracks"][0]
    old = {"track_id": 7, "observations": [{**r, "frame": r["frame"] + 20} for r in actor["observations"]]}
    old["observations"].append({"frame": 21, "bbox": [12, 10, 42, 80]})
    old["observations"][0]["bbox"] = [11, 10, 41, 80]
    bound, reason = build_bound(bundle, contact, evidence, {"tracks": [old]})
    assert reason is None
    assert bound["plan"][0]["anchor_bbox"] == [10, 10, 40, 80]
    assert bound["plan"][1]["anchor_bbox"] == [12, 10, 42, 80]
    assert bound["plan"][1]["anchor_source"] == "original_measured_pose_geometrically_reassociated"


def test_unreviewed_frames_never_become_training_negatives_or_scored_errors():
    from scripts.train_recovered_shoes import restrict_training_to_review, restrict_scoring_to_review
    event = {"review_window": {"start_frame": 20, "end_frame": 30},
             "landings": [{"frame": 25, "foot": "left"}],
             "evidence": {"landing_rows": [{"frame": 10}, {"frame": 25}, {"frame": 40}]}}
    training = restrict_training_to_review(event)
    assert training["evidence"]["landing_rows"] == [{"frame": 25}]
    assert len(event["evidence"]["landing_rows"]) == 3
    prediction = {"landings": [{"frame": 10}, {"frame": 26}, {"frame": 40}], "candidate_count": 3}
    assert restrict_scoring_to_review(event, prediction)["candidate_count"] == 1
    assert prediction["candidate_count"] == 3


def test_same_outer_contact_parameters_applied_to_training_and_validation(monkeypatch):
    import scripts.train_recovered_shoes as trainer
    used = []
    parameters = {"identifier": "exclude_group_a"}
    def predict(record, supplied):
        used.append((record["event"], supplied))
        return {"frame": 12}
    monkeypatch.setattr(trainer, "predict_contact", predict)
    monkeypatch.setattr(trainer, "extract_evidence", lambda inputs, poses: {"bracket": inputs["contact_bracket_frames"]})
    events = [{"clip_id": cid, "match_group": group, "evidence": {}} for cid, group in (("one", "a"), ("two", "b"))]
    recovered = {e["clip_id"]: {"inputs": {"scene_interval_frames": [20, 40]}, "poses": [],
        "record": {"event": e["clip_id"]}, "receipt": {"lossless_shot": {"source_start_frame": 20}}} for e in events}
    output = trainer.events_for_contact_weights(events, recovered, parameters)
    assert [u[0] for u in used] == ["one", "two"]
    assert all(u[1] is parameters for u in used)
    assert all(e["evidence"]["bracket"] == [31, 33] for e in output)


def test_short_track_handoff_requires_both_fragments_to_match_new_measurements():
    actor = {"observations": [{"frame": i, "bbox": [10, 10, 40, 80]} for i in range(20)]}
    tracks = [{"track_id": 1, "observations": [{"frame": i + 20, "bbox": [10, 10, 40, 80]} for i in range(12)]},
              {"track_id": 2, "observations": [{"frame": i + 20, "bbox": [10, 10, 40, 80]} for i in range(10, 20)]}]
    _, audit = associate_measured_actor(tracks, actor, 20)
    assert set(audit["selected_old_track_ids"]) == {1, 2}
    tracks[1]["observations"] = [{"frame": i + 20, "bbox": [10, 10, 40, 80]} for i in range(2, 20)]
    assert associate_measured_actor(tracks, actor, 20)[1]["selected_old_track_ids"] == []


def test_temporal_foot_preserves_order_and_anatomical_antisymmetry():
    import numpy as np
    from src.shoe_temporal_foot import vector_from_rows
    rows = [{"frame": f, "foot": s, "x": np.full(100, f if s == "right" else 0.)}
            for f in range(10, 31) for s in ("left", "right")]
    vector = vector_from_rows(rows, [19, 21], 30.)
    swapped = [{**r, "foot": "left" if r["foot"] == "right" else "right"} for r in rows]
    assert np.allclose(vector, -vector_from_rows(swapped, [19, 21], 30.))
    assert vector[0] < vector[-1]
    assert vector_from_rows(rows[:2], [19, 21], 30.) is None


def test_coarse_external_steps_never_supply_landing_negatives():
    from scripts.train_recovered_shoes import restrict_training_to_review
    event = {"external_foot_only": True, "review_window": None, "landings": [],
             "evidence": {"landing_rows": [{"frame": 99}], "foot_vector": [1, 2]}}
    result = restrict_training_to_review(event)
    assert result["evidence"]["landing_rows"] == []
    assert result["evidence"]["foot_vector"] == [1, 2]


def test_support_control_abstains_on_opposite_simultaneous_plants():
    from scripts.evaluate_support_foot import support_foot_candidate
    prediction = {"automatic_window": {"end_frame": 30},
                  "landings": [{"frame": 28, "foot": "left"}, {"frame": 28, "foot": "right"}]}
    assert support_foot_candidate(prediction, 30.)["striking_foot"] == "unknown"
    prediction["landings"] = [{"frame": 28, "foot": "right"}, {"frame": 34, "foot": "left"}]
    assert support_foot_candidate(prediction, 30.)["striking_foot"] == "left"
