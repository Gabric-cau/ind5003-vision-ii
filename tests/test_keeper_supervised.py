import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.train_keeper_supervised import evaluate, load_events
from scripts.predict_video_keeper import hybrid_direction, predict_analysis
from src.keeper_supervised_model import (FEATURE_NAMES, HEADS, classification_metrics,
                                         extract_features, fit_models, predict_features, predict_keeper)


def keeper(dx=1.0, drop=0.4):
    pre = [[i / 10, 200.0, 300.0, 100.0] for i in range(5)]
    post = [[0.5 + i / 10, 200 + dx * i * 10, 300 + drop * i * 10, 100 - drop * i * 10]
            for i in range(5)]
    return {"status": "ok", "direction_coordinate_system": "source_image_x_right",
            "gk_prekick_stabilization_complete": True, "dive_stabilization_complete": True,
            "selected_candidate": {"qualifies": True, "is_selected_kicker": False,
                "camera_reference_chain_complete": True, "pre_coverage": 1.0, "post_coverage": 1.0,
                "stabilized_pre_trajectory": pre, "stabilized_post_trajectory": post},
            "pose_direction_evidence": {"pose_samples": [{"frame": i, "torso_vector": [dx * i, -3],
                "shoulder_to_ankle_vector": [dx * i, -5]} for i in range(5)]}}


def events():
    rows = []
    for group in ("a", "b", "c", "d"):
        for index, direction in enumerate(("left", "center", "right")):
            data = keeper(index - 1, 0.3 if index != 1 else 0.1)
            rows.append({"clip_id": f"{group}{index}", "match_group": group,
                         "direction": direction, "height": "low" if index != 1 else "middle",
                         "features": extract_features(data),
                         "frozen_rule": {"direction": direction, "height": "unknown"}})
    return rows


def test_features_finite_and_body_relative():
    result = extract_features(keeper())
    assert result["available"] and not result["reasons"]
    assert len(result["x"]) == len(FEATURE_NAMES)
    assert np.isfinite(result["x"]).all()
    assert result["x"][0] == pytest.approx(0.3)


def test_absolute_clock_translation_and_scale_invariance():
    source = keeper()
    changed = copy.deepcopy(source)
    candidate = changed["selected_candidate"]
    for name in ("stabilized_pre_trajectory", "stabilized_post_trajectory"):
        for row in candidate[name]:
            row[0] += 1000
            row[1] = 2 * row[1] + 333
            row[2] = 2 * row[2] - 777
            row[3] *= 2
    for sample in changed["pose_direction_evidence"]["pose_samples"]:
        sample["frame"] += 30000
        for name in ("torso_vector", "shoulder_to_ankle_vector"):
            sample[name] = [2 * value for value in sample[name]]
    np.testing.assert_allclose(extract_features(changed)["x"], extract_features(source)["x"], atol=1e-10)


def test_horizontal_reflection_has_coherent_signed_features():
    positive = np.array(extract_features(keeper(1))["x"])
    negative = np.array(extract_features(keeper(-1))["x"])
    signed = (0, 1, 6, 11, 12)
    np.testing.assert_allclose(positive[list(signed)], -negative[list(signed)])
    invariant = [i for i in range(len(FEATURE_NAMES)) if i not in signed]
    np.testing.assert_allclose(positive[invariant], negative[invariant])


def test_rule_labels_names_and_global_timestamps_never_become_features():
    data = keeper()
    expected = extract_features(data)
    data.update({"dive_direction": "image_left", "dive_height": "high", "clip_id": "special",
                 "match_id": "hidden", "kick_frame": 900000, "human_reference": {"save_height": "middle"}})
    data["pose_direction_evidence"].update({"direction": "image_left", "supported": True})
    for sample in data["pose_direction_evidence"]["pose_samples"]:
        sample["direction"] = "image_right"
    assert extract_features(data) == expected


@pytest.mark.parametrize("change", ("stabilization", "identity", "missing", "nan", "duplicate", "coverage", "clock"))
def test_invalid_geometry_abstains(change):
    data = keeper()
    candidate = data["selected_candidate"]
    if change == "stabilization":
        candidate["camera_reference_chain_complete"] = False
    elif change == "identity":
        data["role"] = {"ambiguous_identity": True}
    elif change == "missing":
        candidate.pop("stabilized_post_trajectory")
    elif change == "nan":
        candidate["stabilized_post_trajectory"][2][1] = float("nan")
    elif change == "duplicate":
        candidate["stabilized_post_trajectory"][2][0] = 0.6
    elif change == "coverage":
        candidate["post_coverage"] = 0.10
    else:
        candidate["stabilized_pre_trajectory"][-1][0] = 0.55
    features = extract_features(data)
    assert not features["available"] and features["x"] is None and features["reasons"]
    result = predict_features(features, {})
    assert result["dive_direction"] == result["dive_height"] == "unknown"


def test_pose_absence_is_explicit_and_does_not_remove_valid_trajectories():
    data = keeper()
    data.pop("pose_direction_evidence")
    result = extract_features(data)
    assert result["available"]
    assert result["x"][-3:] == [0.0, 0.0, 0.0]


def test_fold_scaler_and_weights_ignore_all_held_out_data():
    data = events()
    before = fit_models(data, held_out_match="a")
    for event in data:
        if event["match_group"] == "a":
            event["features"]["x"] = [10000.0] * len(FEATURE_NAMES)
            event["direction"], event["height"] = "left", "middle"
    assert fit_models(data, held_out_match="a") == before
    assert all(not cid.startswith("a") for cid in before["training_ids"])


def test_numpy_json_predictor_matches_sklearn_binary_and_multiclass():
    data = events()
    model = fit_models(data)
    for head, spec in HEADS.items():
        x = np.asarray([e["features"]["x"] for e in data])[:, spec["features"]]
        labels = [e[head] for e in data]
        scaler = StandardScaler().fit(x)
        baseline = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                                      random_state=20261026).fit(scaler.transform(x), labels)
        for event, scores in zip(data, baseline.predict_proba(scaler.transform(x))):
            pred = predict_features(event["features"], model)
            actual = [pred["scores_uncalibrated"][head][name] for name in baseline.classes_]
            np.testing.assert_allclose(actual, scores, atol=1e-12)
    result = predict_keeper(keeper(), model)
    assert result["height"] in ("low", "middle") and result["height_high_class_supported"] is False
    assert result["dive_direction"] == "image_right"


def test_missing_events_stay_in_denominator():
    result = classification_metrics(["left", "right", "left"], ["left", "unknown", "left"],
                                    HEADS["direction"]["classes"])
    assert result["coverage"] == result["all_event_accuracy"] == 2 / 3
    assert result["covered_accuracy"] == 1
    assert result["all_event_macro_f1"] == pytest.approx(1 / 3)


def test_evaluation_four_groups_are_disjoint_and_no_high_class():
    data = events()
    data[0]["features"] = {"available": False, "x": None, "reasons": ["no_track"]}
    report = evaluate(data)
    assert len(report["folds"]) == 4
    assert set(report["predictions"]) == {event["clip_id"] for event in data}
    for fold in report["folds"]:
        assert not set(fold["training_ids"]) & set(fold["validation_ids"])
    assert report["metrics"]["height"]["learned_lomo"]["outputs"] == 11
    assert report["metrics"]["direction"]["training_fold_majority"]["outputs"] == 12
    assert all(p["height"] != "high" for p in report["predictions"].values())
    assert report["metrics"]["direction"]["retain_rule_fill_missing"]["correct"] == 12


def test_hybrid_never_changes_an_existing_rule_decision():
    for rule in ("left", "center", "right"):
        for learned in ("left", "center", "right", "unknown"):
            assert hybrid_direction(rule, learned) == (rule, "existing_supported_rule")
    assert hybrid_direction("unknown", "left") == ("left", "learned_missing_rule_fallback")
    assert hybrid_direction("unknown", "unknown") == ("unknown", "insufficient_evidence")


def test_source_binding_and_reference_provenance():
    mapping = {"clip_id": "a", "clip_sha256": "a" * 64, "source_video_sha256": "b" * 64,
               "fps": 30, "frame_count": 100, "match_id": "match"}
    source = {"events": [{"clip_id": "a", "source_mapping": mapping,
                          "components": {"ball": {"keeper": keeper()}}}]}
    annotation = {"direction_coordinate_system": "source_image_x_right", "field_provenance": {
        field: {"status": "human_annotated", "human_verified": True} for field in ("save_direction", "save_height")}}
    reference = {"events": [{"clip_id": "a", "source_sha256": "a" * 64, "fps": 30, "frame_count": 100,
                            "save_direction": "right", "save_height": "middle", "_shot_save_annotation": annotation}]}
    loaded, bindings = load_events(source, reference)
    assert loaded[0]["features"]["available"] and bindings[0]["label_clip_hash_matches"]
    reference["events"][0]["source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        load_events(source, reference)


def test_unsupported_height_training_label_rejected():
    data = events()
    data[0]["height"] = "high"
    with pytest.raises(ValueError, match="Unsupported height"):
        fit_models(data)


def analysis_fixture(tmp_path):
    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    video = tmp_path / "source/unseen_penalty.mp4"
    video.parent.mkdir()
    video.write_bytes(b"unit-test-source-bytes")
    source = {"clip_id": "unseen_penalty", "clip_sha256": sha(video), "clip_filename": video.name}
    manifest_hash = write("source/source_manifest.json", {"manual_annotations_used": False, "clips": [source]})
    pose_hash = write("pose/unseen_penalty.json", {"clip_id": source["clip_id"], "human_verified": False,
        "new_neural_inference": True, "reference_labels_used_for_prediction": False,
        "inference_identity": {"source_sha256": sha(video), "source_manifest_sha256": manifest_hash}})
    evidence = keeper()
    evidence["dive_direction"] = "image_left"
    write("ball/results/unseen_penalty.json", {"clip_id": source["clip_id"], "human_verified": False,
        "reference_labels_used_for_prediction": False, "keeper": evidence,
        "inference_identity": {"source_sha256": sha(video), "source_manifest_sha256": manifest_hash,
                               "pose_sha256": pose_hash}})
    model = fit_models(events())
    feature_code = Path(__file__).resolve().parents[1] / "src/keeper_supervised_model.py"
    model["evidence_contract"] = {"feature_code_sha256": sha(feature_code)}
    write("deployment.json", model)
    return tmp_path, tmp_path / "deployment.json"


def test_video_cli_predicts_from_bound_automatic_evidence_without_labels(tmp_path):
    directory, model = analysis_fixture(tmp_path)
    result = predict_analysis(directory, model)
    assert result["raw_learned_prediction"]["direction"] == "right"
    assert result["prediction"]["direction"] == "left"
    assert result["prediction"]["direction_method"] == "existing_supported_rule"
    assert result["manual_annotations_used"] is False
    assert result["prediction"]["physical_height_metres"] is None


@pytest.mark.parametrize("corruption", ("video", "pose", "labels", "feature_code"))
def test_video_cli_rejects_unbound_evidence(tmp_path, corruption):
    directory, model = analysis_fixture(tmp_path)
    if corruption == "video":
        (directory / "source/unseen_penalty.mp4").write_bytes(b"changed")
    elif corruption == "pose":
        path = directory / "pose/unseen_penalty.json"
        data = json.loads(path.read_text())
        data["new_neural_inference"] = False
        path.write_text(json.dumps(data))
    elif corruption == "labels":
        path = directory / "ball/results/unseen_penalty.json"
        data = json.loads(path.read_text())
        data["reference_labels_used_for_prediction"] = True
        path.write_text(json.dumps(data))
    else:
        data = json.loads(model.read_text())
        data["evidence_contract"]["feature_code_sha256"] = "0" * 64
        model.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        predict_analysis(directory, model)
