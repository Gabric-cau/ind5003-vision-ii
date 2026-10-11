import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.final_results import build_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "final_analysis.yaml"


def configured_build(tmp_path, *, replacement_key=None, replacement_data=None, ball_changes=None,
                     raw_clips_dir=None):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if replacement_key is not None:
        replacement = tmp_path / f"{replacement_key}.json"
        replacement.write_text(json.dumps(replacement_data), encoding="utf-8")
        config[replacement_key] = str(replacement)
    if ball_changes is not None:
        source = ROOT / config["ball_results_dir"]
        changed = tmp_path / "ball_results"
        shutil.copytree(source, changed)
        path = changed / "col_eng_p01_falcao_col9.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(ball_changes)
        path.write_text(json.dumps(data), encoding="utf-8")
        config["ball_results_dir"] = str(changed)
    config_path = tmp_path / "final_analysis.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "bundle"
    result = build_bundle(ROOT, config_path, output, raw_clips_dir=raw_clips_dir)
    rows = json.loads((output / "features_39.json").read_text(encoding="utf-8"))["rows"]
    return result, rows, output


def input_json(key):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return json.loads((ROOT / config[key]).read_text(encoding="utf-8"))


@pytest.mark.parametrize("problem", ["source_hash", "duplicate_segment", "invalid_time"])
def test_camera_intervals_cannot_mix_sources_or_invalid_windows(tmp_path, problem):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    with (ROOT / config["camera_intervals"]).open(encoding="utf-8-sig", newline="") as stream:
        camera_rows = list(csv.DictReader(stream))
    if problem == "source_hash":
        camera_rows[0]["source_sha256"] = "0" * 64
    elif problem == "duplicate_segment":
        camera_rows.append(dict(camera_rows[0]))
    else:
        camera_rows[0]["end_time_sec"] = camera_rows[0]["start_time_sec"]
    camera_path = tmp_path / "camera.csv"
    with camera_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(camera_rows[0]))
        writer.writeheader()
        writer.writerows(camera_rows)
    config["camera_intervals"] = str(camera_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="camera|Camera"):
        build_bundle(ROOT, config_path, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_missing_optional_camera_input_remains_missing_not_zero(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["camera_intervals"] = str(tmp_path / "absent.csv")
    config["camera_intervals_json"] = str(tmp_path / "absent.json")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    evaluation = build_bundle(ROOT, config_path, tmp_path / "bundle")
    rows = json.loads((tmp_path / "bundle/features_39.json").read_text(encoding="utf-8"))["rows"]
    assert evaluation["camera_interval_diagnostics"]["available"] is False
    assert all(row["camera_interval_count"] is None for row in rows)
    assert len(evaluation["missing_optional_inputs"]) == 2


def test_saved_39_events_keep_predictions_references_and_unknowns_separate(tmp_path):
    evaluation, rows, output = configured_build(tmp_path)
    assert len(rows) == len({row["clip_id"] for row in rows}) == 39
    assert [row["event_id"] for row in rows] == list(range(1, 40))
    assert Counter(row["match_id"] for row in rows) == {
        "col_eng": 10, "cro_den": 10, "esp_rus": 9, "rus_cro": 10,
    }
    assert Counter(row["ball_rule_outcome"] for row in rows) == {
        "goal_likely": 20, "no_goal_likely": 6, "unknown": 13,
    }
    comparison = evaluation["ball_rule_reference_comparison"]
    assert comparison["covered"] == 26
    assert comparison["matching"] == 23
    assert {item["event_id"] for item in comparison["conflicts"]} == {22, 25, 30}
    assert comparison["interpretation"] == "development_reference_agreement_not_test_accuracy"
    camera = evaluation["camera_interval_diagnostics"]
    assert camera["total_intervals"] == 197
    assert camera["clips_with_intervals"] == 39
    assert camera["task_class_candidate_counts"] == {"unknown": 187, "back_of_kicker": 3, "angle_45_right": 7}
    assert camera["playback_status_counts"] == {"unknown": 197}
    assert camera["human_verified_intervals"] == 0
    assert sum(row["camera_interval_count"] for row in rows) == 197
    assert evaluation["accuracy_validated"] is False
    assert evaluation["held_out_test"] is False
    eriksen = next(row for row in rows if row["clip_id"] == "cro_den_p01_eriksen_den10")
    assert eriksen["kicking_foot_candidate"] == "left"
    assert eriksen["kicking_foot_reference"] == "right"
    for clip_id in ("cro_den_p04_kramaric_cro9", "esp_rus_p08_cheryshev_rus6", "esp_rus_p09_aspas_esp17"):
        assert next(row for row in rows if row["clip_id"] == clip_id)["goalkeeper_direction_candidate"] == "unknown"
    assert all(row["goalkeeper_height_candidate"] == "unknown" for row in rows)
    assert all(row["step_count_verified"] is None for row in rows)
    assert all(row["ball_final_position_verified"] is False for row in rows)
    assert all(row["ball_event_final_x_px"] is None and row["ball_event_final_y_px"] is None for row in rows)
    assert all(row["goalkeeper_pre_movement_status"] == "legacy_candidate_requires_recompute_after_stabilization_fix"
               for row in rows)
    assert all(row["source_video_hash_verified"] is False for row in rows)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["raw_source_verification"] == "not_performed"
    assert provenance["table"]["sha256"] == hashlib.sha256((output / "features_39.csv").read_bytes()).hexdigest()
    with (output / "human_review_template.csv").open(newline="", encoding="utf-8") as handle:
        references = list(csv.DictReader(handle))
    assert len(references) == 39
    assert all(row["review_status"] == "pending" and row["step_count_reference"] == "" for row in references)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "noncanonical"])
def test_invalid_canonical_identity_fails_before_export(tmp_path, problem):
    data = input_json("canonical_results")
    if problem == "missing":
        data["rows"].pop()
    elif problem == "duplicate":
        data["rows"].append(dict(data["rows"][0]))
    else:
        data["rows"][0]["clip_id"] = "unknown_match_p01_fake"
    with pytest.raises(ValueError, match="canonical|duplicate"):
        configured_build(tmp_path, replacement_key="canonical_results", replacement_data=data)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("unsafe", [True, None])
def test_ball_label_use_must_be_explicitly_false(tmp_path, unsafe):
    with pytest.raises(ValueError, match="reference-use"):
        configured_build(tmp_path, ball_changes={"outcome_labels_used_for_algorithm": unsafe})
    assert not (tmp_path / "bundle").exists()


def test_foot_predictions_cannot_use_reviewer_labels(tmp_path):
    data = input_json("foot_development_evaluation")
    data["records"][0]["shortcut"]["reviewer_labels_used_for_prediction"] = True
    with pytest.raises(ValueError, match="reference|reviewer|label"):
        configured_build(tmp_path, replacement_key="foot_development_evaluation", replacement_data=data)


def test_unverified_step_claim_cannot_be_counted_as_validated(tmp_path):
    data = input_json("canonical_results")
    data["rows"][0]["steps"]["verified_count"] = 99
    data["rows"][0]["human_verified"] = False
    evaluation, rows, _ = configured_build(tmp_path, replacement_key="canonical_results", replacement_data=data)
    assert rows[0]["step_count_verified"] is None
    assert evaluation["validated_coverage"]["step_count_verified"] == 0


def test_user_direction_override_cannot_fill_missing_automatic_direction(tmp_path):
    data = input_json("goalkeeper_height_summary")
    row = data["rows"][0]
    row.pop("automatic_dive_direction", None)
    row.update(dive_direction="image_right", final_dive_direction="image_right", direction_user_reviewed=True)
    _, rows, _ = configured_build(tmp_path, replacement_key="goalkeeper_height_summary", replacement_data=data)
    assert rows[0]["goalkeeper_direction_candidate"] == "unknown"


def test_changed_foot_references_cannot_reuse_stale_development_comparison(tmp_path):
    data = input_json("foot_references")
    data["reviews"][0]["kicking_foot"] = "left"
    with pytest.raises(ValueError, match="reference|hash|SHA"):
        configured_build(tmp_path, replacement_key="foot_references", replacement_data=data)


def test_empty_trajectory_does_not_create_a_zero_position(tmp_path):
    _, rows, output = configured_build(tmp_path, ball_changes={"trajectory": []})
    assert rows[0]["ball_trajectory_point_count"] == 0
    assert rows[0]["ball_last_observed_x_px"] is None
    assert rows[0]["ball_last_observed_y_px"] is None
    assert rows[0]["ball_last_observed_frame"] is None
    assert rows[0]["ball_event_final_x_px"] is None
    with (output / "features_39.csv").open(newline="", encoding="utf-8") as handle:
        exported = next(csv.DictReader(handle))
    assert exported["ball_last_observed_x_px"] == ""


def test_ball_event_identity_must_match_manifest_order(tmp_path):
    with pytest.raises(ValueError, match="identity mismatch"):
        configured_build(tmp_path, ball_changes={"event_id": 39})


def test_mismatched_source_video_hash_fails_before_export(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "col_eng_p01_falcao_col9.mp4").write_bytes(b"deliberately mismatched test fixture")
    with pytest.raises(ValueError, match="Source video does not match"):
        configured_build(tmp_path, raw_clips_dir=raw)
    assert not (tmp_path / "bundle").exists()


def test_builder_cli_bootstraps_package_from_another_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_final_results.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--raw-clips-dir" in result.stdout


@pytest.mark.parametrize("field", ["partial_runup", "ground_plane", "identity", "crossing"])
def test_string_false_cannot_be_promoted_to_verified_evidence(tmp_path, field):
    if field == "crossing":
        ball = json.loads((ROOT / "outputs/ball_goal_all_events_v2_2026-10-11/results/col_eng_p01_falcao_col9.json").read_text())
        ball["rule_decision"]["whole_ball_crossing_established"] = "false"
        arguments = {"ball_changes": {"rule_decision": ball["rule_decision"]}}
    else:
        key = "goalkeeper_height_summary" if field == "identity" else "canonical_results"
        data = input_json(key)
        if field == "partial_runup":
            data["rows"][0]["steps"]["partial_approach_known"] = "false"
        elif field == "ground_plane":
            data["rows"][0]["homography"]["validated_ground_plane"] = "false"
        else:
            data["rows"][0]["identity_verified"] = "false"
        arguments = {"replacement_key": key, "replacement_data": data}
    with pytest.raises(ValueError, match="JSON boolean"):
        configured_build(tmp_path, **arguments)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_ball_measurement_fails_before_json_export(tmp_path, coordinate):
    with pytest.raises(ValueError, match="Non-finite numeric"):
        configured_build(tmp_path, ball_changes={"trajectory": [{"frame": 71, "x": coordinate, "y": 42.}]})
    assert not (tmp_path / "bundle").exists()


def test_external_same_named_inputs_retain_both_provenance_hashes(tmp_path):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    expected_hashes = set()
    for folder, key in (("canonical", "canonical_results"), ("keeper", "goalkeeper_height_summary")):
        destination = tmp_path / folder / "summary.json"
        destination.parent.mkdir()
        shutil.copy2(ROOT / config[key], destination)
        expected_hashes.add(hashlib.sha256(destination.read_bytes()).hexdigest())
        config[key] = str(destination)
    configuration = tmp_path / "final_analysis.yaml"
    configuration.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "bundle"
    build_bundle(ROOT, configuration, output)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    sources = [item for item in provenance["inputs"] if item["path"].startswith("external/")
               and item["path"].endswith("/summary.json")]
    assert len(sources) == len({item["path"] for item in sources}) == 2
    assert {item["sha256"] for item in sources} == expected_hashes
