"""Synthetic bundles exercise sample identity and evidence-denominator failures."""

import csv
import hashlib
import json

import pytest

from src.eda_analysis import REQUIRED_COLUMNS, load_analysis, run_eda, summarize_analysis


def event_rows(count=39):
    rows = []
    for index in range(count):
        rows.append({
            "clip_id": f"match_p{index + 1:02d}_synthetic",
            "match_id": f"match_{index % 4}", "event_id": index + 1,
            "kicking_foot_candidate": "unknown", "step_count_candidate": "",
            "step_count_verified": "", "partial_runup": "False",
            "runup_ground_plane_validated": "False",
            "goalkeeper_direction_candidate": "unknown", "goalkeeper_height_candidate": "unknown",
            "goalkeeper_height_proxy_candidate": "unknown", "ball_rule_outcome": "unknown",
            "reference_outcome": "unknown", "ball_final_position_verified": "False",
            "source_sha256": "a" * 64, "review_required": "True",
            "ball_unknown_reason": "insufficient_ball_evidence", "camera_interval_count": "",
        })
    return rows


def save_bundle(directory, rows):
    path = directory / "features_39.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = [row["clip_id"] for row in rows]
    evaluation = {"schema_version": 1, "total_events": 39, "held_out_test": False,
                  "accuracy_validated": False, "reference_sources_independently_verified": False}
    evaluation_path = directory / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    (directory / "provenance.json").write_text(json.dumps({
        "schema_version": 1, "artifact_type": "final_analysis_candidate_bundle",
        "canonical_clip_ids": ids, "table": {"path": path.name, "sha256": digest},
        "raw_source_verification": "not_performed", "held_out_test": False,
    }), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({
        "schema_version": "vision2-final-analysis-v1", "clip_ids": ids,
        "features_csv_sha256": digest,
        "evaluation_json_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return path


def test_missing_unknown_and_measured_zero_have_distinct_denominators(tmp_path):
    rows = event_rows()
    rows[0].update(step_count_candidate="0", camera_interval_count="0", reference_outcome="goal", ball_rule_outcome="goal_likely")
    rows[1].update(step_count_candidate="4", reference_outcome="goal", ball_rule_outcome="no_goal_likely", partial_runup="True")
    rows[2].update(reference_outcome="no_goal")
    df, evaluation, _ = load_analysis(save_bundle(tmp_path, rows))
    summary = summarize_analysis(df, evaluation)
    assert summary["steps"]["available"] == 2
    assert summary["steps"]["missing"] == 37
    assert summary["steps"]["distribution"] == {"0": 1, "4": 1}
    assert summary["candidate_coverage"]["camera_interval_count"] == 0
    assert summary["steps"]["known_partial_ids"] == [rows[1]["clip_id"]]
    comparison = summary["ball_reference_comparison"]
    assert comparison["covered"] == 2
    assert comparison["abstentions"] == 37
    assert comparison["covered_reference_agreement"] == 0.5
    assert comparison["coverage"] == 2 / 39
    assert comparison["confusion_matrix"]["goal"] == {"goal": 1, "no_goal": 1}
    assert comparison["conflict_ids"] == [rows[1]["clip_id"]]
    assert summary["validated_coverage"]["step_count_verified"] == 0


@pytest.mark.parametrize("count", [38, 49])
def test_incorrect_event_count_rejected_before_export(tmp_path, count):
    output = tmp_path / "eda"
    with pytest.raises(ValueError, match="39"):
        run_eda(save_bundle(tmp_path, event_rows(count)), output)
    assert not output.exists()


def test_duplicate_or_wrong_clip_identity_rejected(tmp_path):
    rows = event_rows()
    rows[1]["clip_id"] = rows[0]["clip_id"]
    with pytest.raises(ValueError, match="unique"):
        load_analysis(save_bundle(tmp_path, rows))
    path = save_bundle(tmp_path, event_rows())
    provenance_path = tmp_path / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["canonical_clip_ids"][0] = "wrong_event"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="IDs"):
        load_analysis(path)


def test_modified_table_and_modified_evaluation_rejected(tmp_path):
    path = save_bundle(tmp_path, event_rows())
    with path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_analysis(path)
    path = save_bundle(tmp_path, event_rows())
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(evaluation_path.read_text() + "\n")
    with pytest.raises(ValueError, match="Evaluation SHA-256"):
        load_analysis(path)


@pytest.mark.parametrize("column,value", [
    ("step_count_candidate", "unknown"), ("step_count_candidate", "-1"),
    ("step_count_candidate", "2.5"), ("step_count_candidate", "inf"),
    ("partial_runup", "perhaps"), ("source_sha256", "not_a_hash"),
])
def test_malformed_measurement_or_provenance_rejected(tmp_path, column, value):
    rows = event_rows()
    rows[0][column] = value
    with pytest.raises(ValueError):
        load_analysis(save_bundle(tmp_path, rows))


def test_all_abstentions_are_insufficient_not_perfect_miss_accuracy(tmp_path):
    rows = event_rows()
    for row in rows:
        row["reference_outcome"] = "no_goal"
    df, evaluation, _ = load_analysis(save_bundle(tmp_path, rows))
    result = summarize_analysis(df, evaluation)["ball_reference_comparison"]
    assert result["covered"] == 0
    assert result["abstentions"] == 39
    assert result["covered_reference_agreement"] is None
    assert result["status"] == "insufficient_comparable_events"
    assert result["held_out_accuracy"] is None
