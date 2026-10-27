from scripts.integrate_human_reference_2026_10_22 import categorical_summary, single_landing_summary


def test_unavailable_prediction_is_excluded_not_counted_as_zero():
    result = single_landing_summary({"individual_review_diagnostics": [
        {"comparison": {"true_positive": 1, "false_positive": 1, "false_negative": 2},
         "reviewed_visible_count": 3, "visible_candidate_count": 2, "count_absolute_error": 1},
        {"review_status": "observable"},
    ]})
    assert result["events_with_comparable_predictions"] == 1
    assert result["reviewed_events_without_comparable_predictions"] == 1
    assert result["prediction_coverage"] == .5
    assert result["visible_list_count_mae"] == 1
    assert result["recall_against_supplied_list"] == 1 / 3
    assert result["full_runup_count_mae"] is None


def test_unknown_human_label_is_reviewed_but_not_a_categorical_target():
    result = categorical_summary([
        {"clip_id": "a", "prediction": "left", "reference": "left"},
        {"clip_id": "b", "prediction": "right", "reference": "left"},
        {"clip_id": "c", "prediction": "left", "reference": "unknown"},
        {"clip_id": "d", "prediction": "unknown", "reference": "right"},
    ], "prediction", "reference")
    assert result["events"] == 4
    assert result["compared"] == 2
    assert result["coverage"] == .5
    assert result["agreement_on_decided"] == .5
    assert result["disagreements"] == ["b"]
