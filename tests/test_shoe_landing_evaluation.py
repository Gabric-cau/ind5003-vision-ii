import copy
import json

import pytest

from scripts.evaluate_shoe_landings import (
    evaluate,
    freeze_predictions,
    match_landings,
    reference_template,
    validate_prediction,
    validate_references,
    sha256,
)


def prediction(clip_id="event", source="a" * 64, **updates):
    value = {"clip_id": clip_id, "source_sha256": source, "fps": 25,
             "observable_window": {"start_frame": 0, "end_frame": 100},
             "full_approach_visible": False, "count_complete": False,
             "candidate_count": 2, "landings": [{"frame": 20, "foot": "left"},
                                                {"frame": 40, "foot": "right"}],
             "striking_foot": "right"}
    value.update(updates)
    return value


def frozen(tmp_path, values=None):
    paths = []
    for index, value in enumerate(values or [prediction()]):
        path = tmp_path / f"prediction_{index}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    freeze = tmp_path / "freeze.json"
    freeze_predictions(paths, freeze)
    return freeze, paths


def reviewed(reviewer_id, frames=(20, 40), **updates):
    value = {"reviewer_id": reviewer_id, "reviewed_at": "2026-10-13T12:00:00+08:00",
             "annotator_type": "human", "human_verified": True,
             "review_status": "observable", "striking_foot": "right",
             "review_window": {"start_frame": 0, "end_frame": 100},
             "full_approach_visible": False,
             "verified_landings": [{"frame": frame, "foot": foot}
                                   for frame, foot in zip(frames, ("left", "right", "left"))]}
    value.update(updates)
    return value


def evaluation(tmp_path, predictions=None, reviews=None):
    freeze, paths = frozen(tmp_path, predictions)
    template = reference_template(freeze)
    if reviews is not None:
        template["events"][0]["reviews"] = reviews
    references = tmp_path / "references.json"
    references.write_text(json.dumps(template), encoding="utf-8")
    return evaluate(freeze, references), freeze, paths, references


def test_template_is_pending_and_does_not_copy_prediction_labels(tmp_path):
    result, freeze, _, references = evaluation(tmp_path)
    template = json.loads(references.read_text())
    review = template["events"][0]["reviews"][0]
    assert review["reviewer_id"] == ""
    assert review["striking_foot"] == "unknown"
    assert review["verified_landings"] is None
    assert "candidate_count" not in template["events"][0]
    assert template["prediction_freeze_sha256"] == sha256(freeze)
    assert result["reviewed_events"] == 0
    assert result["held_out_test"] is False
    metrics = result["jointly_reviewed_landing_metrics"]
    assert metrics["evaluated_events"] == 0
    assert metrics["visible_window_count_mae"] is None
    assert metrics["event_precision"] is None
    assert result["jointly_reviewed_striking_foot_metrics"]["agreement_on_known_predictions"] is None


def test_prediction_hashes_are_enforced_before_reference_comparison(tmp_path):
    result, freeze, paths, references = evaluation(tmp_path)
    value = json.loads(paths[0].read_text())
    value["striking_foot"] = "left"
    paths[0].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed after freezing"):
        evaluate(freeze, references)


def test_missing_prediction_is_not_observed_zero(tmp_path):
    values = [prediction(candidate_count=None, landings=None),
              prediction("event_zero", "b" * 64, candidate_count=0, landings=[])]
    result, _, _, _ = evaluation(tmp_path, values,
                                [reviewed("person_1"), reviewed("person_2")])
    assert result["proposal_coverage"]["unknown_count_events"] == 1
    assert result["proposal_coverage"]["known_count_events"] == 1
    assert result["jointly_reviewed_landing_metrics"]["evaluated_events"] == 0
    assert result["jointly_reviewed_landing_metrics"]["visible_window_count_mae"] is None
    assert result["jointly_reviewed_landing_metrics"]["unknown_prediction_reference_events"] == 1
    assert result["jointly_reviewed_landing_metrics"]["prediction_coverage_on_jointly_reviewed_events"] == 0


def test_unavailable_window_still_has_a_pending_reference_row(tmp_path):
    result, _, _, references = evaluation(tmp_path, [prediction(candidate_count=None, landings=None,
                                                              observable_window=None, frame_count=101)])
    review = json.loads(references.read_text())["events"][0]["reviews"][0]
    assert review["review_window"] is None
    assert review["verified_landings"] is None
    assert result["events"] == 1
    assert result["reviewed_events"] == 0


def test_human_can_review_a_real_window_when_model_abstained(tmp_path):
    result, _, _, _ = evaluation(tmp_path, [prediction(candidate_count=None, landings=None,
                                                      observable_window=None, frame_count=101)],
                                [reviewed("person_1"), reviewed("person_2")])
    assert result["jointly_reviewed_landing_metrics"]["unknown_prediction_reference_events"] == 1
    assert result["jointly_reviewed_landing_metrics"]["visible_window_count_mae"] is None
    assert result["jointly_reviewed_striking_foot_metrics"]["consensus_events"] == 1


def test_human_window_cannot_exceed_the_input_clip_length(tmp_path):
    with pytest.raises(ValueError, match="exceeds clip frame_count"):
        evaluation(tmp_path, [prediction(candidate_count=None, landings=None,
                                        observable_window=None, frame_count=50)],
                   [reviewed("person_1")])


def test_unknown_pair_has_to_be_consistent():
    with pytest.raises(ValueError, match="both be null"):
        validate_prediction(prediction(candidate_count=0, landings=None))
    with pytest.raises(ValueError, match="must equal"):
        validate_prediction(prediction(candidate_count=1))


def test_single_review_is_diagnostic_not_consensus(tmp_path):
    result, _, _, _ = evaluation(tmp_path, reviews=[reviewed("person_1")])
    assert result["individual_review_diagnostics"][0]["count_absolute_error"] == 0
    assert result["jointly_reviewed_landing_metrics"]["evaluated_events"] == 0
    assert result["jointly_reviewed_landing_metrics"]["visible_window_count_mae"] is None
    assert result["jointly_reviewed_striking_foot_metrics"]["consensus_events"] == 0


def test_two_agreeing_reviews_enable_visible_metrics_only(tmp_path):
    result, _, _, _ = evaluation(tmp_path, reviews=[reviewed("person_1"), reviewed("person_2", (21, 39))])
    metrics = result["jointly_reviewed_landing_metrics"]
    assert metrics["evaluated_events"] == 1
    assert metrics["event_precision"] == 1
    assert metrics["event_recall"] == 1
    assert metrics["visible_window_count_mae"] == 0
    assert metrics["full_runup_count_mae"] is None
    assert result["jointly_reviewed_striking_foot_metrics"]["agreement_on_known_predictions"] == 1


@pytest.mark.parametrize("other", [
    reviewed("person_2", (20,)),
    reviewed("person_2", (20, 60)),
    reviewed("person_2", verified_landings=[{"frame": 20, "foot": "right"}, {"frame": 40, "foot": "right"}]),
    reviewed("person_2", review_status="ambiguous", verified_landings=None),
    reviewed("person_2", review_status="unobservable", verified_landings=None),
    reviewed("person_2", review_window={"start_frame": 10, "end_frame": 100}),
])
def test_review_disagreement_or_unknown_never_becomes_landing_consensus(tmp_path, other):
    result, _, _, _ = evaluation(tmp_path, reviews=[reviewed("person_1"), other])
    assert result["jointly_reviewed_landing_metrics"]["evaluated_events"] == 0
    assert result["jointly_reviewed_landing_metrics"]["visible_window_count_mae"] is None


def test_striking_foot_disagreement_is_separate_from_landing_agreement(tmp_path):
    result, _, _, _ = evaluation(tmp_path, reviews=[reviewed("person_1"), reviewed("person_2", striking_foot="left")])
    assert result["jointly_reviewed_landing_metrics"]["evaluated_events"] == 1
    assert result["jointly_reviewed_striking_foot_metrics"]["reviewer_disagreement_events"] == 1
    assert result["jointly_reviewed_striking_foot_metrics"]["consensus_events"] == 0


def test_three_reviewers_must_agree_pairwise_within_tolerance(tmp_path):
    rows = [reviewed("middle", (20, 40)), reviewed("earlier", (18, 38)), reviewed("later", (22, 42))]
    result, _, _, _ = evaluation(tmp_path, reviews=rows)
    assert result["jointly_reviewed_landing_metrics"]["evaluated_events"] == 0
    assert result["per_event"][0]["landing_reference_status"] == "landing_sequence_disagreement"


def test_unknown_striking_prediction_changes_coverage_not_conditional_agreement(tmp_path):
    result, _, _, _ = evaluation(tmp_path, [prediction(striking_foot="unknown")],
                                [reviewed("person_1"), reviewed("person_2")])
    metrics = result["jointly_reviewed_striking_foot_metrics"]
    assert metrics["consensus_events"] == 1
    assert metrics["coverage_on_consensus_events"] == 0
    assert metrics["agreement_on_known_predictions"] is None


def test_pending_review_cannot_claim_human_identity_or_automatic_labels(tmp_path):
    freeze, _ = frozen(tmp_path)
    reference = reference_template(freeze)
    reference["events"][0]["reviews"][0]["reviewer_id"] = "AI guessed reviewer"
    with pytest.raises(ValueError, match="fabricated"):
        validate_references(reference, {"event": validate_prediction(prediction())}, sha256(freeze))


def test_two_reviewer_ids_cannot_be_the_same_person_entry(tmp_path):
    with pytest.raises(ValueError, match="duplicate reviewer"):
        evaluation(tmp_path, reviews=[reviewed("person_1"), reviewed("person_1")])


@pytest.mark.parametrize("key,value", [
    ("source_sha256", "b" * 64), ("fps", 50), ("frame_clock", "source_seconds")
])
def test_wrong_source_or_frame_clock_is_rejected(tmp_path, key, value):
    freeze, _ = frozen(tmp_path)
    reference = reference_template(freeze)
    reference["events"][0][key] = value
    with pytest.raises(ValueError, match="differs"):
        validate_references(reference, {"event": validate_prediction(prediction())}, sha256(freeze))


def test_duplicate_source_window_is_not_an_independent_sample(tmp_path):
    with pytest.raises(ValueError, match="duplicate source"):
        frozen(tmp_path, [prediction(), prediction("copied_event")])


def test_duplicate_reference_event_cannot_double_count_review(tmp_path):
    freeze, _ = frozen(tmp_path)
    reference = reference_template(freeze)
    reference["events"].append(copy.deepcopy(reference["events"][0]))
    with pytest.raises(ValueError, match="duplicate reference"):
        validate_references(reference, {"event": validate_prediction(prediction())}, sha256(freeze))


def test_ordered_matching_maximizes_cardinality_before_minimizing_error():
    # Greedily matching the exact frame 2 first can consume the wrong reference.
    predicted = [{"frame": 1, "foot": "left"}, {"frame": 2, "foot": "right"}]
    reference = [{"frame": 2, "foot": "left"}, {"frame": 3, "foot": "right"}]
    result = match_landings(predicted, reference, 1)
    assert result["true_positive"] == 2
    assert result["false_positive"] == result["false_negative"] == 0
    assert sum(row["absolute_frame_error"] for row in result["pairs"]) == 2


def test_repeated_proposals_cannot_reuse_one_reference():
    predicted = [{"frame": 10, "foot": "left"}, {"frame": 11, "foot": "left"}, {"frame": 12, "foot": "left"}]
    result = match_landings(predicted, [{"frame": 11, "foot": "unknown"}], 2)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["pairs"][0]["predicted"]["frame"] == 11


def test_zero_observation_is_valid_but_does_not_create_perfect_detection_score(tmp_path):
    result, _, _, _ = evaluation(tmp_path, [prediction(candidate_count=0, landings=[])],
                                [reviewed("person_1", ()), reviewed("person_2", ())])
    metrics = result["jointly_reviewed_landing_metrics"]
    assert metrics["visible_window_count_mae"] == 0
    assert metrics["event_precision"] is None
    assert metrics["event_recall"] is None


def test_partial_review_window_only_scores_proposals_inside_it(tmp_path):
    window = {"start_frame": 30, "end_frame": 100}
    rows = [reviewed(name, review_window=window, verified_landings=[{"frame": 40, "foot": "right"}])
            for name in ("person_1", "person_2")]
    result, _, _, _ = evaluation(tmp_path, reviews=rows)
    assert result["per_event"][0]["predicted_visible_count"] == 1
    assert result["jointly_reviewed_landing_metrics"]["visible_window_count_mae"] == 0
    assert result["jointly_reviewed_landing_metrics"]["full_runup_count_mae"] is None


def test_full_runup_metric_requires_prediction_completeness(tmp_path):
    result, _, _, _ = evaluation(tmp_path, [prediction(full_approach_visible=True, count_complete=False)],
                                [reviewed("person_1", full_approach_visible=True), reviewed("person_2", full_approach_visible=True)])
    assert result["jointly_reviewed_landing_metrics"]["full_runup_evaluated_events"] == 0
