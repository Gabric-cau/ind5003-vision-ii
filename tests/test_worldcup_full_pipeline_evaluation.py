import json

import pytest

from scripts import evaluate_worldcup_full_pipeline as evaluation


def analysis(foot="unknown", goal="unknown", height="unknown"):
    return {
        "required_outputs": {
            "kicking_foot": foot,
            "goal_outcome": goal,
            "keeper_dive_height": {"value": height, "definition": "human_qualitative_image_plane"},
        },
        "auxiliary_models": {"trained_keeper": {"prediction": {"height_model_candidate": height}}},
        "source": {"fps": 50, "frame_count": 100},
        "end_to_end_seconds": 2.0,
        "manual_annotations_used": False,
        "all_stages_executed_successfully": True,
    }


def test_unknowns_are_strict_misses_and_height_uses_ordinal_field():
    predictions = {"wc_17": analysis("right", "goal_likely", "middle"),
                   "wc_18": analysis("unknown", "unknown", "unknown")}
    references = {"wc_17": {"clip_id": "a", "video_sha256": "a" * 64, "foot": "right",
                             "goal_reference": "goal", "keeper_ordinal_height": "middle"},
                  "wc_18": {"clip_id": "b", "video_sha256": "b" * 64, "foot": "left",
                             "goal_reference": "goal", "keeper_ordinal_height": "low"}}
    rows, metrics = evaluation.event_rows(predictions, references)
    assert len(rows) == 2
    assert metrics["kicking_foot"]["correct_events"] == 1
    assert metrics["kicking_foot"]["strict_accuracy_unknown_as_incorrect"] == pytest.approx(.5)
    assert metrics["goal_outcome"]["correct_events"] == 1
    assert metrics["keeper_ordinal_height"]["correct_events"] == 1


def test_reference_schema_and_video_binding_are_checked(tmp_path):
    reference = tmp_path / "references.json"
    reference.write_text(json.dumps({"schema_version": "worldcup8-full-pipeline-references-v1",
        "samples": [{"core_id": "wc_17", "video_sha256": "a" * 64, "foot": "right",
                      "goal_reference": "goal", "keeper_ordinal_height": "low"}]}))
    with pytest.raises(ValueError, match="exactly the completed batch"):
        evaluation.read_references(reference, {"wc_18": {"video_sha256": "a" * 64}})
