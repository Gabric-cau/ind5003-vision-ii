import csv
import pytest

from src.final_results import sha256
from src.fresh_eda import load_bundle
from src.fresh_results import feature_row, summarize, write_json


def bundle(directory, *, mutate=None):
    rows = []
    for index in range(39):
        source = {"clip_id": f"event{index}", "event_id": index + 1, "match_id": "match",
                  "kicker": "kicker", "team": "team", "clip_sha256": "source", "fps": 30.}
        rows.append(feature_row(source, {"pose": {"step_count": 0}},
                                {"reference_outcome": "no_goal", "reference_supported": True}))
    if mutate:
        mutate(rows)
    write_json(directory / "features_39.json", {"events": [{"clip_id": row["clip_id"]} for row in rows]})
    write_json(directory / "summary.json", summarize(rows))
    write_json(directory / "provenance.json", {})
    with (directory / "features_39.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(directory / "manifest.json", {"schema_version": "fresh-course-bundle-v1", "events": 39,
        "artifacts": {name: sha256(directory / name) for name in
                      ("features_39.json", "features_39.csv", "summary.json", "provenance.json")}})


def test_load_preserves_zero_baseline_and_missing_independent_count(tmp_path):
    bundle(tmp_path)
    rows, _, summary = load_bundle(tmp_path)
    assert rows[0]["steps_pose_baseline"] == 0
    assert rows[0]["steps_phase_candidate"] is None
    assert summary["ball_evaluation"]["decided"] == 0


def test_stale_table_is_rejected_before_graphing(tmp_path):
    bundle(tmp_path)
    with (tmp_path / "features_39.csv").open("a") as stream:
        stream.write("stale\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(tmp_path)


def test_unsupported_verification_flag_is_rejected_even_with_fresh_hashes(tmp_path):
    bundle(tmp_path, mutate=lambda rows: rows[0].update(human_verified=True))
    with pytest.raises(ValueError, match="human verification"):
        load_bundle(tmp_path)


def test_missing_source_clock_is_rejected(tmp_path):
    bundle(tmp_path, mutate=lambda rows: rows[0].update(source_fps=None))
    with pytest.raises(ValueError, match="frame rate"):
        load_bundle(tmp_path)
