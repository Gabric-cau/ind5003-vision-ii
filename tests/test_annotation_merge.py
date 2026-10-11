from pathlib import Path
import csv
from scripts.merge_manual_step_reviews import merge


def test_unverified_candidate_cannot_be_exported_as_confirmed(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("match:\n  penalties:\n    - {number: 1, kicker: player}\n", encoding="utf8")
    labels = tmp_path / "labels.csv"
    labels.write_text("clip_id,step_count,landing_frames,confidence,notes\nmatch_p01_player,2,10;20,high,test\n", encoding="utf8")
    result = merge(manifest, tmp_path, [labels], tmp_path / "absent.csv", tmp_path / "out.csv")
    assert result[0]["candidate_step_count"] == 2
    assert result[0]["confirmed_step_count"] is None
    assert result[0]["auto_count_matches_manual"] == ""
    assert result[0]["ground_truth_eligible"] is False
    with (tmp_path / "out.csv").open(newline="") as f:
        assert list(csv.DictReader(f))[0]["confirmed_step_count"] == ""
