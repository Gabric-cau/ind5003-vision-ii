import json
import shutil
from pathlib import Path

import pytest

from scripts.verify_final_bundle import verify


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "outputs" / "final_analysis"


def copy_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    return bundle


def change_json(path, mutation):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutation(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_saved_bundle_consistency_does_not_claim_visual_accuracy():
    result = verify(ROOT, BUNDLE, require_eda=True)
    assert result["events"] == 39
    assert result["input_artifacts_checked"] == 49
    assert result["csv_json_consistent"] is True
    assert result["evaluation_recomputed"] is True
    assert result["eda_recomputed"] is True
    assert result["visual_accuracy_verified"] is False
    assert result["independent_holdout"] is False
    assert result["raw_source_verification"] == "not_performed"


def test_changed_source_hash_is_rejected(tmp_path):
    bundle = copy_bundle(tmp_path)
    change_json(bundle / "provenance.json", lambda data: data["inputs"][0].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="Input artifact missing or changed"):
        verify(ROOT, bundle, require_eda=True)


def test_json_cannot_diverge_from_csv_values(tmp_path):
    bundle = copy_bundle(tmp_path)
    change_json(bundle / "features_39.json", lambda data: data["rows"][0].update(step_count_candidate=99))
    with pytest.raises(ValueError, match="CSV/JSON value mismatch: step_count_candidate"):
        verify(ROOT, bundle, require_eda=True)


def test_eda_from_an_older_table_version_is_rejected(tmp_path):
    bundle = copy_bundle(tmp_path)
    change_json(bundle / "eda" / "summary.json", lambda data: data.update(input_table_sha256="0" * 64))
    with pytest.raises(ValueError, match="EDA summary belongs to a different feature-table version"):
        verify(ROOT, bundle, require_eda=True)


def test_copied_camera_table_cannot_differ_from_source(tmp_path):
    bundle = copy_bundle(tmp_path)
    path = bundle / "camera_intervals_candidates.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("back_of_kicker", "unknown"), encoding="utf-8")
    with pytest.raises(ValueError, match="camera intervals differ"):
        verify(ROOT, bundle, require_eda=True)
