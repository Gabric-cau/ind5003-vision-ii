from pathlib import Path

import pytest
import yaml

from scripts.prepare_course_clips import plan_context_clips


ROOT = Path(__file__).resolve().parents[1]


def test_context_preserves_original_slice_mapping_without_claiming_event_boundaries():
    manifest = yaml.safe_load((ROOT / "configs/penalty_slicing_manifest.yaml").read_text(encoding="utf-8"))
    metadata = {row["video_filename"]: {"fps": 30.0, "frame_count": 50000, "sha256": "a" * 64}
                for row in manifest.values()}
    records = plan_context_clips(manifest, metadata, 6.0, 8.0)
    assert len(records) == len({r["clip_id"] for r in records}) == 39
    first = records[0]
    assert first["source_start_frame"] == 292 * 30
    assert first["original_slice_start_frame_in_clip"] == 180
    assert first["source_end_frame_exclusive"] == 321 * 30
    assert all(not row["semantic_event_end_verified"] for row in records)
    assert all(not row["semantic_event_start_verified"] for row in records)


def test_invalid_context_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        plan_context_clips({}, {}, -1, 0)
