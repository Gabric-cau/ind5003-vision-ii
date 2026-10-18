from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import replay_shoe_landings as replay
from src.shoe_landing_events import LandingConfig, propose_shoe_landings


def _poses(only_normal=False):
    records, hashes, transforms = [], {}, {}
    for frame, x in enumerate([80, 80, 92, 104, 116, 116, 116, 116]):
        points = np.full((33, 2), [130., 80.])
        points[23], points[24] = [110 + 2 * frame, 115], [140 + 2 * frame, 115]
        points[29], points[31] = [x, 195], [x + 8, 195]
        points[30], points[32] = [180, 195], [188, 195]
        hashes[str(frame)] = f"source-pixels-{frame}"
        transforms[str(frame)] = np.eye(3).tolist()
        for reflected in ((False,) if only_normal else (False, True)):
            records.append(dict(frame=frame, reflected=reflected, roi_identity_status="accepted",
                landmarks_source_xy=points.tolist(), landmark_confidence=[1.] * 33,
                anchor_bbox=[0, 0, 260, 200], frame_pixel_sha256=hashes[str(frame)]))
    return records, hashes, transforms


@pytest.fixture
def cohort(tmp_path):
    dense, baseline = tmp_path / "dense", tmp_path / "baseline"
    sources = replay._course_sources()
    flags = dict(original_course_videos_only=True, additional_video_used=False,
                 reference_labels_read=False, human_verified=False, held_out_test=False)
    producer = dense / "source_snapshot/src/synthetic_pose_producer.py"
    producer.parent.mkdir(parents=True)
    producer.write_text("synthetic fixture\n", encoding="utf-8")
    dense_protocol = dict(flags, events=39, pilot=False, plan_only=False,
        selected_clip_ids=list(sources), source_manifest_sha256=replay.MANIFEST_SHA256,
        code_sha256={"src/synthetic_pose_producer.py": replay.sha(producer)}, model={"sha256": "synthetic-model"})
    replay.write(dense / "protocol.json", dense_protocol)
    snapshot = baseline / "source_snapshot/src/shoe_landing_events.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("synthetic baseline fixture\n", encoding="utf-8")
    dense_rows, baseline_rows = [], []
    for index, (cid, source) in enumerate(sources.items()):
        old = dict(clip_id=cid, source_mapping=source, source_sha256=source["clip_sha256"],
            fps=source["fps"], frame_count=source["frame_count"], baseline_ankle_count=None,
            baseline_foot="unknown", striking_foot="unknown", full_approach_visible=False,
            count_complete=False, total_run_up_steps=None, human_verified=False,
            accuracy_validated=False, held_out_test=False, reference_labels_used_for_prediction=False)
        row = dict(clip_id=cid, event_id=source["event_id"], source_sha256=source["clip_sha256"],
                   status="unavailable", reason="independent_contact_unavailable")
        if index < 2:
            records, hashes, transforms = _poses(only_normal=index == 1)
            event_dir = dense / "events" / cid
            replay.write(event_dir / "poses33.json", records)
            inputs = dict(source_mapping=source, fps=source["fps"], interval_frames=[0, 7],
                source_manifest_sha256=replay.MANIFEST_SHA256,
                poses33_sha256=replay.sha(event_dir / "poses33.json"), model_sha256="synthetic-model",
                code_sha256=dense_protocol["code_sha256"], reference_labels_read=False,
                new_neural_inference=True, human_verified=False,
                coordinate_space="original_source_image_pixels", interval_endpoints="inclusive",
                original_camera_motion=transforms, decoded_frame_hashes=hashes,
                input_receipts={"contact_export": {"path": "/deleted/old/contact.json", "sha256": "unavailable"}})
            replay.write(event_dir / "inputs.json", inputs)
            row.update(status="fresh_dense_pose_candidate", interval_frames=[0, 7],
                       inputs_sha256=replay.sha(event_dir / "inputs.json"), poses33_sha256=inputs["poses33_sha256"])
            row.pop("reason")
            landing = propose_shoe_landings(records, source["fps"], [0, 7], transforms, hashes)
            old.update(landing, baseline_ankle_count=4, baseline_foot="right", striking_foot="right",
                dense_foot={"striking_foot": "right", "joint_region_frames": [6], "opaque_frozen_evidence": True},
                dense_inputs_sha256=row["inputs_sha256"], dense_poses_sha256=row["poses33_sha256"])
            if old["candidate_count"]:
                for event in old["landings"]:
                    event["stance_frames"] = event["stance_frames"][:3]
        else:
            old.update(status="unavailable", reason=row["reason"], candidate_count=None,
                       observed_proposal_count=0, landings=None, observable_window=None)
        target = baseline / "events" / cid / "landing_result.json"
        replay.write(target, old)
        baseline_rows.append(dict(clip_id=cid, event_id=source["event_id"], status=old["status"],
            baseline_ankle_count=old["baseline_ankle_count"], baseline_foot=old["baseline_foot"],
            dense_foot=old["striking_foot"], visible_transition_proposals=old["candidate_count"],
            result_sha256=replay.sha(target)))
        dense_rows.append(row)
    replay.write(dense / "summary.json", dict(flags, events=39, rows=dense_rows,
                 source_manifest_sha256=replay.MANIFEST_SHA256, actual_inference_calls=24))
    baseline_protocol = dict(flags, dense_protocol_sha256=replay.sha(dense / "protocol.json"),
        dense_summary_sha256=replay.sha(dense / "summary.json"), landing_settings=asdict(LandingConfig()),
        code_sha256={"src/shoe_landing_events.py": replay.sha(snapshot)})
    replay.write(baseline / "protocol.json", baseline_protocol)
    replay.write(baseline / "summary.json", dict(flags, events=39, rows=baseline_rows))
    return SimpleNamespace(dense_dir=dense, baseline_analysis_dir=baseline, output_dir=tmp_path / "replayed")


def test_replay_preserves_unknowns_and_frozen_foot_without_old_absolute_receipts(cohort):
    summary = replay.replay(cohort)
    assert summary["events"] == len(summary["rows"]) == 39
    assert summary["actual_new_neural_calls"] == 0
    assert summary["foot_results_inherited_unchanged"] is True
    assert summary["landing_thresholds_changed"] is False
    first, unknown = summary["rows"][:2]
    assert first["old_stance_frame_counts"] == [3]
    assert first["new_stance_frame_counts"] == [4]
    assert first["old_stance_seconds"] == pytest.approx([2 / 30])
    assert first["new_stance_seconds"] == pytest.approx([3 / 30])
    assert unknown["visible_transition_proposals"] is None
    assert unknown["new_stance_frame_counts"] is None
    unavailable = summary["rows"][2]
    result = replay.read(cohort.output_dir / "events" / unavailable["clip_id"] / "landing_result.json")
    assert result["candidate_count"] is None and result["landings"] is None
    assert result["observable_window"] is None and result["total_run_up_steps"] is None
    for row in summary["rows"]:
        old = replay.read(cohort.baseline_analysis_dir / "events" / row["clip_id"] / "landing_result.json")
        new = replay.read(cohort.output_dir / "events" / row["clip_id"] / "landing_result.json")
        assert new["striking_foot"] == old["striking_foot"]
        assert new.get("dense_foot") == old.get("dense_foot")
        assert new["source_mapping"] == old["source_mapping"]
        assert replay.sha(cohort.output_dir / "events" / row["clip_id"] / "landing_result.json") == row["result_sha256"]
    assert (cohort.output_dir / "comparison_39.csv").is_file()
    protocol = replay.read(cohort.output_dir / "protocol.json")
    assert protocol["reference_labels_read"] is False
    for name, digest in protocol["code_sha256"].items():
        assert replay.sha(cohort.output_dir / "source_snapshot" / name) == digest


@pytest.mark.parametrize("kind", ["inputs", "poses", "baseline", "snapshot"])
def test_changed_frozen_bytes_are_rejected_before_output_creation(cohort, kind):
    cid = next(iter(replay._course_sources()))
    paths = dict(inputs=cohort.dense_dir / "events" / cid / "inputs.json",
                 poses=cohort.dense_dir / "events" / cid / "poses33.json",
                 baseline=cohort.baseline_analysis_dir / "events" / cid / "landing_result.json",
                 snapshot=cohort.dense_dir / "source_snapshot/src/synthetic_pose_producer.py")
    with paths[kind].open("a", encoding="utf-8") as stream:
        stream.write(" \n")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay.replay(cohort)
    assert not cohort.output_dir.exists()


@pytest.mark.parametrize("corruption", ["external", "partial", "duplicate", "pilot", "labels", "thresholds"])
def test_external_or_partial_cohorts_and_scope_changes_are_rejected(cohort, corruption):
    path = cohort.dense_dir / "summary.json"
    summary = replay.read(path)
    dense_protocol = replay.read(cohort.dense_dir / "protocol.json")
    baseline_protocol = replay.read(cohort.baseline_analysis_dir / "protocol.json")
    if corruption == "external":
        summary["rows"][0]["clip_id"] = "another_match_penalty"
    elif corruption == "partial":
        summary["rows"].pop()
    elif corruption == "duplicate":
        summary["rows"][0] = deepcopy(summary["rows"][1])
    elif corruption == "pilot":
        dense_protocol["pilot"] = True
    elif corruption == "labels":
        dense_protocol["reference_labels_read"] = True
    else:
        baseline_protocol["landing_settings"]["minimum_stance_seconds"] = .01
    replay.write(path, summary)
    replay.write(cohort.dense_dir / "protocol.json", dense_protocol)
    baseline_protocol.update(dense_summary_sha256=replay.sha(path),
                             dense_protocol_sha256=replay.sha(cohort.dense_dir / "protocol.json"))
    replay.write(cohort.baseline_analysis_dir / "protocol.json", baseline_protocol)
    with pytest.raises(ValueError):
        replay.replay(cohort)
    assert not cohort.output_dir.exists()


def test_rehashed_baseline_metadata_mismatch_is_rejected(cohort):
    summary_path = cohort.baseline_analysis_dir / "summary.json"
    summary = replay.read(summary_path)
    cid = summary["rows"][0]["clip_id"]
    path = cohort.baseline_analysis_dir / "events" / cid / "landing_result.json"
    result = replay.read(path)
    result["frame_count"] += 1
    replay.write(path, result)
    summary["rows"][0]["result_sha256"] = replay.sha(path)
    replay.write(summary_path, summary)
    with pytest.raises(ValueError, match="Baseline metadata"):
        replay.replay(cohort)


def test_existing_output_is_never_overwritten(cohort):
    cohort.output_dir.mkdir()
    sentinel = cohort.output_dir / "keep.txt"
    sentinel.write_text("existing output", encoding="utf-8")
    with pytest.raises(FileExistsError):
        replay.replay(cohort)
    assert sentinel.read_text(encoding="utf-8") == "existing output"


def test_output_cannot_be_created_inside_frozen_input(cohort):
    cohort.output_dir = cohort.dense_dir / "new-run"
    with pytest.raises(ValueError, match="separate"):
        replay.replay(cohort)
