"""Produce a foot-only supplement without rewriting the frozen course bundle."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_fresh_shoe_approach_experiment import bind_event, course_manifest, decode_interval, read, sha, write
from src.fresh_results import bind_stage, compact_stage
from src.shoe_approach_fusion import SCHEMA, REGION_SCALE_DEFINITION, guarded_selection, recompute_dense_foot, validate_dense_observations
from src.shoe_pose33_fusion import artifact, load_artifact


def snapshots(directory, hashes, current_files=()):
    bindings = {}
    for name, digest in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Code snapshot paths must stay inside their experiment')
        path = directory / 'source_snapshot' / relative
        if sha(path) != digest or (name in current_files and sha(ROOT / relative) != digest):
            raise ValueError(f'Frozen prediction code changed: {name}')
        bindings[name] = artifact(path)
    if not set(current_files).issubset(hashes):
        raise ValueError('Required frozen prediction modules are missing')
    return bindings


def unique_rows(payload, source_ids, label):
    rows = payload.get('rows', [])
    if (payload.get('events') != 39 or len(rows) != 39
            or len({row['clip_id'] for row in rows}) != 39
            or {row['clip_id'] for row in rows} != source_ids):
        raise ValueError(f'{label} requires the exact distinct original cohort')
    return {row['clip_id']: row for row in rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clips-dir', 'pose-dir', 'phase-dir', 'ball-dir', 'baseline-bundle',
                 'baseline-foot-dir', 'dense-dir', 'analysis-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('Use a new supplement output directory')
    manifest_path = args.clips_dir / 'source_manifest.json'
    _, sources = course_manifest(manifest_path)
    manifest_hash = sha(manifest_path)
    source_ids = {row['clip_id'] for row in sources}
    baseline_path = args.baseline_bundle / 'features_39.json'
    baseline = read(baseline_path)
    baseline_events = baseline.get('events', [])
    baseline_rows = {row['clip_id']: row for row in baseline_events}
    if len(baseline_events) != 39 or set(baseline_rows) != source_ids:
        raise ValueError('Frozen baseline must contain exactly the original 39 events')
    dense_protocol_path, dense_summary_path = args.dense_dir / 'protocol.json', args.dense_dir / 'summary.json'
    analysis_protocol_path, analysis_summary_path = args.analysis_dir / 'protocol.json', args.analysis_dir / 'summary.json'
    dense_protocol, dense_summary = read(dense_protocol_path), read(dense_summary_path)
    analysis_protocol, analysis_summary = read(analysis_protocol_path), read(analysis_summary_path)
    settings = dense_protocol.get('settings', {})
    expected_settings = dict(running_mode='VIDEO', normal_and_mirror_use_independent_models=True, num_poses=1,
        detection_confidence=.5, presence_confidence=.5, landmark_visibility_presence_min=.5,
        crop_margin_body_height=.45, fixed_union_roi_per_event=True,
        actor_anchors='exact_same_frame_original_observations_only', anchor_interpolation=False,
        decode_policy='sequential_from_clip_frame_zero', post_departure_seconds=.2,
        duplicated_pixels_are_independent_support=False)
    if (dense_protocol.get('schema_version') != 'fresh-shoe-approach-pose33-v1'
            or dense_protocol.get('events') != 39 or dense_protocol.get('pilot') is not False
            or dense_protocol.get('plan_only') is not False
            or set(dense_protocol.get('selected_clip_ids', [])) != source_ids
            or dense_protocol.get('source_manifest_sha256') != manifest_hash
            or settings != expected_settings
            or any(dense_protocol.get(key) is not value for key, value in (
                ('original_course_videos_only', True), ('additional_video_used', False),
                ('reference_labels_read', False), ('human_verified', False), ('accuracy_validated', False), ('held_out_test', False)))
            or analysis_protocol.get('dense_protocol_sha256') != sha(dense_protocol_path)
            or analysis_protocol.get('dense_summary_sha256') != sha(dense_summary_path)
            or analysis_protocol.get('baseline_sha256') != sha(baseline_path)
            or any(analysis_protocol.get(key) is not value for key, value in (
                ('original_course_videos_only', True), ('additional_video_used', False),
                ('reference_labels_read', False), ('human_verified', False), ('held_out_test', False)) )):
        raise ValueError('Dense/analysis protocols differ from their frozen source-only experiment')
    dense_rows = unique_rows(dense_summary, source_ids, 'Dense cohort')
    analysis_rows = unique_rows(analysis_summary, source_ids, 'Analysis cohort')
    dense_snapshots = snapshots(args.dense_dir, dense_protocol['code_sha256'], (
        'scripts/run_fresh_shoe_approach_experiment.py', 'src/mediapipe_shoe_evidence.py', 'src/pose_refinement.py'))
    analysis_snapshots = snapshots(args.analysis_dir, analysis_protocol['code_sha256'], (
        'src/kicking_foot_strike_window.py', 'src/mediapipe_shoe_evidence.py',
        'src/kicking_foot_evidence.py', 'src/kicking_foot_evidence_v2.py'))
    model_path = Path(dense_protocol['model']['path'])
    if sha(model_path) != dense_protocol['model']['sha256']:
        raise ValueError('Actual MediaPipe model bytes differ from the frozen inference receipt')
    args.output_dir.mkdir(parents=True)
    code_names = ('scripts/fuse_shoe_approach_foot.py', 'src/shoe_approach_fusion.py',
        'src/shoe_pose33_fusion.py', 'src/fresh_results.py', 'src/kicking_foot_strike_window.py',
        'src/kicking_foot_evidence.py', 'src/kicking_foot_evidence_v2.py',
        'src/mediapipe_shoe_evidence.py', 'src/pose_refinement.py', 'scripts/run_fresh_shoe_approach_experiment.py')
    code_hashes = {}
    for name in code_names:
        target = args.output_dir / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        code_hashes[name] = sha(target)
    common_bindings = dict(source_manifest=artifact(manifest_path), frozen_baseline=artifact(baseline_path),
        dense_protocol=artifact(dense_protocol_path), dense_summary=artifact(dense_summary_path),
        analysis_protocol=artifact(analysis_protocol_path), analysis_summary=artifact(analysis_summary_path),
        dense_code_snapshot=dense_snapshots, analysis_code_snapshot=analysis_snapshots,
        model=artifact(model_path))
    write(args.output_dir / 'protocol.json', dict(schema_version=SCHEMA, events=39,
        source_manifest_sha256=manifest_hash, producer_bindings=common_bindings, code_sha256=code_hashes,
        region_scale_definition=REGION_SCALE_DEFINITION, region_distance_threshold_body=.08,
        selection_rule='existing src.shoe_pose33_fusion.select_candidate, with explicit legacy ambiguity abstention',
        source_pixels_independently_redecoded=True, new_neural_inference_by_fusion=False,
        original_course_videos_only=True, additional_video_used=False,
        main_feature_table_modified=False, step_counts_modified=False, supplement_only=True,
        reference_labels_read=False, human_verified=False, accuracy_validated=False, held_out_test=False,
        development_data_previously_inspected=True, selection_protocol_written_before_fusion_results=True))
    common_bindings['supplement_protocol'] = artifact(args.output_dir / 'protocol.json')
    rows = []
    for source in sources:
        cid = source['clip_id']
        old_path = args.baseline_foot_dir / 'results' / f'{cid}.json'
        old = read(old_path)
        pose_path = args.pose_dir / f'{cid}.json'
        bind_stage('foot', old, source, sha(pose_path), manifest_hash)
        event = baseline_rows[cid]
        if event.get('source_mapping') != source or compact_stage('foot', old) != event['components']['foot']:
            raise ValueError('Frozen baseline foot differs from its validated original producer')
        legacy_binding = old['temporal_evidence']['producer_bindings']['original_neural_result']
        legacy = load_artifact(legacy_binding, 'legacy neural foot')
        old_cohort = load_artifact(old['temporal_evidence']['producer_bindings']['shoe_cohort_protocol'], 'legacy shoe protocol')
        if old_cohort['model_sha256'] != dense_protocol['model']['sha256']:
            raise ValueError('Dense approach uses a different model from the original guarded shoe experiment')
        bound, receipts = bind_event(args, source, manifest_hash)
        dense_entry, analysis_entry = dense_rows[cid], analysis_rows[cid]
        event_dir = args.dense_dir / 'events' / cid
        analysis_result_path = args.analysis_dir / 'events' / cid / 'landing_result.json'
        analysis_result = read(analysis_result_path)
        if (sha(analysis_result_path) != analysis_entry['result_sha256']
                or analysis_result.get('source_mapping') != source
                or analysis_result.get('baseline_foot') != old['kicking_foot_candidate']
                or analysis_entry.get('baseline_foot') != old['kicking_foot_candidate']
                or dense_entry.get('source_sha256') != source['clip_sha256']):
            raise ValueError('Dense analysis result source and baseline producer disagree')
        bindings = dict(common_bindings, baseline_foot=artifact(old_path), legacy_neural_foot=legacy_binding,
                        pose=artifact(pose_path), analysis_result=artifact(analysis_result_path),
                        input_receipts=artifact(event_dir / 'input_receipts.json'))
        if read(event_dir / 'input_receipts.json') != receipts:
            raise ValueError('Dense original input receipts no longer reproduce')
        validation = None
        if bound is None:
            if (dense_entry['status'] != 'unavailable' or dense_entry.get('reason') != receipts['unavailable_reason']
                    or analysis_result.get('striking_foot') != 'unknown'):
                raise ValueError('Unavailable dense event must retain the reproduced abstention')
            bindings['dense_unavailable'] = artifact(event_dir / 'unavailable.json')
            dense_candidate, bracket, region_frames = 'unknown', None, []
        else:
            inputs_path, poses_path, plan_path = event_dir / 'inputs.json', event_dir / 'poses33.json', event_dir / 'roi_plan.json'
            inputs, records, plan = read(inputs_path), read(poses_path), read(plan_path)
            if (dense_entry['status'] != 'fresh_dense_pose_candidate'
                    or sha(inputs_path) != dense_entry['inputs_sha256'] or sha(poses_path) != dense_entry['poses33_sha256']
                    or sha(plan_path) != dense_entry['roi_plan_sha256']
                    or inputs.get('poses33_sha256') != sha(poses_path)
                    or inputs.get('source_manifest_sha256') != manifest_hash
                    or inputs.get('model_sha256') != dense_protocol['model']['sha256']
                    or inputs.get('code_sha256') != dense_protocol['code_sha256']
                    or inputs.get('input_receipts') != receipts):
                raise ValueError('Dense event hashes differ from its cohort, model or source receipts')
            decoded = {str(frame): hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                       for frame, image, _ in decode_interval(bound)}
            if sha(bound['video']) != source['clip_sha256']:
                raise ValueError('Source clip changed during independent decoding')
            validation = validate_dense_observations(records, inputs, plan, bound, decoded)
            for name, value in validation.items():
                if dense_entry.get(name) != value:
                    raise ValueError(f'Dense quality receipt does not reproduce: {name}')
            export_path = args.ball_dir / 'contacts' / f'{cid}.json'
            if sha(export_path) != inputs['input_receipts']['contact_export']['sha256']:
                raise ValueError('Dense contact export bytes changed')
            foot = recompute_dense_foot(records, inputs, read(export_path),
                method_label=analysis_result.get('dense_foot', {}).get('method'))
            if (foot != analysis_result.get('dense_foot') or foot['striking_foot'] != analysis_result.get('striking_foot')
                    or foot['striking_foot'] != analysis_entry.get('dense_foot')
                    or analysis_result.get('dense_inputs_sha256') != sha(inputs_path)
                    or analysis_result.get('dense_poses_sha256') != sha(poses_path)):
                raise ValueError('Dense foot does not reproduce from its restored source observations')
            dense_candidate, bracket, region_frames = foot['striking_foot'], bound['contact_bracket_frames'], foot['joint_region_frames']
            bindings.update(dense_inputs=artifact(inputs_path), dense_poses33=artifact(poses_path),
                            dense_roi_plan=artifact(plan_path), contact_export=artifact(export_path),
                            phase=artifact(args.phase_dir / f'{cid}.json'),
                            camera_motion=artifact(args.phase_dir / 'camera_motion' / f'{cid}.json'))
        selection = guarded_selection(old['kicking_foot_candidate'], dense_candidate, legacy, bracket)
        result = dict(schema_version=SCHEMA, clip_id=cid, source_mapping=source,
            **selection, existing_fusion_candidate=old['kicking_foot_candidate'],
            dense_region_raw_candidate=dense_candidate, joint_shoe_region_frames=region_frames,
            contact_bracket_frames=bracket, region_scale_definition=REGION_SCALE_DEFINITION,
            status='completed_source_bound_foot_supplement', producer_bindings=bindings,
            observation_validation=validation, new_neural_inference=False, supplement_only=True,
            main_feature_table_modified=False, step_counts_modified=False, reference_labels_used_for_prediction=False,
            human_verified=False, accuracy_validated=False, held_out_test=False,
            shoe_ball_contact_verified=False, anatomical_side_verified=False)
        target = args.output_dir / 'results' / f'{cid}.json'
        write(target, result)
        rows.append(dict(clip_id=cid, existing_candidate=old['kicking_foot_candidate'],
                         dense_candidate=selection['dense_region_candidate'], candidate=selection['kicking_foot_candidate'],
                         reason=selection['selection_reason'], result_sha256=sha(target)))
        print(json.dumps(rows[-1]), flush=True)
    with (args.output_dir / 'foot_candidates_39.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(schema_version=SCHEMA, events=39, rows=rows,
        candidate_counts=dict(Counter(row['candidate'] for row in rows)),
        existing_candidate_counts=dict(Counter(row['existing_candidate'] for row in rows)),
        dense_candidate_counts=dict(Counter(row['dense_candidate'] for row in rows)),
        fallback_events=[row['clip_id'] for row in rows if row['reason'] == 'dense_region_fallback_for_existing_unknown'],
        conflict_events=[row['clip_id'] for row in rows if row['reason'] == 'existing_fusion_and_dense_region_conflict'],
        new_neural_inference_by_fusion=False, human_verified=False, accuracy_validated=False, held_out_test=False,
        supplement_only=True, main_feature_table_modified=False, step_counts_modified=False,
        interpretation='Development candidate availability, not verified anatomy, contact or accuracy')
    write(args.output_dir / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    main()
