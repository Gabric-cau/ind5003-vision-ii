"""Summarize frozen alternate-source pilots without merging the course cohort."""

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_fresh_ball_models import compare_event
from src.final_results import sha256
from src.fresh_results import read_json, write_json


def summarize(base, source_manifest, output_dir, refresh_derived_summary=False):
    if output_dir.exists() and not refresh_derived_summary:
        raise FileExistsError('Preserve the comparison in a new output directory')
    if output_dir.exists():
        allowed = {'comparison.json', 'source_manifest.json', 'artifact_receipts.json'}
        if any(p.name not in allowed or not p.is_file() for p in output_dir.iterdir()):
            raise ValueError('Only this derived comparison directory may be refreshed')
    source = read_json(source_manifest)
    records = source['clips']
    if source.get('original_course_recording') is not False or len(records) != 3:
        raise ValueError('This comparison requires the three alternate-source pilot intervals')
    source_hash = sha256(source_manifest)
    bindings = []

    def load(stage, name):
        path = base / stage / name
        value = read_json(path)
        bindings.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path),
                         'bytes': path.stat().st_size})
        return value

    protocols = {label: load(stage, 'protocol.json') for label, stage in
                 [('pose', 'hd_pose_pilot_v1'), ('n', 'hd_ball_n_pilot_v1'),
                  ('m', 'hd_ball_pilot_v1'), ('foot17', 'hd_foot_pilot_v1'),
                  ('shoe33', 'shoe_pose33_hd_kane_guarded_video_pilot_v1')]}
    if any(p['source_manifest_sha256'] != source_hash for p in protocols.values()):
        raise ValueError('A stage is bound to a different source manifest')
    summaries = {label: load(stage, 'summary.json') for label, stage in
                 [('n', 'hd_ball_n_pilot_v1'), ('m', 'hd_ball_pilot_v1'), ('foot17', 'hd_foot_pilot_v1')]}
    rows = []
    for record in records:
        cid = record['clip_id']
        models = {label: load(stage, f'results/{cid}.json') for label, stage in
                  [('n', 'hd_ball_n_pilot_v1'), ('m', 'hd_ball_pilot_v1')]}
        for result in models.values():
            identity = result['inference_identity']
            if (identity['source_sha256'] != record['clip_sha256'] or
                    identity['source_manifest_sha256'] != source_hash or result['clip_id'] != cid):
                raise ValueError('A model prediction is not bound to this pilot clip')
        row = compare_event(models['n'], models['m'])
        row['source_start_frame'] = record['source_start_frame']
        row['source_fps'] = record['fps']
        row['source_interval_video_sec'] = [record['source_start_frame']/record['fps'],
                                             record['source_end_frame_exclusive']/record['fps']]
        for label, stage in [('n', 'hd_ball_n_pilot_v1'), ('m', 'hd_ball_pilot_v1')]:
            contact = load(stage, f'contacts/{cid}.json')
            candidate = contact.get('contact')
            model_row = row[label]
            model_row['candidate_support_score'] = candidate.get('score') if candidate else None
            model_row['exact_contact_frame_established'] = False
            model_row['contact_time_sec_from_frame'] = None
            model_row['contact_source_video_sec_from_frame'] = None
            model_row['contact_source_frame'] = None
            model_row['bracket_source_video_sec'] = None
            model_row['stored_contact_time_sec'] = None
            model_row['stored_time_consistent_with_final_candidate_frame'] = None
            if candidate:
                frame = candidate['contact_frame']
                bracket = candidate['contact_bracket_frames']
                model_row['contact_time_sec_from_frame'] = frame/record['fps']
                model_row['contact_source_frame'] = record['source_start_frame'] + frame
                model_row['contact_source_video_sec_from_frame'] = (record['source_start_frame'] + frame)/record['fps']
                model_row['bracket_source_video_sec'] = [
                    (record['source_start_frame'] + f)/record['fps'] for f in bracket]
                model_row['stored_contact_time_sec'] = candidate.get('contact_time_sec')
                model_row['stored_time_consistent_with_final_candidate_frame'] = math.isclose(
                    candidate.get('contact_time_sec', float('nan')), frame/record['fps'], abs_tol=1e-9)
                model_row['scored_ankle_proximity_frame'] = candidate.get('scored_ankle_proximity_frame')
            model_row['outcome_scope'] = 'unguarded_ball_rule_diagnostic_not_verified_goal'
        pose = load('hd_pose_pilot_v1', f'{cid}.json')
        foot = load('hd_foot_pilot_v1', f'results/{cid}.json')
        if foot['inference_identity']['source_sha256'] != record['clip_sha256']:
            raise ValueError('Foot evidence is not bound to the pilot clip')
        row['pose_diagnostic'] = {key: pose.get(key) for key in
                                  ('status', 'step_count', 'kick_frame', 'kicking_foot', 'selected_track_id')}
        row['pose_diagnostic']['foot_label_is_unverified_legacy_diagnostic'] = True
        row['foot17_guarded'] = {key: foot.get(key) for key in
                                ('kicking_foot_candidate', 'status', 'new_neural_inference',
                                 'normal_pose_frames', 'mirrored_pose_frames')}
        row['foot17_guarded']['contact_bracket_frames'] = foot.get('timing', {}).get('contact_bracket_frames')
        row['strict_keeper_height_included'] = False
        rows.append(row)
    shoe = load('shoe_pose33_hd_kane_guarded_video_pilot_v1', 'result.json')
    shoe_record = next(r for r in records if r['clip_id'] == shoe['clip_id'])
    if protocols['shoe33']['source_sha256'] != shoe_record['clip_sha256']:
        raise ValueError('Shoe evidence is not bound to the pilot clip')
    shoe_compact = {key: shoe.get(key) for key in
                   ('clip_id', 'source_coordinate_shoe_swing_candidate',
                    'source_coordinate_shoe_contact_region_candidate', 'shoe_contact_verified',
                    'anatomical_side_verified', 'ball_identity_verified', 'actual_inference_calls',
                    'detected_pose_calls', 'accepted_roi_pose_calls', 'calls_by_view')}
    shoe_compact['source_video_contact_bracket_sec'] = [
        (shoe_record['source_start_frame'] + f)/shoe_record['fps'] for f in shoe['contact_bracket_frames']]
    shoe_compact['ankle_motion_status'] = shoe['evidence']['ankle_motion']['status']
    shoe_compact['shoe_centroid_motion_status'] = shoe['evidence']['shoe_centroid_motion']['status']
    shoe_compact['local_ball_identity_ambiguous_frames'] = shoe['local_ball_identity_ambiguous_frames']
    included = [base / 'hd_pose_pilot_v1' / 'protocol.json']
    for stage in ('hd_ball_n_pilot_v1', 'hd_ball_pilot_v1', 'hd_foot_pilot_v1'):
        included.extend(base / stage / name for name in ('protocol.json', 'summary.json'))
        for directory in ('results', 'contacts', 'media'):
            included.extend(sorted((base / stage / directory).glob('*')))
    shoe_base = base / 'shoe_pose33_hd_kane_guarded_video_pilot_v1'
    included.extend(shoe_base / name for name in ('protocol.json', 'result.json', 'roi_plan.json', 'poses33.json'))
    included_paths = set(included)
    stage_paths = [base / name for name in ('hd_pose_pilot_v1', 'hd_ball_n_pilot_v1',
                   'hd_ball_pilot_v1', 'hd_foot_pilot_v1', 'shoe_pose33_hd_kane_guarded_video_pilot_v1')]
    excluded = sorted(path for stage in stage_paths for path in stage.rglob('*')
                      if path.is_file() and path not in included_paths)

    def receipt(path):
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                'sha256': sha256(path)}

    public_package = [receipt(path) for path in sorted(included_paths)]
    excluded_package = [receipt(path) for path in excluded]
    public_paths = {item['path'] for item in public_package}
    for binding in bindings:
        binding['included_in_recommended_public_package'] = binding['path'] in public_paths
    summary = {
        'schema_version': 1, 'scope': 'three_existing_penalties_in_alternate_FIFA_source_clock',
        'source_manifest_sha256': source_hash, 'source_url': records[0]['source_url'],
        'source_video_sha256': records[0]['source_video_sha256'],
        'events': 3, 'independent_new_events': 0, 'merged_into_course_39': False,
        'settings_and_code_identical_between_ball_models': True,
        'ball_model_settings': protocols['n']['settings'],
        'ball_model_sha256': {label: next(iter(protocols[label]['model_sha256'].values())) for label in ('n', 'm')},
        'frozen_ball_code_sha256': protocols['n']['code_sha256'],
        'pose_model_sha256': next(iter(protocols['pose']['model_sha256'].values())),
        'pose_settings': protocols['pose']['settings'],
        'foot17_settings': protocols['foot17']['settings'],
        'shoe33_model_sha256': protocols['shoe33']['model_sha256'],
        'shoe33_settings': protocols['shoe33']['settings'],
        'contact_candidates': {label: sum(r[label]['contact'] for r in rows) for label in ('n', 'm')},
        'contact_pairs_n_m': dict(Counter(r['contact_pair'] for r in rows)),
        'guarded_foot17_candidates': dict(Counter(r['foot17_guarded']['kicking_foot_candidate'] for r in rows)),
        'foot17_new_neural_inference_events': summaries['foot17']['new_neural_inference_events'],
        'stored_contact_time_inconsistencies': sum(r[label]['stored_time_consistent_with_final_candidate_frame'] is False
                                                  for r in rows for label in ('n', 'm')),
        'time_mapping': 'official_source_video_sec = (source_start_frame + clip_contact_frame) / source_fps',
        'time_field_note': 'Frozen producers left contact_time_sec at the earlier scored ankle frame. This comparison recomputes final candidate seconds from frames; the producer is fixed separately and frozen inputs remain unchanged.',
        'keeper_height_note': 'Legacy low values in unguarded ball summaries are excluded; no reliable keeper height is established by this pilot.',
        'human_verified': False, 'accuracy_validated': False, 'held_out_test': False,
        'reference_labels_used_for_prediction': False, 'contact_truth_annotated': False,
        'resolution_effect_isolated': False,
        'interpretation': 'Same-event development diagnostics. Alternate edit, resolution, frame rate and manual interval differ from the course recordings; this is not a pure resolution ablation or an accuracy result.',
        'public_artifact_package': {
            'selection_scope': 'Frozen protocols, summaries, results, contact proposals, media, and the compact Kane guarded shoe review inputs. Full frame detections, event windows, inputs, source snapshots and full body pose traces are excluded. Original videos and weights are outside this package.',
            'receipts_file': 'artifact_receipts.json', 'included_files': len(public_package),
            'included_bytes': sum(item['bytes'] for item in public_package),
        },
        'raw_artifacts_not_included': excluded_package,
        'shoe33': shoe_compact, 'rows': rows, 'artifact_bindings': bindings,
        'summary_generator_sha256': sha256(Path(__file__)),
    }
    output_dir.mkdir(parents=True, exist_ok=refresh_derived_summary)
    shutil.copyfile(source_manifest, output_dir / 'source_manifest.json')
    write_json(output_dir / 'comparison.json', summary)
    write_json(output_dir / 'artifact_receipts.json', {
        'schema_version': 1, 'selection_is_recommendation_not_git_stage_receipt': True,
        'source_manifest_sha256': source_hash, 'files': public_package,
        'excluded_files': len(excluded_package),
        'excluded_bytes': sum(item['bytes'] for item in excluded_package),
    })
    print({key: summary[key] for key in ('events', 'contact_candidates', 'guarded_foot17_candidates',
                                       'stored_contact_time_inconsistencies')})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--refresh-derived-summary', action='store_true')
    args = parser.parse_args()
    summarize(args.base_dir.resolve(), args.source_manifest.resolve(), args.output_dir.resolve(),
              args.refresh_derived_summary)


if __name__ == '__main__':
    main()
