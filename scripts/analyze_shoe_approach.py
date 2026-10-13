"""Compare frozen dense shoe observations with the original ankle proposals."""
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

from src.kicking_foot_strike_window import kicking_foot_strike_window
from src.mediapipe_shoe_evidence import coco_samples, shoe_ball_distances
from src.shoe_landing_events import LandingConfig, propose_shoe_landings


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def dense_foot(records, inputs):
    normal = [row for row in records if not row['reflected'] and row['unique_decoded_support']]
    mirror = [row for row in records if row['reflected'] and row['unique_decoded_support']]
    contact = inputs['contact']
    bracket = inputs['contact_bracket_frames']
    # The causal contact export retains the actual detection lists separately.
    export_path = Path(inputs['input_receipts']['contact_export']['path'])
    export = read(export_path)
    if sha(export_path) != inputs['input_receipts']['contact_export']['sha256']:
        raise ValueError('Contact export bytes changed after dense inference')
    payload = export.get('contact', export)
    stationary = payload['stationary_ball_observations']
    departure = payload['departure_ball_observations']
    evidence = kicking_foot_strike_window(coco_samples(normal, shoe_centroids=True),
        coco_samples(mirror, shoe_centroids=True), bracket, inputs['fps'],
        stationary_ball_observations=stationary, departure_ball_observations=departure)
    candidate = evidence['candidate']
    target = next(row['xy'] for row in stationary if row['frame'] == bracket[0])
    scales = [row['anchor_bbox'][3] - row['anchor_bbox'][1] for row in normal
              if row.get('anchor_bbox') and row['frame'] < bracket[0]]
    scale = float(np.median(scales)) if scales else None
    support = []
    if candidate != 'unknown' and scale:
        distances = [shoe_ball_distances(rows, target, scale) for rows in (normal, mirror)]
        near = [{row['frame'] for row in rows if bracket[0] - 1 <= row['frame'] <= bracket[1]
                 and row[candidate]['confident']
                 and row[candidate]['stationary_point_distance_body'] <= .08} for rows in distances]
        paired = {row['frame'] for row in evidence.get('accepted_paired_frames', [])}
        support = sorted(near[0] & near[1] & paired)
    ambiguous = payload.get('seed_ambiguous_frames', []) + export.get('seed_ambiguous_frames', [])
    if any(bracket[0] - 5 <= frame <= bracket[1] for frame in ambiguous):
        candidate, support = 'unknown', []
    return dict(striking_foot=candidate if support else 'unknown', swing_candidate=candidate,
                joint_region_frames=support, evidence=evidence, physical_contact_verified=False,
                method='paired_shoe_strike_with_longer_VIDEO_history_and_approach_median_region_scale')


def analyze(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new analysis directory')
    protocol = read(args.dense_dir / 'protocol.json')
    cohort = read(args.dense_dir / 'summary.json')
    if (protocol.get('original_course_videos_only') is not True
            or protocol.get('additional_video_used') is not False
            or protocol.get('plan_only') is not False
            or cohort.get('events') != 39 or len(cohort.get('rows', [])) != 39):
        raise ValueError('Analysis requires the frozen original-course dense inference cohort')
    baseline = read(args.baseline_bundle / 'features_39.json')
    baseline_rows = {event['clip_id']: event for event in baseline['events']}
    if set(baseline_rows) != {row['clip_id'] for row in cohort['rows']}:
        raise ValueError('Dense and baseline cohorts disagree')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ('scripts/analyze_shoe_approach.py', 'src/shoe_landing_events.py',
             'src/kicking_foot_strike_window.py', 'src/mediapipe_shoe_evidence.py',
             'src/kicking_foot_evidence.py', 'src/kicking_foot_evidence_v2.py',
             'docs/SHOE_APPROACH_PROTOCOL_2026-10-13.md')
    hashes = {}
    for name in names:
        target = args.output_dir / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        hashes[name] = sha(target)
    write(args.output_dir / 'protocol.json', dict(
        schema_version='shoe-approach-analysis-v1', dense_protocol_sha256=sha(args.dense_dir / 'protocol.json'),
        dense_summary_sha256=sha(args.dense_dir / 'summary.json'),
        baseline_sha256=sha(args.baseline_bundle / 'features_39.json'), code_sha256=hashes,
        landing_settings=vars(LandingConfig()), new_neural_inference=False,
        reference_labels_read=False, human_verified=False, held_out_test=False,
        original_course_videos_only=True, additional_video_used=False,
        main_feature_table_modified=False))
    rows = []
    for entry in cohort['rows']:
        cid = entry['clip_id']
        event_dir = args.dense_dir / 'events' / cid
        target = args.output_dir / 'events' / cid
        old = baseline_rows[cid]
        source = old['source_mapping']
        if entry['source_sha256'] != source['clip_sha256']:
            raise ValueError('Baseline and dense source clips differ')
        old_phase = read(args.phase_dir / f'{cid}.json')
        old_foot = old['components']['foot']['kicking_foot_candidate']
        base = dict(clip_id=cid, source_sha256=source['clip_sha256'], fps=source['fps'], frame_count=source['frame_count'],
                    source_mapping=source, full_approach_visible=False,
                    baseline_ankle_count=old_phase.get('candidate_step_count'),
                    baseline_foot=old_foot, human_verified=False, accuracy_validated=False,
                    held_out_test=False, new_neural_inference=False,
                    reference_labels_used_for_prediction=False)
        if entry['status'] != 'fresh_dense_pose_candidate':
            result = dict(base, status=entry['status'], reason=entry.get('reason'),
                candidate_count=None, observed_proposal_count=0, landings=None,
                striking_foot='unknown', observable_window=None, total_run_up_steps=None,
                count_complete=False, dense_input_reason=entry)
        else:
            inputs_path, poses_path = event_dir / 'inputs.json', event_dir / 'poses33.json'
            if sha(inputs_path) != entry['inputs_sha256'] or sha(poses_path) != entry['poses33_sha256']:
                raise ValueError('Dense input or pose bytes changed after inference')
            inputs, records = read(inputs_path), read(poses_path)
            if (inputs['poses33_sha256'] != sha(poses_path)
                    or inputs['source_mapping'] != source
                    or inputs['reference_labels_read'] is not False
                    or inputs['new_neural_inference'] is not True):
                raise ValueError('Dense predictions lack exact source-only input bindings')
            landing = propose_shoe_landings(records, inputs['fps'], inputs['interval_frames'],
                inputs['original_camera_motion'], inputs['decoded_frame_hashes'])
            foot = dense_foot(records, inputs)
            result = {**base, **landing, 'striking_foot': foot['striking_foot'], 'dense_foot': foot,
                'dense_inputs_sha256': sha(inputs_path), 'dense_poses_sha256': sha(poses_path),
                'diagnostic_counts': dict(Counter(item['reason'] for item in landing['diagnostics']))}
        write(target / 'landing_result.json', result)
        rows.append(dict(clip_id=cid, event_id=source['event_id'], status=result['status'],
            baseline_ankle_count=result['baseline_ankle_count'],
            visible_transition_proposals=result['candidate_count'],
            baseline_foot=old_foot, dense_foot=result['striking_foot'],
            paired_unique_pose_frames=len(result.get('paired_unique_pose_frames', [])),
            result_sha256=sha(target / 'landing_result.json')))
    fields = list(rows[0])
    with (args.output_dir / 'comparison_39.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(events=39, rows=rows, status_counts=dict(Counter(row['status'] for row in rows)),
        dense_foot_counts=dict(Counter(row['dense_foot'] for row in rows)),
        events_with_stance_transition_proposals=sum((row['visible_transition_proposals'] or 0) > 0 for row in rows),
        proposal_count_available=sum(row['visible_transition_proposals'] is not None for row in rows),
        proposal_total=sum(row['visible_transition_proposals'] or 0 for row in rows),
        paired_unique_pose_frame_total=sum(row['paired_unique_pose_frames'] for row in rows),
        actual_new_neural_calls=cohort['actual_inference_calls'],
        baseline_foot_available=sum(row['baseline_foot'] != 'unknown' for row in rows),
        dense_foot_available=sum(row['dense_foot'] != 'unknown' for row in rows),
        foot_disagreements=[row['clip_id'] for row in rows if row['baseline_foot'] != 'unknown'
                           and row['dense_foot'] != 'unknown' and row['baseline_foot'] != row['dense_foot']],
        independently_reviewed_events=0, accuracy=None, step_count_mae=None,
        human_verified=False, accuracy_validated=False, held_out_test=False,
        original_course_videos_only=True, additional_video_used=False,
        interpretation='Visible proposal differences and model-history sensitivity; no accuracy gain established')
    write(args.output_dir / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dense-dir', 'phase-dir', 'baseline-bundle', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == '__main__':
    main()
