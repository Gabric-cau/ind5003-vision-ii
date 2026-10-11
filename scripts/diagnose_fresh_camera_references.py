"""Compare source-bound camera reference measurements without new inference."""

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from scripts import run_fresh_camera_experiment as camera
from scripts.run_fresh_camera_experiment_v2 import bound_reference, reference_window
from src.camera_partial_alignment import contiguous
from src.fresh_camera_reference import live_reference_action, source_role_observations


REPRESENTATIVES = ['col_eng_p01_falcao_col9', 'cro_den_p01_eriksen_den10',
                   'cro_den_p05_krohndehli_den2', 'esp_rus_p01_iniesta_esp6',
                   'rus_cro_p02_brozovic_cro11']


def longest(rows):
    return max(map(len, contiguous(rows, .35)), default=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a new diagnostic directory')
    manifest_path = args.clips_dir/'source_manifest.json'
    source = camera.read(manifest_path)
    manifest_hash = camera.sha256(manifest_path)
    results = []
    for record in source['clips']:
        cid = record['clip_id']
        video, pose_path, pose, ball, anchor = bound_reference(args, record, manifest_hash)
        cache_path = args.camera_dir/'clips'/f'{cid}.json'
        cached = camera.read(cache_path)
        if (cached['source_mapping']['clip_sha256'] != record['clip_sha256'] or
                cached['pose_sha256'] != camera.sha256(pose_path)):
            raise ValueError('Camera cache is not from the same source and pose producer')
        row = {'clip_id': cid, 'camera_cache_sha256': camera.sha256(cache_path),
               'source_sha256': record['clip_sha256'], 'pose_sha256': pose['_pose_file_sha256'],
               'contact_frame': anchor['frame'], 'contact_binding': anchor['binding_provenance'],
               'new_neural_inference': False, 'reference_labels_used_for_prediction': False,
               'human_verified': False, 'accuracy_validated': False}
        old_action = cached.get('reference_action') or {'roles': {'kicker': [], 'keeper': []}}
        row['old_reference_support'] = {role: {'usable_samples': len(rows), 'longest_block_samples': longest(rows)}
                                         for role, rows in old_action['roles'].items()}
        row['sample_rejection_counts'] = dict(Counter(
            reason for segment in cached['segments'] for reason in segment['view_evidence']['sample_reasons']))
        if anchor['frame'] is None:
            results.append(row)
            continue
        shot = next(segment for segment in cached['segments']
                    if segment['start_frame'] <= anchor['frame'] < segment['end_frame_exclusive'])
        start, end = reference_window(pose, anchor['frame'], shot)
        new_action = live_reference_action(pose, ball, ball, start, end)
        row['native_reference_support'] = new_action['contiguous_support']
        row['query_alignment_reanalysis'] = []
        for segment in cached['segments']:
            if segment.get('query_action') is not None:
                evidence = camera.speed_evidence(new_action, segment['query_action'], segment['view_candidate'])
                row['query_alignment_reanalysis'].append({'segment_id': segment['segment_id'],
                    'view_candidate': segment['view_candidate'], 'original_evidence': segment['playback_evidence'],
                    'new_evidence': evidence, 'query_neural_measurements_changed': False})
        if cid in REPRESENTATIVES:
            raw = (cached.get('reference_action') or {}).get('raw_pose_observations', [])
            raw_frame = min(raw, key=lambda sample: abs(sample['frame']-anchor['frame'])) if raw else None
            frame_id = raw_frame['frame'] if raw_frame else anchor['frame']
            capture = cv2.VideoCapture(str(video))
            frame = camera.get_frame(capture, frame_id)
            capture.release()
            roles, _ = source_role_observations(pose, ball, ball)
            panel = frame.copy()
            for person in (raw_frame or {}).get('people', []):
                x0, y0, x1, y1 = map(round, person['bbox'])
                cv2.rectangle(panel, (x0, y0), (x1, y1), (30, 40, 235), 2)
                cv2.putText(panel, 'camera v1 '+person['role'], (x0, max(20, y0-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 40, 235), 1)
            actual_rows = []
            for role, observations in roles.items():
                actual = observations.get(frame_id)
                if actual is None:
                    continue
                actual_rows.append({'role': role, 'source_track_id': actual['source_track_id'],
                                    'frame': frame_id, 'bbox': actual['bbox']})
                x0, y0, x1, y1 = map(round, actual['bbox'])
                colour = (235, 185, 15) if role == 'kicker' else (50, 220, 50)
                cv2.rectangle(panel, (x0, y0), (x1, y1), colour, 2)
                cv2.putText(panel, 'native '+role+' id='+str(actual['source_track_id']),
                            (x0, min(panel.shape[0]-8, y1+18)), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1)
            cv2.putText(panel, cid+' source frame='+str(frame_id)+' AI diagnostics, not human gold',
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            image_path = args.output_dir/f'{cid}_frame{frame_id:06d}.jpg'
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), panel):
                raise IOError('Cannot save source QA image')
            row['source_frame_diagnostic'] = {'frame': frame_id,
                'image_file': image_path.name, 'image_sha256': camera.sha256(image_path),
                'camera_people_count': len((raw_frame or {}).get('people', [])),
                'native_actual_observations': actual_rows, 'human_verified': False}
        results.append(row)
    queries = [query for row in results for query in row.get('query_alignment_reanalysis', [])]
    camera.save(args.output_dir/'diagnosis.json', {'rows': results, 'processed_clips': len(results),
        'cohort_clip_ids': [row['clip_id'] for row in results],
        'native_kicker_contiguous_at_least_10': sum(
            row.get('native_reference_support', {}).get('kicker', {}).get('longest_block_samples', 0) >= 10 for row in results),
        'native_keeper_contiguous_at_least_10': sum(
            row.get('native_reference_support', {}).get('keeper', {}).get('longest_block_samples', 0) >= 10 for row in results),
        'query_reanalysis_status_counts': dict(Counter(query['new_evidence']['status'] for query in queries)),
        'new_neural_inference': False, 'reference_labels_used_for_prediction': False,
        'human_verified': False, 'accuracy_validated': False,
        'diagnostic_script_sha256': camera.sha256(Path(__file__))})


if __name__ == '__main__':
    main()
