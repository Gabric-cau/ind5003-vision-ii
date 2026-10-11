"""Reanalyse immutable fresh neural ball detections on CPU, without inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from scripts.run_fresh_ball_experiment import CODE_FILES, load, save, write_contact_sheet
from src.ball_pipeline import _read_window
from src.final_results import sha256
from src.fresh_cache_provenance import validate_cached_neural_producer
from src.fresh_event_analysis import (analyze_fresh_ball, analyze_fresh_keeper, contact_pose,
                                     contact_window, local_event_cache, search_fresh_contact)
from src.runtime import inference_provenance


def producer_record(receipt_path, detections_path, producer):
    return {'inference_identity': deepcopy(producer['inference_identity']),
            'receipt_file': f'producer_results/{producer["clip_id"]}.json',
            'receipt_sha256': sha256(receipt_path),
            'detections_file': f'detections/{producer["clip_id"]}.json',
            'detections_sha256': sha256(detections_path),
            'new_neural_inference': True, 'fresh_model_inference': True}


def replay_provenance(identity, producer):
    return {'inference_identity': identity, 'new_neural_inference': False,
            'new_neural_inference_this_run': False, 'fresh_neural_cache_used': True,
            'fresh_model_inference_this_run': False,
            'original_neural_inference_producer': producer,
            'source_cache_sha256': producer['detections_sha256'],
            'reference_labels_used_for_prediction': False,
            'manual_ball_points_used': False, 'human_verified': False, 'accuracy_validated': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--producer-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pre-sec', type=float, default=.45)
    parser.add_argument('--post-sec', type=float, default=1.5)
    parser.add_argument('--clips', nargs='+')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.pre_sec <= 2 or not 0 < args.post_sec <= 3:
        raise ValueError('Invalid bounded analysis settings')
    if args.output_dir.resolve() == args.producer_dir.resolve():
        raise ValueError('Rule reanalysis needs a separate artifact directory')
    manifest_path = args.clips_dir/'source_manifest.json'
    manifest = load(manifest_path)
    records = {row['clip_id']: row for row in manifest['clips']}
    if len(records) != len(manifest['clips']):
        raise ValueError('Duplicate source clip IDs')
    ids = args.clips or list(records)
    if len(ids) != len(set(ids)) or any(cid not in records for cid in ids):
        raise ValueError('Requested clips must be distinct members of the source manifest')
    if not args.clips and len(ids) != 39:
        raise ValueError('The complete course cohort must contain exactly 39 clips')
    files = list(dict.fromkeys(CODE_FILES+['scripts/replay_fresh_ball_experiment.py',
                                        'src/fresh_cache_provenance.py', 'src/final_results.py']))
    code_hashes = {name: sha256(ROOT/name) for name in files}
    settings = {'execution_mode': 'cpu_rule_reanalysis', 'pre_sec': args.pre_sec,
                'post_sec': args.post_sec}
    manifest_hash = sha256(manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(args.output_dir/'protocol.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'runtime': inference_provenance('cpu', 'cpu'), 'settings': settings,
        'source_manifest_sha256': manifest_hash, 'code_sha256': code_hashes,
        'cohort_clip_ids': ids, 'new_neural_inference': False,
        'new_neural_inference_this_run': False, 'fresh_neural_cache_used': True,
        'cache_policy': 'byte_exact_original_fresh_NN_detections_and_producer_receipts',
        'reference_labels_used_for_prediction': False, 'old_ball_trajectory_input': False,
        'human_verified': False, 'accuracy_validated': False,
        'complete_raw_to_all_features': False, 'final_ball_position_established': False})
    cv2.setNumThreads(3)
    rows = []
    for index, cid in enumerate(ids, 1):
        started = time.monotonic()
        record = records[cid]
        filename = Path(record['clip_filename'])
        if filename.name != str(filename) or filename.suffix.lower() != '.mp4':
            raise ValueError('Source manifest must contain local MP4 basenames')
        video, pose_path = args.clips_dir/filename, args.pose_dir/f'{cid}.json'
        source_hash, pose_hash = sha256(video), sha256(pose_path)
        if source_hash != record['clip_sha256']:
            raise ValueError(f'Source clip hash mismatch: {cid}')
        pose = load(pose_path)
        if (pose.get('clip_id') != cid or pose.get('new_neural_inference') is not True or
                pose.get('reference_labels_used_for_prediction') is not False or
                pose.get('inference_identity', {}).get('source_sha256') != source_hash):
            raise ValueError(f'Pose is not fresh source-only inference: {cid}')
        pose = {**pose, 'video_path': str(video)}
        receipt_path = args.producer_dir/'results'/f'{cid}.json'
        detections_path = args.producer_dir/'detections'/f'{cid}.json'
        receipt, raw_cache = load(receipt_path), load(detections_path)
        producer = producer_record(receipt_path, detections_path, receipt)
        identity = {'source_sha256': source_hash, 'pose_sha256': pose_hash,
                    'model_sha256': producer['inference_identity'].get('model_sha256'),
                    'source_manifest_sha256': manifest_hash,
                    'settings': settings, 'code_sha256': code_hashes,
                    'producer_receipt_sha256': producer['receipt_sha256'],
                    'source_cache_sha256': producer['detections_sha256']}
        metadata = replay_provenance(identity, producer)
        original = {**metadata, 'clip_id': cid,
                    'original_neural_inference_producer': {**producer, 'receipt_file': f'results/{cid}.json'}}
        validate_cached_neural_producer(original, args.producer_dir)
        interval, reason = contact_window(pose, record)
        if interval is None or interval != (raw_cache['start_frame'], raw_cache['end_frame_exclusive']):
            raise ValueError(f'Producer cache does not match the qualified selected shot: {cid}: {reason}')
        if set(raw_cache['detections']) != {str(fid) for fid in range(*interval)}:
            raise ValueError(f'Producer detection coverage is incomplete: {cid}')
        target = args.output_dir/'results'/f'{cid}.json'
        if target.exists():
            result = load(target)
            if not args.resume or result.get('inference_identity') != identity:
                raise FileExistsError(f'Existing reanalysis result is not reusable: {target}')
            validate_cached_neural_producer(result, args.output_dir)
        else:
            for source, destination in ((receipt_path, args.output_dir/producer['receipt_file']),
                                        (detections_path, args.output_dir/producer['detections_file'])):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            validate_cached_neural_producer({**metadata, 'clip_id': cid}, args.output_dir)
            cache = deepcopy(raw_cache)
            cache.update(video_path=str(video), **metadata)
            frame_ids, decoded, fps, width, height = _read_window(str(video), interval[0], interval[1]-1)
            if len(frame_ids) != interval[1]-interval[0] or (width, height) != (cache['width'], cache['height']) or abs(fps-cache['fps']) > 1e-4:
                raise ValueError('Source cache/frame geometry mismatch during reanalysis')
            frames = dict(zip(frame_ids, decoded))
            search = search_fresh_contact(cache, pose, frames)
            search_path = args.output_dir/'contact_search'/f'{cid}.json'
            save(search_path, search)
            if search['selected_candidate'] is None:
                result = {'clip_id': cid, 'status': 'abstain_ball_contact',
                          'reason': 'no_independent_bounded_stationary_ball_departure',
                          'rule_decision': {'rule_outcome': 'unknown'}, 'trajectory': [], 'final_position': None,
                          'keeper': {'status': 'abstain', 'dive_direction': 'unknown', 'dive_height': 'unknown'}}
            else:
                candidate = search['selected_candidate']
                event_pose = contact_pose(pose, candidate)
                event = local_event_cache(cache, event_pose, candidate, args.pre_sec, args.post_sec)
                event_path = args.output_dir/'event_windows'/f'{cid}.json'
                save(event_path, event)
                local_frames = {fid: frame for fid, frame in frames.items()
                                if event['start_frame'] <= fid < event['end_frame_exclusive']}
                result = analyze_fresh_ball(event, event_pose, local_frames, args.output_dir/'inputs')
                result['keeper'] = analyze_fresh_keeper(event_pose, event)
                result['media'] = write_contact_sheet(event, result, local_frames, args.output_dir/'media'/f'{cid}.jpg')
                result.update(status='candidate_event', event_cache_sha256=sha256(event_path))
            result.update(**metadata, source_mapping=record,
                          contact_search_sha256=sha256(search_path),
                          independent_contact_search={key: value for key, value in search.items() if key != 'candidate_windows'},
                          elapsed_sec=round(time.monotonic()-started, 3))
            save(args.output_dir/'contacts'/f'{cid}.json', {
                **metadata, 'clip_id': cid, 'fps': record['fps'], 'source_mapping': record,
                'source_sha256': source_hash, 'pose_sha256': pose_hash,
                'contact': search['selected_candidate'],
                'independent_contact_search': result['independent_contact_search'],
                'ball_departure': result.get('ball_departure')})
            save(target, result)
        if sha256(video) != source_hash or sha256(pose_path) != pose_hash:
            raise AssertionError('Source or pose changed during reanalysis')
        if any(sha256(ROOT/name) != code_hashes[name] for name in files):
            raise AssertionError('Analysis code changed during reanalysis')
        selected = result['independent_contact_search']['selected_candidate'] or {}
        row = {'clip_id': cid, 'status': result['status'],
               'contact_frame': selected.get('contact_frame'), 'contact_bracket_frames': selected.get('contact_bracket_frames'),
               'contact_associated_track_id': selected.get('contact_associated_track_id'),
               'trajectory_points': len(result['trajectory']), 'rule_outcome': result['rule_decision']['rule_outcome'],
               'new_neural_inference': False, 'fresh_neural_cache_used': True}
        rows.append(row)
        save(args.output_dir/'summary.json', {'rows': rows, 'processed_clips': len(rows), 'requested_clips': len(ids),
                                            'new_neural_inference_clips': 0, 'fresh_neural_cache_clips': len(rows),
                                            'all_requested_clips_processed': len(rows) == len(ids),
                                            'human_verified': False, 'accuracy_validated': False})
        print(json.dumps({'completed': index, 'total': len(ids), **row}), flush=True)


if __name__ == '__main__':
    main()
