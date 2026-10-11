"""Continue one fresh measured ball identity to its same-shot observation limit."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from scripts.replay_fresh_ball_experiment import producer_record, replay_provenance
from scripts.run_fresh_ball_experiment import CODE_FILES, load, save
from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_ball_continuity_probe import collision_geometry
from scripts.probe_goal_event_tracking import source_time_indices
from src.ball_continuity_v4 import ContinuityConfig, track_ball_continuity
from src.ball_direction_corridor import DirectionCorridorConfig
from src.final_results import sha256
from src.fresh_ball_endpoint import EndpointPolicy, classify_endpoint
from src.fresh_cache_provenance import validate_cached_neural_producer
from src.fresh_event_analysis import contact_pose, local_event_cache, pose_anchors
from src.runtime import inference_provenance


def fixed_source_prefix(cache, candidate):
    """Reuse the exact independent seed/departure identity, never a new seed."""
    selected = [next(row for row in candidate['stationary_ball_observations']
                     if row['frame'] == candidate['contact_bracket_frames'][0])]
    selected += candidate['departure_ball_observations'][:2]
    measured = {(row['frame'], row['detection_id']): row
                for rows in cache['detections'].values() for row in rows}
    prefix = [measured[(row['frame'], row['detection_id'])] for row in selected]
    if any(row.get('stabilized_xy') is None for row in prefix):
        return None
    return prefix


def boundary_evidence(pose, camera_path, source):
    result = {'reason': pose['shot_end_reason'], 'replay_verified': False,
              'source': 'fresh_pose_source_camera_shot_boundary', 'camera_file_sha256': None}
    if camera_path is None or not camera_path.is_file():
        return result
    camera = load(camera_path)
    if (camera.get('clip_id') != source['clip_id'] or camera.get('source_mapping') != source or
            camera.get('new_neural_inference') is not True):
        raise ValueError('Endpoint camera evidence is not fresh source-bound inference')
    result['camera_file_sha256'] = sha256(camera_path)
    following = next((row for row in camera['segments']
                      if row['start_frame'] <= pose['shot_end_frame'] < row['end_frame_exclusive']), None)
    if following and following['start_frame'] == pose['shot_end_frame']:
        speed = following.get('playback_evidence', {})
        result['following_view_candidate'] = following.get('view_candidate')
        result['following_playback_evidence'] = speed
        if speed.get('status') == 'slow_motion_candidate':
            result['reason'] = 'replay_cut_candidate'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path, help='Optional fresh camera experiment root')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clips', nargs='+')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a new endpoint experiment directory')
    manifest_path = args.clips_dir/'source_manifest.json'
    manifest_hash = sha256(manifest_path)
    records = {row['clip_id']: row for row in load(manifest_path)['clips']}
    ids = args.clips or list(records)
    if len(ids) != len(set(ids)) or any(cid not in records for cid in ids):
        raise ValueError('Endpoint IDs must be distinct source-manifest members')
    if not args.clips and len(ids) != 39:
        raise ValueError('The complete endpoint cohort must contain exactly 39 clips')
    files = list(dict.fromkeys(CODE_FILES+['scripts/run_fresh_endpoint_experiment.py',
        'scripts/replay_fresh_ball_experiment.py', 'scripts/run_fresh_phase_experiment.py',
        'src/fresh_cache_provenance.py', 'src/fresh_ball_endpoint.py', 'src/final_results.py']))
    hashes = {name: sha256(ROOT/name) for name in files}
    policy = EndpointPolicy()
    save(args.output_dir/'protocol.json', {'runtime': inference_provenance('cpu', 'cpu'),
        'cohort_clip_ids': ids, 'source_manifest_sha256': manifest_hash, 'code_sha256': hashes,
        'thresholds': asdict(policy), 'new_neural_inference': False,
        'new_neural_inference_this_run': False, 'fresh_neural_cache_used': True,
        'reference_labels_used_for_prediction': False, 'manual_ball_points_used': False,
        'scope': 'fixed_stationary_departure_identity_extended_to_same_camera_shot_boundary',
        'cross_cut_association_permitted': False, 'final_position_verified': False,
        'human_verified': False, 'accuracy_validated': False})
    cv2.setNumThreads(3)
    rows = []
    for cid in ids:
        source = records[cid]
        filename = Path(source['clip_filename'])
        if filename.name != str(filename):
            raise ValueError('Source clip filename must be a basename')
        video, pose_path = args.clips_dir/filename, args.pose_dir/f'{cid}.json'
        if sha256(video) != source['clip_sha256']:
            raise ValueError('Endpoint video hash mismatch')
        pose = load(pose_path)
        if (pose.get('new_neural_inference') is not True or
                pose.get('reference_labels_used_for_prediction') is not False or
                pose.get('inference_identity', {}).get('source_sha256') != source['clip_sha256']):
            raise ValueError('Endpoint pose is not fresh source-only inference')
        pose['video_path'] = str(video)
        bounded, binding = _bound_contact(args.ball_dir, source, pose_path, pose, manifest_hash)
        if bounded['contact_frame'] is None:
            result = {'clip_id': cid, 'status': 'unknown', 'reason': 'independent_contact_not_established',
                      'final_position': None, 'final_position_verified': False,
                      'contact_binding': binding, 'new_neural_inference': False}
        else:
            ball_result = load(args.ball_dir/'results'/f'{cid}.json')
            candidate = load(args.ball_dir/'contact_search'/f'{cid}.json')['selected_candidate']
            raw_path = args.ball_dir/'detections'/f'{cid}.json'
            raw = load(raw_path)
            original = ball_result.get('original_neural_inference_producer') or {}
            receipt_path = args.ball_dir/original['receipt_file'] if original else args.ball_dir/'results'/f'{cid}.json'
            producer = producer_record(receipt_path, raw_path, load(receipt_path))
            identity = {'source_sha256': source['clip_sha256'], 'pose_sha256': sha256(pose_path),
                        'source_manifest_sha256': manifest_hash,
                        'model_sha256': producer['inference_identity']['model_sha256'],
                        'settings': {'analysis_mode': 'cpu_fixed_prefix_endpoint', 'thresholds': asdict(policy)},
                        'code_sha256': hashes, 'source_cache_sha256': producer['detections_sha256']}
            metadata = replay_provenance(identity, producer)
            for src, destination in ((raw_path, args.output_dir/producer['detections_file']),
                                     (receipt_path, args.output_dir/producer['receipt_file'])):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, destination)
            validate_cached_neural_producer({**metadata, 'clip_id': cid}, args.output_dir)
            raw.update(video_path=str(video), **metadata)
            event_pose = contact_pose(pose, candidate)
            before = (candidate['contact_frame']-candidate['contact_bracket_frames'][0])/raw['fps']
            after = (raw['end_frame_exclusive']-1-candidate['contact_frame'])/raw['fps']
            cache = local_event_cache(raw, event_pose, candidate, before, after)
            prefix = fixed_source_prefix(cache, candidate)
            anchors, goal = pose_anchors(event_pose, cache)
            regions = collision_geometry(cache, {})
            if prefix is None or goal is None:
                path, tracking = [], {'tracking_status': 'abstain_missing_prefix_camera_or_goal_reference'}
            else:
                frame_metadata = {frame: {'coordinate_supported': cache['camera']['transforms'].get(frame) is not None}
                                  for frame in cache['detections']}
                path, tracking = track_ball_continuity(cache['detections'], cache['kick_frame'], anchors, goal,
                    initial_path=prefix, frame_metadata=frame_metadata, collision_regions=regions,
                    config=ContinuityConfig(), corridor_config=DirectionCorridorConfig())
                if [(row['frame'], row['detection_id']) for row in path[:len(prefix)]] != [(row['frame'], row['detection_id']) for row in prefix]:
                    raise AssertionError('Endpoint continuation replaced its independent identity prefix')
            timing = source_time_indices(cache)
            boundary = boundary_evidence(pose, args.camera_dir/'clips'/f'{cid}.json' if args.camera_dir else None, source)
            result = classify_endpoint(path, cache['detections'], cache['camera']['transforms'], cache['kick_frame'],
                cache['fps'], cache['end_frame_exclusive'], cache['width'], cache['height'],
                identity_supported=tracking.get('tracking_status') == 'candidate',
                boundary_reason=boundary['reason'], ambiguous_seed_frames=candidate.get('seed_ambiguous_frames', []),
                people=cache['people'], duplicate_frames=[frame for frame, row in timing.items() if row['duplicate_frame']], policy=policy)
            input_path = args.output_dir/'inputs'/f'{cid}.json'
            save(input_path, {'clip_id': cid, 'start_frame': cache['start_frame'],
                 'end_frame_exclusive': cache['end_frame_exclusive'], 'contact': candidate,
                 'detections': cache['detections'], 'camera': cache['camera'], 'source_time_indices': timing})
            result.update(**metadata, clip_id=cid, source_mapping=source,
                          trajectory=path, tracking=tracking, contact_binding=binding,
                          contact_frame=cache['kick_frame'], contact_bracket_frames=candidate['contact_bracket_frames'],
                          boundary_evidence=boundary, complete_same_shot_detection_cache_used=True,
                          endpoint_input_sha256=sha256(input_path))
        result.update(source_sha256=source['clip_sha256'], pose_sha256=sha256(pose_path),
                      source_manifest_sha256=manifest_hash, human_verified=False,
                      accuracy_validated=False, reference_labels_used_for_prediction=False)
        if sha256(video) != source['clip_sha256']:
            raise AssertionError('Endpoint source changed during reanalysis')
        save(args.output_dir/'results'/f'{cid}.json', result)
        row = {'clip_id': cid, 'status': result['status'], 'reason': result['reason'],
               'last_observed_frame': result.get('last_observed_frame'),
               'stationary_end_candidate': result.get('stationary_end_candidate'),
               'final_position': None, 'new_neural_inference': False}
        rows.append(row)
        print(json.dumps(row), flush=True)
        save(args.output_dir/'summary.json', {'rows': rows, 'processed_clips': len(rows),
            'requested_clips': len(ids), 'status_counts': dict(Counter(row['status'] for row in rows)),
            'new_neural_inference_clips': 0, 'verified_final_position_clips': 0,
            'human_verified': False, 'accuracy_validated': False})


if __name__ == '__main__':
    main()
