"""Fresh camera inference with source-bound native live pose references."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts import run_fresh_camera_experiment as v1
from scripts.run_fresh_phase_experiment import _bound_contact
from src.camera_timing_v2 import assign_kit_roles, jersey_hist
from src.fresh_camera_reference import live_reference_action, source_role_observations
from src.runtime import inference_provenance, resolve_device


def bound_reference(args, record, manifest_hash):
    cid = record['clip_id']
    video = args.clips_dir / record['clip_filename']
    pose_path = args.pose_dir / f'{cid}.json'
    pose = v1.read(pose_path)
    if (Path(record['clip_filename']).name != record['clip_filename'] or
            v1.sha256(video) != record['clip_sha256'] or
            pose.get('clip_id') != cid or pose.get('new_neural_inference') is not True or
            pose.get('reference_labels_used_for_prediction') is not False or
            pose.get('inference_identity', {}).get('source_sha256') != record['clip_sha256']):
        raise ValueError('Camera source/pose producer mismatch')
    pose['_pose_file_sha256'] = v1.sha256(pose_path)
    contact, provenance = _bound_contact(args.ball_dir, record, pose_path, pose, manifest_hash)
    anchor = {'frame': None, 'source': 'source_bound_contact_unknown', 'human_verified': False,
              'pose_proposal_for_comparison_only': pose.get('kick_frame'),
              'binding_provenance': provenance}
    ball_result = {}
    if v1.qualified_scene(pose) and contact.get('status') == 'tentative':
        ball_result = v1.read(args.ball_dir / 'results' / f'{cid}.json')
        actor = contact['contact_associated_track_id']
        anchor.update(frame=contact['contact_frame'],
            source='fresh_ball_stationary_seed_and_departure_candidate',
            contact_bracket_frames=contact['contact_bracket_frames'],
            contact_associated_track_id=actor, contact_frame_is_estimated=True,
            exact_contact_frame_established=False,
            original_pose_selected_track_id=pose['selected_track_id'],
            actor_reselected_by_ball_evidence=actor != pose['selected_track_id'])
    return video, pose_path, pose, ball_result, anchor


def reference_window(pose, frame, shot=None):
    start = max(pose['shot_start_frame'], frame-round(1.4*pose['fps']))
    end = min(pose['shot_end_frame'], frame+round(pose['fps']))
    if shot:
        start = max(start, shot['start_frame'])
        end = min(end, shot['end_frame_exclusive'])
    return start, end


def native_live_people(frame, frame_id, pose, roles, references):
    height, width = frame.shape[:2]
    people = []
    for track in pose['tracks']:
        observation = next((row for row in track['observations'] if row['frame'] == frame_id), None)
        if observation is None:
            continue
        bbox = observation['bbox']
        confidence = np.asarray(observation['keypoint_confidences'])
        person = {**observation, 'height_fraction': (bbox[3]-bbox[1])/height,
                  'area_fraction': (bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(width*height),
                  'face_visible_fraction': float(np.mean(confidence[:5] > .4)),
                  'jersey_hist': jersey_hist(frame, bbox, observation['keypoints'], confidence),
                  'measurement_source': 'same_shot_source_pose_producer',
                  'source_pose_sha256': pose['_pose_file_sha256']}
        assign_kit_roles([person], references)
        person['role'] = 'unknown'
        for role, observations in roles.items():
            row = observations.get(frame_id)
            if row is not None and row['source_track_id'] == track['track_id']:
                person.update(role=role+'_candidate', role_source='same_shot_bound_source_track')
        people.append(person)
    return people


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--clips', nargs='+')
    parser.add_argument('--plan-reference-only', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a new experiment directory')
    source_path = args.clips_dir / 'source_manifest.json'
    source = v1.read(source_path)
    manifest_hash = v1.sha256(source_path)
    records = {record['clip_id']: record for record in source['clips']}
    ids = args.clips or list(records)
    if (len(records) != len(source['clips']) or len(ids) != len(set(ids)) or
            any(cid not in records for cid in ids) or (not args.clips and len(ids) != 39)):
        raise ValueError('Distinct source-manifest members and a full 39-clip cohort are required')
    device, model, model_check = None, None, None
    if not args.plan_reference_only:
        if not args.model:
            parser.error('--model is required for fresh cross-shot camera inference')
        from ultralytics import YOLO
        device = resolve_device(args.device)
        model = YOLO(str(args.model))
        model_check = v1.validate_pose_model(model)
        model._visionii_device = device
    protocol = {'version': 'source_bound_live_pose_reference_v2', 'cohort_clip_ids': ids,
        'source_manifest_sha256': manifest_hash, 'reference_labels_used_for_prediction': False,
        'human_verified': False, 'accuracy_validated': False, 'held_out_test': False,
        'plan_reference_only': args.plan_reference_only, 'model_task_validation': model_check,
        'runtime': inference_provenance(args.device, device, [args.model]) if model else None,
        'settings': {'cross_shot_imgsz': 960, 'confidence': .25, 'action_sampling_fps_target': 8,
            'live_reference': 'exact_source_pose_track_under_contact_and_shot_binding'},
        'code_sha256': {name: v1.sha256(ROOT/name) for name in (
            'scripts/run_fresh_camera_experiment_v2.py', 'scripts/run_fresh_camera_experiment.py',
            'scripts/run_fresh_phase_experiment.py', 'src/fresh_camera_reference.py',
            'src/camera_timing_v2.py', 'src/camera_view_guard.py', 'src/camera_partial_alignment.py',
            'src/fresh_cache_provenance.py', 'src/goalkeeper_pipeline.py',
            'src/goal_geometry.py', 'src/goalkeeper_role_association.py',
            'scripts/camera_timing_context.py', 'scripts/camera_timing_features.py')}}
    v1.save(args.output_dir/'protocol.json', protocol)
    clips = []
    cv2.setNumThreads(2)
    for index, cid in enumerate(ids):
        started = time.monotonic()
        record = records[cid]
        video, pose_path, pose, ball, anchor = bound_reference(args, record, manifest_hash)
        contact = anchor['frame']
        action = None
        if contact is not None:
            start, end = reference_window(pose, contact)
            action = live_reference_action(pose, ball, ball, start, end)
        if args.plan_reference_only:
            result = {'clip_id': cid, 'source_mapping': record, 'contact_anchor': anchor,
                'reference_action': action, 'new_neural_inference': False,
                'fresh_pose_producer_measurements_reused': action is not None,
                'human_verified': False, 'accuracy_validated': False}
            v1.save(args.output_dir/'reference_plans'/f'{cid}.json', result)
            clips.append(result)
            print(json.dumps({'completed': index+1, 'total': len(ids), 'clip_id': cid,
                'support': action['contiguous_support'] if action else None}), flush=True)
            continue
        scan = v1.scan_video(video)
        v1.validate_timeline(scan, record)
        cap = cv2.VideoCapture(str(video))
        reference_pose = {**pose, 'selected_track_id': anchor.get('contact_associated_track_id', pose['selected_track_id'])}
        references = v1.appearance_references(cap, reference_pose, contact) if contact is not None else {
            'kicker_histograms': [], 'keeper_histograms': [], 'identity_verified': False,
            'samples': [], 'rejection': 'no_supported_contact_reference_window'}
        reference = next((shot for shot in scan['segments'] if contact is not None and
                          shot['start_frame'] <= contact < shot['end_frame_exclusive']), None)
        roles = {'kicker': {}, 'keeper': {}}
        if reference:
            start, end = reference_window(pose, contact, reference)
            action = live_reference_action(pose, ball, ball, start, end)
            roles, _ = source_role_observations(pose, ball, ball)
        segments = []
        neural_calls = 0
        for shot in scan['segments']:
            samples = []
            for frame_id in shot['sample_frame_ids']:
                frame = v1.get_frame(cap, frame_id)
                if contact is not None and pose['shot_start_frame'] <= frame_id < pose['shot_end_frame']:
                    people = native_live_people(frame, frame_id, pose, roles, references)
                    measurement_source = 'same_shot_source_pose_producer'
                else:
                    people = assign_kit_roles(v1.infer(model, frame, device), references)
                    neural_calls += 1
                    measurement_source = 'fresh_cross_shot_camera_neural_inference'
                goal = v1.detect_goal_frame(frame, max_goal_width_fraction=.98, max_post_height_fraction=.85)
                selected = goal.get('selected')
                samples.append({'frame': frame_id, 'width': scan['width'], 'height': scan['height'],
                    'people': people, 'measurement_source': measurement_source, 'goal': selected,
                    'goal_camera': v1.goal_camera_hypotheses(selected['corners'], scan['width'], scan['height']) if selected else {},
                    'foreground_net_cue': v1.near_net(frame),
                    'closeup_context': v1.closeup_context(frame, people)})
            view = v1.classify_guarded(samples)
            query = None
            speed = {'status': 'unknown', 'reason': 'same_action_alignment_not_attempted',
                     'identity_verified': False, 'reference_speed_verified': False}
            if reference and shot['start_frame'] > contact and view['view_candidate'] in v1.REPLAY_VIEWS:
                query = v1.action_sequence(cap, model, references, shot['start_frame']+2,
                    min(shot['end_frame_exclusive']-2, shot['start_frame']+round(8*scan['fps'])), scan['fps'], device)
                neural_calls += len(query['raw_pose_observations'])
                speed = v1.speed_evidence(action, query, view['view_candidate'])
            source_start = record['source_start_frame']+shot['start_frame']
            source_end = record['source_start_frame']+shot['end_frame_exclusive']
            segments.append({**shot, 'samples': samples, 'view_evidence': view,
                'view_candidate': view['view_candidate'], 'playback_evidence': speed,
                'task_class_candidate': v1.combine_task_class(view['view_candidate'], speed),
                'query_action': query, 'course_source_start_frame': source_start,
                'course_source_end_frame_exclusive': source_end,
                'course_source_start_sec': source_start/record['fps'],
                'course_source_end_sec_exclusive': source_end/record['fps'],
                'human_verified': False, 'accuracy_validated': False})
        cap.release()
        if v1.sha256(video) != record['clip_sha256'] or v1.sha256(pose_path) != pose['_pose_file_sha256']:
            raise AssertionError('Camera video or source pose changed during inference')
        result = {'clip_id': cid, 'source_mapping': record, 'pose_sha256': pose['_pose_file_sha256'],
            'fps': scan['fps'], 'source_frames': scan['frame_count'], 'camera_scan': scan,
            'timeline_coverage': scan['timeline_coverage'], 'segments': segments,
            'references': references, 'reference_action': action, 'contact_anchor': anchor,
            'new_neural_inference': neural_calls > 0, 'new_neural_inference_calls': neural_calls,
            'fresh_pose_producer_measurements_reused': action is not None,
            'human_verified': False, 'accuracy_validated': False,
            'elapsed_sec': round(time.monotonic()-started, 3)}
        v1.save(args.output_dir/'clips'/f'{cid}.json', result)
        clips.append(result)
        print(json.dumps({'completed': index+1, 'total': len(ids), 'clip_id': cid,
                          'segments': len(segments), 'elapsed_sec': result['elapsed_sec']}), flush=True)
    all_segments = [segment for clip in clips for segment in clip.get('segments', [])]
    v1.save(args.output_dir/'summary.json', {'events': len(clips),
        'source_bound_contact_reference_clips': sum(clip['reference_action'] is not None for clip in clips),
        'contiguous_kicker_reference_at_least_10': sum(
            (clip['reference_action'] or {}).get('contiguous_support', {}).get('kicker', {}).get('longest_block_samples', 0) >= 10 for clip in clips),
        'contiguous_keeper_reference_at_least_10': sum(
            (clip['reference_action'] or {}).get('contiguous_support', {}).get('keeper', {}).get('longest_block_samples', 0) >= 10 for clip in clips),
        'segments': len(all_segments),
        'view_candidates': dict(Counter(row['view_candidate'] for row in all_segments)),
        'task_class_candidates': dict(Counter(row['task_class_candidate'] for row in all_segments)),
        'playback_candidates': dict(Counter(row['playback_evidence']['status'] for row in all_segments)),
        'new_neural_inference_clips': sum(clip['new_neural_inference'] for clip in clips),
        'human_verified': False, 'accuracy_validated': False})


if __name__ == '__main__':
    main()
