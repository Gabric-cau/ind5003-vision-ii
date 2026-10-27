"""Fresh event-window detections and retained ball/keeper rules on course clips."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.ball_pipeline import _read_window
from src.ball_specialist import predict_images
from src.final_results import sha256
from src.fresh_event_analysis import contact_window, partition_predictions, merge_ball_candidates, analyze_fresh_ball, analyze_fresh_keeper, search_fresh_contact, local_event_cache, contact_pose
from src.goal_geometry import detect_goal_frame
from src.runtime import inference_provenance, resolve_device

CODE_FILES = ['scripts/run_fresh_ball_experiment.py', 'src/fresh_event_analysis.py', 'src/ball_specialist.py',
              'src/ball_pipeline.py', 'src/ball_continuity_v4.py', 'src/ball_continuity.py',
              'src/ball_direction_corridor.py', 'src/ball_mask_seed.py', 'src/ball_shot_focus.py',
              'src/ball_keeper_rebound.py', 'src/ball_goal_decision.py', 'src/event_ball_tracking.py',
              'src/goal_event_tracker_v2.py', 'src/goal_plane_rectification.py', 'src/goal_geometry.py',
              'src/motion_stabilization.py', 'src/goalkeeper_pipeline.py', 'src/goalkeeper_analysis.py',
              'src/goalkeeper_role_association.py', 'src/goalkeeper_observation_guard.py',
              'src/goalkeeper_direction_evidence.py', 'src/goalkeeper_pose_direction.py', 'src/person_appearance.py',
              'src/field_masks.py',
              'src/kick_contact.py',
              'scripts/probe_goal_event_tracking.py', 'scripts/run_ball_continuity_probe.py',
              'scripts/run_ball_shot_focus_review.py', 'scripts/review_ball_keeper_rebound.py',
              'scripts/detect_all_ball_goals.py', 'src/runtime.py']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_detection_model(model):
    names = model.names
    name = lambda index: names.get(index, names.get(str(index))) if isinstance(names, dict) else names[index]
    if model.task != 'detect' or name(0) != 'person' or name(32) != 'sports ball':
        raise ValueError('Fresh ball inference requires a COCO detection model with person=0 and sports ball=32')


def validate_ball_specialist(model):
    names = ({int(key): value for key, value in model.names.items()}
             if isinstance(model.names, dict) else dict(enumerate(model.names)))
    if model.task != 'detect' or len(names) != 1 or names.get(0) not in {'ball', 'sports ball'}:
        raise ValueError('Ball specialist must be a single-class detection model with ball=0')


def partition_specialist(ids, frames, specialist, settings):
    config = settings['ball_specialist']
    rows = predict_images(specialist, frames, config['mode'], config['imgsz'],
                          settings['device'], settings['batch_size'])
    predictions = [SimpleNamespace(boxes=SimpleNamespace(
        xyxy=np.asarray([item['bbox'] for item in items], dtype=float).reshape(-1, 4),
        conf=np.asarray([item['confidence'] for item in items], dtype=float),
        cls=np.zeros(len(items), dtype=int))) for items in rows]
    return partition_predictions(ids, predictions, frames, mode='ball_specialist_'+config['mode'],
                                 ball_class_id=0, person_class_id=None)


def propose_crops(pose, goals, kick, width, height):
    options = [(abs(int(f)-kick), int(f), g['selected']) for f, g in goals.items() if g.get('selected')]
    crops = [('full', [0, 0, width, height])]
    if options:
        _, _, goal = min(options, key=lambda q: (q[0], q[1]))
        x0, y0, x1, y1 = goal['bbox']
        gw, gh = x1-x0, y1-y0
        crops.append(('goal_roi', [max(0, int(x0-.6*gw)), max(0, int(y0-.3*gh)),
                                  min(width, int(x1+.6*gw)), min(height, int(y1+.8*gh))]))
    excluded = set((pose.get('goal_line_role_exclusions') or {}).get('excluded_track_ids', []))
    field_ids = {row['track_id'] for row in pose.get('track_candidates', [])
                 if float(row.get('field_support') or 0) >= .20 and row['track_id'] not in excluded}
    points = [o['keypoints'][j] for track in pose['tracks'] if track['track_id'] in field_ids
              for o in track.get('observations', []) for j in (15, 16)
              if len(o.get('keypoint_confidences', [])) > j and o['keypoint_confidences'][j] >= .25]
    if points:
        pts = np.asarray(points)
        pad = max(80, .12*height)
        bbox = [max(0, int(pts[:, 0].min()-pad)), max(0, int(pts[:, 1].min()-pad)),
                min(width, int(pts[:, 0].max()+pad)), min(height, int(pts[:, 1].max()+pad))]
        if 0 < (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) < .80*width*height:
            crops.append(('automatic_ankle_roi', bbox))
    return [(mode, bbox) for mode, bbox in crops if bbox[2] > bbox[0] and bbox[3] > bbox[1]]


def detect_event(video, pose, record, model, settings, *, ball_specialist=None):
    interval, reason = contact_window(pose, record)
    if interval is None:
        return None, {}, reason
    if ball_specialist is not None:
        validate_ball_specialist(ball_specialist)
    start, end = interval
    ids, frames, fps, width, height = _read_window(str(video), start, end-1)
    if len(ids) != end-start:
        raise ValueError('Fresh event source did not fully decode')
    if not np.isclose(fps, pose['fps'], rtol=1e-5):
        raise ValueError('Decoded video/pose fps mismatch')
    goals = {str(fid): detect_goal_frame(frame) for fid, frame in zip(ids, frames)}
    proposal = pose.get('kick_frame')
    crops = propose_crops(pose, goals, proposal if proposal is not None else (start+end)//2, width, height)
    detections, people, raw = ({str(fid): [] for fid in ids} for _ in range(3))
    for mode, bbox in crops:
        x0, y0, x1, y1 = bbox
        for index in range(0, len(frames), settings['batch_size']):
            batch = frames[index:index+settings['batch_size']]
            batch_ids = ids[index:index+settings['batch_size']]
            views = [frame[y0:y1, x0:x1] for frame in batch]
            predictions = model.predict(views, imgsz=settings['imgsz'], conf=settings['confidence'],
                        classes=[0, 32], device=settings['device'], verbose=False)
            balls, persons, observations = partition_predictions(batch_ids, predictions, batch, offset=(x0, y0), mode=mode)
            for fid in batch_ids:
                key = str(fid)
                detections[key].extend(balls[key]); people[key].extend(persons[key]); raw[key].extend(observations[key])
    if ball_specialist is not None:
        for index in range(0, len(frames), settings['batch_size']):
            batch = frames[index:index+settings['batch_size']]
            batch_ids = ids[index:index+settings['batch_size']]
            balls, _, observations = partition_specialist(batch_ids, batch, ball_specialist, settings)
            for fid in batch_ids:
                key = str(fid)
                detections[key].extend(balls[key])
                raw[key].extend(observations[key])
    # A long shot can have several camera-chain breaks. Contact search and
    # event analysis establish their own fresh, explicitly bounded chains.
    camera = {'transforms': {}, 'scope': 'estimated_independently_in_global_contact_search_windows'}
    for fid, frame in zip(ids, frames):
        key = str(fid)
        merged = merge_ball_candidates(detections[key], fid)
        for candidate in merged:
            candidate['stabilized_xy'] = None
            candidate['motion_mean_abs_gray'] = None
        detections[key] = merged
    cache = {'clip_id': pose['clip_id'], 'status': 'cached_candidates', 'video_path': str(video),
            'source_sha256': record['clip_sha256'], 'start_frame': start, 'end_frame_exclusive': end,
            'pose_kick_proposal': pose.get('kick_frame'), 'fps': fps, 'width': width, 'height': height,
            'source_mapping': record, 'contact': {'contact_frame': None,
             'contact_associated_track_id': pose['selected_track_id'], 'source': reason, 'human_verified': False},
            'goal_proposals': goals, 'goal_roi': next((bbox for mode, bbox in crops if mode == 'goal_roi'), None),
            'camera': camera, 'detections': detections, 'people': people, 'raw_neural_detections': raw,
            'inference_crops': [{'mode': mode, 'bbox': bbox, 'coordinate_transform': 'translation_to_original_clip_pixels'} for mode, bbox in crops],
            'source_times': {str(f): {'clip_frame': f, 'clip_time_sec': f/fps,
               'course_recording_frame': record['source_start_frame']+f,
               'course_recording_time_sec': (record['source_start_frame']+f)/record['fps']} for f in ids},
            'fresh_model_inference': True, 'old_ball_trajectory_input': False}
    if ball_specialist is not None:
        cache['ball_specialist'] = {**settings['ball_specialist'], 'combination': 'merge_with_coco_ball_candidates',
                                    'person_detection_source': 'original_coco_model',
                                    'existing_ball_candidate_gates_retained': True,
                                    'coordinate_transform': 'tile_translation_to_original_clip_pixels'}
    return cache, dict(zip(ids, frames)), reason


def write_contact_sheet(cache, result, frames, path):
    selected = {q['frame']: q for q in result['trajectory']}
    samples = [min(cache['end_frame_exclusive']-1, max(cache['start_frame'], cache['kick_frame']+round(s*cache['fps'])))
               for s in (-.2, 0, .12, .3, .5, .8)]
    tiles = []
    for fid in samples:
        scene = cv2.resize(frames[fid], (640, 360))
        scale = np.array([640/cache['width'], 360/cache['height']])
        goal = cache['goal_proposals'].get(str(fid), {}).get('selected')
        if goal:
            cv2.polylines(scene, [np.rint(np.asarray(goal['corners'])*scale).astype(np.int32)], True, (220, 190, 50), 1)
        for q in cache['detections'][str(fid)]:
            cv2.circle(scene, tuple(np.rint([q['x']*scale[0], q['y']*scale[1]]).astype(int)), 5, (220, 130, 60), 1)
        if fid in selected:
            q = selected[fid]
            cv2.circle(scene, tuple(np.rint([q['x']*scale[0], q['y']*scale[1]]).astype(int)), 9, (70, 240, 80), 2)
        cv2.putText(scene, f'{cache["clip_id"]} f{fid}', (10, 24), 0, .45, (245, 245, 245), 1, cv2.LINE_AA)
        tiles.append(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack([np.hstack(tiles[j:j+2]) for j in (0, 2, 4)])):
        raise ValueError('Contact sheet write failed')
    return {'contact_sheet': path.name, 'sha256': sha256(path), 'sample_frames': samples,
            'visual_review_status': 'pending', 'human_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ball-specialist', type=Path,
                        help='Optional trained single-class ball detector; original COCO model remains active')
    parser.add_argument('--specialist-mode', choices=['full', 'tiles'], default='full')
    parser.add_argument('--specialist-imgsz', type=int, default=1280)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--pre-sec', type=float, default=.45)
    parser.add_argument('--post-sec', type=float, default=1.5)
    parser.add_argument('--clips', nargs='+')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if min(args.batch_size, args.imgsz, args.specialist_imgsz) <= 0 or not 0 < args.pre_sec <= 2 or not 0 < args.post_sec <= 3:
        raise ValueError('Invalid bounded inference settings')
    source_manifest = args.clips_dir/'source_manifest.json'
    source = load(source_manifest)
    records = {record['clip_id']: record for record in source['clips']}
    if len(records) != len(source['clips']):
        raise ValueError('Duplicate source clip IDs')
    ids = args.clips or list(records)
    if len(ids) != len(set(ids)) or any(cid not in records for cid in ids):
        raise ValueError('Requested clips must be distinct members of the source manifest')
    if not args.clips and len(ids) != 39:
        raise ValueError('The complete course cohort must contain exactly 39 clips')
    device = resolve_device(args.device)
    settings = {'imgsz': args.imgsz, 'batch_size': args.batch_size, 'device': device,
                'confidence': .01, 'pre_sec': args.pre_sec, 'post_sec': args.post_sec}
    code_hashes = {name: sha256(ROOT/name) for name in CODE_FILES}
    model_hash, manifest_hash = sha256(args.model), sha256(source_manifest)
    model_paths = [args.model]
    specialist_hash = None
    if args.ball_specialist is not None:
        specialist_hash = sha256(args.ball_specialist)
        model_paths.append(args.ball_specialist)
        settings['ball_specialist'] = {'mode': args.specialist_mode, 'imgsz': args.specialist_imgsz,
            'weights_sha256': specialist_hash, 'native_ball_class_id': 0, 'canonical_ball_class_id': 32,
            'person_class_id': None, 'confidence': .05, 'nms_iou': .6, 'max_det': 20,
            'tile_edge': 960, 'tile_overlap': .25, 'includes_full_frame': True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = inference_provenance(args.device, device, model_paths)
    protocol.update(created_at_utc=datetime.now(timezone.utc).isoformat(), settings=settings,
                    source_manifest_sha256=manifest_hash, cohort_clip_ids=ids, code_sha256=code_hashes,
                    old_caches_used=False, reference_labels_used_for_prediction=False, manual_ball_points_used=False,
                    human_verified=False, accuracy_validated=False, held_out_test=False,
                    scope='fresh_full_selected_shot_detection_global_contact_search_and_bounded_event_rules',
                    full_raw_clip_coverage=False, complete_raw_to_all_features=False,
                    final_ball_position_established=False, whole_ball_crossing_established=False)
    save(args.output_dir/'protocol.json', protocol)
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    validate_detection_model(model)
    ball_specialist = None
    if args.ball_specialist is not None:
        ball_specialist = YOLO(str(args.ball_specialist))
        validate_ball_specialist(ball_specialist)
    cv2.setNumThreads(3)
    rows = []
    for index, cid in enumerate(ids, 1):
        record = records[cid]
        filename = Path(record['clip_filename'])
        if filename.name != str(filename) or filename.suffix.lower() != '.mp4':
            raise ValueError('Source manifest must contain local MP4 basenames')
        video, pose_path = args.clips_dir/filename, args.pose_dir/f'{cid}.json'
        if sha256(video) != record['clip_sha256']:
            raise ValueError(f'Source clip hash mismatch: {cid}')
        pose = load(pose_path)
        if pose.get('clip_id') != cid or pose.get('new_neural_inference') is not True:
            raise ValueError(f'Pose is not a fresh per-clip inference result: {cid}')
        if pose.get('reference_labels_used_for_prediction') is not False:
            raise ValueError(f'Pose prediction provenance is not source-only: {cid}')
        if pose.get('inference_identity', {}).get('source_sha256') != record['clip_sha256']:
            raise ValueError(f'Pose source hash mismatch: {cid}')
        pose = {**pose, 'video_path': str(video)}
        identity = {'source_sha256': record['clip_sha256'], 'pose_sha256': sha256(pose_path),
                    'model_sha256': model_hash, 'source_manifest_sha256': manifest_hash,
                    'settings': settings, 'code_sha256': code_hashes}
        if specialist_hash is not None:
            identity['ball_specialist_sha256'] = specialist_hash
        target = args.output_dir/'results'/f'{cid}.json'
        if target.exists():
            result = load(target)
            if not args.resume or result.get('inference_identity') != identity:
                raise FileExistsError(f'Existing fresh event result is not reusable: {target}')
        else:
            started = time.monotonic()
            cache, frames, reason = detect_event(video, pose, record, model, settings,
                                                ball_specialist=ball_specialist)
            if cache is None:
                result = {'clip_id': cid, 'status': 'abstain_contact', 'reason': reason,
                          'new_neural_inference': False, 'rule_decision': {'rule_outcome': 'unknown'},
                          'trajectory': [], 'final_position': None,
                          'keeper': {'status': 'abstain', 'dive_direction': 'unknown', 'dive_height': 'unknown'}}
            else:
                cache['pose_sha256'] = identity['pose_sha256']
                cache['runtime'] = inference_provenance(args.device, device, model_paths, [video])
                cache['inference_identity'] = identity
                save(args.output_dir/'detections'/f'{cid}.json', cache)
                search = search_fresh_contact(cache, pose, frames)
                save(args.output_dir/'contact_search'/f'{cid}.json', search)
                if search['selected_candidate'] is None:
                    result = {'clip_id': cid, 'status': 'abstain_ball_contact', 'reason': 'no_independent_stationary_ball_departure',
                              'rule_decision': {'rule_outcome': 'unknown'}, 'trajectory': [], 'final_position': None,
                              'keeper': {'status': 'abstain', 'dive_direction': 'unknown', 'dive_height': 'unknown'}}
                else:
                    event_pose = contact_pose(pose, search['selected_candidate'])
                    event = local_event_cache(cache, event_pose, search['selected_candidate'], args.pre_sec, args.post_sec)
                    event['pose_sha256'] = identity['pose_sha256']
                    save(args.output_dir/'event_windows'/f'{cid}.json', event)
                    local_frames = {fid: frame for fid, frame in frames.items() if event['start_frame'] <= fid < event['end_frame_exclusive']}
                    result = analyze_fresh_ball(event, event_pose, local_frames, args.output_dir/'inputs')
                    result['keeper'] = analyze_fresh_keeper(event_pose, event)
                    result['media'] = write_contact_sheet(event, result, local_frames, args.output_dir/'media'/f'{cid}.jpg')
                    result.update(status='candidate_event', event_cache_sha256=sha256(args.output_dir/'event_windows'/f'{cid}.json'))
                search_summary = {key: value for key, value in search.items() if key != 'candidate_windows'}
                result.update(new_neural_inference=True, independent_contact_search=search_summary,
                              contact_search_sha256=sha256(args.output_dir/'contact_search'/f'{cid}.json'),
                              source_cache_sha256=sha256(args.output_dir/'detections'/f'{cid}.json'))
            result.update(inference_identity=identity, source_mapping=record,
                          elapsed_sec=round(time.monotonic()-started, 3),
                          human_verified=False, accuracy_validated=False, reference_labels_used_for_prediction=False)
            if sha256(video) != record['clip_sha256']:
                raise AssertionError('Source video changed during fresh inference')
            selected_contact = (result.get('independent_contact_search') or {}).get('selected_candidate')
            save(args.output_dir/'contacts'/f'{cid}.json', {
                'clip_id': cid, 'fps': record['fps'], 'source_mapping': record,
                'source_sha256': identity['source_sha256'], 'pose_sha256': identity['pose_sha256'],
                'inference_identity': identity, 'contact': selected_contact,
                'independent_contact_search': result.get('independent_contact_search'),
                'ball_departure': result.get('ball_departure'),
                'new_neural_inference': result['new_neural_inference'],
                'reference_labels_used_for_prediction': False, 'human_verified': False,
                'accuracy_validated': False})
            save(target, result)
        row = {'clip_id': cid, 'status': result['status'], 'new_neural_inference': result['new_neural_inference'],
               'reason': result.get('reason'), 'rule_outcome': result['rule_decision']['rule_outcome'],
               'trajectory_points': len(result['trajectory']), 'keeper_direction': result['keeper']['dive_direction'],
               'keeper_height': result['keeper']['dive_height'], 'elapsed_sec': result['elapsed_sec']}
        rows.append(row)
        save(args.output_dir/'summary.json', {'rows': rows, 'processed_clips': len(rows), 'requested_clips': len(ids),
                         'new_neural_inference_clips': sum(q['new_neural_inference'] for q in rows),
                         'all_requested_clips_processed': len(rows) == len(ids),
                         'human_verified': False, 'accuracy_validated': False})
        print(json.dumps({'completed': index, 'total': len(ids), **row}), flush=True)


if __name__ == '__main__':
    main()
