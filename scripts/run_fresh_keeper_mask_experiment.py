"""Plan or infer NEW automatic-role SAM2 masks on a fixed source-pixel ROI."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.run_fresh_keeper_height_experiment import _association, _camera
from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write
from src.fresh_keeper_height import _mask_guard, _points, connected_camera_prefix
from src.goalkeeper_mask_experiment import pose_seed
from src.goalkeeper_role_association import select_goalkeeper_role


def plan_keeper_mask(pose, event, camera, role, association):
    kick, fps = event['kick_frame'], pose['fps']
    start, end = pose['shot_start_frame'], pose['shot_end_frame']
    _, stop = connected_camera_prefix(camera, start, end)
    result = dict(status='abstain', new_neural_inference=False, reason=None,
                  source_shape=[pose['frame_height'], pose['frame_width']],
                  keeper_track_id=role.get('seed_track_id'), contact_frame_candidate=kick,
                  contact_bracket_frames=event['contact'].get('contact_bracket_frames'),
                  camera_prefix_stop_frame=stop if stop < end else None)
    if stop <= kick:
        result['reason'] = 'camera_reference_lost_before_contact'
        return result
    if role.get('status') != 'candidate_supported':
        result['reason'] = 'automatic_keeper_role_not_supported'
        return result
    if role['seed_track_id'] == event['contact'].get('contact_associated_track_id'):
        result['reason'] = 'keeper_role_is_independent_striker'
        return result
    observations = (association or {}).get('observations', [])
    seeds = [row for row in observations if kick-round(.5*fps) <= row['frame'] < min(kick, stop)
             and row.get('association_source_track_id') in role.get('same_person_seed_ids', [role['seed_track_id']])
             and _points(row, [5, 6, 11, 12]) is not None]
    if not seeds:
        result['reason'] = 'no_confident_pre_contact_keeper_torso_seed'
        return result
    seed = min(seeds, key=lambda row: row['frame'])
    goals = [goal['selected']['bbox'] for goal in event['goal_proposals'].values()
             if goal.get('selected') and isinstance(goal['selected'].get('bbox'), list)]
    if not goals:
        result['reason'] = 'no_source_goal_roi'
        return result
    boxes = goals+[row['bbox'] for row in observations if seed['frame'] <= row['frame'] < stop]
    boxes = np.asarray(boxes, float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError('Invalid source boxes for fixed keeper inference ROI')
    goals = np.asarray(goals, float)
    goal_width = float(np.max(goals[:, 2])-np.min(goals[:, 0]))
    goal_height = float(np.max(goals[:, 3])-np.min(goals[:, 1]))
    lower = boxes[:, :2].min(0)-[.25*goal_width, .25*goal_height]
    upper = boxes[:, 2:].max(0)+[.25*goal_width, .60*goal_height]
    height, width = result['source_shape']
    x0, y0 = max(0, int(np.floor(lower[0]/2)*2)), max(0, int(np.floor(lower[1]/2)*2))
    x1, y1 = min(width, int(np.ceil(upper[0]/2)*2)), min(height, int(np.ceil(upper[1]/2)*2))
    x1 -= (x1-x0) % 2
    y1 -= (y1-y0) % 2
    if x1-x0 < 32 or y1-y0 < 32 or stop-seed['frame'] < 6:
        result['reason'] = 'insufficient_same_shot_mask_window_or_roi'
        return result
    result.update(status='planned_source_role_mask_inference', reason=None,
                  seed_frame=seed['frame'], end_frame_exclusive=stop,
                  expected_frame_count=stop-seed['frame'], inference_roi_xyxy=[x0, y0, x1, y1],
                  seed_bbox_source_xyxy=seed['bbox'],
                  seed_bbox_roi_xyxy=(np.asarray(seed['bbox'])-[x0, y0, x0, y0]).tolist(),
                  seed_torso_keypoints_source=np.asarray(seed['keypoints'])[[5, 6, 11, 12]].tolist(),
                  seed_min_torso_confidence=float(np.min(np.asarray(seed['keypoint_confidences'])[[5, 6, 11, 12]])),
                  prompt_source='automatic_pre_contact_goal_line_role_bbox_not_human_identity',
                  frame_mapping='mask local frame i -> source clip frame seed_frame+i',
                  human_verified=False, physical_ground_plane_calibrated=False)
    return result


def _lossless_roi_video(video, destination, plan, fps):
    x0, y0, x1, y1 = plan['inference_roi_xyxy']
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'FFV1'), fps, (x1-x0, y1-y0))
    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise ValueError('Lossless FFV1 source crop video backend unavailable')
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, plan['seed_frame'])
        for frame in range(plan['seed_frame'], plan['end_frame_exclusive']):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode original keeper source frame {frame}')
            crop = np.ascontiguousarray(image[y0:y1, x0:x1])
            writer.write(crop)
            frames.append(dict(frame=frame, crop_bgr_sha256=hashlib.sha256(crop.tobytes()).hexdigest()))
    finally:
        cap.release()
        writer.release()
    check = cv2.VideoCapture(str(destination))
    try:
        for row in frames:
            ok, crop = check.read()
            if not ok or hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest() != row['crop_bgr_sha256']:
                raise ValueError('Encoded ROI video does not preserve exact decoded source pixels')
        if check.read()[0]:
            raise ValueError('Unexpected extra lossless ROI video frame')
    finally:
        check.release()
    return frames


def run_masks(model_path, device, imgsz, video, destination, plan, pose, role, association, identity):
    import torch
    import ultralytics
    from ultralytics.models.sam import SAM2VideoPredictor
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no silent CPU fallback')
    destination.mkdir(parents=True, exist_ok=True)
    crop_path = destination/'source_roi_lossless.avi'
    frames = _lossless_roi_video(video, crop_path, plan, pose['fps'])
    settings = dict(model=str(model_path.resolve()), device=device, imgsz=imgsz,
                    task='segment', mode='predict', conf=.1, save=False, verbose=False,
                    vid_stride=1, batch=1)
    predictor = SAM2VideoPredictor(overrides=settings)
    runtime = dict(torch_version=torch.__version__, ultralytics_version=ultralytics.__version__,
                   predictor_python_sha256=_hash(Path(inspect.getfile(SAM2VideoPredictor))),
                   requested_device=device, cuda_available=torch.cuda.is_available())
    generator = predictor(source=str(crop_path), stream=True, bboxes=[plan['seed_bbox_roi_xyxy']])
    x0, y0, x1, y1 = plan['inference_roi_xyxy']
    by_frame = {row['frame']: row for row in association['observations']}
    own = set(role.get('same_person_seed_ids', [role['seed_track_id']]))
    records, qcs, previous, initial_area, stop = [], {}, None, None, None
    started = time.monotonic()
    for index, prediction in enumerate(generator):
        if index >= len(frames):
            raise ValueError('SAM2 emitted extra frames beyond the source crop')
        model = predictor.model
        if not (hasattr(model, 'memory_attention') and hasattr(model, 'memory_encoder')):
            raise ValueError('SAM2 temporal model is required, not SAM1/detection/pose')
        if tuple(prediction.orig_shape) != (y1-y0, x1-x0):
            raise ValueError('SAM2 source shape does not match exact fixed ROI')
        masks = prediction.masks.data.cpu().numpy() if prediction.masks is not None else None
        if masks is None or len(masks) != 1 or masks.shape[1:] != (y1-y0, x1-x0):
            raise ValueError('One original-ROI-coordinate keeper mask per frame is required')
        local = masks[0] > .5
        mask = np.zeros(plan['source_shape'], bool)
        mask[y0:y1, x0:x1] = local
        frame = frames[index]['frame']
        observed = by_frame.get(frame)
        excluded = np.zeros_like(mask, np.uint8)
        for track in pose['tracks']:
            if track['track_id'] in own or track['track_id'] == (observed or {}).get('association_source_track_id'):
                continue
            for row in track['observations']:
                if row['frame'] == frame:
                    excluded |= pose_seed(row, mask.shape)
        if initial_area is None:
            initial_area = int(mask.sum())
        qc = _mask_guard(mask, observed, first=previous is None, previous=previous,
                         scale=max(20, plan['seed_bbox_source_xyxy'][3]-plan['seed_bbox_source_xyxy'][1]),
                         initial_area=initial_area, excluded=excluded, roi=plan['inference_roi_xyxy'])
        qcs[str(frame)] = qc
        if stop is None and not qc['valid']:
            stop = dict(frame=frame, reason=qc['reason'])
        if stop is None:
            previous = mask
        image_path = destination/f'{frame:06d}.png'
        if not cv2.imwrite(str(image_path), mask.astype(np.uint8)*255):
            raise OSError('Could not write source-coordinate binary keeper mask')
        records.append(dict(frame=frame, path=image_path.name, sha256=_hash(image_path),
                            crop_bgr_sha256=frames[index]['crop_bgr_sha256']))
    if len(records) != len(frames):
        raise ValueError('SAM2 mask frame count differs from exact source window')
    manifest = dict(schema_version='fresh-keeper-height-mask-v1',
        clip_id=pose['clip_id'], source_sha256=identity['source_clip_sha256'],
        pose_sha256=identity['pose_sha256'], event_cache_sha256=identity['event_cache_sha256'],
        keeper_track_id=role['seed_track_id'], coordinate_system='full_source_pixels',
        frame_height=pose['frame_height'], frame_width=pose['frame_width'],
        new_neural_inference=True, reference_labels_used_for_prediction=False,
        model_sha256=_hash(model_path), seed_frame=plan['seed_frame'], masks=records,
        inference_roi_xyxy=plan['inference_roi_xyxy'], source_plan=plan,
        source_roi_video_sha256=_hash(crop_path), mask_identity_qc=qcs,
        strict_identity_prefix_stop=stop, runtime=runtime,
        inference_identity=identity, inference_elapsed_seconds=time.monotonic()-started,
        human_verified=False, accuracy_validated=False, physical_ground_plane_calibrated=False)
    _write(destination/'mask_manifest.json', manifest)
    return dict(status='new_source_role_mask_inference_completed', masks=len(records),
                strict_identity_prefix_stop=stop, manifest_sha256=_hash(destination/'mask_manifest.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path)
    parser.add_argument('--clips', nargs='+', required=True, help='Explicit small pilot IDs')
    parser.add_argument('--model', type=Path, help='Existing official Ultralytics-compatible SAM2 checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--plan-only', action='store_true', help='No model import/loading or neural inference')
    args = parser.parse_args(argv)
    if not args.plan_only and (args.model is None or not args.model.is_file()):
        parser.error('Actual inference requires an existing --model checkpoint')
    if args.imgsz != 1024:
        parser.error('This pilot uses the SAM2 trained 1024-pixel encoder contract')
    if any(args.output_dir.resolve() == path.resolve() for path in
           (args.clips_dir, args.pose_dir, args.ball_dir, args.camera_dir) if path):
        parser.error('Output cannot overwrite source/inference inputs')
    manifest_path = args.clips_dir/'source_manifest.json'
    manifest_hash = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = {row['clip_id']: row for row in manifest['clips']}
    if (len(sources) != len(manifest['clips']) or len(set(args.clips)) != len(args.clips)
            or any(cid not in sources for cid in args.clips)):
        raise ValueError('Pilot IDs must be distinct source-manifest members')
    code_paths = ('scripts/run_fresh_keeper_mask_experiment.py', 'scripts/run_fresh_keeper_height_experiment.py',
                  'src/fresh_keeper_height.py', 'src/goalkeeper_role_association.py',
                  'src/goalkeeper_sam2.py', 'src/goalkeeper_mask_experiment.py',
                  'src/person_appearance.py', 'src/motion_stabilization.py',
                  'scripts/run_fresh_phase_experiment.py', 'scripts/run_fresh_pitch_experiment.py')
    code_hashes = {name: _hash(ROOT/name) for name in code_paths}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'camera_motion').mkdir(exist_ok=True)
    protocol = dict(schema_version='fresh-keeper-mask-pilot-v1', source_manifest_sha256=manifest_hash,
                    code_sha256=code_hashes, cohort=args.clips, plan_only=args.plan_only,
                    model_sha256=_hash(args.model) if args.model and args.model.is_file() else None,
                    requested_device=args.device, imgsz=args.imgsz, reference_labels_read=False,
                    human_verified=False, held_out_test=False)
    _write(args.output_dir/'protocol.json', protocol)
    cv2.setNumThreads(2)
    plans = []
    for cid in args.clips:
        source, pose_path = sources[cid], args.pose_dir/(cid+'.json')
        video = args.clips_dir/source['clip_filename']
        if _hash(video) != source['clip_sha256']:
            raise ValueError('Source clip bytes do not match manifest')
        pose, provenance = _pose_input(pose_path, source)
        plan = dict(clip_id=cid, status='abstain', reason=provenance['status'], new_neural_inference=False)
        if pose is not None:
            selection = pose['shot_selection']
            chosen = [attempt for attempt in selection['attempts']
                      if attempt.get('shot_index') == selection['selected_shot_index']]
            if len(chosen) != 1 or chosen[0].get('scene_rejections') != []:
                raise ValueError('Mask source shot scene guards are incomplete/rejected')
            if pose.get('reference_labels_used_for_prediction') is not False:
                raise ValueError('Pose inference used reference labels')
            contact, contact_provenance = _bound_contact(args.ball_dir, source, pose_path, pose, manifest_hash)
            if contact['status'] == 'tentative':
                event_path = args.ball_dir/'event_windows'/(cid+'.json')
                event = json.loads(event_path.read_text())
                identity = dict(source_manifest_sha256=manifest_hash, source_clip_sha256=source['clip_sha256'],
                                pose_sha256=_hash(pose_path), event_cache_sha256=_hash(event_path),
                                shot_interval=[pose['shot_start_frame'], pose['shot_end_frame']], code_sha256=code_hashes)
                camera, camera_provenance = _camera(video, pose, identity, args.camera_dir,
                    args.output_dir/'camera_motion'/(cid+'.json'))
                geometries = {int(frame): goal for frame, goal in event['goal_proposals'].items()}
                role = select_goalkeeper_role(pose, geometries, contact['contact_frame'])
                association = _association(video, pose, role, contact['contact_frame'])
                plan = plan_keeper_mask(pose, event, camera, role, association)
                plan.update(clip_id=cid, inference_identity=identity, pose_provenance=provenance,
                            contact_provenance=contact_provenance, camera_provenance=camera_provenance,
                            model_loaded_and_checked=False, reference_labels_read=False,
                            automatic_role=role, automatic_association=association)
                if plan['status'] == 'planned_source_role_mask_inference' and not args.plan_only:
                    plan['inference'] = run_masks(args.model, args.device, args.imgsz, video,
                        args.output_dir/cid, plan, pose, role, association, identity)
                    plan['new_neural_inference'] = True
                    plan['model_loaded_and_checked'] = True
            else:
                plan['reason'] = contact['warnings'][0]
        _write(args.output_dir/(cid+'_plan.json'), plan)
        plans.append(dict(clip_id=cid, status=plan['status'], reason=plan.get('reason'),
                          expected_frame_count=plan.get('expected_frame_count'),
                          new_neural_inference=plan['new_neural_inference']))
        print(json.dumps(plans[-1]), flush=True)
    _write(args.output_dir/'summary.json', dict(rows=plans, events=len(plans),
        plan_only=args.plan_only, neural_inference_events=sum(row['new_neural_inference'] for row in plans),
        human_verified=False, accuracy_validated=False))
    return plans


if __name__ == '__main__':
    main()
