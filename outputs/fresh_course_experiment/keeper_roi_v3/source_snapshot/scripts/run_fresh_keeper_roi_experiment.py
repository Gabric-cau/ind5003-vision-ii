"""Compare fresh rotated keeper-ROI pose inference with the frozen full-frame pose."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.run_fresh_keeper_height_experiment import _association, _camera, _diagnostic
from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write as write_json
from src.fresh_keeper_height import analyze_height
from src.goalkeeper_role_association import deduplicate_pose_people, select_goalkeeper_role, unrotate_pose
from src.runtime import inference_provenance, resolve_device


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def plan_roi(pose, event, role):
    if role.get('status') != 'candidate_supported':
        return None
    kick = event['kick_frame']
    seed = next(t for t in pose['tracks'] if t['track_id'] == role['seed_track_id'])
    pre = [o for o in seed['observations'] if kick-round(.5*pose['fps']) <= o['frame'] < kick]
    goals = [g['selected']['bbox'] for g in event['goal_proposals'].values()
             if g.get('selected') and isinstance(g['selected'].get('bbox'), list)]
    if len(pre) < 3 or not goals:
        return None
    boxes = np.asarray(goals+[o['bbox'] for o in pre], float)
    scale = float(np.median([o['bbox'][3]-o['bbox'][1] for o in pre]))
    if not np.isfinite(boxes).all() or scale < 20:
        return None
    lower = boxes[:, :2].min(0)-[.4*scale, .4*scale]
    upper = boxes[:, 2:].max(0)+[.4*scale, .8*scale]
    roi = [max(0, int(lower[0])), max(0, int(lower[1])),
           min(pose['frame_width'], int(np.ceil(upper[0]))),
           min(pose['frame_height'], int(np.ceil(upper[1])))]
    return dict(roi_xyxy=roi, seed_track_id=role['seed_track_id'],
                start_frame=max(pose['shot_start_frame'], kick-round(.5*pose['fps'])),
                end_frame_exclusive=min(pose['shot_end_frame'], kick+round(1.5*pose['fps'])+1),
                selection='automatic pre-contact role and visible goal geometry',
                rotations=[0, 90, 270], source_pixels_are_not_upscaled_evidence=True)


def same_pre_person(candidate, original):
    a, b = np.asarray(candidate['bbox'], float), np.asarray(original['bbox'], float)
    area = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    iou = area/max(1., np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area)
    p, q = np.asarray(candidate['keypoints'], float), np.asarray(original['keypoints'], float)
    indices = [5, 6, 11, 12]
    pc, qc = np.asarray(candidate['keypoint_confidences']), np.asarray(original['keypoint_confidences'])
    if iou < .45 or min(pc[indices]) < .25 or min(qc[indices]) < .25:
        return False
    separation = np.linalg.norm(p[indices].reshape(2, 2, 2).mean(1)
                                -q[indices].reshape(2, 2, 2).mean(1), axis=1).max()
    return separation/max(20., b[3]-b[1]) < .20


def augment_pose(pose, detections, plan, kick):
    merged = deepcopy(pose)
    replacements = []
    # Refine existing identities; a fresh ROI alias cannot bypass role exclusions.
    for track in merged['tracks']:
        for index, old in enumerate(track['observations']):
            if old['frame'] < kick and track['track_id'] != plan['seed_track_id']:
                continue
            matches = [o for o in detections.get(old['frame'], []) if same_pre_person(o, old)]
            indices = [5, 6, 11, 12, 15, 16] if old['frame'] < kick else [5, 6, 11, 12]
            best = max([old]+matches, key=lambda o: float(np.min(np.asarray(o['keypoint_confidences'])[indices])))
            if best is not old:
                track['observations'][index] = dict(old, **deepcopy(best),
                    original_source_track_id=track['track_id'])
                if old['frame'] < kick:
                    replacements.append(old['frame'])
    return merged, replacements


def infer_roi(model, video, plan, *, device, imgsz):
    x0, y0, x1, y1 = plan['roi_xyxy']
    cap = cv2.VideoCapture(str(video))
    rows, receipts = {}, []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open keeper source video')
        cap.set(cv2.CAP_PROP_POS_FRAMES, plan['start_frame'])
        for frame in range(plan['start_frame'], plan['end_frame_exclusive']):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode keeper source frame {frame}')
            if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame+1:
                raise ValueError('Decoder frame clock differs from source mapping')
            crop = np.ascontiguousarray(image[y0:y1, x0:x1])
            views = [crop, cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                     cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)]
            predictions = model.predict(views, imgsz=imgsz, conf=.08, iou=.8,
                                        max_det=20, device=device, verbose=False)
            observed = []
            for angle, prediction in zip(plan['rotations'], predictions):
                if prediction.boxes is None or prediction.keypoints is None:
                    continue
                boxes, points = unrotate_pose(prediction.boxes.xyxy.cpu().numpy(),
                    prediction.keypoints.xy.cpu().numpy(), x1-x0, y1-y0, angle)
                boxes[:, [0, 2]] += x0
                boxes[:, [1, 3]] += y0
                points[:, :, 0] += x0
                points[:, :, 1] += y0
                confidence = prediction.keypoints.conf.cpu().numpy()
                for j in range(len(boxes)):
                    # A cropped-off body cannot supply an artificial landing.
                    if min(boxes[j][0]-x0, boxes[j][1]-y0,
                           x1-boxes[j][2], y1-boxes[j][3]) <= 3:
                        continue
                    observed.append(dict(frame=frame, bbox=boxes[j].tolist(),
                        keypoints=points[j].tolist(), keypoint_confidences=confidence[j].tolist(),
                        box_confidence=float(prediction.boxes.conf[j].cpu()),
                        source_roi=f'fresh_goal_role_roi_rotation_{angle}'))
            rows[frame] = deduplicate_pose_people(observed)
            receipts.append(dict(frame=frame, crop_bgr_sha256=hashlib.sha256(crop.tobytes()).hexdigest()))
    finally:
        cap.release()
    return rows, receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['clips-dir', 'pose-dir', 'ball-dir', 'output-dir', 'model']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--clips', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('Use a new ROI experiment output directory')
    manifest_path = args.clips_dir/'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {row['clip_id']: row for row in manifest['clips']}
    if len(set(args.clips)) != len(args.clips) or any(cid not in sources for cid in args.clips):
        raise ValueError('Distinct source-manifest clip IDs required')
    device = resolve_device(args.device)
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(3)
    cv2.setNumThreads(2)
    model = YOLO(str(args.model))
    if model.task != 'pose':
        raise ValueError('A pose model is required')
    code = ['scripts/run_fresh_keeper_roi_experiment.py', 'src/fresh_keeper_height.py',
            'src/goalkeeper_role_association.py', 'scripts/run_fresh_keeper_height_experiment.py']
    protocol = dict(runtime=inference_provenance(args.device, device, [args.model]),
        cohort=args.clips, source_manifest_sha256=_hash(manifest_path),
        code_sha256={name: _hash(ROOT/name) for name in code}, imgsz=args.imgsz,
        reference_labels_used_for_prediction=False, new_neural_inference=True,
        held_out_test=False, human_verified=False, accuracy_validated=False,
        scope='fixed source-role ROI pose ablation; original contact and camera evidence unchanged')
    protocol['identity_policy'] = 'refine same-frame original person only; preserve all original source track IDs'
    _write(args.output_dir/'protocol.json', protocol)
    (args.output_dir/'camera_motion').mkdir()
    rows = []
    for cid in args.clips:
        started = time.monotonic()
        source = sources[cid]
        video, pose_path = args.clips_dir/source['clip_filename'], args.pose_dir/(cid+'.json')
        if _hash(video) != source['clip_sha256']:
            raise ValueError('Source video hash mismatch')
        pose, provenance = _pose_input(pose_path, source)
        if pose is None:
            raise ValueError('Fresh source-bound pose required')
        contact, binding = _bound_contact(args.ball_dir, source, pose_path, pose, _hash(manifest_path))
        if contact['status'] != 'tentative':
            raise ValueError('This ROI pilot requires an independently supported contact')
        event_path = args.ball_dir/'event_windows'/(cid+'.json')
        event = json.loads(event_path.read_text())
        role = select_goalkeeper_role(pose, {int(f): g for f, g in event['goal_proposals'].items()}, event['kick_frame'])
        plan = plan_roi(pose, event, role)
        if plan is None:
            rows.append(dict(clip_id=cid, status='unavailable_automatic_role_roi'))
            continue
        identity = dict(source_manifest_sha256=_hash(manifest_path),
            source_clip_sha256=source['clip_sha256'], pose_sha256=provenance['sha256'],
            event_cache_sha256=_hash(event_path), shot_interval=[pose['shot_start_frame'], pose['shot_end_frame']],
            code_sha256=protocol['code_sha256'])
        camera, camera_provenance = _camera(video, pose, identity, None,
                                           args.output_dir/'camera_motion'/(cid+'.json'))
        baseline_association = _association(video, pose, role, event['kick_frame'])
        baseline = analyze_height(pose, event, camera=camera, role=role, association=baseline_association)
        detections, receipts = infer_roi(model, video, plan, device=device, imgsz=args.imgsz)
        _write(args.output_dir/'detections'/(cid+'.json'), dict(plan=plan,
               detections=detections, source_pixel_receipts=receipts, inference_identity=identity,
               new_neural_inference=True, model_sha256=_hash(args.model)))
        augmented, replacements = augment_pose(pose, detections, plan, event['kick_frame'])
        association = _association(video, augmented, role, event['kick_frame'])
        result = analyze_height(augmented, event, camera=camera, role=role, association=association)
        result.update(clip_id=cid, inference_identity=identity, original_contact_binding=binding,
            camera_provenance=camera_provenance, roi_plan=plan, pre_reference_replaced_frames=replacements,
            new_neural_inference=True, human_verified=False, accuracy_validated=False,
            baseline=baseline, elapsed_seconds=time.monotonic()-started)
        diagnostic = args.output_dir/'source_diagnostics'/(cid+'.jpg')
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        result['source_diagnostic'] = _diagnostic(diagnostic, video, augmented, event, result)
        _write(args.output_dir/'results'/(cid+'.json'), result)
        row = dict(clip_id=cid, baseline_reason=baseline['reason'], roi_reason=result['reason'],
            baseline_observed_frames=baseline.get('observed_keeper_frame_count', 0),
            roi_observed_frames=result.get('observed_keeper_frame_count', 0),
            baseline_signal=baseline['observed_image_plane_height_candidate'],
            roi_signal=result['observed_image_plane_height_candidate'],
            baseline_ordinal=baseline['ordinal_pose_height_candidate'],
            roi_ordinal=result['ordinal_pose_height_candidate'],
            replaced_pre_frames=len(replacements), elapsed_seconds=result['elapsed_seconds'])
        rows.append(row)
        print(json.dumps(row), flush=True)
        _write(args.output_dir/'summary.json', dict(rows=rows, processed_clips=len(rows),
            requested_clips=len(args.clips), new_neural_inference=True,
            roi_signal_counts=dict(Counter(r.get('roi_signal', 'unknown') for r in rows)),
            human_verified=False, accuracy_validated=False, held_out_test=False))


if __name__ == '__main__':
    main()
