"""Run official MediaPipe 33-point inference on hashed cached kicker ROI frames."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kicking_foot_strike_window import kicking_foot_strike_window
from src.mediapipe_shoe_evidence import (
    SHOE_PAIRS, coco_samples, restore_landmarks, shoe_ball_distances, source_body_box,
)
from scripts.run_fresh_foot_experiment import prepare_tasks, timing_and_actor
from src.pose_refinement import select_anchored_detection


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    roi_input = parser.add_mutually_exclusive_group(required=True)
    roi_input.add_argument('--foot-dir', type=Path)
    roi_input.add_argument('--pose-input', type=Path, help='Rich source-only pose JSON for new ROI preparation')
    parser.add_argument('--pose-dir', type=Path, help='Exact original rich pose directory for cached ROI rival identities')
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--clip-id', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--running-mode', choices=('IMAGE', 'VIDEO'), default='IMAGE')
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve experiment evidence')
    if args.foot_dir and args.pose_dir is None:
        parser.error('--pose-dir is required with cached --foot-dir to retain rival-actor evidence')
    contact_path = args.ball_dir / 'contacts' / f'{args.clip_id}.json'
    contact = read(contact_path)
    producer_path = args.foot_dir / 'results' / f'{args.clip_id}.json' if args.foot_dir else args.pose_input
    producer = read(producer_path)
    source = producer['source_mapping']
    video = args.clips_dir / source['clip_filename']
    manifest_path = args.clips_dir / 'source_manifest.json'
    if (producer.get('new_neural_inference') is not True
            or producer.get('reference_labels_used_for_prediction') is not False
            or sha(video) != source['clip_sha256']
            or producer['inference_identity']['source_sha256'] != source['clip_sha256']):
        raise ValueError('Pilot needs original qualified, hashed, source-only ROI/contact evidence')
    manifest_rows = [row for row in read(manifest_path)['clips'] if row['clip_id'] == args.clip_id]
    if len(manifest_rows) != 1 or manifest_rows[0]['clip_sha256'] != source['clip_sha256']:
        raise ValueError('Pilot manifest does not bind the selected source')
    if args.foot_dir:
        crop_path = args.foot_dir / 'poses' / f'{args.clip_id}.json'
        if (producer.get('upstream_scene_rejections') != []
                or producer.get('model_task_validation', {}).get('actual_task') != 'pose'
                or sha(manifest_path) != producer['inference_identity']['source_manifest_sha256']
                or sha(contact_path) != producer['inference_identity']['contact_sha256']):
            raise ValueError('Cached ROI producer lacks matching qualified source/contact identity')
        tasks = [row for row in read(crop_path) if row['reflected'] is False and row.get('selection_reason') == 'accepted']
        bracket = producer['timing']['contact_bracket_frames']
        roi_hash = sha(crop_path)
        pose_path = args.pose_dir / f'{args.clip_id}.json'
        if sha(pose_path) != producer['inference_identity']['pose_sha256']:
            raise ValueError('Rival actors require the exact original pose inference bytes')
        rival_pose = read(pose_path)
        actor_id = producer['timing']['actor_track_id_candidate']
    else:
        if (contact['inference_identity']['pose_sha256'] != sha(producer_path)
                or contact['inference_identity']['source_sha256'] != source['clip_sha256']
                or contact.get('reference_labels_used_for_prediction') is not False):
            raise ValueError('New ROI contact is not tied to the exact pose/source bytes')
        timing = timing_and_actor(producer, contact, False)
        if timing['contact_bracket_frames'] is None:
            raise ValueError('New ROI requires independent ball-departure timing')
        _, all_tasks, rejected = prepare_tasks(video, producer, timing)
        tasks = [row for row in all_tasks if row['reflected'] is False]
        bracket = timing['contact_bracket_frames']
        roi_hash = None
        pose_path, rival_pose, actor_id = producer_path, producer, timing['actor_track_id_candidate']
    if not tasks or len(set(row['frame'] for row in tasks)) != len(tasks):
        raise ValueError('Pilot needs distinct cached normal ROI frames')
    frame_ids = [row['frame'] for row in tasks]
    if (any(not isinstance(frame, int) or isinstance(frame, bool) for frame in frame_ids)
            or frame_ids != sorted(set(frame_ids))):
        raise ValueError('Source frames must be unique increasing integers')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Source clip cannot be opened')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_dimensions = (producer['plan']['source_pixel_dimensions'] if args.foot_dir
                           else [producer['frame_width'], producer['frame_height']])
    if (not np.isfinite(fps) or fps <= 0 or abs(fps - source['fps']) > .01
            or [frame_width, frame_height] != expected_dimensions
            or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != source['frame_count']
            or frame_ids[0] < 0 or frame_ids[-1] >= source['frame_count']):
        raise ValueError('Source decoder clock/dimensions/frame count disagree with provenance')
    timestamps = [round(frame / fps * 1000) for frame in frame_ids]
    if timestamps != sorted(set(timestamps)):
        raise ValueError('VIDEO timestamps must be strictly increasing per independent view')
    bounds = np.asarray([row['crop_xyxy'] for row in tasks])
    fixed_roi = [int(bounds[:, 0].min()), int(bounds[:, 1].min()),
                 int(bounds[:, 2].max()), int(bounds[:, 3].max())]
    if not (0 <= fixed_roi[0] < fixed_roi[2] <= frame_width
            and 0 <= fixed_roi[1] < fixed_roi[3] <= frame_height):
        raise ValueError('Fixed union ROI must lie within source pixels')
    tasks = [dict(row, planned_original_crop_xyxy=row['crop_xyxy'], crop_xyxy=fixed_roi) for row in tasks]
    rival_boxes = {}
    for track in rival_pose['tracks']:
        if track['track_id'] == actor_id:
            continue
        for observation in track['observations']:
            if observation['frame'] in frame_ids:
                rival_boxes.setdefault(observation['frame'], []).append(observation['bbox'])
    payload = contact.get('contact', contact)
    first, departure = bracket
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_names = ('scripts/run_fresh_shoe_pose_experiment.py', 'src/mediapipe_shoe_evidence.py',
                  'src/kicking_foot_strike_window.py', 'src/kicking_foot_evidence.py',
                  'src/kicking_foot_evidence_v2.py', 'scripts/run_fresh_foot_experiment.py',
                  'src/pose_refinement.py', 'requirements-pose33.txt')
    code_hashes = {}
    for name in code_names:
        dest = args.output_dir / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        code_hashes[name] = sha(dest)
    protocol = dict(schema_version='fresh-shoe-pose33-pilot-v2', clip_id=args.clip_id,
        source_sha256=sha(video), source_manifest_sha256=sha(manifest_path),
        roi_cache_sha256=roi_hash, original_neural_producer_sha256=sha(producer_path),
        rival_pose_sha256=sha(pose_path), actor_track_id_candidate=actor_id,
        contact_sha256=sha(contact_path), model_sha256=sha(args.model), model_bytes=args.model.stat().st_size,
        model_source='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
        documentation='https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python',
        code_sha256=code_hashes, new_neural_inference=True,
        runtime=dict(python=sys.version, platform=platform.platform(), delegate='CPU',
                     packages={name: importlib.metadata.version(name) for name in ('mediapipe', 'numpy', 'opencv-contrib-python', 'scipy')}),
        settings=dict(running_mode=args.running_mode, num_poses=1, detection_confidence=.5,
                      presence_confidence=.5, landmark_visibility_presence_min=.5,
                      shoe_contact_region_maximum_body_distance=.08,
                      fixed_union_roi_xyxy=fixed_roi, roi_reference_changes_between_frames=False),
        roi_basis='original source-only COCO person ROI; no event foot labels',
        prediction_scope='source-coordinate shoe-swing and local stationary-region hypotheses',
        world_coordinates_used=False, reference_labels_read=False,
        human_verified=False, accuracy_validated=False, held_out_test=False,
        development_data_previously_inspected=True)
    write(args.output_dir / 'protocol.json', protocol)
    write(args.output_dir / 'roi_plan.json', tasks)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model), delegate=python.BaseOptions.Delegate.CPU),
        running_mode=getattr(vision.RunningMode, args.running_mode), num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5)
    records = []
    previous_timestamps = {False: None, True: None}
    started = time.monotonic()
    with ExitStack() as stack:
        models = {reflected: stack.enter_context(vision.PoseLandmarker.create_from_options(options))
                  for reflected in (False, True)}
        for task in tasks:
            frame = task['frame']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Source frame cannot be decoded: {frame}')
            if (image.shape[:2] != (frame_height, frame_width)
                    or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (frame + 1)) > .5
                    or abs(cap.get(cv2.CAP_PROP_POS_MSEC) - frame / fps * 1000) > max(1., 100 / fps)):
                raise ValueError('Decoded source frame index/timestamp does not match the planned clock')
            left, top, right, bottom = task['crop_xyxy']
            crop = image[top:bottom, left:right]
            for reflected in (False, True):
                timestamp_ms = round(frame / fps * 1000)
                if previous_timestamps[reflected] is not None and timestamp_ms <= previous_timestamps[reflected]:
                    raise ValueError('Per-view VIDEO timestamps are not strictly increasing')
                previous_timestamps[reflected] = timestamp_ms
                sample = np.ascontiguousarray(crop[:, ::-1] if reflected else crop)
                rgb = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
                model_input = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                output = (models[reflected].detect(model_input) if args.running_mode == 'IMAGE'
                          else models[reflected].detect_for_video(model_input, timestamp_ms))
                row = dict(frame=frame, reflected=reflected, crop_xyxy=task['crop_xyxy'],
                           anchor_bbox=task['anchor_bbox'], pose_count=len(output.pose_landmarks),
                           rival_source_boxes=rival_boxes.get(frame, []),
                           timestamp_ms=timestamp_ms, decoded_timestamp_ms=cap.get(cv2.CAP_PROP_POS_MSEC),
                           landmarks_source_xy=None, landmark_confidence=None,
                           roi_identity_status='no_detected_pose')
                if len(output.pose_landmarks) == 1:
                    landmarks = output.pose_landmarks[0]
                    normalized = np.asarray([[item.x, item.y] for item in landmarks])
                    visibility = np.asarray([float(item.visibility) for item in landmarks])
                    presence = np.asarray([float(item.presence) for item in landmarks])
                    points, confidence = restore_landmarks(normalized, np.minimum(visibility, presence),
                                                           task['crop_xyxy'], reflected)
                    body_box = source_body_box(points, confidence)
                    selected, identity = select_anchored_detection([dict(bbox=body_box)], task['anchor_bbox'],
                        rivals=rival_boxes.get(frame, [])) if body_box else (None, 'insufficient_confident_body_for_roi_identity')
                    row.update(original_normalized_xy=normalized.tolist(), original_visibility=visibility.tolist(),
                               original_presence=presence.tolist())
                    row.update(roi_identity_status=identity, predicted_source_body_bbox=body_box)
                    if selected is not None:
                        row.update(landmarks_source_xy=points.tolist(), landmark_confidence=confidence.tolist())
                records.append(row)
    cap.release()
    write(args.output_dir / 'poses33.json', records)
    normal = [row for row in records if not row['reflected']]
    mirror = [row for row in records if row['reflected']]
    stationary, departing = payload['stationary_ball_observations'], payload['departure_ball_observations']
    evidence = {}
    for name, use_shoe in (('ankle_motion', False), ('shoe_centroid_motion', True)):
        evidence[name] = kicking_foot_strike_window(coco_samples(normal, shoe_centroids=use_shoe),
            coco_samples(mirror, shoe_centroids=use_shoe), bracket, fps,
            stationary_ball_observations=stationary, departure_ball_observations=departing)
    target = next(row['xy'] for row in stationary if row['frame'] == first)
    scale = float(np.median([row['anchor_bbox'][3] - row['anchor_bbox'][1] for row in normal if row['frame'] < first]))
    distances = {name: shoe_ball_distances(rows, target, scale) for name, rows in (('normal', normal), ('mirrored_inverse_mapped', mirror))}
    candidate = evidence['shoe_centroid_motion']['candidate']
    near_support = {}
    for name, rows in distances.items():
        near_support[name] = {side: [row['frame'] for row in rows if first - 1 <= row['frame'] <= departure
                                    and row[side]['confident'] and row[side]['stationary_point_distance_body'] <= .08]
                              for side in SHOE_PAIRS}
    accepted_frames = {row['frame'] for row in evidence['shoe_centroid_motion'].get('accepted_paired_frames', [])}
    joint_near_frames = (sorted(set(near_support['normal'][candidate])
                        & set(near_support['mirrored_inverse_mapped'][candidate]) & accepted_frames)
                        if candidate != 'unknown' else [])
    contact_candidate = candidate if joint_near_frames else 'unknown'
    ambiguous_frames = sorted(set(producer.get('local_seed_identity_ambiguous_frames', [])
        + [frame for frame in payload.get('seed_ambiguous_frames', []) if first - 5 <= frame <= departure]))
    if ambiguous_frames:
        candidate, contact_candidate = 'unknown', 'unknown'
    result = dict(clip_id=args.clip_id, source_mapping=source, contact_bracket_frames=bracket,
        source_fps=fps, source_coordinate_shoe_swing_candidate=candidate,
        source_coordinate_shoe_contact_region_candidate=contact_candidate,
        shoe_contact_verified=False, anatomical_side_verified=False, ball_identity_verified=False,
        new_neural_inference=True, actual_inference_calls=len(records),
        detected_pose_calls=sum(row['pose_count'] == 1 for row in records),
        accepted_roi_pose_calls=sum(row['landmarks_source_xy'] is not None for row in records),
        calls_by_view=dict(Counter('mirror' if row['reflected'] else 'normal' for row in records)),
        elapsed_seconds=time.monotonic() - started, evidence=evidence,
        stationary_ball_reference=dict(frame=first, xy=target), observed_preparation_scale_source_px=scale,
        shoe_ball_distances=distances, shoe_region_near_bracket_frames=near_support,
        joint_near_shoe_and_paired_pose_frames=joint_near_frames,
        local_ball_identity_ambiguous_frames=ambiguous_frames,
        reference_labels_used_for_prediction=False, human_verified=False,
        accuracy_validated=False, held_out_test=False,
        interpretation='Independent 33-point model pilot; anatomical channels and shoe proximity require source review')
    write(args.output_dir / 'result.json', result)
    print(json.dumps({name: result[name] for name in ('clip_id', 'source_coordinate_shoe_swing_candidate',
        'source_coordinate_shoe_contact_region_candidate', 'actual_inference_calls', 'detected_pose_calls',
        'accepted_roi_pose_calls', 'elapsed_seconds')}, indent=2))
    return result


if __name__ == '__main__':
    main()
