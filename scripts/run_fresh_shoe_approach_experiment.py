"""Infer paired 33-point poses throughout source-bound visible approaches.

Only the original course cohort is eligible. Every planned frame is decoded
sequentially; missing source actor observations remain missing rather than
extending identity through gaps. This produces observations, not step truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_fresh_pitch_experiment import _pose_input
from src.mediapipe_shoe_evidence import restore_landmarks, source_body_box
from src.pose_refinement import crop_bounds, select_anchored_detection


COURSE_SOURCES = {'col_eng_pen.mp4', 'cro_den_pen.mp4', 'esp_rus_pen.mp4', 'rus_cro_pen.mp4'}
CODE_FILES = (
    'scripts/run_fresh_shoe_approach_experiment.py',
    'scripts/run_fresh_phase_experiment.py', 'scripts/run_fresh_pitch_experiment.py',
    'scripts/run_penalty_phase_experiment.py', 'src/fresh_cache_provenance.py',
    'src/mediapipe_shoe_evidence.py', 'src/pose_refinement.py',
    'src/motion_stabilization.py', 'src/penalty_phases.py', 'src/phase_step_events.py',
    'src/style_aware_steps.py', 'src/terminal_footwork.py', 'src/final_results.py',
    'src/step_pipeline.py', 'src/camera_angles.py', 'src/kicker_analysis.py',
    'src/video_utils.py', 'src/homography.py', 'src/pitch_calibration.py',
    'requirements-pose33.txt', 'docs/SHOE_APPROACH_PROTOCOL_2026-10-13.md',
)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def receipt(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_box(box, width, height) -> bool:
    array = np.asarray(box, float)
    return (array.shape == (4,) and np.isfinite(array).all()
            and 0 <= array[0] < array[2] <= width
            and 0 <= array[1] < array[3] <= height)


def course_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = read(path)
    sources = manifest.get('sources') or {}
    rows = manifest.get('clips') or []
    if (set(sources) != COURSE_SOURCES or len(rows) != 39
            or len({row.get('clip_id') for row in rows}) != 39):
        raise ValueError('Dense approach experiment requires the original four-video, 39-event manifest')
    for row in rows:
        original = sources.get(row.get('source_filename'), {})
        if (row.get('source_video_sha256') != original.get('sha256')
                or not _integer(row.get('source_start_frame'))
                or not _integer(row.get('source_end_frame_exclusive'))
                or not _integer(row.get('frame_count'))
                or row['source_end_frame_exclusive'] - row['source_start_frame'] != row['frame_count']
                or not 0 <= row['source_start_frame'] < row['source_end_frame_exclusive'] <= original['frame_count']
                or Path(row.get('clip_filename', '')).name != row.get('clip_filename')):
            raise ValueError('Course clip does not bind its original source frame interval')
    return manifest, rows


def bind_event(args, source: dict, manifest_hash: str) -> tuple[dict | None, dict]:
    cid = source['clip_id']
    video = args.clips_dir / source['clip_filename']
    pose_path = args.pose_dir / f'{cid}.json'
    phase_path = args.phase_dir / f'{cid}.json'
    camera_path = args.phase_dir / 'camera_motion' / f'{cid}.json'
    provenance = {'source': receipt(video), 'source_mapping': source}
    if provenance['source']['sha256'] != source['clip_sha256']:
        raise ValueError(f'Course clip hash mismatch: {cid}')
    pose, pose_receipt = _pose_input(pose_path, source)
    provenance['pose'] = pose_receipt
    if pose is None:
        return None, dict(provenance, unavailable_reason=pose_receipt['status'])
    if (pose.get('reference_labels_used_for_prediction') is not False
            or not np.isfinite(pose['fps']) or pose['fps'] <= 0
            or not np.isclose(pose['fps'], source['fps'])
            or pose['source_frame_count'] != source['frame_count']):
        raise ValueError('Original pose must preserve source-only inference and exact source clock')
    if not phase_path.exists():
        return None, dict(provenance, unavailable_reason='original_phase_unavailable')
    phase = read(phase_path)
    provenance['phase'] = receipt(phase_path)
    identity = phase.get('inference_identity', {})
    scene = [pose['shot_start_frame'], pose['shot_end_frame']]
    if (any(not _integer(frame) for frame in scene)
            or not 0 <= scene[0] < scene[1] <= source['frame_count']):
        raise ValueError('Original selected scene must lie within the hashed course clip')
    if (phase.get('clip_id') != cid or phase.get('old_reference_labels_used') is not False
            or phase.get('user_step_labels_used') is not False
            or identity.get('source_manifest_sha256') != manifest_hash
            or identity.get('source_clip_sha256') != source['clip_sha256']
            or identity.get('pose_sha256') != pose_receipt['sha256']
            or identity.get('shot_interval') != scene
            or phase.get('pose_input', {}).get('sha256') != pose_receipt['sha256']):
        raise ValueError('Phase input must bind the unchanged original source and actor inference')
    contact, contact_provenance = _bound_contact(args.ball_dir, source, pose_path, pose, manifest_hash)
    provenance['contact'] = contact_provenance
    if contact['status'] != 'tentative':
        return None, dict(provenance, unavailable_reason='independent_contact_unavailable')
    bracket = contact['contact_bracket_frames']
    phase_contact = phase.get('contact', {})
    actor_id = contact['contact_associated_track_id']
    if (phase_contact.get('contact_bracket_frames') != bracket
            or phase_contact.get('contact_associated_track_id') != actor_id
            or phase.get('analysis_actor_track_id') != actor_id):
        raise ValueError('Phase actor and contact bracket disagree with independent ball observations')
    contact_path = args.ball_dir / 'contacts' / f'{cid}.json'
    if contact_path.exists():
        contact_document = read(contact_path)
        document_identity = contact_document.get('inference_identity', {})
        payload = contact_document.get('contact', {})
        if (document_identity.get('source_sha256') != source['clip_sha256']
                or document_identity.get('pose_sha256') != pose_receipt['sha256']
                or document_identity.get('source_manifest_sha256') != manifest_hash
                or payload.get('contact_bracket_frames') != bracket
                or payload.get('contact_associated_track_id') != actor_id):
            raise ValueError('Contact export must retain the same source, actor and causal bracket')
        provenance['contact_export'] = receipt(contact_path)
    onset = phase.get('phase_result', {}).get('approach_start_frame')
    if not _integer(onset) or not scene[0] <= onset < bracket[0] < bracket[1] < scene[1]:
        return None, dict(provenance, unavailable_reason='visible_approach_onset_unavailable')
    if not camera_path.exists() or sha(camera_path) != phase.get('camera_motion_sha256'):
        raise ValueError('Original phase camera-motion artifact is missing or changed')
    camera = read(camera_path)
    if camera.get('inference_identity') != identity or camera.get('shot_interval') != scene:
        raise ValueError('Camera-motion source scene identity mismatch')
    provenance['camera_motion'] = receipt(camera_path)
    actor = next((row for row in pose.get('tracks', []) if row.get('track_id') == actor_id), None)
    if actor is None:
        return None, dict(provenance, unavailable_reason='ball_associated_actor_unavailable')
    width, height = pose['frame_width'], pose['frame_height']
    start, end = onset, min(scene[1] - 1, bracket[1] + round(.2 * pose['fps']))
    actor_by_frame = {}
    for observation in actor.get('observations', []):
        frame = observation.get('frame')
        if not _integer(frame) or not scene[0] <= frame < scene[1]:
            raise ValueError('Original actor observation outside selected source scene')
        if frame in actor_by_frame:
            raise ValueError('Original actor has duplicate same-frame observations')
        if _finite_box(observation.get('bbox'), width, height):
            actor_by_frame[frame] = observation
    bounds = [crop_bounds(row['bbox'], width, height, margin=.45)
              for frame, row in actor_by_frame.items() if start <= frame <= end]
    if not bounds:
        return None, dict(provenance, unavailable_reason='no_exact_actor_observations_in_approach')
    bounds = np.asarray(bounds, int)
    fixed_roi = [int(bounds[:, 0].min()), int(bounds[:, 1].min()),
                 int(bounds[:, 2].max()), int(bounds[:, 3].max())]
    rival_by_frame = {}
    for track in pose.get('tracks', []):
        if track.get('track_id') == actor_id:
            continue
        for observation in track.get('observations', []):
            frame = observation.get('frame')
            if _integer(frame) and start <= frame <= end and _finite_box(observation.get('bbox'), width, height):
                rival_by_frame.setdefault(frame, []).append(observation['bbox'])
    plan = [dict(frame=frame, crop_xyxy=fixed_roi,
                 anchor_bbox=actor_by_frame.get(frame, {}).get('bbox'),
                 anchor_source='observed' if frame in actor_by_frame else 'missing_no_interpolation',
                 rival_source_boxes=rival_by_frame.get(frame, []))
            for frame in range(start, end + 1)]
    return dict(source_mapping=source, video=video, fps=pose['fps'], width=width, height=height,
                scene_interval_frames=scene, interval_frames=[onset, bracket[0]],
                inference_interval_frames=[start, end], contact_bracket_frames=bracket,
                actor_track_id=actor_id, original_pose_selected_track_id=pose['selected_track_id'],
                plan=plan, fixed_roi=fixed_roi, original_camera_motion=camera,
                original_phase_result=phase['phase_result'], contact=contact), provenance


def decode_interval(bound: dict):
    """Decode from frame zero so requested frame indices never depend on seeks."""
    cap = cv2.VideoCapture(str(bound['video']))
    try:
        source = bound['source_mapping']
        if (not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - bound['fps']) > .01
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != source['frame_count']
                or [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
                   != [bound['width'], bound['height']]):
            raise ValueError('Source decoder metadata disagrees with original pose and manifest')
        start, end = bound['inference_interval_frames']
        for frame in range(end + 1):
            ok, image = cap.read()
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if (not ok or image.shape[:2] != (bound['height'], bound['width'])
                    or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (frame + 1)) > .5
                    or not np.isfinite(timestamp_ms)
                    or abs(timestamp_ms - frame / bound['fps'] * 1000) > max(1., 100 / bound['fps'])):
                raise ValueError(f'Sequential source frame clock or dimensions disagree at frame {frame}')
            if frame >= start:
                yield frame, image, float(timestamp_ms)
    finally:
        cap.release()


def infer_event(bound: dict, model_path: Path) -> tuple[list[dict], dict, dict]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=.5, min_pose_presence_confidence=.5,
    )
    plan = {row['frame']: row for row in bound['plan']}
    records, hashes, first_hash_frame = [], {}, {}
    previous_timestamps = {False: None, True: None}
    call_count = 0
    with ExitStack() as stack:
        models = {reflected: stack.enter_context(vision.PoseLandmarker.create_from_options(options))
                  for reflected in (False, True)}
        for frame, image, decoded_timestamp in decode_interval(bound):
            task = plan[frame]
            pixel_hash = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            hashes[str(frame)] = pixel_hash
            duplicate = first_hash_frame.get(pixel_hash)
            first_hash_frame.setdefault(pixel_hash, frame)
            left, top, right, bottom = bound['fixed_roi']
            crop = image[top:bottom, left:right]
            for reflected in (False, True):
                timestamp_ms = round(frame / bound['fps'] * 1000)
                previous = previous_timestamps[reflected]
                if previous is not None and timestamp_ms <= previous:
                    raise ValueError('Per-view VIDEO timestamps must be strictly increasing')
                previous_timestamps[reflected] = timestamp_ms
                row = dict(task, reflected=reflected, timestamp_ms=timestamp_ms,
                           decoded_timestamp_ms=decoded_timestamp, frame_pixel_sha256=pixel_hash,
                           unique_decoded_support=duplicate is None, duplicate_of_frame=duplicate,
                           pose_count=0, neural_call=False, landmarks_source_xy=None, landmark_confidence=None,
                           roi_identity_status='original_actor_observation_unavailable')
                if task['anchor_bbox'] is not None:
                    rgb = cv2.cvtColor(np.ascontiguousarray(crop[:, ::-1] if reflected else crop), cv2.COLOR_BGR2RGB)
                    output = models[reflected].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                    call_count += 1
                    row.update(neural_call=True, pose_count=len(output.pose_landmarks), roi_identity_status='no_detected_pose')
                    if len(output.pose_landmarks) == 1:
                        landmarks = output.pose_landmarks[0]
                        normalized = np.asarray([[item.x, item.y] for item in landmarks])
                        visibility = np.asarray([float(item.visibility) for item in landmarks])
                        presence = np.asarray([float(item.presence) for item in landmarks])
                        points, confidence = restore_landmarks(normalized, np.minimum(visibility, presence), bound['fixed_roi'], reflected)
                        body_box = source_body_box(points, confidence)
                        selected, status = (select_anchored_detection([dict(bbox=body_box)], task['anchor_bbox'],
                                            rivals=task['rival_source_boxes']) if body_box else
                                            (None, 'insufficient_confident_body_for_roi_identity'))
                        row.update(original_normalized_xy=normalized.tolist(), original_visibility=visibility.tolist(),
                                   original_presence=presence.tolist(), predicted_source_body_bbox=body_box,
                                   roi_identity_status=status)
                        if selected is not None:
                            row.update(landmarks_source_xy=points.tolist(), landmark_confidence=confidence.tolist())
                records.append(row)
    quality = dict(actual_inference_calls=call_count, decoded_interval_frames=len(hashes),
                   distinct_decoded_pixel_frames=len(first_hash_frame),
                   repeated_decoded_pixel_frames=len(hashes) - len(first_hash_frame),
                   accepted_roi_pose_calls=sum(row['roi_identity_status'] == 'accepted' for row in records),
                   unique_pixel_accepted_roi_pose_calls=sum(row['roi_identity_status'] == 'accepted'
                                                            and row['unique_decoded_support'] for row in records),
                   roi_identity_status_counts=dict(Counter(row['roi_identity_status'] for row in records)),
                   original_actor_missing_frames=[row['frame'] for row in bound['plan'] if row['anchor_bbox'] is None])
    return records, hashes, quality


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clips-dir', 'pose-dir', 'phase-dir', 'ball-dir', 'model', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip-id', action='append', help='Explicit pilot IDs; omitted rows remain visible in cohort summary')
    parser.add_argument('--plan-only', action='store_true', help='Bind source inputs and register interval/ROI plans without neural inference')
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve evidence')
    if any(args.output_dir.resolve() == path.resolve() for path in (args.clips_dir, args.pose_dir, args.phase_dir, args.ball_dir)):
        parser.error('Output must not overwrite experiment inputs')
    manifest_path = args.clips_dir / 'source_manifest.json'
    _, sources = course_manifest(manifest_path)
    manifest_hash = sha(manifest_path)
    selected = set(args.clip_id or [row['clip_id'] for row in sources])
    if ((args.clip_id and len(args.clip_id) != len(selected))
            or not selected.issubset({row['clip_id'] for row in sources})):
        parser.error('Pilot clip IDs must be distinct original course-manifest members')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {}
    for name in CODE_FILES:
        dest = args.output_dir / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        code_hashes[name] = sha(dest)
    protocol = dict(schema_version='fresh-shoe-approach-pose33-v1', created_at_utc=datetime.now(timezone.utc).isoformat(),
        events=39, selected_clip_ids=sorted(selected), pilot=args.clip_id is not None,
        source_manifest_sha256=manifest_hash, model=receipt(args.model), code_sha256=code_hashes,
        model_source='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
        settings=dict(running_mode='VIDEO', normal_and_mirror_use_independent_models=True, num_poses=1,
                      detection_confidence=.5, presence_confidence=.5, landmark_visibility_presence_min=.5,
                      crop_margin_body_height=.45, fixed_union_roi_per_event=True,
                      actor_anchors='exact_same_frame_original_observations_only', anchor_interpolation=False,
                      decode_policy='sequential_from_clip_frame_zero', post_departure_seconds=.2,
                      duplicated_pixels_are_independent_support=False),
        runtime=dict(python=sys.version, platform=platform.platform(), delegate='CPU',
                     packages={name: importlib.metadata.version(name) for name in ('mediapipe', 'numpy', 'opencv-contrib-python', 'scipy')}),
        plan_only=args.plan_only, original_course_videos_only=True, additional_video_used=False,
        reference_labels_read=False, human_verified=False, accuracy_validated=False, held_out_test=False,
        full_approach_visibility_verified=False, world_coordinates_used=False,
        development_data_previously_inspected=True, protocol_registered_before_inference=True,
        scope='Source-coordinate observed visible-approach poses; no physical landing or full-approach count truth')
    write(args.output_dir / 'protocol.json', protocol)
    rows = []
    started = time.monotonic()
    cv2.setNumThreads(2)
    for source in sources:
        cid = source['clip_id']
        event_dir = args.output_dir / 'events' / cid
        row = dict(clip_id=cid, event_id=source['event_id'], status='not_selected_explicit_pilot',
                   source_sha256=source['clip_sha256'], actual_inference_calls=0,
                   human_verified=False, accuracy_validated=False)
        if cid not in selected:
            write(event_dir / 'unavailable.json', row)
        else:
            bound, provenance = bind_event(args, source, manifest_hash)
            write(event_dir / 'input_receipts.json', provenance)
            if bound is None:
                row.update(status='unavailable', reason=provenance['unavailable_reason'])
                write(event_dir / 'unavailable.json', dict(row, input_receipts=provenance))
            else:
                write(event_dir / 'roi_plan.json', bound['plan'])
                inputs = {key: bound[key] for key in ('source_mapping', 'fps', 'scene_interval_frames',
                    'interval_frames', 'inference_interval_frames', 'contact_bracket_frames', 'actor_track_id',
                    'original_pose_selected_track_id', 'original_camera_motion', 'original_phase_result', 'contact')}
                inputs.update(schema_version='fresh-shoe-approach-inputs-v1', source_manifest_sha256=manifest_hash,
                              input_receipts=provenance, code_sha256=code_hashes, model_sha256=protocol['model']['sha256'],
                              coordinate_space='original_source_image_pixels', interval_endpoints='inclusive',
                              full_approach_visible=False, full_approach_visibility_verified=False,
                              reference_labels_read=False, human_verified=False, accuracy_validated=False)
                if args.plan_only:
                    row.update(status='bound_plan_only', planned_source_frames=len(bound['plan']),
                               planned_inference_calls=2 * sum(item['anchor_bbox'] is not None for item in bound['plan']))
                    inputs.update(new_neural_inference=False, decoded_frame_hashes={})
                else:
                    event_started = time.monotonic()
                    records, hashes, quality = infer_event(bound, args.model)
                    write(event_dir / 'poses33.json', records)
                    inputs.update(new_neural_inference=True, decoded_frame_hashes=hashes,
                                  poses33_sha256=sha(event_dir / 'poses33.json'))
                    row.update(status='fresh_dense_pose_candidate', **quality,
                               elapsed_seconds=time.monotonic() - event_started,
                               poses33_sha256=inputs['poses33_sha256'])
                write(event_dir / 'inputs.json', inputs)
                row.update(interval_frames=bound['interval_frames'], inference_interval_frames=bound['inference_interval_frames'],
                           inputs_sha256=sha(event_dir / 'inputs.json'), roi_plan_sha256=sha(event_dir / 'roi_plan.json'))
                write(event_dir / 'result.json', row)
        rows.append(row)
        write(args.output_dir / 'progress.json', dict(processed=len(rows), events=39, rows=rows))
        print(json.dumps(row), flush=True)
    summary = dict(schema_version='fresh-shoe-approach-summary-v1', events=39, rows=rows,
        actual_inference_calls=sum(row['actual_inference_calls'] for row in rows),
        neural_events=sum(row['actual_inference_calls'] > 0 for row in rows),
        status_counts=dict(Counter(row['status'] for row in rows)), source_manifest_sha256=manifest_hash,
        elapsed_seconds=time.monotonic() - started, reference_labels_read=False,
        human_verified=False, accuracy_validated=False, held_out_test=False,
        original_course_videos_only=True, additional_video_used=False,
        interpretation='Fresh pose availability and unique decoded support; not step count or foot accuracy')
    write(args.output_dir / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    main()
