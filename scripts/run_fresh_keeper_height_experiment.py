"""CPU replay of source-bound goalkeeper torso height; performs no GPU inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.run_fresh_phase_experiment import _bound_contact, _prefix_motion
from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write
from src.fresh_keeper_height import analyze_height, connected_camera_prefix
from src.goalkeeper_role_association import select_goalkeeper_role, associate_locked_goalkeeper
from src.motion_stabilization import estimate_background_motion
from src.person_appearance import torso_histogram


def load_bound_masks(directory, *, clip_id, source_sha, pose_sha, event_sha, keeper_id, shape):
    """Accept only newly inferred full-source masks bound to this exact role."""
    path = directory / clip_id / 'mask_manifest.json'
    if not path.exists():
        return None, dict(status='unavailable', new_neural_inference=False)
    manifest = json.loads(path.read_text())
    expected = dict(schema_version='fresh-keeper-height-mask-v1', clip_id=clip_id,
                    source_sha256=source_sha, pose_sha256=pose_sha,
                    event_cache_sha256=event_sha, keeper_track_id=keeper_id,
                    coordinate_system='full_source_pixels',
                    frame_height=shape[0], frame_width=shape[1],
                    new_neural_inference=True, reference_labels_used_for_prediction=False)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Fresh keeper mask manifest source/pose/contact/role identity mismatch')
    if not isinstance(manifest.get('model_sha256'), str) or len(manifest['model_sha256']) != 64:
        raise ValueError('Fresh keeper masks must record their model SHA256')
    roi = manifest.get('inference_roi_xyxy')
    if roi is not None and (not isinstance(roi, list) or len(roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
            or not 0 <= roi[0] < roi[2] <= shape[1] or not 0 <= roi[1] < roi[3] <= shape[0]):
        raise ValueError('Invalid source-coordinate mask inference ROI')
    masks, files = {}, []
    root = path.parent.resolve()
    for record in manifest.get('masks', []):
        frame = record.get('frame')
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame in masks:
            raise ValueError('Invalid or duplicate mask source frame')
        image_path = (root / record['path']).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError('Mask path must resolve inside its clip artifact directory')
        if _hash(image_path) != record.get('sha256'):
            raise ValueError('Fresh keeper mask file hash mismatch')
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != shape:
            raise ValueError('Mask must be decoded at full source dimensions')
        if not set(np.unique(image)).issubset({0, 1, 255}):
            raise ValueError('Expected binary visible-body mask, not an RGB/probability image')
        masks[frame] = image > 0
        files.append(dict(frame=frame, path=str(image_path), sha256=record['sha256']))
    if not masks or min(masks) != manifest.get('seed_frame'):
        raise ValueError('Mask seed must be the earliest source mask frame')
    return masks, dict(status='source_bound_new_mask_candidate', manifest_path=str(path.resolve()),
                       manifest_sha256=_hash(path), files=files, model_sha256=manifest['model_sha256'],
                       new_neural_inference=True, human_verified=False, inference_roi_xyxy=roi)


def _camera(video, pose, identity, camera_dir, output_path):
    start, end = pose['shot_start_frame'], pose['shot_end_frame']
    path = camera_dir / (pose['clip_id']+'.json') if camera_dir else None
    if path is not None and path.exists():
        motion = json.loads(path.read_text())
        old = motion.get('inference_identity', {})
        for key in ('source_manifest_sha256', 'source_clip_sha256', 'pose_sha256'):
            if old.get(key) != identity[key]:
                raise ValueError('Full-shot camera artifact source/pose identity mismatch')
        if old.get('shot_interval') != [start, end]:
            raise ValueError('Full-shot camera artifact shot interval mismatch')
        connected_camera_prefix(motion, start, end)
        return motion, dict(path=str(path.resolve()), sha256=_hash(path), new_cpu_estimation=False)
    boxes = {}
    for track in pose['tracks']:
        for row in track['observations']:
            boxes.setdefault(row['frame'], []).append(row['bbox'])
    motion = _prefix_motion(estimate_background_motion(str(video), start, end, boxes_by_frame=boxes), start, end)
    motion['inference_identity'] = identity
    _write(output_path, motion)
    return motion, dict(path=str(output_path.resolve()), sha256=_hash(output_path), new_cpu_estimation=True)


def _association(video, pose, role, kick):
    if role.get('status') != 'candidate_supported':
        return None
    start = max(pose['shot_start_frame'], kick-round(.5*pose['fps']))
    end = pose['shot_end_frame']
    by_frame = {}
    for track in pose['tracks']:
        for row in track['observations']:
            if start <= row['frame'] < end:
                by_frame.setdefault(row['frame'], []).append((track['track_id'], row))
    histograms, cap = {}, cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame in range(start, end):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode source frame for keeper identity: {frame}')
            for tid, row in by_frame.get(frame, []):
                histograms[(frame, tid)] = torso_histogram(image, row)
    finally:
        cap.release()
    lock = associate_locked_goalkeeper(pose, role, histograms, kick, pre_sec=.5,
                                      post_sec=(end-kick-1)/pose['fps'])
    lock['source_pixels_decoded'] = True
    lock['appearance_features_are_not_person_identity_truth'] = True
    return lock


def _diagnostic(path, video, pose, cache, result):
    phase = result.get('pose_only_phase', {})
    reference = result.get('reference')
    role = result.get('role', {})
    candidates = [cache['kick_frame']-6, cache['kick_frame'], cache['kick_frame']+6]
    candidates += phase.get('supporting_frames', [])
    candidates += [phase.get('onset_frame'), phase.get('landing_frame')]
    frames = sorted({frame for frame in candidates if isinstance(frame, int)
                     and pose['shot_start_frame'] <= frame < pose['shot_end_frame']})[:8]
    observations = {row['frame']: row for row in (result.get('association') or {}).get('observations', [])}
    seed = next((track for track in pose['tracks'] if track['track_id'] == role.get('seed_track_id')), {})
    for row in seed.get('observations', []):
        observations.setdefault(row['frame'], row)
    tiles, cap = [], cv2.VideoCapture(str(video))
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode keeper diagnostic source frame {frame}')
            row = observations.get(frame)
            if row:
                box = np.rint(row['bbox']).astype(int)
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 215, 255), 2)
                points, conf = np.asarray(row['keypoints']), np.asarray(row['keypoint_confidences'])
                for joint in (5, 6, 11, 12, 15, 16):
                    if conf[joint] >= .4:
                        cv2.circle(image, tuple(np.rint(points[joint]).astype(int)), 4, (255, 0, 255), -1)
                if np.min(conf[[5, 6, 11, 12]]) >= .4:
                    center = points[[5, 6, 11, 12]].mean(0)
                    cv2.circle(image, tuple(np.rint(center).astype(int)), 5, (0, 255, 255), -1)
            tile = cv2.resize(image, (640, 360))
            strip = np.full((45, 640, 3), 30, np.uint8)
            text = f'frame {frame} / {frame/pose["fps"]:.3f}s  keeper seed {role.get("seed_track_id")}'
            cv2.putText(strip, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1)
            tiles.append(np.vstack([strip, tile]))
    finally:
        cap.release()
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 30))
    if tiles and not cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])):
        raise OSError('Could not write keeper source diagnostic')
    return dict(path=str(path.resolve()), sha256=_hash(path), source_frames=frames,
                human_verified=False, includes_ground_reference=bool(reference)) if tiles else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--ball-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path, help='Optional source-bound full-shot camera JSON directory')
    parser.add_argument('--mask-dir', type=Path, help='Optional per-clip fresh binary masks with hash manifests')
    parser.add_argument('--mask-clips', nargs='+', help='Only reuse masks for these exact contact/pose-bound clip IDs')
    parser.add_argument('--clips', nargs='+', help='Explicit pilot IDs; omitted means every source-manifest event')
    args = parser.parse_args(argv)
    if any(args.output_dir.resolve() == path.resolve() for path in
           (args.clips_dir, args.pose_dir, args.ball_dir, args.camera_dir, args.mask_dir) if path):
        parser.error('Output must be distinct from inference inputs')
    manifest_path = args.clips_dir / 'source_manifest.json'
    manifest_hash = _hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sources = {row['clip_id']: row for row in manifest['clips']}
    if len(sources) != len(manifest['clips']):
        raise ValueError('Duplicate source clip IDs')
    clip_ids = args.clips or list(sources)
    if len(set(clip_ids)) != len(clip_ids) or any(cid not in sources for cid in clip_ids):
        raise ValueError('Pilot IDs must be distinct source-manifest events')
    if args.mask_clips and (not args.mask_dir or len(set(args.mask_clips)) != len(args.mask_clips)
                           or not set(args.mask_clips).issubset(clip_ids)):
        parser.error('--mask-clips requires --mask-dir and distinct IDs within this experiment')
    code_paths = ('src/fresh_keeper_height.py', 'scripts/run_fresh_keeper_height_experiment.py',
                  'src/goalkeeper_dive_phase.py', 'src/goalkeeper_sam2.py',
                  'src/goalkeeper_role_association.py', 'src/person_appearance.py',
                  'src/motion_stabilization.py', 'scripts/run_fresh_phase_experiment.py',
                  'scripts/run_fresh_pitch_experiment.py')
    code_hashes = {name: _hash(ROOT/name) for name in code_paths}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'camera_motion').mkdir(exist_ok=True)
    (args.output_dir/'source_diagnostics').mkdir(exist_ok=True)
    protocol = dict(schema_version='fresh-keeper-height-v1', cohort=clip_ids,
                    source_manifest_sha256=manifest_hash, code_sha256=code_hashes,
                    neural_inference_performed_by_this_script=False, reference_labels_read=False,
                    human_verified=False, accuracy_validated=False, held_out_test=False,
                    mask_clip_ids=args.mask_clips)
    _write(args.output_dir/'protocol.json', protocol)
    cv2.setNumThreads(2)
    rows = []
    for cid in clip_ids:
        source = sources[cid]
        video, pose_path = args.clips_dir/source['clip_filename'], args.pose_dir/(cid+'.json')
        if _hash(video) != source['clip_sha256']:
            raise ValueError('Source video SHA mismatch')
        pose, provenance = _pose_input(pose_path, source)
        identity = dict(source_manifest_sha256=manifest_hash, source_clip_sha256=source['clip_sha256'],
                        pose_sha256=provenance.get('sha256'), code_sha256=code_hashes)
        result = dict(dive_height='unknown', strict_height_candidate='unknown',
                      observed_image_plane_height_candidate='unknown',
                      ordinal_pose_height_candidate='unknown', reason=provenance['status'])
        if pose is not None:
            if pose.get('reference_labels_used_for_prediction') is not False:
                raise ValueError('Fresh keeper height pose must not use reference labels')
            if not np.isclose(pose['fps'], source['fps']):
                raise ValueError('Pose/source FPS mismatch')
            contact, contact_provenance = _bound_contact(args.ball_dir, source, pose_path, pose, manifest_hash)
            result['contact_provenance'] = contact_provenance
            if contact['status'] == 'tentative':
                event_path = args.ball_dir/'event_windows'/(cid+'.json')
                event = json.loads(event_path.read_text())
                if event.get('source_sha256') != source['clip_sha256']:
                    raise ValueError('Fresh keeper event source mismatch')
                if (event.get('width'), event.get('height')) != (pose['frame_width'], pose['frame_height']):
                    raise ValueError('Fresh keeper event source dimensions mismatch')
                identity['event_cache_sha256'] = _hash(event_path)
                identity['shot_interval'] = [pose['shot_start_frame'], pose['shot_end_frame']]
                motion, camera_provenance = _camera(video, pose, identity, args.camera_dir,
                                                    args.output_dir/'camera_motion'/(cid+'.json'))
                geometries = {int(frame): goal for frame, goal in event['goal_proposals'].items()}
                role = select_goalkeeper_role(pose, geometries, contact['contact_frame'])
                association = _association(video, pose, role, contact['contact_frame'])
                masks, mask_provenance = (load_bound_masks(args.mask_dir, clip_id=cid,
                    source_sha=source['clip_sha256'], pose_sha=identity['pose_sha256'],
                    event_sha=identity['event_cache_sha256'], keeper_id=role.get('seed_track_id'),
                    shape=(pose['frame_height'], pose['frame_width'])) if args.mask_dir
                    and (args.mask_clips is None or cid in args.mask_clips)
                    else (None, dict(status='not_requested', new_neural_inference=False)))
                result = analyze_height(pose, event, camera=motion, association=association, masks=masks,
                                        role=role, mask_roi_xyxy=mask_provenance.get('inference_roi_xyxy'))
                result.update(contact_provenance=contact_provenance, camera_provenance=camera_provenance,
                              mask_provenance=mask_provenance,
                              source_diagnostic=_diagnostic(args.output_dir/'source_diagnostics'/(cid+'.jpg'),
                                                            video, pose, event, result))
            else:
                result['reason'] = 'independent_ball_contact_not_established'
        result.update(clip_id=cid, event_id=source['event_id'], inference_identity=identity,
                      pose_provenance=provenance, human_verified=False, accuracy_validated=False,
                      physical_height_metres=None, held_out_test=False, reference_labels_read=False,
                      new_neural_inference=False, schema_version='fresh-keeper-height-v1')
        _write(args.output_dir/(cid+'.json'), result)
        rows.append(dict(clip_id=cid, strict_height_candidate=result['strict_height_candidate'],
                         observed_image_plane_height_candidate=result['observed_image_plane_height_candidate'],
                         ordinal_pose_height_candidate=result['ordinal_pose_height_candidate'], reason=result['reason']))
        print(json.dumps(rows[-1]), flush=True)
    summary = dict(events=len(rows), rows=rows, strict_candidate_counts=dict(Counter(
        row['strict_height_candidate'] for row in rows)), ordinal_candidate_counts=dict(Counter(
        row['ordinal_pose_height_candidate'] for row in rows)), human_verified=False,
        observed_image_plane_candidate_counts=dict(Counter(
            row['observed_image_plane_height_candidate'] for row in rows)),
        accuracy_validated=False, held_out_test=False, physical_height_metres_estimated=False)
    _write(args.output_dir/'summary.json', summary)
    return summary


if __name__ == '__main__':
    main()
