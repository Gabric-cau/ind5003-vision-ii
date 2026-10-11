"""Render source-hashed fresh SAM2 masks for diagnosis, without changing labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.run_fresh_keeper_height_experiment import load_bound_masks
from scripts.run_fresh_phase_experiment import _bound_contact
from scripts.run_fresh_pitch_experiment import _hash, _pose_input, _write


def _contain(image, width, height):
    scale = min(width/image.shape[1], height/image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    out = np.full((height, width, 3), 25, np.uint8)
    x, y = (width-resized.shape[1])//2, (height-resized.shape[0])//2
    out[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clips-dir', 'pose-dir', 'ball-dir', 'mask-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--clips', nargs='+', required=True)
    args = parser.parse_args(argv)
    manifest_path = args.clips_dir/'source_manifest.json'
    sources = {row['clip_id']: row for row in json.loads(manifest_path.read_text())['clips']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cid in args.clips:
        source, pose_path = sources[cid], args.pose_dir/(cid+'.json')
        video = args.clips_dir/source['clip_filename']
        if _hash(video) != source['clip_sha256']:
            raise ValueError('Keeper diagnostic source video SHA mismatch')
        pose, _ = _pose_input(pose_path, source)
        if pose is None:
            raise ValueError('Keeper diagnostic pose scene/source is not supported')
        contact, _ = _bound_contact(args.ball_dir, source, pose_path, pose, _hash(manifest_path))
        if contact['status'] != 'tentative':
            raise ValueError('Keeper diagnostic independent contact is not supported')
        path = args.mask_dir/cid/'mask_manifest.json'
        record = json.loads(path.read_text())
        event_path = args.ball_dir/'event_windows'/(cid+'.json')
        masks, provenance = load_bound_masks(args.mask_dir, clip_id=cid,
            source_sha=source['clip_sha256'], pose_sha=_hash(pose_path), event_sha=_hash(event_path),
            keeper_id=record['keeper_track_id'], shape=(pose['frame_height'], pose['frame_width']))
        stop = record.get('strict_identity_prefix_stop') or {}
        first, last = min(masks), max(masks)
        kick = contact['contact_frame']
        desired = [first, kick, kick+6, kick+12, last]
        if stop.get('frame') is not None:
            desired += [stop['frame']-2, stop['frame'], min(last, stop['frame']+2)]
        else:
            desired += [kick+18, kick+24, kick+30]
        frames = sorted({frame for frame in desired if frame in masks})
        observations = {row['frame']: row for row in (record['source_plan'].get('automatic_association') or {}).get('observations', [])}
        roi = record['inference_roi_xyxy']
        x0, y0, x1, y1 = roi
        lossless_path = path.parent/'source_roi_lossless.avi'
        if _hash(lossless_path) != record['source_roi_video_sha256']:
            raise ValueError('Actual GPU lossless source crop artifact hash mismatch')
        gpu_pixels = {}
        crop_capture = cv2.VideoCapture(str(lossless_path))
        try:
            for mask_record in record['masks']:
                ok, crop = crop_capture.read()
                if not ok or hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest() != mask_record['crop_bgr_sha256']:
                    raise ValueError('Actual lossless GPU source pixels differ from frame receipt')
                if mask_record['frame'] in frames:
                    gpu_pixels[mask_record['frame']] = crop
        finally:
            crop_capture.release()
        tiles, meta, cap = [], [], cv2.VideoCapture(str(video))
        try:
            for frame in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Cannot decode fresh keeper source frame')
                local_crop = np.ascontiguousarray(image[y0:y1, x0:x1])
                local_crop_sha = hashlib.sha256(local_crop.tobytes()).hexdigest()
                mask_record = next(row for row in record['masks'] if row['frame'] == frame)
                difference = np.abs(local_crop.astype(float)-gpu_pixels[frame])
                # MP4 YUV-to-BGR rounding can differ between FFmpeg builds.
                # The ROI shown here uses the hash-verified actual GPU pixels.
                image[y0:y1, x0:x1] = gpu_pixels[frame]
                original_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                mask = masks[frame]
                overlay = image.copy()
                tint = np.full_like(image, (35, 215, 90))
                overlay[mask] = cv2.addWeighted(image, .55, tint, .45, 0)[mask]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (20, 255, 120), 1)
                cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), (30, 160, 255), 2)
                row = observations.get(frame)
                if row:
                    points, conf = np.asarray(row['keypoints']), np.asarray(row['keypoint_confidences'])
                    for joint in (5, 6, 11, 12, 15, 16):
                        if conf[joint] >= .4:
                            cv2.circle(overlay, tuple(np.rint(points[joint]).astype(int)), 4, (255, 50, 230), -1)
                ys, xs = np.nonzero(mask)
                if len(xs):
                    bx0, by0 = max(x0, int(xs.min())-35), max(y0, int(ys.min())-35)
                    bx1, by1 = min(x1, int(xs.max())+36), min(y1, int(ys.max())+36)
                else:
                    bx0, by0, bx1, by1 = roi
                tile = np.hstack([_contain(overlay, 480, 270), _contain(overlay[by0:by1, bx0:bx1], 320, 270)])
                strip = np.full((38, 800, 3), 25, np.uint8)
                qc = record.get('mask_identity_qc', {}).get(str(frame), {})
                label = f'f{frame} / {frame/pose["fps"]:.3f}s role {record["keeper_track_id"]} QC: {qc.get("reason")}'
                cv2.putText(strip, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .46, (235, 235, 235), 1)
                tiles.append(np.vstack([strip, tile]))
                meta.append(dict(frame=frame, source_recording_frame=source['source_start_frame']+frame,
                                 source_bgr_sha256=original_sha, crop_bgr_sha256=mask_record['crop_bgr_sha256'],
                                 local_mp4_crop_bgr_sha256=local_crop_sha,
                                 local_mp4_vs_gpu_crop_absolute_difference=dict(
                                     mean=float(difference.mean()), p99=float(np.percentile(difference, 99)),
                                     maximum=float(difference.max())),
                                 mask_sha256=mask_record['sha256'], mask_pixels=int(mask.sum()),
                                 bbox_source_pixels=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if len(xs) else None,
                                 mask_identity_qc=qc))
        finally:
            cap.release()
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        destination = args.output_dir/(cid+'_source_masks.jpg')
        if not cv2.imwrite(str(destination), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])):
            raise OSError('Cannot write fresh keeper source/mask diagnostic')
        _write(args.output_dir/(cid+'_source_masks.json'), dict(clip_id=cid, frames=meta,
            source_clip_sha256=source['clip_sha256'], pose_sha256=_hash(pose_path),
            mask_provenance=provenance, source_sheet_sha256=_hash(destination),
            diagnostic_script_sha256=_hash(Path(__file__)), human_verified=False,
            accuracy_validated=False, actual_source_pixels_verified_against_gpu_crops=True))
        print(json.dumps(dict(clip_id=cid, frames=len(frames), sheet=str(destination))), flush=True)


if __name__ == '__main__':
    main()
