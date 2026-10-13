"""Package alternate-source intervals without treating them as contact annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    source_hash = sha256(args.source)
    if source_hash != plan['source_sha256']:
        raise ValueError('Source file does not match the frozen acquisition hash')
    canonical = json.loads(args.canonical_manifest.read_text(encoding='utf-8'))
    events = {record['clip_id']: record for record in canonical['clips']}
    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if (width, height, fps, count) != (plan['source_width'], plan['source_height'],
                                    plan['container_fps'], plan['decoded_frames']):
        raise ValueError('Source metadata differs from the inspected acquisition')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    endpoints = {}
    for interval in plan['clips']:
        if interval['view'] != 'main_live_candidate':
            continue
        clip_id = interval['canonical_clip_id']
        event = events[clip_id]
        first = interval['start_frame']
        end = interval['end_frame_exclusive']
        target = args.output_dir / f'{clip_id}.mp4'
        command = [str(args.ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                   '-ss', str(first / fps), '-i', str(args.source), '-frames:v', str(end - first),
                   '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '10', '-pix_fmt', 'yuv420p',
                   '-threads', '2', '-fps_mode', 'passthrough', '-movflags', '+faststart', str(target)]
        subprocess.run(command, check=True)
        decoded = cv2.VideoCapture(str(target))
        output_count = 0
        output_first = None
        output_last = None
        while True:
            ok, bgr = decoded.read()
            if not ok:
                break
            if output_first is None:
                output_first = bgr
            output_last = bgr
            output_count += 1
        output_fps = decoded.get(cv2.CAP_PROP_FPS)
        decoded.release()
        if output_count != end - first or output_fps != fps:
            raise ValueError(f'Output frame count or fps differs: {clip_id}')
        endpoints[first] = output_first
        endpoints[end - 1] = output_last
        command_record = ['FFMPEG' if part == str(args.ffmpeg) else
                          args.source.name if part == str(args.source) else
                          target.name if part == str(target) else part for part in command]
        record = {key: event[key] for key in
                  ('clip_id', 'match_id', 'event_id', 'kicker', 'team', 'penalty_number')}
        record.update({
            'slice_start_sec': first / fps, 'slice_end_sec': end / fps,
            'source_filename': args.source.name, 'source_video_sha256': source_hash,
            'source_url': plan['source_url'], 'source_title': plan['source_title'],
            'source_origin': 'public_official_FIFA_alternate_edit',
            'fps': fps, 'width': width, 'height': height,
            'source_start_frame': first, 'source_end_frame_exclusive': end,
            'frame_count': output_count, 'original_slice_start_frame_in_clip': 0,
            'original_slice_end_frame_in_clip': output_count,
            'semantic_event_start_verified': False, 'semantic_event_end_verified': False,
            'context_is_not_event_boundary_ground_truth': True,
            'interval_basis': plan['interval_basis'], 'clip_sha256': sha256(target),
            'clip_bytes': target.stat().st_size, 'clip_filename': target.name,
            'ffmpeg_command_portable_paths': command_record,
            'reencoded': True, 'decoded_pixel_identity_verified': False,
            'contact_truth_annotated': False, 'accuracy_validated': False,
        })
        records.append(record)
        print(f'Prepared {clip_id}: {output_count} frames', flush=True)
    # Sequential source decoding checks the temporal endpoints without random-seek errors.
    cap = cv2.VideoCapture(str(args.source))
    index = 0
    comparisons = {}
    while index <= max(endpoints):
        ok, bgr = cap.read()
        if not ok:
            raise ValueError('Source decode stopped before the required endpoint')
        if index in endpoints:
            differences = bgr.astype(np.float32) - endpoints[index].astype(np.float32)
            mse = float(np.mean(differences * differences))
            comparisons[index] = {'source_frame': index, 'pixel_mean_absolute_error_8bit':
                                  float(np.mean(np.abs(differences))),
                                  'psnr_db': float(10 * np.log10(255 * 255 / mse)) if mse else None,
                                  'decoded_pixels_identical': bool(np.array_equal(bgr, endpoints[index]))}
        index += 1
    cap.release()
    for record in records:
        record['endpoint_frame_comparisons'] = [comparisons[record['source_start_frame']],
                                                comparisons[record['source_end_frame_exclusive'] - 1]]
    summary = {
        'schema_version': 1, 'dataset': 'FIFA_alternate_source_HD_manual_pilot_2026-10-13',
        'manifest_sha256': sha256(args.plan),
        'source_hash_reference_origin': 'new_public_FIFA_acquisition_hash_not_original_course_source',
        'sources': {args.source.name: {'sha256': source_hash, 'bytes': args.source.stat().st_size,
                   'fps': fps, 'frame_count': count, 'width': width, 'height': height,
                   'source_url': plan['source_url'], 'source_title': plan['source_title']}},
        'ffmpeg_version': subprocess.check_output([str(args.ffmpeg), '-version'], text=True).splitlines()[0],
        'encoding': {'codec': 'libx264', 'crf': 10, 'resize': False, 'audio_included': False,
                     'reencoded': True, 'lossless': False},
        'old_experiment_caches_compatible': False, 'human_verified': False,
        'original_course_recording': False, 'contact_truth_annotated': False,
        'alternate_edit_synchronized_to_course': False, 'clips': records,
    }
    (args.output_dir / 'source_manifest.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--canonical-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
