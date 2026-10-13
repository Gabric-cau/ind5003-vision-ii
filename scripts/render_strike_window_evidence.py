"""Render source-frame leg evidence for decided strike-window diagnostics."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_fresh_foot_v2 import cached_samples
from scripts.run_fresh_pitch_experiment import _hash, _write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--foot-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    out_dir = args.experiment_dir / 'evidence'
    out_dir.mkdir(exist_ok=True)
    records = []
    summary = json.loads((args.experiment_dir / 'summary.json').read_text())
    for row in summary['rows']:
        if row['candidate'] == 'unknown':
            continue
        path = args.experiment_dir / 'results' / f"{row['clip_id']}.json"
        result = json.loads(path.read_text())
        source = result['source_mapping']
        video = args.clips_dir / source['clip_filename']
        if _hash(video) != source['clip_sha256']:
            raise ValueError('Evidence source video does not match experiment')
        pose_path = args.foot_dir / 'poses' / f"{row['clip_id']}.json"
        if _hash(pose_path) != result['cached_pose_sha256']:
            raise ValueError('Evidence poses do not match experiment')
        normal, mirror = cached_samples(json.loads(pose_path.read_text()))
        views = [{item['frame']: item for item in rows} for rows in (normal, mirror)]
        evidence = result['temporal_evidence']
        frames = [item['frame'] for item in evidence['accepted_paired_frames']]
        points = np.concatenate([np.asarray(views[0][frame]['keypoints'])[[11, 12, 13, 14, 15, 16]]
                                 for frame in frames])
        target = np.asarray(evidence['stationary_distance_reference']['xy'])
        bounds = np.concatenate([points, target[None]])
        left, top = np.floor(bounds.min(axis=0) - 20).astype(int)
        right, bottom = np.ceil(bounds.max(axis=0) + 20).astype(int)
        cap = cv2.VideoCapture(str(video))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
        tile_w, tile_h = 280, 330
        sheet = np.full((2 * tile_h + 70, len(frames) * tile_w, 3), 245, np.uint8)
        cv2.putText(sheet, f"{row['clip_id']} | candidate: {row['candidate']} | source-pose diagnostic",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, 'Red: COCO left channel. Blue: COCO right channel. Yellow: last stationary ball reference.',
                    (12, 49), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
        factor = min(tile_w / (right - left), (tile_h - 40) / (bottom - top))
        for column, frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError('Evidence source frame could not be decoded')
            for view_index, lookup in enumerate(views):
                crop = image[top:bottom, left:right].copy()
                pts = np.asarray(lookup[frame]['keypoints']) - [left, top]
                for indices, color in (([11, 13, 15], (50, 50, 230)), ([12, 14, 16], (230, 100, 30))):
                    for a, b in zip(indices, indices[1:]):
                        cv2.line(crop, tuple(np.rint(pts[a]).astype(int)), tuple(np.rint(pts[b]).astype(int)), color, 1)
                    for index in indices:
                        cv2.circle(crop, tuple(np.rint(pts[index]).astype(int)), 2, color, -1)
                cv2.circle(crop, tuple(np.rint(target - [left, top]).astype(int)), 4, (0, 235, 235), 1)
                resized = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                x = column * tile_w + (tile_w - resized.shape[1]) // 2
                y = 70 + view_index * tile_h + 34
                sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                label = 'normal' if view_index == 0 else 'mirror inverse'
                cv2.putText(sheet, f'{label} frame {frame}', (column * tile_w + 8, y - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1, cv2.LINE_AA)
        cap.release()
        destination = out_dir / f"{row['clip_id']}.jpg"
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError('Evidence sheet could not be saved')
        records.append(dict(clip_id=row['clip_id'], image=destination.name, sha256=_hash(destination),
                            source_video_sha256=source['clip_sha256'], source_frames=frames,
                            crop_xyxy=[int(value) for value in (left, top, right, bottom)], source_frame_pixels_resampled=True,
                            human_verified=False, anatomical_side_verified=False))
    _write(out_dir / 'manifest.json', dict(records=records, render_script_sha256=_hash(Path(__file__)),
                                         scope='Visible source-frame and pose-channel review, not verified shoe contact'))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
