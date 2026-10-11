"""Show actual cached normal/mirror leg joints on their hashed source video."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.inspect_fresh_foot_contact import contain
from scripts.run_fresh_pitch_experiment import _hash, _write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clips-dir', 'foot-dir', 'replay-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--clips', nargs='+', required=True)
    args = parser.parse_args(argv)
    sources = {row['clip_id']: row for row in json.loads((args.clips_dir/'source_manifest.json').read_text())['clips']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cid in args.clips:
        source = sources[cid]
        replay_path = args.replay_dir/'results'/(cid+'.json')
        result = json.loads(replay_path.read_text())
        crop_path = args.foot_dir/'poses'/(cid+'.json')
        if _hash(crop_path) != result['cropped_pose_sha256']:
            raise ValueError('Paired source diagnostic cropped-pose bytes differ from replay')
        video = args.clips_dir/source['clip_filename']
        if _hash(video) != source['clip_sha256'] or result['inference_identity']['source_sha256'] != source['clip_sha256']:
            raise ValueError('Paired source diagnostic source video identity mismatch')
        raw = {(row['frame'], row['reflected']): row for row in json.loads(crop_path.read_text())}
        first, departure = result['temporal_evidence']['contact_bracket_frames']
        frames = [first-2, first-1, first, departure, departure+1, departure+2, departure+4, departure+6]
        frames = sorted({frame for frame in frames if (frame, False) in raw and (frame, True) in raw})
        tiles, metadata, cap = [], [], cv2.VideoCapture(str(video))
        try:
            for frame in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Cannot decode the cached pose source frame')
                pixel_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                annotated = image.copy()
                bounds = raw[(frame, False)]['crop_xyxy']
                record = dict(frame=frame, source_recording_frame=source['source_start_frame']+frame,
                    decoded_source_bgr_sha256=pixel_sha, crop_xyxy=bounds, observations=[])
                for reflected in (False, True):
                    row = raw[(frame, reflected)]
                    observed = row.get('selected')
                    record['observations'].append(row)
                    if observed is None:
                        continue
                    points, confidence = np.asarray(observed['keypoints']), np.asarray(observed['keypoint_confidences'])
                    for side, joints, color in (('L', [11, 13, 15], (220, 160, 30)),
                                                 ('R', [12, 14, 16], (40, 110, 245))):
                        for a, b in zip(joints, joints[1:]):
                            if min(confidence[[a, b]]) >= .4:
                                cv2.line(annotated, tuple(np.rint(points[a]).astype(int)),
                                         tuple(np.rint(points[b]).astype(int)), color, 1)
                        if confidence[joints[-1]] >= .4:
                            point = tuple(np.rint(points[joints[-1]]).astype(int))
                            cv2.circle(annotated, point, 4 if reflected else 2, color, 1 if reflected else -1)
                x0, y0, x1, y1 = bounds
                tile = np.hstack([contain(annotated, 480, 270), contain(annotated[y0:y1, x0:x1], 320, 270)])
                strip = np.full((46, 800, 3), 25, np.uint8)
                label = f'f{frame}; source {source["source_start_frame"]+frame}; motion {result["kicking_foot_candidate"]}'
                cv2.putText(strip, label, (8, 18), 0, .46, (240, 240, 240), 1)
                cv2.putText(strip, 'L blue / R orange; normal filled / mirror outlined; AI anatomy unverified', (8, 38), 0, .39, (240, 240, 240), 1)
                tiles.append(np.vstack([strip, tile]))
                metadata.append(record)
        finally:
            cap.release()
        while len(tiles)%2:
            tiles.append(np.zeros_like(tiles[0]))
        path = args.output_dir/(cid+'_paired_foot_source.jpg')
        if not cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])):
            raise OSError('Could not save paired foot source diagnostic')
        _write(args.output_dir/(cid+'_paired_foot_source.json'), dict(clip_id=cid,
            source_clip_sha256=source['clip_sha256'], cropped_pose_sha256=_hash(crop_path),
            replay_result_sha256=_hash(replay_path), diagnostic_script_sha256=_hash(Path(__file__)),
            source_sheet_sha256=_hash(path), frames=metadata, human_verified=False,
            accuracy_validated=False, original_inference_bgr_pixels_not_stored=True))
        print(json.dumps(dict(clip_id=cid, source_frames=len(frames), sheet=str(path))), flush=True)


if __name__ == '__main__':
    main()
