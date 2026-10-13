"""Render source pixels beside normal/inverse-mirror shoe candidates for review."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-dir', type=Path, required=True)
    parser.add_argument('--clips-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = []
    cohort_protocol = read(args.cohort_dir / 'protocol.json')
    summary = read(args.cohort_dir / 'summary.json')
    for entry in summary['rows']:
        if entry['shoe_region_candidate'] == 'unknown':
            continue
        event_dir = args.cohort_dir / 'events' / entry['clip_id']
        result_path, pose_path = event_dir / 'result.json', event_dir / 'poses33.json'
        result, poses = read(result_path), read(pose_path)
        protocol = read(event_dir / 'protocol.json')
        if (sha(result_path) != entry['result_sha256']
                or protocol['code_sha256']['scripts/run_fresh_shoe_pose_experiment.py'] !=
                   cohort_protocol['code_sha256']['scripts/run_fresh_shoe_pose_experiment.py']
                or protocol['settings']['roi_reference_changes_between_frames'] is not False):
            raise ValueError('Review needs an exact fixed-ROI cohort member')
        source = result['source_mapping']
        video = args.clips_dir / source['clip_filename']
        if sha(video) != source['clip_sha256'] or sha(video) != protocol['source_sha256']:
            raise ValueError('Review source does not match the inference')
        inferred = {(row['frame'], row['reflected']): row for row in poses}
        if len(inferred) != len(poses) or len(poses) != result['actual_inference_calls']:
            raise ValueError('Duplicate or incorrectly counted pose calls')
        if sum(row['pose_count'] == 1 for row in poses) != result['detected_pose_calls']:
            raise ValueError('Detected pose counts disagree')
        if sum(row['landmarks_source_xy'] is not None for row in poses) != result['accepted_roi_pose_calls']:
            raise ValueError('Accepted ROI counts disagree')
        support = result['joint_near_shoe_and_paired_pose_frames']
        focus = support[0]
        frames = [focus - 1, focus, focus + 1]
        left, top, right, bottom = protocol['settings']['fixed_union_roi_xyxy']
        cap = cv2.VideoCapture(str(video))
        decoded = {}
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, pixels = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (frame + 1)) > .5:
                raise ValueError('Review decode clock mismatch')
            decoded[frame] = pixels
        cap.release()
        tile_w, tile_h, banner = 380, 380, 470
        board = np.full((banner + 3 * tile_h, 3 * tile_w, 3), 245, np.uint8)
        cv2.putText(board, f"{entry['clip_id']} | {entry['shoe_region_candidate']} shoe-region candidate",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(board, 'Source pixels / normal / inverse mirror; model review, not contact truth',
                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
        scene = decoded[focus].copy()
        cv2.rectangle(scene, (left, top), (right - 1, bottom - 1), (0, 235, 235), 2)
        scene = cv2.resize(scene, (720, 405), interpolation=cv2.INTER_AREA)
        board[58:463, 210:930] = scene
        pixel_receipts = []
        for column, frame in enumerate(frames):
            original = decoded[frame][top:bottom, left:right].copy()
            pixel_receipts.append(dict(frame=frame,
                source_pixels_sha256=hashlib.sha256(decoded[frame].tobytes()).hexdigest(),
                original_crop_pixels_sha256=hashlib.sha256(original.tobytes()).hexdigest()))
            for view in range(3):
                crop = original.copy()
                label = 'source pixels'
                if view:
                    row = inferred.get((frame, view == 2))
                    label = 'normal' if view == 1 else 'inverse mirror'
                    if row is None or row['landmarks_source_xy'] is None:
                        label += ' - no accepted pose'
                    else:
                        points = np.asarray(row['landmarks_source_xy']) - [left, top]
                        conf = np.asarray(row['landmark_confidence'])
                        for indices, color in (([23, 25, 27, 29, 31], (40, 40, 235)),
                                               ([24, 26, 28, 30, 32], (235, 90, 30))):
                            for a, b in zip(indices, indices[1:]):
                                if min(conf[a], conf[b]) >= .5:
                                    cv2.line(crop, tuple(np.rint(points[a]).astype(int)),
                                             tuple(np.rint(points[b]).astype(int)), color, 1)
                            for index in indices:
                                if conf[index] >= .5:
                                    cv2.circle(crop, tuple(np.rint(points[index]).astype(int)), 2, color, -1)
                        target = np.asarray(result['stationary_ball_reference']['xy']) - [left, top]
                        cv2.circle(crop, tuple(np.rint(target).astype(int)), 4, (0, 235, 235), 1)
                scale = min((tile_w - 12) / crop.shape[1], (tile_h - 44) / crop.shape[0])
                enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                x = column * tile_w + (tile_w - enlarged.shape[1]) // 2
                y = banner + view * tile_h + 34
                board[y:y + enlarged.shape[0], x:x + enlarged.shape[1]] = enlarged
                cv2.putText(board, f'{label} | f{frame}', (column * tile_w + 6, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
        target_path = args.output_dir / f"{entry['clip_id']}.png"
        if not cv2.imwrite(str(target_path), board):
            raise ValueError('Review board could not be saved')
        receipt.append(dict(clip_id=entry['clip_id'], shoe_region_candidate=entry['shoe_region_candidate'],
            supporting_frames=support, rendered_frames=frames, original_crop_xyxy=[left, top, right, bottom],
            source_sha256=source['clip_sha256'], result_sha256=sha(result_path), pose_sha256=sha(pose_path),
            protocol_sha256=sha(event_dir / 'protocol.json'), image=target_path.name,
            image_sha256=sha(target_path), source_pixel_receipts=pixel_receipts,
            crop_resampled_for_display=True, human_verified=False, anatomical_side_verified=False,
            shoe_contact_verified=False, accuracy_validated=False,
            counts_match_saved_calls=True))
    (args.output_dir / 'receipt.json').write_text(json.dumps(dict(records=receipt,
        renderer_sha256=sha(Path(__file__)), interpretation='Review of pixels and predictions, not human labels'),
        indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
