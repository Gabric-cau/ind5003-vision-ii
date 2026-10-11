"""Original-pixel comparisons for association changes; no outcome labels."""
from pathlib import Path
import argparse
import sys
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / '.deps_ball_mask'))
import cv2
from PIL import Image
from scripts.build_ball_detection_seed_evidence import evidence_tile
from scripts.run_ball_direction_corridor_review import load, save, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); out = args.directory.resolve(); rows = []
    for row in load(out / 'summary.json')['rows']:
        cid = row['clip_id']; frames = row['changed_source_frames']
        if not frames: continue
        r = load(out / 'results' / f'{cid}.json')
        new = {q['frame']: q for q in r['trajectory_in_focus']}
        old = {q['frame']: q for q in r['control_same_window']['trajectory_in_focus']}
        selected = frames if len(frames) <= 6 else frames[:3] + frames[-3:]
        panel = Image.new('RGB', (600, 270 * len(selected)), (25, 25, 25))
        cap = cv2.VideoCapture(r['source_video_path']); originals = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(selected))
        for f in range(min(selected), max(selected) + 1):
            ok, bgr = cap.read(); assert ok
            if f in selected: originals[f] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        evidence = []
        for i, f in enumerate(selected):
            for j, (label, points) in enumerate([('CONTROL', old), ('NARROW LANE', new)]):
                q = points.get(f)
                tile = evidence_tile(originals[f], q, f'{cid} f{f} {label}')
                if q is None:
                    # Replace the seed-specific inherited text in this blank tile.
                    from PIL import ImageDraw
                    tile = Image.new('RGB', (300, 270), (25, 25, 25)); draw = ImageDraw.Draw(tile)
                    draw.text((7, 7), f'f{f} {label}', fill='white')
                    draw.text((15, 125), 'No selected observation', fill=(255, 160, 160))
                panel.paste(tile, (300 * j, 270 * i))
            evidence.append({'frame': f, 'control_id': old[f]['detection_id'] if f in old else None,
                             'new_id': new[f]['detection_id'] if f in new else None,
                             'region': 'incoming' if f <= r['phase']['main_end_frame'] else 'short_context'})
        name = f'{cid}_changes.jpg'; panel.save(out / 'media' / name, quality=95)
        rows.append({'clip_id': cid, 'panel': name, 'frames': evidence, 'source_sha256': r['source_sha256']})
        print(f'{cid}: original pixels {selected}', flush=True)
    save(out / 'changed_pixel_evidence.json', {'rows': rows, 'human_verified': False,
         'note': '80x80 original-pixel crops at each selected candidate center; enlarged with nearest-neighbor sampling'})
    save(out / 'pixel_evidence_protocol.json', {'created_at': datetime.now(timezone.utc).isoformat(),
         'code_sha256': {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                         'scripts/build_ball_detection_seed_evidence.py': sha(ROOT / 'scripts/build_ball_detection_seed_evidence.py')},
         'outcome_labels_used': False, 'used_for_association': False, 'human_verified': False})
    (out / 'source_snapshot' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__': main()
