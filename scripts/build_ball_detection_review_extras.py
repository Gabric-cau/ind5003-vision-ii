"""Evidence selection for visual QA only; never feeds inference or association."""
from pathlib import Path
import argparse
import json
from PIL import Image, ImageDraw
from build_ball_detection_seed_evidence import source_frame, evidence_tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True)
    out = parser.parse_args().result_dir
    # Approximate visible-ball locations select already cached detections for inspection.
    checks = [
        ('esp_rus_p07_ramos_esp15', 110, (915, 445)),
        ('esp_rus_p08_cheryshev_rus6', 63, (841, 443)),
        ('esp_rus_p09_aspas_esp17', 100, (906, 440)),
        ('rus_cro_p08_vida_cro21', 85, (967, 404)),
    ]
    tiles, records = [], []
    for cid, frame, xy in checks:
        cache = json.loads((out / 'detections' / f'{cid}.json').read_text(encoding='utf-8'))
        ds = cache['detections'][str(frame)]
        d = min(ds, key=lambda q: (q['x'] - xy[0]) ** 2 + (q['y'] - xy[1]) ** 2)
        im = source_frame(cache['video_path'], frame)
        tile = evidence_tile(im, d, cid)
        # Mark the center of the detector's actual measurement, not the inspection hint.
        ImageDraw.Draw(tile).ellipse((146, 141, 154, 149), outline='red', width=1)
        name = f'{cid}_raw_ball_check.jpg'
        tile.save(out / 'media' / name)
        records.append({'clip_id': cid, 'detection': d, 'image': 'media/' + name,
                        'selection': 'assistant inspection of fresh raw candidates only',
                        'used_as_inference_input': False, 'human_verified': False})
        tiles.append(tile)
    grid = Image.new('RGB', (1200, 270), (25, 25, 25))
    for i, tile in enumerate(tiles):
        grid.paste(tile, (300 * i, 0))
    grid.save(out / 'media' / 'raw_ball_checks.jpg')
    extras = [
        ('esp_rus_p01_iniesta_esp6', 54), ('esp_rus_p01_iniesta_esp6', 67),
        ('esp_rus_p02_smolov_rus10', 120), ('esp_rus_p02_smolov_rus10', 137),
        ('rus_cro_p05_fernandes_rus2', 96), ('rus_cro_p04_kovacic_cro8', 104),
    ]
    other_tiles = []
    for cid, frame in extras:
        result = json.loads((out / 'results' / f'{cid}.json').read_text(encoding='utf-8'))
        ds = result['variants']['detector_and_motion']['trajectory']
        d = next((q for q in ds if q['frame'] == frame), None)
        cache = json.loads((out / 'detections' / f'{cid}.json').read_text(encoding='utf-8'))
        im = source_frame(cache['video_path'], frame)
        tile = evidence_tile(im, d, cid)
        name = f'{cid}_extra_f{frame}.jpg'
        tile.save(out / 'media' / name)
        other_tiles.append(tile)
        records.append({'clip_id': cid, 'frame': frame, 'detection': d,
                        'image': 'media/' + name, 'selection': 'current mixed association',
                        'used_as_inference_input': False, 'human_verified': False})
    grid = Image.new('RGB', (900, 540), (25, 25, 25))
    for i, tile in enumerate(other_tiles):
        grid.paste(tile, (300 * (i % 3), 270 * (i // 3)))
    grid.save(out / 'media' / 'additional_association_checks.jpg')
    (out / 'extra_evidence.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
