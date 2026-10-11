"""Render raw detector crops for assistant QA; read-only inference inputs."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
from build_ball_detection_seed_evidence import source_frame, evidence_tile

OUT = Path('outputs/ball_detection_all_2026-10-11')
for cid, frame in [('esp_rus_p08_cheryshev_rus6', 64), ('rus_cro_p08_vida_cro21', 87)]:
    c = json.loads((OUT / 'detections' / f'{cid}.json').read_text(encoding='utf-8'))
    im = source_frame(c['video_path'], frame)
    box = (700, 290, 1110, 490)
    crop = im.crop(box).resize((820, 400), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(crop)
    for d in c['detections'][str(frame)]:
        if box[0] <= d['x'] < box[2] and box[1] <= d['y'] < box[3]:
            x, y = (d['x'] - box[0]) * 2, (d['y'] - box[1]) * 2
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline='red', width=1)
            draw.text((x + 8, y), f"{d['detection_id']} {d['confidence']:.3f}", fill='yellow')
    crop.save(OUT / 'media' / f'{cid}_raw_overview_f{frame}.jpg')
