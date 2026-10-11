"""Render original-pixel evidence for each independently detected ball seed."""
from pathlib import Path
import argparse
import json
import io
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = ROOT / 'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'


def source_frame(path, frame):
    data = subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-i', str(path),
        '-vf', f'select=eq(n\\,{frame})', '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
        check=True, capture_output=True).stdout
    return Image.open(io.BytesIO(data)).convert('RGB')


def evidence_tile(im, d, label, size=(300, 270)):
    tile = Image.new('RGB', size, (25, 25, 25))
    draw = ImageDraw.Draw(tile)
    draw.text((7, 7), label, fill='white')
    if d:
        x, y = round(d['x']), round(d['y'])
        crop = im.crop((x - 40, y - 40, x + 40, y + 40))
        tile.paste(crop.resize((200, 200), Image.Resampling.NEAREST), (50, 45))
        draw.text((7, 27), f"f{d['frame']} conf={d['confidence']:.3f} id={d['detection_id']}", fill='white')
    else:
        draw.text((15, 125), 'No multi-frame ball seed', fill=(255, 160, 160))
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.result_dir
    tiles, evidence = [], []
    for file in sorted((out / "results").glob("*.json")):
        r = json.loads(file.read_text(encoding="utf-8"))
        ds = r["stationary_seed"]
        c = json.loads(Path(r["cache_path"]).read_text(encoding="utf-8"))
        if not ds:
            evidence.append({"clip_id": file.stem, "seed_available": False})
            tile = evidence_tile(None, None, file.stem)
        else:
            d = max(ds, key=lambda q: q["confidence"])
            im = source_frame(c['video_path'], d['frame'])
            tile = evidence_tile(im, d, file.stem)
            evidence.append({"clip_id": file.stem, "seed_available": True, "frame": d["frame"],
                             "detection_id": d["detection_id"], "xy": [d["x"], d["y"]],
                             "confidence": d["confidence"], "is_ground_truth": False})
        tile.save(out / 'media' / f'{file.stem}_seed.jpg')
        tiles.append(tile)
    for k in range(0, len(tiles), 12):
        group = list(tiles[k:k + 12])
        while len(group) % 4:
            group.append(Image.new('RGB', (300, 270), (25, 25, 25)))
        grid = Image.new('RGB', (1200, 270 * (len(group) // 4)), (25, 25, 25))
        for j, tile in enumerate(group):
            grid.paste(tile, (j % 4 * 300, j // 4 * 270))
        grid.save(out / 'media' / f'seed_grid_{k // 12 + 1}.jpg')
    (out / "seed_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
