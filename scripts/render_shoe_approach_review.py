"""Render original-source observations and frozen visible-landing comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def board(video, inputs, poses, result, target):
    interval = inputs['interval_frames']
    focus = set(np.rint(np.linspace(*interval, 6)).astype(int).tolist())
    for event in result.get('landings') or []:
        focus.update(event['stance_frames'][:3])
    focus.update(result['dense_foot']['joint_region_frames'][:3])
    frames = sorted(focus)
    lookup = {(row['frame'], row['reflected']): row for row in poses}
    tile_w, tile_h = 320, 310
    columns = 3
    canvas = np.full((72 + int(np.ceil(len(frames) / columns)) * tile_h, columns * tile_w, 3), 248, np.uint8)
    cv2.putText(canvas, result['clip_id'], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 30, 30), 1)
    cv2.putText(canvas, 'Original pixels / model overlay; visible proposals, not verified ground contacts',
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .48, (70, 70, 70), 1)
    cap = cv2.VideoCapture(str(video))
    receipts = []
    try:
        for index, frame in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, pixels = cap.read()
            if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - (frame + 1)) > .5:
                raise ValueError('Review source frame clock mismatch')
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if digest != inputs['decoded_frame_hashes'][str(frame)]:
                raise ValueError('Review pixels differ from the exact inferred source frame')
            normal, mirror = lookup[(frame, False)], lookup[(frame, True)]
            box = normal['anchor_bbox']
            height = box[3] - box[1]
            x0, y0, x1, y1 = (max(0, int(box[0] - .15 * height)), max(0, int(box[1] - .12 * height)),
                              min(pixels.shape[1], int(box[2] + .15 * height)),
                              min(pixels.shape[0], int(box[3] + .10 * height)))
            crop = pixels[y0:y1, x0:x1].copy()
            painted = crop.copy()
            for row, thickness in ((mirror, 1), (normal, 2)):
                if row['landmarks_source_xy'] is None:
                    continue
                points = np.asarray(row['landmarks_source_xy']) - [x0, y0]
                confidence = np.asarray(row['landmark_confidence'])
                for indices, color in (([23, 25, 27, 29, 31], (45, 65, 220)),
                                       ([24, 26, 28, 30, 32], (180, 140, 20))):
                    for first, second in zip(indices, indices[1:]):
                        if min(confidence[first], confidence[second]) >= .5:
                            cv2.line(painted, tuple(np.rint(points[first]).astype(int)),
                                     tuple(np.rint(points[second]).astype(int)), color, thickness)
            pair = np.hstack((crop, painted))
            scale = min((tile_w - 16) / pair.shape[1], (tile_h - 58) / pair.shape[0])
            tile = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            ox, oy = index % columns * tile_w, 72 + index // columns * tile_h
            source_seconds = (inputs['source_mapping']['source_start_frame'] + frame) / inputs['fps']
            cv2.putText(canvas, f'f{frame} / source {source_seconds:.3f}s', (ox + 8, oy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 40, 40), 1)
            labels = [event['foot'] for event in result.get('landings') or [] if frame in event['stance_frames']]
            label = 'stance support: ' + ', '.join(labels) if labels else 'sampled observation'
            if frame in result['dense_foot']['joint_region_frames']:
                label += ' / shoe region'
            cv2.putText(canvas, label, (ox + 8, oy + 39), cv2.FONT_HERSHEY_SIMPLEX, .38, (70, 70, 70), 1)
            canvas[oy + 48:oy + 48 + tile.shape[0], ox + (tile_w - tile.shape[1]) // 2:
                   ox + (tile_w - tile.shape[1]) // 2 + tile.shape[1]] = tile
            receipts.append(dict(frame=frame, source_seconds=source_seconds, decoded_pixels_sha256=digest,
                                 source_crop_xyxy=[x0, y0, x1, y1]))
    finally:
        cap.release()
    if not cv2.imwrite(str(target), canvas):
        raise ValueError('Source review board could not be written')
    return receipts


PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IND5003 / Visible Footwork</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f7;color:#222;font:14px Arial,sans-serif;letter-spacing:0}header{background:#fff;border-bottom:1px solid #d5d8db;padding:18px 22px}h1{font-size:20px;margin:0}main{max-width:1200px;margin:auto;padding:20px}.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:16px}select{max-width:100%;padding:8px;border:1px solid #b9bfc4;border-radius:4px;background:#fff;font:inherit}.status{color:#655c32;font-size:12px}.metrics{display:flex;gap:24px;flex-wrap:wrap;border-top:1px solid #d5d8db;border-bottom:1px solid #d5d8db;padding:12px 0;margin-bottom:16px}.metric{display:grid;gap:5px}.metric span{color:#687178;font-size:12px}.metric strong{font-size:16px;font-weight:500}img{display:block;max-width:100%;height:auto;margin:0 auto}a{color:#176b5e}nav{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:18px;font-size:12px}.missing{padding:30px 0;color:#666}details{margin:14px 0}summary{cursor:pointer}pre{overflow:auto;max-height:350px;background:#fff;border:1px solid #d5d8db;padding:12px;font-size:11px}.table{overflow:auto;margin-top:25px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:9px;border-bottom:1px solid #d8dcdf;text-align:left;white-space:nowrap}th{background:#fff}.chart{margin-top:24px}@media(max-width:650px){header{padding:14px}main{padding:14px}.controls{display:grid;gap:8px}.metrics{gap:18px}.metric strong{font-size:14px}}
img[hidden],.missing[hidden]{display:none}
</style></head><body><header><h1>IND5003 / Visible Footwork</h1></header><main>
<nav><a href="../comparison_39.csv" download>Comparison CSV</a><a href="../human_reference_template.json" download>Reference JSON</a><a href="../evaluation_pending.json">Evaluation</a></nav>
<div class="controls"><select id="event" aria-label="Penalty event"></select><span class="status" id="status"></span></div><div class="metrics" id="metrics"></div>
<img id="evidence" alt="Original course frames and paired anatomical model overlays"><div id="missing" class="missing" hidden></div><details><summary>Source and Prediction Records</summary><pre id="record"></pre></details>
<div class="table"><table><thead><tr><th>Event</th><th>Ankle extrema</th><th>Stance proposals</th><th>Previous foot</th><th>Dense foot</th></tr></thead><tbody id="rows"></tbody></table></div>
<img class="chart" src="comparison.png" alt="Baseline extrema and conservative transition proposals for the same course events">
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),sel=document.getElementById('event');const v=x=>x===null||x===undefined?'Unknown':String(x);for(const r of data){const o=document.createElement('option');o.value=r.clip_id;o.textContent=r.name;sel.append(o);const tr=document.createElement('tr');for(const val of [r.name,r.baseline_ankle_count,r.candidate_count,r.baseline_foot,r.striking_foot]){const td=document.createElement('td');td.textContent=v(val);tr.append(td)}document.getElementById('rows').append(tr)}
function choose(){const r=data.find(x=>x.clip_id===sel.value),image=document.getElementById('evidence');image.hidden=!r.image;if(r.image)image.src=r.image;const missing=document.getElementById('missing');missing.hidden=!!r.image;missing.textContent=r.reason||'No source-bound approach observations';document.getElementById('status').textContent=r.status.replaceAll('_',' ');document.getElementById('record').textContent=JSON.stringify(r.record,null,2);const metrics=document.getElementById('metrics');metrics.replaceChildren();for(const [name,val]of [['Ankle extrema',r.baseline_ankle_count],['Stance proposals (subset)',r.candidate_count],['Previous foot',r.baseline_foot],['Dense foot',r.striking_foot],['Human review','Pending']]){const d=document.createElement('div');d.className='metric';const span=document.createElement('span'),strong=document.createElement('strong');span.textContent=name;strong.textContent=v(val);d.append(span,strong);metrics.append(d)}}sel.onchange=choose;choose();
</script></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis-dir', 'dense-dir', 'clips-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new review output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = read(args.analysis_dir / 'summary.json')
    records, data = [], []
    for row in sorted(summary['rows'], key=lambda row: row['event_id']):
        cid = row['clip_id']
        path = args.analysis_dir / 'events' / cid / 'landing_result.json'
        if sha(path) != row['result_sha256']:
            raise ValueError('Frozen landing prediction changed before rendering')
        result = read(path)
        image = None
        if (args.dense_dir / 'events' / cid / 'poses33.json').exists():
            event_dir = args.dense_dir / 'events' / cid
            inputs, poses = read(event_dir / 'inputs.json'), read(event_dir / 'poses33.json')
            if sha(event_dir / 'inputs.json') != result['dense_inputs_sha256'] or sha(event_dir / 'poses33.json') != result['dense_poses_sha256']:
                raise ValueError('Dense source evidence changed before rendering')
            video = args.clips_dir / inputs['source_mapping']['clip_filename']
            if sha(video) != result['source_sha256']:
                raise ValueError('Review video differs from dense inference')
            image = cid + '.jpg'
            receipts = board(video, inputs, poses, result, args.output_dir / image)
            records.append(dict(clip_id=cid, image=image, image_sha256=sha(args.output_dir / image),
                landing_result_sha256=sha(path), source_sha256=result['source_sha256'], frames=receipts,
                enlarged_for_display=True, human_verified=False))
        data.append(dict(clip_id=cid, name=result['source_mapping']['kicker'].replace('_', ' ').title(),
                         image=image, **{key: result.get(key) for key in ('baseline_ankle_count', 'candidate_count',
                             'baseline_foot', 'striking_foot', 'status', 'reason')}, record=result))
    fig, ax = plt.subplots(figsize=(10, 12))
    y = np.arange(len(data))
    old = np.array([np.nan if item['baseline_ankle_count'] is None else item['baseline_ankle_count'] for item in data])
    new = np.array([np.nan if item['candidate_count'] is None else item['candidate_count'] for item in data])
    ax.scatter(old, y - .12, color='#3d6a9b', label='Ankle extrema (baseline)', s=23)
    ax.scatter(new, y + .12, color='#ad544d', label='Paired stance proposals (subset)', marker='s', s=23)
    ax.set_yticks(y, [item['name'] for item in data], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Automatic proposals; neither series is verified total steps')
    ax.set_title('Same 39 course events / different observation criteria', fontsize=13)
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'comparison.png', dpi=150)
    plt.close(fig)
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (args.output_dir / 'index.html').write_text(PAGE.replace('__DATA__', payload), encoding='utf-8')
    write(args.output_dir / 'receipt.json', dict(records=records, renderer_sha256=sha(Path(__file__)),
        index_sha256=sha(args.output_dir / 'index.html'), comparison_sha256=sha(args.output_dir / 'comparison.png'),
        original_course_videos_only=True, additional_video_used=False, human_verified=False))
    print(args.output_dir / 'index.html')


if __name__ == '__main__':
    main()
