"""39-clip mask/radius ablation on fresh raw candidates; no manual point inputs."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import html
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.deps_ball_mask'))
import cv2
import numpy as np
from src.ball_mask_seed import MaskSeedConfig, white_mask, measure_mask_support, choose_seed, associate_with_seed
from src.ball_continuity import ContinuityConfig, motion_forecasts
from src.goal_plane_rectification import project_image_point
from scripts.probe_goal_event_tracking import motion_blobs, source_time_indices
from scripts.run_ball_continuity_probe import collision_geometry
from scripts.build_ball_detection_seed_evidence import evidence_tile, source_frame

cv2.setNumThreads(2)
FFMPEG = ROOT / 'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract_support(c, cfg):
    cap = cv2.VideoCapture(c['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, c['start_frame'])
    evidence = {}
    for f in range(c['start_frame'], c['end_frame_exclusive']):
        ok, fr = cap.read()
        if not ok:
            raise ValueError('Source decode incomplete')
        if c['kick_frame'] - 12 <= f <= c['kick_frame'] + 3:
            mask = white_mask(fr, cfg)
            for d in c['detections'][str(f)]:
                evidence[d['detection_id']] = measure_mask_support(mask, d, cfg)
    cap.release()
    return evidence


def run(c, baseline, cfg):
    evidence = extract_support(c, cfg)
    timing = source_time_indices(c)
    blobs = motion_blobs(c)  # Existing white motion candidates; identical across variants.
    candidates = {f: [{**d, **timing[int(f)]} for d in ds + blobs.get(f, [])]
                  for f, ds in c['detections'].items()}
    before = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
    metadata = {f: {'coordinate_supported': c['camera']['transforms'].get(f) is not None} for f in candidates}
    regions = collision_geometry(c, {})
    anchors, goal, variants = baseline['anchors'], baseline['goal_reference_center'], {}
    for name, use_mask, radius in [('radius_only_35', False, cfg.tight_foot_radius_px),
                                 ('mask_only_85', True, 85.),
                                 ('mask_and_radius_35', True, cfg.tight_foot_radius_px)]:
        seed, diag = choose_seed(c['detections'], c['kick_frame'], anchors, evidence, use_mask=use_mask, radius=radius)
        path, tracking = associate_with_seed(candidates, c['kick_frame'], anchors, goal, seed, diag,
            config=ContinuityConfig(), frame_metadata=metadata, collision_regions=regions)
        repeat, repeated = associate_with_seed(candidates, c['kick_frame'], anchors, goal, seed, diag,
            config=ContinuityConfig(), frame_metadata=metadata, collision_regions=regions)
        assert path == repeat and tracking == repeated
        valid = {(d['frame'], d['detection_id']) for ds in candidates.values() for d in ds}
        assert all((d['frame'], d['detection_id']) in valid for d in path)
        variants[name] = {'seed': seed, 'seed_diagnostics': diag, 'trajectory': path,
            'tracking': tracking, 'forecasts': motion_forecasts(path, c['start_frame'], c['end_frame_exclusive'],
                config=ContinuityConfig(), model_states=tracking.get('motion_states')),
            'deterministic_replay': True, 'manual_ball_points_used': False}
    assert hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest() == before
    old_seed = baseline['stationary_seed']
    rejected = [d['detection_id'] for d in old_seed if not evidence[d['detection_id']]['mask_support']]
    return {'clip_id': c['clip_id'], 'mask_evidence': evidence, 'variants': variants,
        'old_seed_mask_rejected_ids': rejected, 'old_seed_point_count': len(old_seed),
        'input_candidates_unchanged': True, 'all_flight_neural_candidates_preserved': True,
        'old_ball_trajectory_used_for_inference': False, 'baseline_used_only_for_comparison': True,
        'accuracy_validated': False, 'human_verified': False, 'outcome': 'unknown',
        'source_sha256': c['source_sha256'], 'source_video_path': c['video_path']}


def draw_text(im, value, xy, color=(230, 235, 240), scale=.42):
    cv2.putText(im, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def crop(frame, d, size=160):
    if d is None:
        return np.full((size, size, 3), 25, np.uint8)
    x, y = round(d['x']), round(d['y'])
    p = cv2.copyMakeBorder(frame, 40, 40, 40, 40, cv2.BORDER_CONSTANT)
    return cv2.resize(p[y:y + 80, x:x + 80], (size, size), interpolation=cv2.INTER_NEAREST)


def render(c, baseline, result, out, mask_config=None):
    cid = c['clip_id']
    variant = result['variants']['mask_and_radius_35']
    old = {p['frame']: p for p in baseline['variants']['detector_and_motion']['trajectory']}
    now = {p['frame']: p for p in variant['trajectory']}
    seeds = {p['frame']: p for p in variant.get('effective_stationary_seed', variant['seed'])}
    old_seeds = {p['frame']: p for p in baseline['stationary_seed']}
    predicted = {p['frame']: p for p in variant['forecasts']}
    samples = [min(c['end_frame_exclusive'] - 1, max(c['start_frame'], c['kick_frame'] + n))
               for n in [-5, 0, 5, 10, 20, 40]]
    if cid == 'esp_rus_p01_iniesta_esp6':
        samples[-2] = 67
    video = out / 'media' / f'{cid}.mp4'
    cmd = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', '1120x500', '-r', str(c['fps'] / 3), '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '21', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(c['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, c['start_frame'])
    tiles = {}
    scale = np.array([640 / c['width'], 360 / c['height']])
    for f in range(c['start_frame'], c['end_frame_exclusive']):
        ok, fr = cap.read()
        assert ok
        im = np.full((500, 1120, 3), 25, np.uint8)
        scene = cv2.resize(fr, (640, 360))
        b, s, p = old.get(f) or old_seeds.get(f), seeds.get(f), now.get(f)
        for d, colour, radius in [(b, (255, 145, 50), 11), (s, (30, 220, 255), 9), (p, (70, 255, 70), 7)]:
            if d:
                xy = tuple(np.rint(np.array([d['x'], d['y']]) * scale).astype(int))
                cv2.circle(scene, xy, radius, colour, 2)
        pred = predicted.get(f)
        tf = c['camera']['transforms'].get(str(f))
        if pred and tf is not None:
            xy = project_image_point(np.linalg.inv(np.array(tf)), pred['predicted_stabilized_xy'])
            if xy is not None:
                cv2.drawMarker(scene, tuple(np.rint(np.array(xy) * scale).astype(int)), (25, 165, 255), cv2.MARKER_CROSS, 10)
        im[40:400, :640] = scene
        draw_text(im, f'{cid} | source frame {f} | automatic kick {c["kick_frame"]}', (10, 22), scale=.49)
        d = p or s
        im[70:230, 660:820] = crop(fr, d)
        draw_text(im, 'New selected observation', (650, 52))
        draw_text(im, 'none' if d is None else f'{d["detection_id"]} ({d["x"]:.1f},{d["y"]:.1f})', (650, 252), scale=.38)
        if d:
            actual_mask = white_mask(fr, mask_config)
            mask_rgb = cv2.cvtColor(actual_mask, cv2.COLOR_GRAY2BGR)
            im[290:450, 660:820] = crop(mask_rgb, d)
        draw_text(im, 'White pixel mask (not identity)', (650, 276), scale=.37)
        draw_text(im, 'Earlier baseline target', (860, 52))
        im[70:230, 880:1040] = crop(fr, b)
        draw_text(im, 'none' if b is None else b['detection_id'], (875, 252), scale=.38)
        draw_text(im, 'Green: mask-seeded track | Blue: earlier baseline | Yellow: new seed', (10, 426), scale=.43)
        draw_text(im, 'Orange +: prediction only | Masks affect seed eligibility, flight candidates retained', (10, 452), scale=.4)
        draw_text(im, 'Experimental, no manual points, no goal/no-goal or final-position decision', (10, 477), scale=.4)
        proc.stdin.write(im.tobytes())
        if f in samples:
            tiles[f] = im.copy()
            cv2.imwrite(str(out / 'media' / f'{cid}_f{f}.jpg'), im)
    cap.release()
    proc.stdin.close()
    err = proc.stderr.read().decode(errors='replace')
    if proc.wait():
        raise RuntimeError(err)
    cap = cv2.VideoCapture(str(video))
    count = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        assert fr.shape == (500, 1120, 3)
        count += 1
    cap.release()
    assert count == c['end_frame_exclusive'] - c['start_frame']
    sheet = np.vstack([np.hstack([tiles[samples[i]], tiles[samples[i + 1]]]) for i in range(0, 6, 2)])
    cv2.imwrite(str(out / 'media' / f'{cid}_sheet.jpg'), sheet)
    # All three ablation representatives are actual original pixels.
    for name, v in result['variants'].items():
        seed = v.get('effective_stationary_seed', v['seed'])
        d = max(seed, key=lambda q: q['confidence']) if seed else None
        im = source_frame(c['video_path'], d['frame']) if d else None
        evidence_tile(im, d, f'{cid} {name}').save(out / 'media' / f'{cid}_{name}_seed.jpg')
    return {'video': video.name, 'decoded_frames': count, 'sample_frames': samples,
            'video_sha256': sha(video), 'contact_sheet': f'{cid}_sheet.jpg'}


def report(out, rows):
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>球起点掩膜对照实验</title><style>
body{max-width:1200px;margin:25px auto;padding:0 15px;background:#15191f;color:#e2e7ed;font:16px/1.65 system-ui}a{color:#9bd0ff}video,img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #55616f}details{margin:20px 0;padding:14px;border:1px solid #55616f}summary{cursor:pointer;font-weight:650}.seeds{display:flex;flex-wrap:wrap;gap:8px}.seeds figure{width:300px;margin:0}.warn{color:#ffc4a0}</style>
<h1>球的静止起点：掩膜与脚附近范围对照</h1>
<p>复用上轮重新推理得到的39条神经检测候选，从原视频像素计算白色／低饱和度掩膜。旧轨迹只供画面对照，没有输入新关联。</p>
<p>三组对照：只把脚附近半径从85缩到35像素；只加入白色掩膜（半径仍85）；掩膜加35像素半径。后续连续关联参数相同，所有飞行神经候选保留，没有用白色硬掩膜删掉飞行球。原有运动候选已经采用白色与运动差分。</p>
<p>绿色为“掩膜加35像素”的新关联，蓝色是上轮含运动关联，黄色是新静止起点，橙色加号仅预测。右侧白色区域是像素掩膜，不是足球身份标签。</p>
<p class="warn">本轮针对起点选择。缺失、反弹、球门附近误关联、自动结果与最终位置尚未验收。当前窗口仍受自动主镜头结束限制；没有按进球后／结果确定后偷偷截去错误。</p>
<p id="review-status">AI视觉核对结果待写入。点数和回放解码检查不代表准确率。</p>'''
    body += f'<p>已完成 {len(rows)} / 39 条。</p><table><tr><th>片段</th><th>原起点被掩膜排除点数</th><th>缩半径起点</th><th>只掩膜起点</th><th>掩膜＋半径起点</th><th>新路径结束帧</th></tr>'
    for r in rows:
        body += f'<tr><td><a href="#{r["clip_id"]}">{r["clip_id"]}</a></td><td>{r["old_mask_rejected"]}/{r["old_seed_n"]}</td>'
        for name in ['radius_only_35', 'mask_only_85', 'mask_and_radius_35']:
            body += f'<td>{r["variants"][name]["seed_n"]}</td>'
        body += f'<td>{r["variants"]["mask_and_radius_35"]["end"]}</td></tr>'
    body += '</table>'
    for r in rows:
        cid = r['clip_id']
        body += f'<details id="{cid}"><summary>{cid}</summary><div class="seeds">'
        for name, label in [('radius_only_35', '只收紧脚附近范围'), ('mask_only_85', '只加像素掩膜'), ('mask_and_radius_35', '掩膜＋脚附近范围')]:
            body += f'<figure><img loading="lazy" src="media/{cid}_{name}_seed.jpg"><figcaption>{label}</figcaption></figure>'
        body += f'</div><video controls preload="none" src="media/{cid}.mp4"></video><img loading="lazy" src="media/{cid}_sheet.jpg"><a href="results/{cid}.json">候选来源与三组关联数据</a></details>'
    body += '<script>window.addEventListener("hashchange",()=>{const el=document.getElementById(location.hash.slice(1));if(el&&el.tagName==="DETAILS"){el.open=true;el.scrollIntoView();}});</script></html>'
    (out / 'report.html').write_text(body, encoding='utf-8')
    save(out / 'summary.json', {'processed_clips': len(rows), 'rows': rows, 'accuracy_validated': False, 'human_verified': False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=ROOT / 'outputs/ball_detection_all_2026-10-11')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out, inp = args.output_dir.resolve(), args.input_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    for folder in ['results', 'media', 'source_snapshot']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    protocol = load(inp / 'protocol.json')
    ids = protocol['canonical_clips']
    assert len(ids) == len(set(ids)) == 39
    cfg = MaskSeedConfig()
    files = [Path(__file__), ROOT / 'src/ball_mask_seed.py', ROOT / 'src/ball_continuity.py',
             ROOT / 'src/event_ball_tracking.py', ROOT / 'scripts/probe_goal_event_tracking.py']
    for file in files:
        (out / 'source_snapshot' / file.name).write_bytes(file.read_bytes())
    save(out / 'protocol.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'canonical_clips': ids, 'mask_config': asdict(cfg), 'continuity_config': asdict(ContinuityConfig()),
        'input_candidates': str(inp / 'detections'), 'new_neural_inference': False,
        'fresh_pixel_masks': True, 'old_ball_trajectory_used_for_inference': False,
        'old_result_usage': 'automatic pose anchors and goal reference; old path for display only',
        'manual_ball_seed_used': False, 'outcome_labels_used': False,
        'event_endpoint_not_changed': True, 'full_raw_clip_coverage': False,
        'opencv': cv2.__version__, 'numpy': np.__version__, 'accuracy_validated': False,
        'code_sha256': {str(file.relative_to(ROOT)): sha(file) for file in files}})
    priority = ['esp_rus_p09_aspas_esp17', 'esp_rus_p01_iniesta_esp6', 'esp_rus_p07_ramos_esp15',
                'esp_rus_p08_cheryshev_rus6', 'rus_cro_p08_vida_cro21']
    ordered = priority + [cid for cid in ids if cid not in priority]
    rows = []
    for i, cid in enumerate(ordered, 1):
        begun = time.monotonic()
        print(f'[{i}/39] {cid}', flush=True)
        c, base = load(inp / 'detections' / f'{cid}.json'), load(inp / 'results' / f'{cid}.json')
        assert sha(c['video_path']) == c['source_sha256']
        result = run(c, base, cfg)
        result['media'] = render(c, base, result, out)
        assert sha(c['video_path']) == c['source_sha256']
        save(out / 'results' / f'{cid}.json', result)
        row = {'clip_id': cid, 'old_mask_rejected': len(result['old_seed_mask_rejected_ids']),
            'old_seed_n': len(base['stationary_seed']), 'elapsed_seconds': round(time.monotonic() - begun, 2),
            'variants': {name: {'seed_n': len(v['seed']), 'path_n': len(v['trajectory']),
                'end': v['trajectory'][-1]['frame'] if v['trajectory'] else None,
                'status': v['tracking']['tracking_status']} for name, v in result['variants'].items()},
            **result['media']}
        rows.append(row)
        print(json.dumps(row), flush=True)
        report(out, rows)
    assert len(rows) == 39


if __name__ == '__main__':
    main()
