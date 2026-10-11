"""All-39 bounded-shot association replay and causal corridor overlays."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / '.deps_ball_mask'))
import cv2
import numpy as np
from src.ball_continuity_v4 import ContinuityConfig, track_ball_continuity
from src.ball_direction_corridor import DirectionCorridorConfig, corridor_forecasts
from src.ball_mask_seed import associate_with_seed, MaskSeedConfig, white_mask, measure_mask_support
from src.ball_shot_focus import ShotFocusConfig, focus_shot
from src.goal_plane_rectification import project_image_point
from scripts.run_ball_mask_seed_experiment import crop, draw_text, FFMPEG
from scripts.run_ball_mask_continuity_review import collision_geometry
from scripts.run_ball_shot_focus_review import goal_polygon, REASONS

cv2.setNumThreads(2)
PARENT = ROOT / 'outputs/ball_mask_review_ready_2026-10-11'
CACHE = ROOT / 'outputs/ball_detection_all_2026-10-11/detections'


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p, data): Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def current_pixel_evidence(c, pool):
    # Motion difference can contain the previous ball location. Measure the
    # current raw image at the proposed box, without changing candidate IDs.
    cfg = MaskSeedConfig(maximum_saturation=110, minimum_value=135, maximum_green_excess=25)
    cap = cv2.VideoCapture(c['video_path']); cap.set(cv2.CAP_PROP_POS_FRAMES, c['start_frame'])
    evidence = {}
    try:
        for f in sorted(int(k) for k in pool):
            ok, raw = cap.read(); assert ok
            mask = white_mask(raw, cfg)
            for q in pool[str(f)]:
                if 'motion_blob' in q.get('sources', []):
                    evidence[q['detection_id']] = measure_mask_support(mask, q, cfg)
    finally: cap.release()
    enriched = {f: [{**q, **({'current_pixel_mask_support': evidence[q['detection_id']]['mask_support']}
                   if q['detection_id'] in evidence else {})} for q in ds] for f, ds in pool.items()}
    return enriched, evidence


def replay(out):
    if out.exists(): raise FileExistsError(out)
    for folder in ['results', 'media', 'source_snapshot']: (out / folder).mkdir(parents=True)
    ids = load(PARENT / 'protocol.json')['canonical_clips']
    assert len(ids) == len(set(ids)) == 39
    cfg, lane_cfg, focus_cfg = ContinuityConfig(), DirectionCorridorConfig(), ShotFocusConfig()
    files = [Path(__file__), ROOT / 'src/ball_direction_corridor.py', ROOT / 'src/ball_continuity_v4.py',
             ROOT / 'src/ball_mask_seed.py', ROOT / 'src/ball_shot_focus.py',
             ROOT / 'src/event_ball_tracking.py', ROOT / 'src/goal_event_tracker_v2.py',
             ROOT / 'src/goal_plane_rectification.py', ROOT / 'scripts/run_ball_shot_focus_review.py',
             ROOT / 'scripts/run_ball_mask_continuity_review.py', ROOT / 'scripts/run_ball_mask_seed_experiment.py',
             ROOT / 'tests/test_ball_direction_corridor.py']
    for file in files: (out / 'source_snapshot' / file.name).write_bytes(file.read_bytes())
    save(out / 'protocol.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'canonical_clips': ids,
        'continuity_config': asdict(cfg), 'corridor_config': asdict(lane_cfg), 'focus_config': asdict(focus_cfg),
        'parent_protocol_sha256': sha(PARENT / 'protocol.json'), 'parent_inputs': str(PARENT / 'inputs'),
        'neural_inference_repeated': False, 'tracker_reassociated': True,
        'same_raw_candidates_as_parent': True, 'seed_source': 'unchanged automatic mask seed',
        'fresh_current_pixel_evidence_for_motion_blobs': True,
        'current_pixel_mask_config': asdict(MaskSeedConfig(maximum_saturation=110, minimum_value=135, maximum_green_excess=25)),
        'current_pixel_policy': 'soft reward penalty only; strict motion-bootstrap mask disabled after all-39 replay; no raw candidate removed; neural candidates unchanged',
        'manual_ball_points_used': False, 'outcome_labels_used': False, 'visual_review_used_for_inference': False,
        'replay_window': 'original start to kick+0.85 seconds, bounded by primary camera end',
        'full_raw_clip_coverage': False, 'human_verified': False, 'accuracy_validated': False,
        'code_sha256': {str(f.relative_to(ROOT)): sha(f) for f in files}})
    rows = []
    for i, cid in enumerate(ids, 1):
        inp_file = PARENT / 'inputs' / f'{cid}.json'; old_file = PARENT / 'results' / f'{cid}.json'
        inp, old, c = load(inp_file), load(old_file), load(CACHE / f'{cid}.json')
        variant = old['variants']['mask_and_radius_35']; seed, sd = variant['seed'], variant['seed_diagnostics']
        assert sha(c['video_path']) == c['source_sha256'] == inp['source_sha256'] == old['source_sha256']
        assert sha(CACHE / f'{cid}.json') == inp['raw_neural_cache_sha256']
        candidate_hash = hashlib.sha256(json.dumps(inp['candidates'], sort_keys=True).encode()).hexdigest()
        assert candidate_hash == inp['candidate_hash'] == old['candidate_hash']
        end = min(c['end_frame_exclusive'], c['kick_frame'] + math.ceil(focus_cfg.maximum_analysis_seconds * c['fps']) + 1)
        pool = {f: ds for f, ds in inp['candidates'].items() if c['start_frame'] <= int(f) < end}
        pool, pixel_evidence = current_pixel_evidence(c, pool)
        meta = {f: {'coordinate_supported': c['camera']['transforms'].get(f) is not None} for f in pool}
        kwargs = dict(tracker=track_ball_continuity, config=cfg, frame_metadata=meta,
                      collision_regions=collision_geometry(c, {}))
        p, d = associate_with_seed(pool, c['kick_frame'], inp['anchors'], inp['goal_center'], seed, sd,
                                  corridor_config=lane_cfg, **kwargs)
        repeat, rd = associate_with_seed(pool, c['kick_frame'], inp['anchors'], inp['goal_center'], seed, sd,
                                        corridor_config=lane_cfg, **kwargs)
        control, cd = associate_with_seed(pool, c['kick_frame'], inp['anchors'], inp['goal_center'], seed, sd, **kwargs)
        assert p == repeat and d == rd
        assert candidate_hash == hashlib.sha256(json.dumps(inp['candidates'], sort_keys=True).encode()).hexdigest()
        allowed = {(q['frame'], q['detection_id']) for ds in pool.values() for q in ds}
        assert all((q['frame'], q['detection_id']) in allowed for q in p)
        effective = [q for q in seed if q['frame'] <= p[0]['frame']] if p and d['tracking_status'] == 'candidate' else seed
        polygon = goal_polygon(c)
        kept, phase = focus_shot(p, effective, inp['goal_center'], c['kick_frame'], c['fps'], end,
                                 goal_polygon=polygon, config=focus_cfg)
        control_kept, control_phase = focus_shot(control, effective, inp['goal_center'], c['kick_frame'], c['fps'], end,
                                                 goal_polygon=polygon, config=focus_cfg)
        forecasts = corridor_forecasts(p, d.get('motion_states', []), c['start_frame'], phase['focus_end_frame'] + 1,
                         config=lane_cfg, max_gap_frames=cfg.max_gap_frames, reachable_speed_ratio=cfg.reachable_speed_ratio)
        new_by_frame = {q['frame']: q['detection_id'] for q in kept}
        old_by_frame = {q['frame']: q['detection_id'] for q in control_kept}
        changed = [f for f in sorted(set(new_by_frame) | set(old_by_frame)) if new_by_frame.get(f) != old_by_frame.get(f)]
        r = {'clip_id': cid, 'source_video_path': c['video_path'], 'source_sha256': c['source_sha256'],
             'parent_inputs_sha256': sha(inp_file), 'parent_result_sha256': sha(old_file), 'candidate_hash': candidate_hash,
             'seed': effective, 'automatic_seed': seed, 'trajectory': p, 'trajectory_in_focus': kept, 'tracking': d,
             'current_pixel_evidence': pixel_evidence,
             'phase': phase, 'corridor_forecasts': forecasts,
             'control_same_window': {'trajectory': control, 'trajectory_in_focus': control_kept, 'tracking': cd, 'phase': control_phase},
             'changed_source_frames': changed, 'two_replays_equal': True, 'candidate_pool_unchanged': True,
             'outcome': 'unknown', 'human_verified': False, 'accuracy_validated': False, 'media': None}
        save(out / 'results' / f'{cid}.json', r)
        rows.append({'clip_id': cid, 'main_end': phase['main_end_frame'], 'focus_end': phase['focus_end_frame'],
                     'reason': phase['reason'], 'control_main_end': control_phase['main_end_frame'],
                     'changed_source_frames': changed, 'selected_count': len(kept), 'control_selected_count': len(control_kept),
                     'outside_corridor_rejections': d.get('rejected_outside_forward_corridor', 0), 'media': None})
        print(f'[{i}/39] {cid}: main {control_phase["main_end_frame"]}->{phase["main_end_frame"]}; changed {changed}', flush=True)
    save(out / 'summary.json', {'processed_clips': len(rows), 'rows': rows, 'human_verified': False, 'accuracy_validated': False})
    save(out / 'validation.json', {'association_clips': 39, 'two_replays_equal': True, 'input_candidates_unchanged': True,
          'selected_points_are_source_candidates': True, 'source_hashes_match': True,
          'rendered_clips': 0, 'browser_interaction_verified': False, 'human_verified': False, 'accuracy_validated': False})
    report(out, rows)


def render_one(c, r, out):
    cid = c['clip_id']; phase = r['phase']; end = phase['focus_end_frame']; main = phase['main_end_frame']
    video = out / 'media' / f'{cid}.mp4'
    proc = subprocess.Popen([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', '1120x500', '-r', str(c['fps'] / 3), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-threads', '2', '-crf', '21', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(c['video_path']); cap.set(cv2.CAP_PROP_POS_FRAMES, c['start_frame'])
    points = {q['frame']: q for q in r['trajectory_in_focus']}; seeds = {q['frame']: q for q in r['seed']}
    earlier = {q['frame']: q for q in r['control_same_window']['trajectory_in_focus']}
    bands = {q['frame']: q for q in r['corridor_forecasts']}
    samples = [min(end, max(c['start_frame'], f)) for f in [c['kick_frame'] - 3, c['kick_frame'] + 4, main - 3, main, main + 2, end]]
    scale = np.array([640 / c['width'], 360 / c['height']]); tiles = {}
    try:
        for f in range(c['start_frame'], end + 1):
            ok, raw = cap.read(); assert ok
            im = np.full((500, 1120, 3), 25, np.uint8); scene = cv2.resize(raw, (640, 360))
            band = bands.get(f); transform = c['camera']['transforms'].get(str(f))
            if band is not None and transform is not None:
                inverse = np.linalg.inv(np.asarray(transform, float))
                polygon = [project_image_point(inverse, q) for q in band['polygon_stabilized_xy']]
                if all(q is not None for q in polygon):
                    poly = np.rint(np.array(polygon) * scale).astype(np.int32)
                    layer = scene.copy(); cv2.fillPoly(layer, [poly], (220, 190, 40))
                    scene = cv2.addWeighted(scene, .87, layer, .13, 0)
                    cv2.polylines(scene, [poly], True, (220, 190, 40), 1)
            old = earlier.get(f)
            if old is not None: cv2.circle(scene, tuple(np.rint(np.array([old['x'], old['y']]) * scale).astype(int)), 10, (255, 100, 50), 1)
            p = points.get(f) or seeds.get(f); context = main is not None and f > main
            color = (20, 190, 255) if context else (70, 255, 70)
            if p is not None:
                cv2.circle(scene, tuple(np.rint(np.array([p['x'], p['y']]) * scale).astype(int)), 7, color, 2)
                im[75:235, 665:825] = crop(raw, p)
            elif old is not None:
                im[75:235, 665:825] = crop(raw, old)
            im[40:400, :640] = scene
            draw_text(im, f'{cid} | source frame {f} | kick {c["kick_frame"]}', (10, 23), scale=.48)
            draw_text(im, 'CONTACT CONTEXT' if context else 'INCOMING SHOT: finite forward corridor', (10, 426), color, scale=.47)
            draw_text(im, 'Green: new candidate | Blue: same-window control | Cyan: search corridor', (10, 452), scale=.44)
            draw_text(im, 'Corridor is prediction only; goal / miss remains UNKNOWN', (10, 477), scale=.46)
            draw_text(im, 'Raw pixels at selected candidate', (655, 54), scale=.44)
            draw_text(im, p['detection_id'] if p is not None else 'No new observation', (655, 260), scale=.44)
            draw_text(im, f'Corridor full width: {2 * band["half_width_px"]:.1f} px' if band else 'No active corridor', (655, 298), scale=.44)
            draw_text(im, f'Main end {main} | Context end {end}', (655, 330), scale=.44)
            draw_text(im, 'Direction updates from prior observations', (655, 362), scale=.37)
            draw_text(im, 'Single side candidate cannot reset the lane', (655, 394), scale=.37)
            proc.stdin.write(im.tobytes())
            if f in samples: tiles[f] = im.copy()
    finally:
        cap.release(); proc.stdin.close()
    error = proc.stderr.read().decode(errors='replace')
    if proc.wait(): raise RuntimeError(error)
    cap = cv2.VideoCapture(str(video)); count = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        assert frame.shape == (500, 1120, 3); count += 1
    cap.release(); assert count == end - c['start_frame'] + 1
    sheet = np.vstack([np.hstack([tiles[samples[j]], tiles[samples[j + 1]]]) for j in (0, 2, 4)])
    cv2.imwrite(str(out / 'media' / f'{cid}_sheet.jpg'), sheet)
    return {'video': video.name, 'sheet': f'{cid}_sheet.jpg', 'decoded_frames': count,
            'source_start_frame': c['start_frame'], 'source_end_frame_exclusive': end + 1,
            'video_sha256': sha(video), 'sample_source_frames': samples}


def report(out, rows):
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>球：窄通道连续关联</title><style>body{max-width:1200px;margin:24px auto;padding:0 16px;background:#15191f;color:#e4eaf0;font:16px/1.65 system-ui}a{color:#a6d4ff}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border:1px solid #52616f;padding:8px}video,img{width:100%}details{border:1px solid #52616f;margin:15px 0;padding:12px}summary{cursor:pointer;font-weight:650}.note{color:#ffd3a4}</style>
<h1>球：窄通道连续关联</h1>
<p>沿已观测到的最近运动方向建立有限宽度通道。宽度采用之前候选的球尺寸与测量余量，漏帧时少量增加，半宽不超过32像素；横向误差不再随球速成比例扩大。通道随连续观测逐步调整方向，保留缓慢弯曲；拟接入的新目标不能用自己的大框扩大通道。</p>
<p>正常关联必须处在向前通道内。通道外至少55°的大转向进入独立假设，仍需之前球位置附近有自动碰撞几何线索、至少3个实测点支持；仅有守门员附近候选或单帧跳点不足以换轨。碰撞几何仍未经确认，不能把换向当作是否进球的标签。</p>
<p>另外复测当前帧的运动候选框是否有亮色、低饱和度像素。缺乏支持的运动点只降低得分；复测发现硬掩膜会提前丢失部分轨迹，本轮关闭该硬门槛。所有原始候选仍保留，神经模型候选不受该掩膜约束，白线或衣物仍可能通过。</p>
<p>绿色：窄通道＋当前像素支持的新候选；蓝色：相同原始候选、相同短窗口的原关联对照；青色通道：当前帧关联前、仅由更早观测建立的预测范围。橙色：球门附近最多约0.2秒的短观察。球门后尾段不用于没进判定。</p>
<p class="note">这是关联实验，39段完整运行和规则检查不代表准确率。窄通道可能漏掉遮挡后的真球，同一通道内的衣物仍可能误跟；目前没有人工逐帧真值，是否进球仍为unknown。原始候选池、此前种子及旧报告保留。</p>
<p><a href="protocol.json">运行协议</a> · <a href="validation.json">检查记录</a> · <a href="tests.json">规则检查</a> · <a href="visual_review.json">AI原像素抽样核对</a> · <a href="README.md">复现与限制</a></p>'''
    rendered = sum(bool(q['media']) for q in rows)
    doc += f'<p>关联复测 {len(rows)}/39；已生成并逐帧解码视频 {rendered}/39。主要窗口上限为自动触球后0.85秒，沿用球门参考范围和短观察策略。</p>'
    doc += '<table><tr><th>片段</th><th>原／新主要结束帧</th><th>改变帧数</th><th>新边界</th></tr>'
    for rr in rows:
        cid = rr['clip_id']; doc += f'<tr><td><a href="#{cid}">{cid}</a></td><td>{rr["control_main_end"]} / {rr["main_end"]}</td><td>{len(rr["changed_source_frames"])}</td><td>{REASONS[rr["reason"]]}</td></tr>'
    doc += '</table>'
    notes_file = out / 'visual_review.json'
    notes = {q['clip_id']: q for q in load(notes_file).get('rows', [])} if notes_file.exists() else {}
    for rr in rows:
        cid = rr['clip_id']; doc += f'<details id="{cid}"><summary>{cid}</summary><p>{REASONS[rr["reason"]]}；改变帧：{rr["changed_source_frames"]}。</p>'
        if cid in notes: doc += f'<p class="note">AI抽样核对：{html.escape(notes[cid]["note"])}</p>'
        if rr['media']:
            m = rr['media']; doc += f'<video controls preload="none" src="media/{m["video"]}"></video><img loading="lazy" src="media/{m["sheet"]}">'
        else: doc += '<p>视频生成待完成。</p>'
        doc += f'<p><a href="results/{cid}.json">通道、预测及关联诊断</a> · <a href="../ball_shot_focus_2026-10-11/report.html#{cid}">此前短窗口</a> · <a href="../ball_mask_review_ready_2026-10-11/report.html#{cid}">完整窗口对照</a></p></details>'
    doc += '<script>function openTarget(){const e=document.getElementById(location.hash.slice(1));if(e){e.open=true;e.scrollIntoView();}}window.addEventListener("hashchange",openTarget);openTarget();</script></html>'
    (out / 'report.html').write_text(doc, encoding='utf-8')


def render_existing(out):
    protocol = load(out / 'protocol.json')
    for rel, digest in protocol['code_sha256'].items():
        assert sha(ROOT / rel) == sha(out / 'source_snapshot' / Path(rel).name) == digest
    summary = load(out / 'summary.json')
    for i, row in enumerate(summary['rows'], 1):
        cid = row['clip_id']; result_file = out / 'results' / f'{cid}.json'; r = load(result_file)
        c = load(CACHE / f'{cid}.json'); assert sha(c['video_path']) == r['source_sha256']
        if row['media'] is None:
            row['media'] = r['media'] = render_one(c, r, out); save(result_file, r)
            save(out / 'summary.json', summary)
        report(out, summary['rows'])
        print(f'[{i}/39] video {cid}: {row["media"]["decoded_frames"]} decoded', flush=True)
    validation = load(out / 'validation.json')
    validation.update(rendered_clips=39, decoded_video_frames=sum(q['media']['decoded_frames'] for q in summary['rows']),
                      media_rows=[{'clip_id': q['clip_id'], **q['media']} for q in summary['rows']])
    save(out / 'validation.json', validation)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--render-existing', action='store_true'); parser.add_argument('--association-only', action='store_true')
    args = parser.parse_args(); out = args.output_dir.resolve()
    if args.render_existing: render_existing(out)
    else:
        replay(out)
        if not args.association_only: render_existing(out)


if __name__ == '__main__': main()
