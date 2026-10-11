"""Validate all retest artifacts and publish separate assistant visual findings."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import html
import json
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=Path, required=True)
    out = parser.parse_args().result_dir.resolve()
    protocol, summary = load(out / 'protocol.json'), load(out / 'summary.json')
    ids = protocol['canonical_clips']
    reviews = load(out / 'visual_review.json')
    assert len(ids) == len(set(ids)) == len(reviews) == 39
    assert {r['clip_id'] for r in reviews} == set(ids)
    assert {p.stem for p in (out / 'results').glob('*.json')} == set(ids)
    assert {p.stem for p in (out / 'detections').glob('*.json')} == set(ids)
    assert summary['processed_clips'] == 39 and not summary['errors']
    assert protocol['old_ball_trajectory_input'] is False
    assert sha(protocol['weights']) == protocol['weights_sha256']
    for file, value in protocol['code_sha256'].items():
        assert sha(ROOT / file) == value, file
        assert sha(out / 'source_snapshot' / Path(file).name) == value, file
    seed_evidence = {r['clip_id']: r for r in load(out / 'seed_evidence.json')}
    raw_checks = {r['clip_id']: r for r in load(out / 'extra_evidence.json')
                  if r['selection'].startswith('assistant inspection')}
    assert len(raw_checks) == 4
    by_id = {r['clip_id']: r for r in reviews}
    tables, cards, total_frames, sampled = [], [], 0, 0
    labels = {'identity_drift': '抽样确认跟错目标',
              'partial_no_observed_drift': '抽样部分跟上，尚未验收全段',
              'contact_ambiguous': '接触附近身份待核对',
              'seed_only': '真球起点后未建立飞行段',
              'wrong_seed_no_flight': '起点选错，未建立飞行段'}
    for cid in ids:
        r, c = load(out / 'results' / f'{cid}.json'), load(out / 'detections' / f'{cid}.json')
        v = by_id[cid]
        assert r['fresh_model_inference'] and c['fresh_model_inference']
        assert not r['old_ball_trajectory_input'] and not c['old_ball_trajectory_input']
        assert sha(c['video_path']) == c['source_sha256']
        n = c['end_frame_exclusive'] - c['start_frame']
        total_frames += n
        sampled += len(set(r['summary']['sample_frames']))
        assert r['summary']['decoded_frames'] == n
        assert sha(out / 'media' / r['summary']['video']) == r['summary']['video_sha256']
        known = {(d['frame'], d['detection_id']) for ds in c['detections'].values() for d in ds}
        for name, variant in r['variants'].items():
            assert variant['old_trajectory_supplied'] is False and variant['deterministic_replay']
            assert [d['frame'] for d in variant['trajectory']] == sorted(set(d['frame'] for d in variant['trajectory']))
            if name == 'detector_only':
                assert all((d['frame'], d['detection_id']) in known for d in variant['trajectory'])
            for d in variant['trajectory']:
                assert c['start_frame'] <= d['frame'] < c['end_frame_exclusive']
        for f in v['evidence_frames'] + v['drift_evidence_frames']:
            assert c['start_frame'] <= f < c['end_frame_exclusive']
        if cid in raw_checks:
            d, pic = raw_checks[cid]['detection'], raw_checks[cid]['image']
            seed_ok = False
        else:
            se = seed_evidence[cid]
            d = next(q for q in c['detections'][str(se['frame'])] if q['detection_id'] == se['detection_id'])
            pic, seed_ok = f'media/{cid}_seed.jpg', True
        assert (d['frame'], d['detection_id']) in known
        v['verified_neural_detection'] = {'frame': d['frame'], 'detection_id': d['detection_id'],
            'xy': [d['x'], d['y']], 'confidence': d['confidence'], 'sources': d['sources'],
            'image': pic, 'reviewer': 'assistant_visual_review', 'human_verified': False}
        v['seed_representative_is_ball'] = seed_ok
        v['seed_representative_image'] = f'media/{cid}_seed.jpg'
        for image_path in [pic, v['seed_representative_image'], 'media/' + r['summary']['contact_sheet']]:
            with Image.open(out / image_path) as im:
                im.verify()
        seed_text = '代表帧是真球' if seed_ok else '代表帧选错'
        label = labels[v['flight_tracking']]
        tables.append(f'<tr data-kind="{v["flight_tracking"]}" data-search="{cid}">'
            f'<td><a href="#{cid}">{cid}</a></td><td>有，第{d["frame"]}帧</td>'
            f'<td class="{"bad" if not seed_ok else ""}">{seed_text}</td><td>{label}</td></tr>')
        evidence = ', '.join(map(str, v['evidence_frames']))
        drift = ', '.join(map(str, v['drift_evidence_frames'])) or '未确认'
        cards.append(f'<details id="{cid}" data-kind="{v["flight_tracking"]}" data-search="{cid}">'
            f'<summary>{cid} · {label}</summary><p>{html.escape(v["note"])}</p>'
            f'<p>真球证据帧：{evidence}；跟错目标证据帧：{drift}。'
            f'一次神经检测的核对记录：{d["detection_id"]}，原模型分数 {d["confidence"]:.3f}。</p>'
            f'<div class="pair"><figure><img src="{pic}" loading="lazy"><figcaption>原像素真球检测证据</figcaption></figure>'
            f'<figure><img src="media/{cid}_seed.jpg" loading="lazy"><figcaption>自动起点代表帧：{seed_text}</figcaption></figure></div>'
            f'<video controls preload="none" src="media/{r["summary"]["video"]}"></video>'
            f'<img src="media/{r["summary"]["contact_sheet"]}" loading="lazy">'
            f'<p><a href="results/{cid}.json">关联与预测数据</a> · <a href="detections/{cid}.json">新推理原始候选</a></p></details>')
    counts = dict(Counter(r['flight_tracking'] for r in reviews))
    audit = {'created_at': datetime.now(timezone.utc).isoformat(), 'canonical_clips': 39,
        'fresh_inference_clips': 39, 'source_hash_verified_clips': 39,
        'video_hash_verified_clips': 39, 'rendered_and_decoded_clips': 39,
        'source_window_frames': total_frames, 'assistant_visual_review_clips': 39,
        'contact_sheet_unique_frames_total': sampled,
        'ball_seen_at_least_once_clips': sum(r['ball_detected'] == 'yes' for r in reviews),
        'correct_seed_representative_clips': sum(r['seed_representative_is_ball'] for r in reviews),
        'wrong_seed_representative_clips': sum(not r['seed_representative_is_ball'] for r in reviews),
        'tracking_visual_categories': counts, 'full_frame_by_frame_identity_validated': False,
        'human_verified': False, 'accuracy_validated': False, 'accuracy_estimate': None,
        'all_raw_clip_frames_retested': False, 'old_ball_trajectory_input': False,
        'review_scope': 'assistant reviewed every stationary-seed representative, every six-frame contact sheet, and targeted extra original-pixel crops; prediction alone is not detection; sampled drift is a failure finding, not an error-rate estimate'}
    save(out / 'visual_review.json', reviews)
    save(out / 'validation.json', audit)
    summary['visual_review'] = 'assistant_sampled_all_39_complete'
    summary['visual_review_summary'] = audit
    save(out / 'summary.json', summary)
    with (out / 'visual_review.csv').open('w', encoding='utf-8-sig', newline='') as file:
        fields = ['clip_id', 'ball_detected', 'verified_neural_frame', 'verified_neural_detection_id',
                  'seed_representative_is_ball', 'flight_tracking', 'evidence_frames', 'drift_evidence_frames',
                  'note', 'reviewer', 'human_verified']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for r in reviews:
            writer.writerow({k: (';'.join(map(str, r[k])) if k.endswith('_frames') else r.get(k))
                for k in fields if not k.startswith('verified_neural_')} | {
                    'verified_neural_frame': r['verified_neural_detection']['frame'],
                    'verified_neural_detection_id': r['verified_neural_detection']['detection_id']})
    options = ''.join(f'<option value="{key}">{value}</option>' for key, value in labels.items())
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>39条球识别独立重跑与原帧核对</title><style>
body{{max-width:1180px;margin:30px auto;background:#15191f;color:#e1e6ed;font:16px/1.65 system-ui,sans-serif;padding:0 18px}}
a{{color:#91c9ff}}h1{{line-height:1.3}}.lead{{font-size:22px}}.box{{padding:16px;background:#202b38;border-radius:8px}}.bad{{color:#ffb69e}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #485461;padding:8px;text-align:left}}
details{{margin:20px 0;border:1px solid #485461;padding:14px}}summary{{cursor:pointer;font-weight:650}}video,img{{width:100%;height:auto}}
.pair{{display:flex;gap:20px;flex-wrap:wrap}}figure{{width:300px;margin:0}}figcaption{{text-align:center}}input,select{{padding:10px;margin:12px 8px 12px 0;background:#222b35;color:#eee;border:1px solid #68798d;border-radius:4px}}[hidden]{{display:none!important}}
@media(max-width:700px){{table{{display:block;overflow:auto}}h1{{font-size:27px}}}}
</style><h1>39 条球识别：重新推理复测</h1>
<p class="lead">39 条都至少有一次真球检测；持续跟踪仍未解决。</p>
<div class="box"><p>本轮从原视频重新推理，旧球轨迹没有输入。39 条主射门窗口全部完成，共 {total_frames:,} 帧，处理错误为 0；39 个回放全部解码通过。</p>
<p>AI 逐条查看静止起点代表帧、每条 6 帧对照图及必要的附加原像素截图：确认 39 条各有至少一次神经检测命中真球；自动起点代表帧中 35 条为球，4 条选错；含运动关联在 30 条中出现明确跟错目标。6 条抽样部分跟上，1 条接触附近身份待核对，2 条未建立飞行段。</p></div>
<p>“找到一次真球”主要由触球前证据支持，不等于飞行全过程识别成功。以上是 AI 抽样失败分析，没有用户人工真值，也没有计算准确率、召回率或全段轨迹合格率；6 条抽样未见漂移的片段也未做全帧验收。</p>
<p>范围是清单中的全部 39 条标准点球，每条采用现有自动触球/主镜头提议，从触球前 0.4 秒到后 2.3 秒并受主镜头结束限制。它没有遍历原片中的全部机位和回放；Aspas 的自动窗口只有 27 帧。检测使用本地 YOLO11n，全图加自动球门区域，低阈值候选逐帧保留。没有读取旧球轨迹、人工球点或进球标签；本轮没有修改检测器或跟踪参数。</p>
<p>绿色：检测＋运动候选关联；蓝色：纯神经检测关联；黄色：自动静止起点；灰色：原始神经检测候选；橙色加号：仅预测。右侧放大优先显示绿色，否则显示蓝色/起点/预测，请结合主画面颜色查看。运动亮斑本身不等于识别为足球，橙色预测不计为检测。</p>
<p class="bad">关键回归：Iniesta 第 54 帧检测到飞行球，第 67 帧仍错误关联到网与广告。此前保留正确旧轨迹的报告继续保留，这次复测表明从头运行的关联仍会失败。</p>
<p><a href="validation.json">范围与验证记录</a> · <a href="visual_review.csv">逐条核对表</a> · <a href="protocol.json">冻结模型与参数</a></p>
<p>输入片段或球员名筛选；点击片段查看慢速回放和原像素证据。</p>
<input id="search" placeholder="如 iniesta / ramos" aria-label="筛选片段"><select id="kind" aria-label="筛选问题"><option value="">全部片段</option>{options}</select>
<table><thead><tr><th>片段</th><th>至少一次真球</th><th>自动起点</th><th>含运动关联抽样结论</th></tr></thead><tbody>{''.join(tables)}</tbody></table>
{''.join(cards)}
<details><summary>全部自动起点代表帧（含选错的 4 条）</summary>{''.join(f'<img loading="lazy" src="media/seed_grid_{k}.jpg">' for k in range(1,5))}</details>
<script>const search=document.getElementById('search'),kind=document.getElementById('kind');function filter(){{const q=search.value.trim().toLowerCase();document.querySelectorAll('[data-search]').forEach(el=>el.hidden=!el.dataset.search.toLowerCase().includes(q)||(kind.value&&el.dataset.kind!==kind.value));}}search.addEventListener('input',filter);kind.addEventListener('change',filter);window.addEventListener('hashchange',()=>{{const el=document.getElementById(location.hash.slice(1));if(el&&el.tagName==='DETAILS'){{search.value='';kind.value='';filter();el.open=true;el.scrollIntoView();}}}});</script></html>'''
    (out / 'report.html').write_text(body, encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
