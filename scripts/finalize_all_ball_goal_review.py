"""Publish source-pixel AI assessments beside frozen automatic suggestions.

Visual assessments are supplied separately after the algorithm replay. They
never become inputs to detection, association, or the automatic decision.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps_ball_mask'))
from PIL import Image


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


CONFLICT_NOTES = {
    22: '反弹分支选到了守门员头部及手部附近的候选；原画面显示真球进入左侧网底。自动反弹证据的球身份不可靠。',
    25: '扑救区域的白色候选仍在门框投影内，算法给出了倾向进球；原画面显示球被挡回门前。候选身份及门前深度仍有问题。',
    30: '被挡后的球在图像上仍处于门框内，二维投影被当成入网线索；原画面显示球返回门前。门框内的像素位置不能直接证明越过球门线。',
}
RULE_NAMES = {'goal_likely': '倾向进球', 'no_goal_likely': '倾向未进', 'unknown': '待定'}
VISUAL_NAMES = {'goal': '进球', 'no_goal': '未进'}
REASONS = {
    'continuous_projected_mouth_sequence': '多帧候选连续位于门框投影内；球门线深度未确立。',
    'mouth_sequence_before_later_turn': '先出现门框内连续候选，再发生换向；后来的换向未作为未进证据。',
    'supported_short_keeper_rebound': '门前预测接触附近出现多帧大角度分支；球身份及真实接触仍需核对。',
    'finite_ball_outside_mouth_near_arrival': '到达门前时，多帧球的有限范围位于横梁上方或侧柱外。',
    'insufficient_ball_identity_or_arrival_evidence': '球身份、到达位置或连续观察不足。',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dense-source-dir', type=Path, required=True)
    parser.add_argument('--test-log', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    protocol, automatic, visual = [load(out / name) for name in
        ['protocol.json', 'summary.json', 'visual_assessment.json']]
    ids = protocol['canonical_clips']
    assert len(ids) == len(set(ids)) == 39
    assert len(automatic['rows']) == len(visual['rows']) == 39
    assert visual['human_verified'] is False and visual['is_ground_truth'] is False
    assert not visual['outcome_labels_used_for_algorithm']
    # Freeze and verify all rule outputs before attaching any review metadata.
    frozen = {cid: sha(out / 'results' / f'{cid}.json') for cid in ids}
    test_log = args.test_log.read_text(encoding='utf-8-sig')
    match = re.search(r'Ran (\d+) tests', test_log)
    assert match and '\nOK' in test_log.replace('\r\n', '\n')
    test_count = int(match.group(1))
    for relative, expected in protocol['code_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
        assert sha(out / 'source_snapshot' / Path(relative).name) == expected, relative
    for source in args.dense_source_dir.glob('event_*_dense_*.jpg'):
        shutil.copy2(source, out / 'media' / source.name)
    shutil.copy2(args.test_log, out / 'validation_tests.log')
    shutil.copy2(__file__, out / 'source_snapshot' / Path(__file__).name)
    dense_builder = ROOT / 'tmp/build_goal_dense_panels.py'
    if dense_builder.exists():
        shutil.copy2(dense_builder, out / 'source_snapshot' / dense_builder.name)

    rows = []
    for index, (cid, a, v) in enumerate(zip(ids, automatic['rows'], visual['rows']), 1):
        assert a['event_id'] == v['event'] == index and a['clip_id'] == cid
        assert v['outcome'] in VISUAL_NAMES and not v.get('needs_dense_review', False)
        r = load(out / 'results' / f'{cid}.json')
        assert sha(Path(r['source_video_path'])) == r['source_sha256']
        fresh = Path(protocol['fresh_candidates_dir']) / f'{cid}.json'
        assert sha(fresh) == r['fresh_candidates_sha256']
        assert not r['outcome_labels_used_for_algorithm'] and not r['manual_ball_points_used']
        rebound = r['rebound']
        if rebound['turn_confirmed']:
            support = rebound['support_frames']
            assert len(support) >= rebound['config']['minimum_support_points']
            assert len(set(support)) == len(support)
            assert all(b-a <= rebound['config']['maximum_gap_frames'] for a,b in zip(support,support[1:]))
        conflict = a['rule_outcome'] != 'unknown' and a['rule_outcome'] != v['outcome'] + '_likely'
        panels = [out / 'media' / f'event_{index:02d}.jpg']
        panels += sorted((out / 'media').glob(f'event_{index:02d}_dense_*.jpg'))
        for panel in panels:
            with Image.open(panel) as im:
                im.verify()
        rows.append({
            **a, 'visual_outcome': v['outcome'], 'visual_evidence_frames': v['frames'],
            'visual_reason': v['reason'], 'reviewer': visual['reviewer'],
            'human_verified': False, 'is_ground_truth': False, 'accuracy_validated': False,
            'rule_visual_conflict': conflict,
            'conflict_note': CONFLICT_NOTES.get(index, '自动规则与AI画面判断不同，需要核对候选球身份和球门线前后的位置。') if conflict else '',
            'dense_source_reviewed': v.get('dense_source_reviewed', False),
            'source_sha256': r['source_sha256'],
            'review_panels_sha256': {str(p.relative_to(out)).replace('\\','/'): sha(p) for p in panels},
        })
    # No outcome decision or trajectory has been modified by finalization.
    assert frozen == {cid: sha(out / 'results' / f'{cid}.json') for cid in ids}
    visual_counts = Counter(r['visual_outcome'] for r in rows)
    rule_counts = Counter(r['rule_outcome'] for r in rows)
    conflicts = [r['event_id'] for r in rows if r['rule_visual_conflict']]
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'clips': 39,
        'visual_counts': dict(visual_counts), 'rule_counts': dict(rule_counts),
        'rule_visual_conflict_event_ids': conflicts, 'visual_review_complete': True,
        'human_verified': False, 'is_ground_truth': False, 'accuracy_validated': False,
        'outcome_labels_used_for_algorithm': False, 'manual_ball_points_used': False,
        'algorithm_records_frozen_sha256': frozen,
        'test_count': test_count, 'tests_passed': True,
        'fully_decoded_replay_clips': 39,
        'fully_decoded_replay_frames': sum(r['video_decoded_frames'] for r in rows),
        'visual_assessment_sha256': sha(out / 'visual_assessment.json'),
        'replay_sha256': {f'media/{cid}.mp4': sha(out / 'media' / f'{cid}.mp4') for cid in ids},
        'rows': rows,
    }
    save(out / 'review_summary.json', summary)
    with (out / 'outcomes.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['event_id','clip_id','visual_outcome','rule_outcome','rule_visual_conflict',
                  'visual_evidence_frames','visual_reason','human_verified','is_ground_truth']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'visual_evidence_frames': ','.join(map(str,row['visual_evidence_frames']))})

    e = html.escape
    table, cards = [], []
    for r in rows:
        index, cid = r['event_id'], r['clip_id']
        v, rule, conflict = r['visual_outcome'], r['rule_outcome'], r['rule_visual_conflict']
        cls = 'conflict' if conflict else ('pending' if rule == 'unknown' else '')
        attributes = f'data-outcome="{v}" data-rule="{rule}" data-conflict="{int(conflict)}"'
        status = '规则冲突' if conflict else ('自动待定' if rule == 'unknown' else '方向一致')
        table.append(f'<tr class="{cls}" {attributes}><td>{index:02d}</td>'
            f'<td><a href="#{cid}" class="clip-link">{e(cid)}</a></td>'
            f'<td><b class="{v}">{VISUAL_NAMES[v]}</b></td><td>{RULE_NAMES[rule]}</td>'
            f'<td>{status}</td><td>{e(r["visual_reason"])}</td></tr>')
        raw = f'media/event_{index:02d}.jpg'
        dense = sorted((out / 'media').glob(f'event_{index:02d}_dense_*.jpg'))
        dense_html = ''.join(f'<img loading="lazy" src="media/{p.name}" alt="第{index}段原画面连续帧">' for p in dense)
        selected = sorted((out / 'media').glob(f'event_{index:02d}_selected_*.jpg'))
        candidates = ''.join(f'<img loading="lazy" src="media/{p.name}" alt="算法所选候选原像素">' for p in selected)
        warning = f'<p class="warning">{e(r["conflict_note"])}</p>' if conflict else ''
        frames = '、'.join(map(str,r['visual_evidence_frames']))
        cards.append(f'''<details class="event {cls}" id="{cid}" {attributes}>
<summary>{index:02d} · {e(cid)} · <span class="{v}">{VISUAL_NAMES[v]}（AI画面判断）</span> · {status}</summary>
<p>{e(r['visual_reason'])} 原片证据帧：{frames}。</p>{warning}
<video controls preload="none" src="media/{cid}.mp4"></video>
<p class="subtle">慢放为原速的1/3。绿圈是当前关联候选，蓝圈是先前候选，青线是门框提案；圈出的候选仍可能跟错。</p>
<details><summary>查看原画面事件帧（无检测圈）</summary><img loading="lazy" src="{raw}" alt="第{index}段原画面事件帧">{dense_html}</details>
<details><summary>算法证据：{RULE_NAMES[rule]}</summary>
<p>{e(REASONS.get(r['reason'],r['reason']))} 自动与画面判断：{status}。</p>
{candidates}<p><a href="results/{cid}.json">关联候选与冻结的规则结果 JSON</a></p></details>
</details>''')
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>39段罚球是否进球 · 事件检测与画面复核</title>
<style>body{margin:0;background:#12171c;color:#e7edf2;font:16px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:28px 24px 70px}h1{font-size:28px;margin:0 0 12px}h2{font-size:21px;margin-top:28px}a{color:#8ad0ff}.counts{display:flex;gap:16px;flex-wrap:wrap}.count{background:#202a32;border:1px solid #3c4a55;border-radius:8px;padding:12px 22px}.count b{font-size:30px;display:block}.goal{color:#80e4a5}.no_goal{color:#ffbd84}.subtle{color:#adb9c3;font-size:14px}.warning{border-left:4px solid #ed7979;background:#33252a;padding:10px 14px}.filters{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}button{border:1px solid #566775;border-radius:5px;background:#202a32;color:#e7edf2;padding:9px 15px;cursor:pointer}button[aria-pressed=true]{background:#27627d;border-color:#83ceef}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #35414c;padding:10px 9px;text-align:left;vertical-align:top}th{background:#25313b}td:first-child{width:35px}td:nth-child(2){min-width:235px}td:nth-child(3),td:nth-child(4),td:nth-child(5){white-space:nowrap}.conflict{background:#33252a}.pending{background:#29291e}details.event{margin:14px 0;padding:13px 16px;border:1px solid #465361;border-radius:6px;scroll-margin-top:12px}summary{cursor:pointer;font-weight:bold}details details{margin:15px 0}video{width:100%;max-width:1120px;background:black}img{display:block;width:100%;max-width:1200px;height:auto;margin:12px 0}.hidden{display:none!important}code{word-break:break-all}</style></head><body><main>
<h1>39段罚球：是否进球</h1>
<p>依据原片的短事件画面，AI逐段判断并保留证据帧。主要看球到达门框时的位置，以及接触后的去向；进网后的反弹不改判未进。</p>
<div class="counts"><div class="count"><b class="goal">VISUAL_GOALS</b>进球 · AI画面判断</div><div class="count"><b class="no_goal">VISUAL_MISSES</b>未进 · AI画面判断</div><div class="count"><b>RULE_UNKNOWN</b>自动规则待定</div><div class="count"><b>CONFLICT_COUNT</b>自动规则与画面判断冲突</div></div>
<p>自动规则输出：RULE_GOALS段倾向进球，RULE_MISSES段倾向未进，RULE_UNKNOWN段待定。表中的进球/未进来自AI画面判读，尚未经过人工逐段确认；不能把39段复核完成当作自动模型的准确率。</p>
<p class="warning">RULE_CONFLICT_DESCRIPTION 球身份误关联和门前/网后深度是当前自动判断的主要问题，详情已单独标出。</p>
<div class="filters"><button data-filter="all" aria-pressed="true">全部39段</button><button data-filter="miss" aria-pressed="false">未进球</button><button data-filter="pending" aria-pressed="false">自动待定</button><button data-filter="conflict" aria-pressed="false">规则冲突</button></div>
<p id="visible-count" class="subtle">显示39段。点击片名展开对应回放。</p>
<div class="scroll"><table><thead><tr><th>序号</th><th>片段</th><th>AI画面判断</th><th>自动规则</th><th>核对状态</th><th>画面理由</th></tr></thead><tbody>TABLE_ROWS</tbody></table></div>
<h2>逐段证据与慢放</h2>EVENT_CARDS
<details><summary>复现与验证</summary><p>本次运行新推理NEW_INFERENCE段神经检测，另REUSED_INFERENCE段复用同源检测缓存；39段均运行了本轮关联和规则判断。TEST_COUNT项测试通过，39段回放共DECODED_COUNT帧完成逐帧解码。画面判断记录单独保存，没有写回检测、关联或规则输入。</p>
<p>门框单应变换只表达图像中的位置，不能确立球整体越过球门线。此报告未使用人工球坐标或官方比赛结果作算法输入，也未计算带独立人工真值的准确率。</p>
<p><a href="outcomes.csv">全部结果 CSV</a> · <a href="review_summary.json">复核汇总与文件哈希 JSON</a> · <a href="visual_assessment.json">AI画面判断记录</a> · <a href="protocol.json">运行协议与源码哈希</a> · <a href="validation_tests.log">测试日志</a></p></details>
</main><script>const buttons=[...document.querySelectorAll('[data-filter]')];function accept(el,f){return f==='all'||(f==='miss'&&el.dataset.outcome==='no_goal')||(f==='pending'&&el.dataset.rule==='unknown')||(f==='conflict'&&el.dataset.conflict==='1')}function filter(f){buttons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.filter===f)));document.querySelectorAll('tr[data-outcome],details.event').forEach(el=>el.classList.toggle('hidden',!accept(el,f)));document.querySelector('#visible-count').textContent='显示'+document.querySelectorAll('tr[data-outcome]:not(.hidden)').length+'段。点击片名展开对应回放。'}buttons.forEach(b=>b.addEventListener('click',()=>filter(b.dataset.filter)));function reveal(id){const el=document.getElementById(id);if(el){filter('all');el.open=true;el.scrollIntoView({behavior:'smooth',block:'start'})}}document.querySelectorAll('.clip-link').forEach(a=>a.addEventListener('click',()=>reveal(a.hash.slice(1))));if(location.hash)reveal(location.hash.slice(1));</script></body></html>'''
    substitutions = {'VISUAL_GOALS':visual_counts['goal'], 'VISUAL_MISSES':visual_counts['no_goal'],
        'RULE_UNKNOWN':rule_counts['unknown'], 'RULE_GOALS':rule_counts['goal_likely'],
        'RULE_MISSES':rule_counts['no_goal_likely'], 'CONFLICT_COUNT':len(conflicts),
        'TEST_COUNT':test_count,'DECODED_COUNT':summary['fully_decoded_replay_frames'],
        'TABLE_ROWS':'\n'.join(table),'EVENT_CARDS':'\n'.join(cards),
        'NEW_INFERENCE':protocol['new_neural_inference_clips'],
        'REUSED_INFERENCE':protocol['reused_fresh_neural_inference_clips'],
        'RULE_CONFLICT_DESCRIPTION':('仍有冲突：第'+ '、'.join(map(str,conflicts))+'段。') if conflicts else '本轮规则方向与已有AI画面判断一致，但仍需独立人工核对。'}
    for token, value in substitutions.items():
        page = page.replace(token,str(value))
    (out / 'report.html').write_text(page,encoding='utf-8')
    refs = set(re.findall(r'(?:src|href)="([^"]+)"',page))
    local = [x for x in refs if not x.startswith('#')]
    assert all((out/x).is_file() for x in local)
    save(out/'validation_report.json',{
        'canonical_clips':39,'source_hashes_verified':39,'fresh_cache_hashes_verified':39,
        'rule_records_unchanged_by_visual_review':True,'all_rule_code_snapshot_hashes_verified':True,
        'rebound_independent_support_count_and_frame_gaps_verified':True,
        'tests_passed':test_count,'replays_fully_decoded':39,'replay_frames_decoded':summary['fully_decoded_replay_frames'],
        'html_local_file_references_verified':len(local),'raw_panel_files_verified':sum(len(r['review_panels_sha256']) for r in rows),
        'html_rendered_in_browser':False,'human_verified':False,'accuracy_validated':False,
    })
    print(json.dumps({k:summary[k] for k in ['visual_counts','rule_counts','rule_visual_conflict_event_ids','test_count','fully_decoded_replay_frames']},ensure_ascii=False))


if __name__ == '__main__':
    main()
