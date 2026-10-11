"""Publish separate AI visual notes and validate a completed review package."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import re
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();out=args.directory.resolve()
    protocol=load(out/'protocol.json');summary=load(out/'summary.json');review=load(out/'visual_review.json')
    ids=protocol['canonical_clips'];assert len(ids)==len(set(ids))==39
    assert len(review)==39 and set(r['clip_id'] for r in review)==set(ids)
    assert len(summary['rows'])==39
    checked=[]
    for row in review:
        r=load(out/'results'/f'{row["clip_id"]}.json')
        v=r['variants']['mask_and_radius_35'];path={q['detection_id']:q for q in v['trajectory']}
        seed={q['detection_id']:q for q in v.get('effective_stationary_seed',v['seed'])}
        assert row['seed_evidence']['detection_id'] in seed
        assert row['seed_evidence']['frame']==seed[row['seed_evidence']['detection_id']]['frame']
        assert not row['human_verified']
        assert row['reviewer']=='assistant_visual_review'
        assert row['flight_ball_visible_in_sample']==bool(row['flight_evidence'])
        for q in row['flight_evidence']:assert q['detection_id'] in path and q['frame']==path[q['detection_id']]['frame']
        inp=load(out/'inputs'/f'{row["clip_id"]}.json')
        pool=inp['candidates'];allowed={(q['frame'],q['detection_id']) for ds in pool.values() for q in ds}
        assert all((q['frame'],q['detection_id']) in allowed for q in v['trajectory'])
        assert sha(Path(r['source_video_path']))==r['source_sha256']
        assert r['deterministic_replay'] and r['input_candidates_unchanged'] and r['flight_neural_candidates_retained']
        assert sha(out/'media'/r['media']['video'])==r['media']['video_sha256']
        checked.append({'clip_id':row['clip_id'],'visual_evidence_ids_present':True,
          'current_candidate_membership':True,'source_hash_verified':True,'rendered_video_hash_verified':True})
    for rel,h in protocol['code_sha256'].items():
        assert sha(ROOT/rel)==h
        assert sha(out/'source_snapshot'/Path(rel).name)==h
    postprocessing=['scripts/build_ball_mask_review_evidence.py',
                    'scripts/render_ball_review_samples.py',
                    'scripts/finalize_ball_mask_review.py',
                    'scripts/refine_ball_mask_review.py',
                    'tests/test_ball_mask_continuity.py']
    protocol['review_tools_sha256']={rel:sha(ROOT/rel) for rel in postprocessing}
    for rel in postprocessing:(out/'source_snapshot'/Path(rel).name).write_bytes((ROOT/rel).read_bytes())
    save(out/'protocol.json',protocol)
    counts={'seed_representatives_show_ball':sum(r['seed_representative_is_ball'] for r in review),
      'selected_flight_ball_visible_in_sample':sum(r['flight_ball_visible_in_sample'] for r in review),
      'reviewed_clips':39,'human_verified':False,'full_trajectory_accuracy':None}
    save(out/'visual_review_summary.json',counts)
    with (out/'visual_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['clip_id','seed_frame','seed_is_ball','flight_sample_has_ball','flight_frames','remaining_issue','reviewer','human_verified'])
        w.writeheader()
        for r in review:w.writerow({'clip_id':r['clip_id'],'seed_frame':r['seed_evidence']['frame'],
          'seed_is_ball':r['seed_representative_is_ball'],'flight_sample_has_ball':r['flight_ball_visible_in_sample'],
          'flight_frames':','.join(str(q['frame']) for q in r['flight_evidence']),
          'remaining_issue':r['remaining_issue'],'reviewer':r['reviewer'],'human_verified':False})
    doc=(out/'report.html').read_text(encoding='utf-8')
    status=f'AI原像素抽样核对：39条起点代表帧中，{counts["seed_representatives_show_ball"]}条可见足球；{counts["selected_flight_ball_visible_in_sample"]}条的选中飞行段至少有一个抽样帧可见球。<strong>这不代表整条轨迹准确率</strong>，未进行人工逐帧标注。'
    doc=re.sub(r'<p id="review-status">.*?</p>',f'<p id="review-status">{status}</p>',doc)
    doc=doc.replace('<h1>球：掩膜与连续运动优化复测</h1>','<h1>球：新版验收报告</h1>')
    addition='<p>已修复晚出生的起跑假设覆盖已锁定轨迹的入口；至少5个候选支持后才锁定起跑段，避免两个杂点阻止真球起跑。高分静止候选与起跑回推不一致时也会排除该起跑假设。</p><p>原先 Ramos、Cheryshev、Aspas、Vida 的错误起点均已纠正；Iniesta 第67帧当前候选重新保留。统一提高起跑速度门槛后，Uribe 从跟跑动球员恢复到早期飞行中的球，其余38条选中点未变。这个门槛针对本批原视频的像素尺度；跨分辨率或慢速球仍需重新验证。预测只用于搜索，从未填成真实检测点。</p>'
    doc=doc.replace('<p id="review-status">',addition+'<p id="review-status">')
    by_clip={r['clip_id']:r for r in review}
    table='<table><tr><th>片段</th><th>起点证据帧</th><th>飞行球抽样证据帧</th><th>待解决的问题</th></tr>'
    for rr in summary['rows']:
        r=by_clip[rr['clip_id']];cid=r['clip_id']
        frames=', '.join(str(q['frame']) for q in r['flight_evidence']) or '未确认'
        table+=f'<tr><td><a href="#{cid}">{cid}</a></td><td>{r["seed_evidence"]["frame"]}</td><td>{frames}</td><td>{html.escape(r["remaining_issue"])}</td></tr>'
    table+='</table>'
    doc=re.sub(r'<table>.*?</table>',table,doc,count=1,flags=re.S)
    doc=doc.replace('</table>', '</table><p><a href="visual_review.csv">逐片段视觉核对表</a> · <a href="visual_review.json">证据ID与核对来源</a> · <a href="tests.json">算法检查</a> · <a href="README.md">复现方法与限制</a></p>',1)
    for r in review:
        cid=r['clip_id'];frames=', '.join(str(q['frame']) for q in r['flight_evidence']) or '未确认'
        note=f'起点代表帧：{r["seed_evidence"]["frame"]}；选中飞行球证据帧：{frames}。\n{r["remaining_issue"]}'
        doc=doc.replace(f'<p class="review" id="review-{cid}">抽样核对待完成。</p>',f'<p class="review" id="review-{cid}">{html.escape(note)}</p><p>助手视觉核对的原像素样本（未人工逐帧标注）：</p><img loading="lazy" src="media/{cid}_visual_review_sample.jpg" alt="起点与飞行球的视觉抽样证据"><p>按距离自动抽取的候选样本，可能包含误跟：</p><img loading="lazy" src="media/{cid}_midflight_check.jpg" alt="原像素候选抽样，身份需要核对">')
    doc=doc.replace('<p class="warn">','<p class="warn">球的轨迹、最终位置与进球判定仍未全部完成。',1)
    doc=doc.replace('</html>','<p>视频已解码核对，HTML资源引用已检查；本地浏览器交互未通过工具实测。</p></body></html>')
    (out/'report.html').write_text(doc,encoding='utf-8')
    refs=[]
    for ref in re.findall(r'(?:href|src)="([^"]+)"',doc):
        if ref.startswith('#'):continue
        if '://' in ref:continue
        p=out/ref;assert p.is_file(),ref;refs.append(ref)
        if p.suffix.lower() in {'.jpg','.png'}:
            with Image.open(p) as im:im.verify()
    validation=load(out/'validation.json');assert validation['processed_clips']==39
    assert all(r['decoded_video_frames']==r['expected_video_frames'] for r in validation['rows'])
    validation.update({'artifact_references_checked':len(refs),'source_snapshot_hashes_match':True,
      'selected_visual_evidence_checked':checked,'total_decoded_frames':sum(r['decoded_video_frames'] for r in validation['rows']),
      'algorithm_tests':load(out/'tests.json'),'browser_interaction_verified':False,'accuracy_validated':False,
      'human_verified':False,'full_raw_clip_coverage':False})
    save(out/'validation.json',validation)
    print(json.dumps({'review_counts':counts,'decoded_frames':validation['total_decoded_frames'],
      'html_references':len(refs),'success':True},ensure_ascii=False))


if __name__=='__main__':main()
