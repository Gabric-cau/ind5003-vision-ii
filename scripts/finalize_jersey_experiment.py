"""Validate, retain diagnostic boundaries, and report the isolated jersey experiment."""
from pathlib import Path
import sys,io,json,html,unittest,subprocess,importlib.metadata,ast
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_jersey_gallery import DEFAULT as OUT
from scripts.run_camera_partial_pose_experiment import DEFAULT as OLD,BASE,read,save,sha
from scripts.search_player_evidence import search_records


def interval_overlap(a,b):
    return a['clip_id']==b['clip_id'] and a['segment_id']==b['segment_id'] and max(a['start_time_sec'],b['start_time_sec'])<min(a['end_time_sec'],b['end_time_sec'])


def run_tests():
    patterns=('test_camera_partial_alignment.py','test_camera_view_intervals.py','test_camera_view_guard.py',
         'test_goal_temporal_tracking.py','test_camera_timing_v2.py','test_goal_geometry.py',
         'test_goal_plane_rectification.py','test_goal_event_tracker.py','test_jersey_evidence.py',
         'test_upright_text_crop.py','test_jersey_camera_guard.py','test_player_evidence_search.py')
    results={}
    for name,suite in [('targeted',unittest.TestSuite(unittest.TestLoader().discover(str(ROOT/'tests'),pattern=p) for p in patterns)),
                       ('full',unittest.TestLoader().discover(str(ROOT/'tests')))]:
        stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (OUT/'validation'/f'{name}_tests.log').write_text(stream.getvalue(),encoding='utf-8')
        results[name]={'entries':r.testsRun,'passed':r.testsRun-len(r.errors)-len(r.failures)-len(r.skipped),
                      'skipped':len(r.skipped),'errors':len(r.errors),'failures':len(r.failures),
                      'error_names':[str(t) for t,e in r.errors],'success':r.wasSuccessful()}
    return results


def main():
    (OUT/'validation').mkdir(exist_ok=True)
    tests=run_tests();index=read(OUT/'evaluation_v1/retrieval_index.json')
    summary=read(OUT/'evaluation_v1/summary.json');integration=read(OUT/'camera_integration_v2/summary.json')
    old=read(OLD/'review_candidates.json');new=read(OUT/'camera_integration_v2/camera_candidates.json')['intervals']
    findings=read(OUT/'visual_qa/ai_findings.json');qa=read(OUT/'visual_qa/index.json')['records']
    queue=[];quarantine=[]
    for r in new:
        prior=[q for q in old['quarantined'] if interval_overlap(r,q) and r['view_candidate']==q['view_candidate']]
        notes=[{'source':'prior_AI_diagnostic','reason':q['ai_diagnostic_flags']} for q in prior]
        for f in findings['findings']:
            if f['kind'] not in ('systematic_font_misread','text_ownership_ambiguous','occlusion_review'):continue
            qr=qa[f['qa_index']]
            track=next(t for t in index['tracklets'] if (t['clip_id'],t['segment_id'],t['tracklet_id'])==(qr['clip_id'],qr['segment_id'],qr['tracklet_id']))
            if interval_overlap(r,track):notes.append({'source':'new_AI_diagnostic','qa_index':f['qa_index'],'reason':f['reason']})
        row={**r,'ai_diagnostic_flags':notes,'human_verified':False,'accuracy_validated':False}
        (quarantine if prior else queue).append(row)
    save(OUT/'review_candidates.json',{'automatic_count':len(new),'pending_review_count':len(queue),
         'prior_AI_quarantine_count':len(quarantine),'pending_review':queue,'quarantined':quarantine,
         'note':'Pending is not clean, correct, or verified; OCR font and text-ownership warnings are preserved. No playback-speed conclusion.'})
    hash_checks=[];crop_count=0;crop_ok=True;cohort_ok=True
    for p in sorted((OUT/'gallery').glob('*.json')):
        g=read(p);cid=p.stem
        hash_checks.append({'clip_id':cid,'raw_unchanged':sha(ROOT/'data/raw_clips'/f'{cid}.mp4')==g['source_sha256']})
        images=[k for s in g['shots'] for t in s['tracklets'] for k in t['keyframes']]
        crop_count+=len(images);crop_ok &= all(sha(OUT/k['image'])==k['image_sha256'] for k in images)
        expected=[(s['segment_id'],t['tracklet_id'],k['image'],k['frame']) for s in g['shots'] for t in s['tracklets'] for k in t['keyframes']]
        for variant,protocol in [('ocr','ocr_protocol.json'),('ocr_upright','upright_ocr_protocol.json')]:
            d=read(OUT/variant/p.name)
            actual=[(t['segment_id'],t['tracklet_id'],k['image'],k['frame']) for t in d['tracklets'] for k in t['observations']]
            cohort_ok &= actual==expected and d['gallery_sha256']==sha(p) and d['protocol_sha256']==sha(OUT/protocol)
    previous_zips={
      'VisionII_Current_Deliverable_2026-10-10.zip':'1e4166bbbc6c9b9729b7cc4ed142c64bae1fac8c824bac8ed8dd8c6c0d404343',
      'VisionII_Optimization_2026-10-11_Experimental.zip':'f278c436ca89a7a84dc6d0605f361ce98a9cb15f888041555c57aa75f1f4039a'}
    zip_checks={n:sha(ROOT/'deliverables'/n)==h for n,h in previous_zips.items()}
    cli={}
    for name,args in [('falcao',['--number','9','--name','Falcao']),('modric_cross_clip',['--number','10','--name','Modric']),('absent99',['--number','99'])]:
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/search_player_evidence.py'),*args],capture_output=True,text=True,encoding='utf-8',check=True)
        cli[name]=json.loads(result.stdout)
        save(OUT/'validation'/f'search_{name}.json',cli[name])
    if cli['falcao']['total_matches']<1 or not any('vida' in r['clip_id'] for r in cli['modric_cross_clip']['results']):raise ValueError('Search regression')
    examples=[];(OUT/'examples').mkdir(exist_ok=True)
    encoder=ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    for row in integration['new_or_changed_intervals']:
        cid=row['clip_id'];target=OUT/'examples'/f'{cid}_recovered_candidate.mp4'
        if not target.exists():
            cmd=[str(encoder),'-hide_banner','-loglevel','error','-n','-ss',str(row['start_time_sec']),'-i',str(ROOT/'data/raw_clips'/f'{cid}.mp4'),
                 '-t',str(row['end_time_sec']-row['start_time_sec']),'-map','0:v:0','-an','-c:v','libx264','-preset','fast','-crf','23',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
            subprocess.run(cmd,capture_output=True,check=True)
        cap=cv2.VideoCapture(str(target));reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded+=1
        cap.release()
        if decoded!=reported or decoded==0:raise ValueError('Incomplete example decode')
        examples.append({'clip_id':cid,'start_sec':row['start_time_sec'],'end_sec':row['end_time_sec'],'file':target.name,
                         'sha256':sha(target),'reported_frames':reported,'decoded_frames':decoded,'source_frames_only':True})
    save(OUT/'examples/decode_validation.json',examples)
    source_files=[ROOT/n for n in SOURCE_FILES]
    for path in source_files:
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
    jsons=list(OUT.rglob('*.json'))
    for path in jsons:json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
    validation={'tests':tests,'raw_hash_checks':hash_checks,'raw_hashes_all_match':all(r['raw_unchanged'] for r in hash_checks),
       'crop_count':crop_count,'crop_hashes_all_match':crop_ok,'default_and_upright_cohorts_identical':cohort_ok,
       'previous_zip_hashes_unchanged':zip_checks,'source_AST_files':len(source_files),'strict_JSON_files_checked':len(jsons),
       'search_smoke_cases':{k:v['total_matches'] for k,v in cli.items()},'example_videos_fully_decoded':len(examples),
       'python':sys.version,'interpreter':sys.executable,'accuracy_estimate':None,'release_replaced':False}
    save(OUT/'validation/summary.json',validation)
    if not tests['targeted']['success'] or not crop_ok or not cohort_ok or not all(zip_checks.values()) or not validation['raw_hashes_all_match']:raise ValueError('Validation failed')
    write_report(summary,integration,validation,queue,quarantine,examples,index)
    print(json.dumps({k:v for k,v in validation.items() if k!='raw_hash_checks'},ensure_ascii=False,indent=2))


SOURCE_FILES=['src/jersey_evidence.py','src/upright_text_crop.py','src/jersey_camera_guard.py',
    'scripts/build_jersey_gallery.py','scripts/run_jersey_ocr_worker.py','scripts/run_jersey_upright_ocr.py',
    'scripts/evaluate_jersey_retrieval.py','scripts/refine_jersey_camera_candidates.py','scripts/render_jersey_evidence_qa.py',
    'scripts/search_player_evidence.py','scripts/finalize_jersey_experiment.py',
    'tests/test_jersey_evidence.py','tests/test_upright_text_crop.py','tests/test_jersey_camera_guard.py','tests/test_player_evidence_search.py']


def write_report(s,integration,v,queue,quarantine,examples,index):
    cards=''
    for r in examples:
        cards+=f'<article><h3>{html.escape(r["clip_id"])}</h3><p>原视频 {r["start_sec"]:.2f}–{r["end_sec"]:.2f} 秒</p><video controls preload="metadata" src="examples/{r["file"]}"></video></article>'
    sources=read(OUT/'research_sources.json')['sources']
    source_html=''.join(f'<li><a href="{html.escape(x["url"])}">{html.escape(x["title"])}</a><br>{html.escape(x["local_use"])}<br><small>{html.escape(x["not_used"])}</small></li>' for x in sources)
    gallery=''
    for r in index['tracklets']:
        if r['upright_query']['status']=='unknown':continue
        e=r['upright_evidence'];names=' / '.join(n['text'] for n in e['name_tokens']);first=r['keyframes'][0]
        gallery+=f'<article><h3>{html.escape(r["clip_id"])} · s{r["segment_id"]} / t{r["tracklet_id"]}</h3><p>OCR号码：{e["number"]}；文字：{html.escape(names)}<br>与所在视频名单对比：{r["upright_query"]["status"]}<br>{r["start_time_sec"]:.2f}–{r["end_time_sec"]:.2f} 秒；全部待验证</p><a href="{first["image"]}"><img loading="lazy" src="{first["image"]}" alt="候选文字截图，不是真值"></a></article>'
    t=v['tests']
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vision II 球衣证据检索优化</title><style>
body{{max-width:1120px;margin:30px auto;padding:0 22px;font:16px/1.75 system-ui,sans-serif;color:#203b4b}}h1,h2,h3{{line-height:1.35}}h2{{margin-top:36px}}a{{color:#126d82}}.warn{{padding:16px;background:#fff0ce;border-left:5px solid #b67d0d}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd8df;padding:10px;text-align:left}}th{{background:#edf4f7}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:20px}}article{{border:1px solid #d8e2e7;padding:16px;overflow-wrap:anywhere}}video{{width:100%}}img{{max-width:100%;height:220px;object-fit:contain}}pre{{white-space:pre-wrap;background:#eff4f6;padding:18px}}small{{color:#52646e}}.contact img{{height:auto;width:100%}}</style></head><body>
<h1>Vision II：球衣证据检索与认人防护</h1><p>2026-10-11 · 独立增量实验 · 检索、实现、同输入对照、AI诊断与回归检查</p>
<div class="warn">本轮没有新增人工标注，没有得到识别准确率。名单号码仅用于检索，不作为图像真值。全部结果仍为候选；模型置信分数不是身份正确概率。原视频和前两份ZIP均未修改。</div>
<h2>1. 实际结果</h2><table><tr><th>同一批输入</th><th>默认通用OCR</th><th>保留直立文字方向</th></tr><tr><td>片段 / 镜头 / 人物短轨迹 / 截图</td><td colspan="2">39 / 197 / 598 / 1,912（完全相同）</td></tr><tr><td>有高置信数字候选的截图</td><td>95</td><td>133</td></tr><tr><td>有跨帧重复号码的短轨迹</td><td>19</td><td>33</td></tr><tr><td>有重复文字匹配本片名单的片段</td><td>7 / 39</td><td>15 / 39</td></tr></table>
<p>这是<b>文字证据覆盖</b>，不是正确识别人次或准确率。598条是碎片化人物轨迹，不是598位球员；球员、队友、守门员均可能出现。短轨迹来自缓存采样帧，并非197个镜头逐帧全部扫描。15个片段中的匹配仍可能有文字归属歧义。</p>
<p>接入保守机位判断后，自动空间区间从38增至40：原38个全部保留，新增下方2个近景候选。继续隔离上轮2个已知AI疑点，留下{len(queue)}个待复核候选（不是{len(queue)}个正确结果）。其他OCR字体/多人遮挡警告也保留。</p><div class="cards">{cards}</div>
<p>截取的是原视频画面，未生成新视角或插帧；H.264编码，两段均逐帧完整解码。约0.93秒的区间只表示连续采样支持范围，不是完整慢动作镜头。</p>
<h2>2. 方法：先读画面，再按人检索</h2><ol><li>汇合原镜头3点采样与已有10Hz窗口。在单个镜头内用双向最佳框重叠关联，相近候选有歧义或间隔超过0.4秒则断开。号码、姓名、队服匹配均不参与关联。</li><li>用肩/髋姿态截躯干；关键点不足时回退人物框。按时间分箱、清晰度和尺寸每轨迹选最多6帧，选帧阶段不看OCR结果。</li><li>使用本地RapidOCR/PP-OCRv4读文字。默认实现会把高宽比≥1.5的文本框旋转90°，不利于直立的单个球衣数字；新增透视裁剪保留其方向，同模型、同输入做对照。</li><li>置信分数≥0.85、至少2个独立帧、跨时≥0.2秒、数字投票比例≥75%才产生重复号码候选；同一帧多个号码不选择“恰好匹配名单”的那个。</li><li>最后与名单查询匹配，或者跨视频搜索姓名/号码。<code>--number 10 --name Modric</code>确实在Vida片段里检索到了MODRIC+10。所在视频主罚者并不等于画面里的每个人。</li><li>在原有大人物/队服候选基础上，重复文字仅能放宽“草地不足”这一条限制，不推翻多人竞争限制；号码冲突只能触发复核，不能自动删掉候选。</li></ol>
<h2>3. 失败实验也保留：不能一味提高通过率</h2><p>初版硬性否决与名单号码不一致的片段，但AI检查发现Cuadrado的11重复读成71、Muriel的14重复读成74。这说明跨帧一致仍会系统性出错。现已放弃此否决策略；<code>evaluation_v1/camera_changes.json</code>是失败实验，不是最终输出。应使用<code>camera_integration_v2</code>及带AI标记的<code>review_candidates.json</code>。</p>
<p>Vida拥抱镜头中还存在文字属于遮挡者而非被跟踪人物的情况。没有强行按名单把错误OCR改成“正确”号码，没有伪造人工确认。目标Modric片段中原有错误队服候选仍隔离；检索到另一处Modric不等于已经修复该镜头的身份关联。</p>
<div class="warn">尚未解决：相同动作跨机位匹配、真正慢动作倍速、精确45°角分类、稳定跨镜头ReID。球轨迹、步数与进球判断本轮未重算，不能据此声称它们提高。本轮使用已经查看过的开发片段，未做独立留出测试。</div>
<h2>4. 代码来源与检索依据</h2><p>这是参考研究分解方式后编写的轻量基线，不是复现论文完整网络或使用其成绩。识别器为已有通用PP-OCRv4，未下载SoccerNet权重；本轮未向云端上传视频。</p><ul>{source_html}</ul>
<p><code>src/jersey_evidence.py</code>：短轨迹、ROI、选帧、投票、查询；<code>src/upright_text_crop.py</code>：OpenCV四点透视裁剪保留方向；<code>src/jersey_camera_guard.py</code>：保守接入；<code>scripts/search_player_evidence.py</code>：跨片段文字查询。工程规则为本项目编写，库/模型版本及哈希单独保存。</p>
<h2>5. 证据与验证</h2><p>相关回归测试{t['targeted']['entries']}项通过。全项目unittest：{t['full']['entries']}个收集项，{t['full']['passed']}通过、{t['full']['errors']}个导入错误（环境缺pytest），不能说全项目全绿。39段原视频、1,912张截图哈希与同输入检查通过；前两份交付ZIP哈希保持不变。</p>
<p><a href="README.txt">运行说明</a> · <a href="validation/summary.json">完整校验</a> · <a href="validation/targeted_tests.log">相关测试</a> · <a href="validation/full_tests.log">全项目日志</a> · <a href="review_candidates.json">候选与隔离记录</a> · <a href="visual_qa/ai_findings.json">AI诊断记录</a> · <a href="visual_qa/index.html">全部34条文字证据联系图</a> · <a href="validation/search_modric_cross_clip.json">Modric跨视频检索实例</a></p>
<h2>6. 可检索证据目录</h2><p>以下仅列非unknown的34条；完整598条（含未识别）在<code>evaluation_v1/retrieval_index.json</code>。卡片中的图只是该轨迹的首张选帧，未必是最清楚的号码帧；支持读数的图片请看联系图及完整JSON。使用浏览器查找可搜索姓名或视频ID，CLI可按号码/姓名严格检索。</p><div class="cards">{gallery}</div>
</body></html>'''
    (OUT/'report.html').write_text(report,encoding='utf-8')
    qa_html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>文字证据AI检查</title><style>body{max-width:1300px;margin:30px auto;font:18px/1.7 system-ui}img{width:100%;height:auto}</style><h1>34条文字候选 · AI静态诊断</h1><p>非人工标注；不是准确率统计。左侧原场景，右侧两张支持帧。看过开发样例后调整策略，不是独立测试。</p><p><a href="ai_findings.json">发现记录</a> · <a href="../report.html">返回报告</a></p>'
    qa_html+=''.join(f'<h2>{p.stem}</h2><img src="{p.name}" alt="文字证据联系图" loading="lazy">' for p in sorted((OUT/'visual_qa').glob('*.jpg')))
    (OUT/'visual_qa/index.html').write_text(qa_html+'</html>',encoding='utf-8')


if __name__=='__main__':main()
