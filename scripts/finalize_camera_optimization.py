"""Package reproducible diagnostic evidence, never manufacture accuracy labels."""
from pathlib import Path
import sys,io,json,html,unittest,hashlib,zipfile
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import DEFAULT,read,save,sha


def overlap(a,b):
    return (a['clip_id']==b['clip_id'] and a['segment_id']==b['segment_id'] and
            a['view_candidate']==b['view_candidate'] and
            max(a['start_frame'],b['start_frame'])<min(a['end_frame_exclusive'],b['end_frame_exclusive']))


def main():
    out=DEFAULT;validation=out/'validation';validation.mkdir(exist_ok=True)
    patterns=('test_camera_partial_alignment.py','test_camera_view_intervals.py','test_camera_view_guard.py',
         'test_goal_temporal_tracking.py','test_camera_timing_v2.py','test_goal_geometry.py',
         'test_goal_plane_rectification.py','test_goal_event_tracker.py')
    suite=unittest.TestSuite(unittest.TestLoader().discover(str(ROOT/'tests'),pattern=p) for p in patterns)
    stream=io.StringIO();run=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (validation/'targeted_tests.log').write_text(stream.getvalue(),encoding='utf-8')
    fullstream=io.StringIO()
    full=unittest.TextTestRunner(stream=fullstream,verbosity=2).run(unittest.TestLoader().discover(str(ROOT/'tests')))
    (validation/'all_tests.log').write_text(fullstream.getvalue(),encoding='utf-8')
    guarded=read(out/'guarded_views_v2/summary.json');old=read(out/'dense_views_v1/summary.json')
    finding=read(out/'visual_qa/ai_findings.json');partial=read(out/'evaluation_v1/summary.json')
    suspicious=[(e['interval_index'],e['reason']) for e in finding['identity_contradictions_or_ambiguities']+finding['geometry_concerns']]
    retained=[];quarantined=[]
    for r in guarded['intervals']:
        notes=[{'ai_finding_interval_index':i,'reason':reason} for i,reason in suspicious if overlap(r,old['intervals'][i])]
        row={**r,'ai_diagnostic_flags':notes,'human_verified':False,'accuracy_validated':False}
        (quarantined if notes else retained).append(row)
    save(out/'review_candidates.json',{'status':'experimental_candidates_not_verified_results',
         'automated_candidate_count':len(guarded['intervals']), 'quarantine_source':'explicit AI diagnostic review, not automatic model or human labels',
         'candidates_without_known_ai_flags':retained,'quarantined':quarantined,
         'warning':'Passing a field/crowd filter does not prove identity. No playback speed verified.'})
    bad=finding['identity_contradictions_or_ambiguities']
    survivors=[e['interval_index'] for e in bad if any(overlap(r,old['intervals'][e['interval_index']]) for r in guarded['intervals'])]
    archive=ROOT/'deliverables/VisionII_Current_Deliverable_2026-10-10.zip'
    expected='1e4166bbbc6c9b9729b7cc4ed142c64bae1fac8c824bac8ed8dd8c6c0d404343'
    package=read(ROOT/'outputs/camera_timing_retest_2026-10-10_01/camera_timings_candidates.json')
    raw_checks=[{'clip_id':c['clip_id'],'matches':sha(c['raw_video_path'])==c['source_sha256']} for c in package['clips']]
    validation_result={'targeted_tests':run.testsRun,'targeted_success':run.wasSuccessful(),
        'full_unittest_entries':full.testsRun,'full_unittest_failures':len(full.failures),'full_unittest_errors':len(full.errors),
        'full_unittest_error_names':[str(t) for t,e in full.errors],
        'previous_deliverable_zip_sha256':sha(archive),'previous_deliverable_zip_unchanged':sha(archive)==expected,
        'canonical_raw_video_hash_checks':raw_checks,'all_canonical_raw_videos_unchanged':all(c['matches'] for c in raw_checks),
        'AI_contact_sheets':{p.name:sha(p) for p in sorted((out/'visual_qa').glob('*.jpg'))},
        'AI_review_scope':'54 intervals x 3 static frames, not human ground truth; continuous motion not verified'}
    save(validation/'summary.json',validation_result)
    stats={'partial_pose_observations_rescued':104,'old_cohort_sample_eligible_before':7,'old_cohort_sample_eligible_after':7,
        'real_slowdown_candidates':len(partial['slowdown_candidates']), 'dense_spatial_intervals_before_guard':54,
        'guarded_spatial_intervals':len(guarded['intervals']), 'AI_identity_warning_cases_in_development_data':len(bad),
        'AI_identity_warning_cases_removed_by_automatic_guard':len(bad)-len(survivors),
        'AI_identity_warning_cases_remaining':survivors, 'AI_quarantined_after_guard':len(quarantined),
        'candidates_without_known_ai_flags':len(retained),
        'scope':'Numbers are candidate coverage / diagnostic counts, NOT accuracy. Engineering rules designed after viewing development examples.',
        'human_ground_truth_records':0,'release_replaced':False}
    save(out/'comparison_summary.json',stats)
    label={'kicker_closeup':'球员近景候选（身份仍待验证）','back_of_kicker':'主罚者背后候选',
           'angle_45_right':'主罚者右斜方候选（不保证45°）','back_of_goalkeeper':'守门员背后候选'}
    table=''.join(f'<tr><td>{html.escape(r["clip_id"])}</td><td>{label[r["view_candidate"]]}</td><td>{r["start_time_sec"]:.2f}–{r["end_time_sec"]:.2f}</td><td>unknown</td></tr>' for r in retained)
    flags=''.join(f'<li>{html.escape(r["clip_id"])} / {r["start_time_sec"]:.2f}–{r["end_time_sec"]:.2f} 秒：{html.escape(r["ai_diagnostic_flags"][0]["reason"])}</li>' for r in quarantined)
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vision II 优化实验 2026-10-11</title>
<style>body{{font:16px/1.75 system-ui,sans-serif;max-width:1120px;margin:30px auto;padding:0 22px;color:#233742}}h1,h2{{line-height:1.3}}h2{{margin-top:34px}}.warn{{background:#fff0cf;padding:18px;border-left:5px solid #b37b04}}.ok{{background:#e6f3ee;padding:18px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:9px;border:1px solid #d2dce1}}th{{background:#edf3f6}}code{{background:#edf3f6;padding:2px 5px}}a{{color:#126c88}}video{{width:100%;max-width:520px}}img{{max-width:100%}}.cards{{display:flex;flex-wrap:wrap;gap:18px}}.card{{flex:1;min-width:280px}}pre{{white-space:pre-wrap;font-size:13px;background:#f3f6f7;padding:16px}}</style></head><body>
<h1>Vision II：继续优化的实测结果</h1><p>2026-10-11 · 镜头机位 / 慢动作 / 身份误判 · 增量实验版</p>
<div class="warn">本轮没有“人工复核数据”。以下画面检查由 AI 完成，不是真值。没有测得真实视频准确率；没有把候选数或单元测试通过数当作准确率。原39条视频和昨天成果ZIP均保持不变。</div>
<h2>一、真正改善了什么</h2><ol>
<li><b>修复判断分支漏检：</b>原程序进入“守门员较大”的分支后，即使无法确认背后视角，也跳过后面的球门几何判断。新实验模块允许继续检查几何，12个采样点得到空间分类回退。Cuadrado 的 <b>11.13–12.07秒</b> 从未知恢复为主罚者背后候选。不是确认整段镜头或确认慢动作。</li>
<li><b>输出镜头内有效区间：</b>每段至少4个连续支持采样，相邻间隔不大于0.35秒；未知或变类立即分开。Kane 的 <b>9.70–11.23秒</b> 保留为近景候选，后半段构图变化不再使前段全部丢失。</li>
<li><b>新增误认防护：</b>HSV草地候选掩膜排除人物框后，结合多个同色大人物、密集前景等条件，拒绝弱身份片段。54个未经保护的候选变为38个自动候选，并非“只错了16条”。</li></ol>
<p>AI检查54段的首/中/末3帧，发现14段有认人矛盾或明显多人歧义。自动防护阻断其中13段对应的原区间；仍有1段错误队服候选残留。这是看过开发样例后制定规则的诊断结果，不是独立测试准确率。</p>
<div class="warn">防护有代价：Henderson、Uribe、Brozovic、Dzagoev及Kuzyaev末段等真实球员特写样例，也会因看不到足够草地/多人入镜而丢失支持。草地不是身份识别器；下一步应接球衣号码或可靠人物关联，而非继续扩大颜色阈值。</div>
<h2>二、哪些实验没有达到目标</h2><table><tr><th>实验</th><th>实际观测</th><th>结论</th></tr>
<tr><td>部分关节可用就分析</td><td>重新测量3593帧；多保留104个人物姿态观测。原镜头配对中足量样本仍为7组；扩大范围后完整/部分方案均为21组。</td><td>减少单帧缺失，但尚未增加可用动作配对。</td></tr>
<tr><td>有时间顺序约束的匹配</td><td>152个角色配对：151不确定、1无相对减速证据、0减速候选。93个不同事件对照中仅24组达到最低样本数，0组接受。</td><td>没有证明真实慢动作识别提升。拒绝全部错误对照也不等于有足够识别能力。</td></tr>
<tr><td>时间缩放数值对照</td><td>30组自序列改时间戳为2倍：8组减速候选、19不确定、3无减速；30组倒序2倍全部不确定。</td><td>只证明有限数值行为；这些不是新增慢动作视频，敏感性仍不足。</td></tr>
<tr><td>球门连续光流跟踪</td><td>Cuadrado单镜头试验20个稠密采样点，球门几何由7点增至9点；单独仍未生成足够连续机位区间。</td><td>有局部几何补全，但还不是可发布的完整跟踪提升。</td></tr></table>
<h2>三、代码和方法来源</h2><ul>
<li><code>src/camera_partial_alignment.py</code>：逐关节缺失掩码；只比较可见腿部角度；有序平移+恒定时间缩放网格，含倒序/随机打乱/正常速度对照。项目自行实现，参考 <a href="https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_DiagonalMatching.html">FMP 对角子序列与时间缩放匹配</a> 的通用思路；不是DTW，不是跨视角3D恢复。</li>
<li><code>src/camera_view_intervals.py</code>：连续采样投票切成较短区间，不将局部证据延伸到整个镜头。</li>
<li><code>src/camera_view_guard.py</code>：项目编写的HSV人物外草地掩膜、同色多人/密集前景排除，以及守门员分支失败后的几何回退。</li>
<li><code>src/goal_temporal_tracking.py</code>：调用OpenCV Shi-Tomasi、金字塔Lucas-Kanade、前后向一致性与RANSAC相似变换，参考 <a href="https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html">OpenCV光流教程</a>；球门区域掩膜、白线支持、检测年龄和拒绝规则为项目实现。只在图像球门平面跟踪，不是球场俯视变换。</li></ul>
<h2>四、保留的证据和输出</h2><p>38个自动候选再经独立记录的AI诊断隔离2个已知可疑区间（1个队服身份矛盾、1个角度歧义）。剩余{len(retained)}个是<b>尚无已记录AI疑点的候选</b>，不是{len(retained)}个人工正确结果。慢动作状态全部保留unknown。</p>
<ul>{flags}</ul><p><a href="review_candidates.json">候选/隔离记录JSON</a> · <a href="guarded_views_v2/summary.json">自动防护原始统计</a> · <a href="visual_qa/ai_findings.json">AI观察原文</a> · <a href="evaluation_v1/report.html">姿态对齐详细实验</a> · <a href="goal_tracking_pilot_v1/comparison.jpg">球门跟踪对照图（红=独立检测，绿=跟踪）</a></p>
<table><tr><th>视频</th><th>空间候选</th><th>原片内秒数 [开始,结束)</th><th>速度</th></tr>{table}</table>
<h2>五、验证与复现</h2><p>本次相关测试 <b>{run.testsRun}项{'通过' if run.wasSuccessful() else '未全部通过'}</b>。全项目unittest收集{full.testsRun}项，失败{len(full.failures)}、错误{len(full.errors)}；完整原因保留在日志，不能说全项目全绿。原始39个视频SHA-256逐一一致：{validation_result['all_canonical_raw_videos_unchanged']}；昨天交付ZIP哈希一致：{validation_result['previous_deliverable_zip_unchanged']}。</p>
<p><a href="validation/targeted_tests.log">本轮相关测试</a> · <a href="validation/all_tests.log">全项目测试日志</a> · <a href="validation/summary.json">哈希与检查范围</a></p>
<pre>在原项目目录、已有环境中运行（不是脱离项目的独立安装包）：
python -X utf8 scripts/run_camera_partial_pose_experiment.py --cohort all --resume
# 后续实验输出目录不可覆盖；需新命名或保存后按脚本参数另建目录。
# 每个实验目录的 protocol.json 保留代码/输入哈希与规则。
# 已存档 evaluation_v1、dense_views_v1、guarded_views_v2 不得互相覆盖。
# 精确复现命令与本轮文件清单见实验包 README.txt。</pre>
<h2>六、下一轮应优先做什么</h2><p>首先解决“是不是本次主罚者”，尤其同队球员、观众穿球衣，以及上游颜色参考混入背景的问题；其次做球衣号码/跨帧身份关联，再用带可信身份的相同动作片段估计速度。球门角度应同时考虑角点误差，避免假精确的45°。当前先交付独立实验和证据，不用这批未验证结果覆盖昨天的候选成果。</p>
</body></html>'''
    (out/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps({'comparison':stats,'validation':{k:v for k,v in validation_result.items() if k not in ('canonical_raw_video_hash_checks','AI_contact_sheets')}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
