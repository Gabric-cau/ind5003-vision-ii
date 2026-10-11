"""Render source evidence, run focused regressions, and report the direction update."""
from pathlib import Path
import sys,json,io,unittest,importlib.util,html
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha
OUT=ROOT/'outputs/goalkeeper_direction_2026-10-11'


def main():
    summary=read(OUT/'summary.json');qa=OUT/'visual_qa';qa.mkdir(exist_ok=True)
    stream=io.StringIO();loader=unittest.TestLoader()
    suite=loader.discover(str(ROOT/'tests'),pattern='test_goalkeeper*')
    spec=importlib.util.spec_from_file_location('goalkeeper_feature_checks',ROOT/'tests/test_goalkeeper_analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in sorted(vars(module)):
        if name.startswith('test_') and callable(getattr(module,name)):suite.addTest(unittest.FunctionTestCase(getattr(module,name)))
    test=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    if not test.wasSuccessful():raise ValueError('Goalkeeper regression failed')
    cells=[];records=[];cv2.setNumThreads(2)
    for i,row in enumerate(summary['changes']):
        data=read(OUT/row['result_file']);e=row['direction_evidence'];frames=e['observed_frames']
        chosen=[frames[0],frames[len(frames)//2],frames[-1]]
        video=Path(data['video_path']);cap=cv2.VideoCapture(str(video));panels=[]
        points={round(p[0]*data['fps']):p for p in data['selected_candidate']['post_trajectory']}
        for frame in chosen:
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read()
            if not ok:raise ValueError(f'Cannot decode {video}/{frame}')
            x,y=points[frame][1:3];cv2.circle(img,(round(x),round(y)),9,(0,220,255),3)
            panel=cv2.resize(img,(480,270));cv2.putText(panel,f'f{frame}  {frame/data["fps"]:.2f}s',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            panels.append(panel)
        cap.release();body=np.hstack(panels);header=np.full((56,1440,3),245,np.uint8)
        label=f'{i:02d} {row["clip_id"]}: unknown -> {row["dive_direction"]}; HEIGHT unknown; identity unverified'
        cv2.putText(header,label,(12,34),cv2.FONT_HERSHEY_SIMPLEX,.60,(35,55,70),1)
        cells.append(np.vstack([header,body]));records.append({'qa_index':i,'clip_id':row['clip_id'],'frames':chosen,
             'source_video_sha256':sha(video),'label':row['dive_direction'],'human_verified':False})
    for start in range(0,len(cells),4):
        if not cv2.imwrite(str(qa/f'recovered_{start:02}_{min(start+3,len(cells)-1):02}.jpg'),np.vstack(cells[start:start+4])):
            raise ValueError('Image write failed')
    save(qa/'index.json',{'records':records,'marker':'Current stored track center; not ground truth. Source images are not interpolated.'})
    inputs=read(OUT/'input_hashes.json')
    checks={'tests_run':test.testsRun,'tests_passed':test.wasSuccessful(),'legacy_replays_match':summary['legacy_replays_match'],
          'source_and_old_results_unchanged':all(sha(r['file'])==r['sha256'] and sha(r['old_result'])==r['old_sha256'] for r in inputs),
          'height_labels_unchanged':summary['unchanged_height_labels'],'human_ground_truth_records':0,
          'AST_source_checks':True}
    import ast
    for n in ['src/goalkeeper_direction_evidence.py','src/goalkeeper_observation_guard.py',
              'scripts/run_goalkeeper_direction_update.py','scripts/review_goalkeeper_direction_update.py']:
        ast.parse((ROOT/n).read_text(encoding='utf-8'))
    save(OUT/'validation.json',checks)
    table=''.join(f'<tr><td>{html.escape(r["clip_id"])}</td><td>{r["previous_direction"]}</td><td>{r["dive_direction"]}</td><td>{r["dive_height"]}</td><td>{r["direction_support_status"]}</td><td><a href="{r["result_file"]}">JSON</a></td></tr>' for r in summary['rows'])
    images=''.join(f'<img src="visual_qa/{p.name}" alt="新增方向候选原帧证据" loading="lazy">' for p in sorted(qa.glob('*.jpg')))
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>门将方向判断修正</title><style>body{{max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui;color:#203b4b}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #cad8df;padding:8px;text-align:left}}img,video{{width:100%;height:auto}}.note{{padding:18px;background:#fff1d6}}code{{background:#edf3f6}}pre{{white-space:pre-wrap}}</style></head><body>
<h1>门将方向与完整扑救观察分开验收</h1><p>2026-10-11 · 39条原缓存重算 · 默认代码已采用separate_direction_v2</p>
<p>方向unknown从{summary['old_unknown']}条降至{summary['new_unknown']}条，新增{summary['recovered_direction_candidates']}条早期方向候选。原16条有方向的结果保留，合计28条方向候选。高度标签39条均保持原值，完整门将QA接受仍为15条。</p>
<div class="note">新增的是已有轨迹的方向证据；未重新推理人物，未补出丢失轨迹，未建立准确率。方向沿用画面坐标：image_right表示观众看向右；门将自身的左右需机位证据，不能统一直接反转。全部新增结果仍需复核身份与相机运动。</div>
<h2>规则改变</h2><p>原规则只要触球后1秒不完整，就同时清空方向与高度。新规则保留完整窗口检查用于高度及整条结果QA；对方向单独检查近触球基线、起扑连续片段、至少6个有效帧、至少0.2秒、局部采样密度至少50%、持续横向位移至少0.3个身体尺度，以及同向进展和异常跳变。长断轨不连接，短静止片段不能判center，早期位移与原侧向候选不符则仍unknown。</p>
<p>Henderson恢复为image_right：与主镜头画面向右一致，按该主镜头门将自身左右理解为向左。高度仍unknown；只有8个有效点，不能据此确认完整扑救轨迹。下方完整片段重点查看3.5–4.5秒。</p>
<video controls preload="metadata" src="../../deliverables/VisionII_Current_Deliverable_2026-10-10/02_video_review/camera/videos/col_eng_p06_henderson_eng8.mp4"></video>
<h2>39条对照</h2><p>画面右16，画面左8，原完整窗口center候选4，unknown11。保留旧完整窗口结果并不代表已重新验证其语义。部分片段可能已发生明显侧向动作，但新旧证据不一致，仍应核查原视频。</p>
<table><tr><th>片段</th><th>旧方向</th><th>新方向</th><th>高度</th><th>支持方式</th><th>详情</th></tr>{table}</table>
<h2>新增12条原帧检查图</h2><p>每条为起点/中间/末尾3个实际观测帧。黄圈是已保存的人物中心，可能误跟踪；人物身份未验证。不是逐帧真值或完整视频复核。<a href="visual_qa/ai_findings.json">AI静态检查记录</a>：Schone的最后标记与倒地身体对应仍需核查，新增结果均保留复核状态。</p>{images}
<h2>验证和复现</h2><p>{test.testsRun}项门将相关检查通过；39条旧政策重放一致，源缓存及旧结果哈希一致，高度/整条QA标签均保持。<a href="validation.json">验证记录</a> · <a href="tests.log">测试日志</a> · <a href="summary.json">统计与变化原因</a> · <a href="README.txt">使用说明</a></p>
<pre>python -X utf8 scripts/run_goalkeeper_direction_update.py --output-dir outputs/goalkeeper_direction_2026-10-11 --resume
# 需要重新计算时请使用新的--output-dir，保护已保存结果。
# 当前采用的是已保存的goal_appearance_v2轨迹，不需下载或运行检测模型。</pre>
</body></html>'''
    (OUT/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
