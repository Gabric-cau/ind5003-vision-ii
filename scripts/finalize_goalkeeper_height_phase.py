"""Validate, freeze and package the 39-clip phase experiment without promotion."""
from pathlib import Path
from collections import Counter
import sys, shutil, json, io, unittest, zipfile, html
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha
from src.goalkeeper_dive_phase import mask_phase_candidates
from src.goalkeeper_height_fusion import contact_bracket_hip_height
SOURCE = ROOT/'outputs/goalkeeper_height_phase_2026-10-11/all39_v2'
OUT = ROOT/'outputs/goalkeeper_height_phase_2026-10-11/release_v2'
BASE = ROOT/'outputs/goalkeeper_sam2_update_2026-10-11/results_v4_release'
DISPLAY = ROOT/'outputs/goalkeeper_user_review_2026-10-11/goalkeeper_display_results_39.json'


def main():
    summary=read(SOURCE/'summary.json'); protocol=read(SOURCE/'protocol.json')
    if summary['experiment_clips'] != 39: raise ValueError('Incomplete batch')
    if OUT.exists(): raise FileExistsError('Choose a new frozen release')
    original=read(DISPLAY); rows=summary['rows']
    unchanged=all(all(r[k]==v for k,v in old.items()) for old,r in zip(original['rows'],rows))
    if not unchanged: raise ValueError('Original direction/height/QA/user review changed')
    for name,h in protocol['code_sha256'].items():
        if sha(ROOT/name)!=h: raise ValueError('Experiment source changed during run')
    hashes=[]; video_checks=[]; replay=[]; sensitivity=[]; audit=[]; fused=[]
    for trial in summary['experiments']:
        cid=trial['clip_id']; dest=SOURCE/'clips'/cid
        for p,h in read(dest/'input_hashes.json').items():
            if sha(Path(p)) != h: raise ValueError('Experiment input changed')
            hashes.append(dict(path=p,sha256=h))
        if not trial['review_video']: continue
        cap=cv2.VideoCapture(str(SOURCE/trial['review_video'])); count=0
        while cap.read()[0]: count+=1
        cap.release(); expected=read(dest/'video_validation.json')['expected_frames']
        video_checks.append(dict(clip_id=cid,decoded=count,expected=expected,passed=count==expected))
        seg=read(dest/'segmentation.json'); pose=read(dest/'pose_candidates.json')
        audit.append(dict(clip_id=cid,source_window=seg['frame_window'],requested_end=seg['requested_last'],
                          source_censored=seg['source_right_censored'],
                          connected_mask_frames=trial['phase_evidence']['connected_mask_frames'],
                          mask_frames=trial['phase_evidence']['total_mask_frames'],
                          pose_candidate_frames=len(pose)))
        se=trial['phase_evidence']['silhouette_phase']
        scale=read(dest/'height_reference.json')['body_scale_px'] if (dest/'height_reference.json').exists() else None
        if scale is None:
            continue
        b=read(BASE/'results_39'/f'{cid}.json')
        fps=b.get('fps',30.); kick=b['kick_frame']
        evidence=trial['phase_evidence']
        fusion=contact_bracket_hip_height(evidence.get('samples',[]),read(dest/'silhouette_measurements.json'),
                                         se,evidence.get('reference'),fps,evidence.get('landing_frame'))
        fused.append(dict(clip_id=cid,previous_height=trial['previous_height'],evidence=fusion))
        again=mask_phase_candidates(read(dest/'silhouette_measurements.json'),kick,fps,scale)
        replay.append(dict(clip_id=cid,passed=again==se or
                     ('camera_motion_unreliable' in se['reasons'] and not se['supported'])))
        if se.get('supported'):
            peak=se['proxy_peak_body']; bands=[]
            for low,high in [(.30,.70),(.35,.75),(.40,.80)]:
                bands.append('low' if peak<low else 'high' if peak>high else 'mid')
            sensitivity.append(dict(clip_id=cid,proxy_peak_body=peak,threshold_variants=bands,
                                    category_stable=len(set(bands))==1,
                                    note='Engineering band sensitivity, not accuracy validation'))
    suites=['tests.test_goalkeeper_dive_phase','tests.test_goalkeeper_height_fusion','tests.test_goalkeeper_sam2','tests.test_goalkeeper_mask_experiment',
            'tests.test_goalkeeper_role_association','tests.test_goalkeeper_pose_direction',
            'tests.test_goalkeeper_direction_evidence','tests.test_goalkeeper_observation_guard','tests.test_goalkeeper_user_review']
    stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    if not result.wasSuccessful() or not all(v['passed'] for v in video_checks+replay): raise ValueError('QA failed')
    shutil.copytree(SOURCE,OUT,ignore=shutil.ignore_patterns('phase_staging.mp4'))
    fusion_by_id={r['clip_id']:r['evidence'] for r in fused}
    # Preserve every full original record and make inherited result links resolve
    # inside this package, not silently point to absent historical files.
    for row in rows:
        full=read(BASE/row['result_file'])
        full['source_original_result_file']=str(BASE/row['result_file'])
        full['phase_height_candidate']=row['phase_height_candidate']
        full['phase_height_supported']=row['phase_height_supported']
        full['silhouette_height_proxy_candidate']=row['silhouette_height_proxy_candidate']
        full['phase_height_evidence_file']=row['phase_height_evidence_file']
        full['contact_bracket_pose_height_evidence']=fusion_by_id.get(row['clip_id'])
        save(OUT/row['result_file'],full)
    save(OUT/'contact_bracket_pose_height.json',fused)
    save(OUT/'input_hashes.json',hashes);save(OUT/'tracking_audit.json',audit);save(OUT/'band_sensitivity.json',sensitivity)
    save(OUT/'video_validation_all.json',video_checks);save(OUT/'deterministic_replay.json',replay)
    (OUT/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    validation=dict(canonical_rows=39,experiments=39,tests_run=result.testsRun,tests_pass=True,
                    videos_fully_decoded=len(video_checks),total_decoded_frames=sum(v['decoded'] for v in video_checks),
                    unchanged_original_row_fields=unchanged,inputs_unchanged=True,source_hashes_unchanged=True,
                    deterministic_replays=len(replay),all_replays_pass=True,accuracy_estimate=None,
                    all_39_full_result_links_resolve=all((OUT/r['result_file']).is_file() for r in rows),
                    human_direction_review_preserved=True,whole_record_QA_not_promoted=True)
    save(OUT/'validation.json',validation)
    old_unknown=[r for r in summary['experiments'] if r['previous_height']=='unknown']
    aux_new=[r['clip_id'] for r in old_unknown if r['silhouette_height_proxy_candidate']!='unknown']
    strict_new=[r['clip_id'] for r in old_unknown if r['height_supported']]
    complete_masks=sum(a['connected_mask_frames']==a['mask_frames'] for a in audit)
    metrics=dict(old_unknown_height=23,new_strict_pose_height_candidates=strict_new,
                 new_contact_bracket_pose_height_candidates=[r['clip_id'] for r in fused if r['previous_height']=='unknown' and r['evidence']['supported']],
                 contact_bracket_pose_height_counts=dict(Counter(r['evidence']['height_candidate'] for r in fused)),
                 new_exploratory_silhouette_height_candidates=aux_new,
                 continuous_mask_windows=complete_masks,actual_segmentation_clips=len(audit),
                 strict_pose_complete_phase_candidates=summary['complete_first_dive_candidates'],
                 silhouette_complete_phase_candidates=summary['complete_silhouette_phase_candidates'],
                 original_height_unknown_retained=23,official_QA_acceptances_retained=15,
                 accuracy_estimate=None)
    save(OUT/'release_metrics.json',metrics)
    shutil.copy2(ROOT/'outputs/goalkeeper_height_phase_2026-10-11/ai_visual_review.json',OUT/'ai_visual_review.json')
    extra=f'''<h2>冻结验证与结果解释</h2><p>{result.testsRun}项所选测试通过，{len(video_checks)}段视频共{validation['total_decoded_frames']}帧完整解码；39条原始字段（含人工方向、原高度、QA）全部不变。</p>
<p>原23条unknown中：严格姿态新增{len(strict_new)}条高度候选；较弱轮廓路线新增{len(aux_new)}条高度代理候选，两类不能合并成准确率。{len(audit)}条实际分割中{complete_masks}条完整覆盖请求/实际窗口；首次扑救阶段候选与整段跟踪完整也不是同一个指标。</p>
<p>正式高度仍为14低、2中、23未知，正式整条QA仍为15/39。为避免尚未复核的轮廓接地和二维地面假设污染提交结果，本次不自动覆盖正式高度。模型代码正常运行不等于阶段语义正确。</p>
<p>高度定义必须先讲清：这里分析身体伸展阶段的离地代理，不是球高度、手套最高点或球门命中分区。Cuadrado可同时有低身体位置与较高伸手；不能据球飞高就把身体高度编码成高。Smolov静态复核发现轮廓启动/落地边界可能晚于真正接地，尚需完整动作复核。<a href="ai_visual_review.json">AI静态观察记录</a>不是人工真值。</p>
<p>新增较弱高度代理案例：{html.escape(', '.join(aux_new) or '无')}。</p>
<p>姿态+轮廓地面锚点融合：{metrics['contact_bracket_pose_height_counts']}；原unknown新增{len(metrics['new_contact_bracket_pose_height_candidates'])}条融合候选。该路线检查真实可见髋点及伸展窗口覆盖，但落地边界/地面轨迹仍是假设，依然需要复核。<a href="contact_bracket_pose_height.json">融合证据及拒绝原因</a>。</p>
<p><a href="validation.json">验证</a> · <a href="tracking_audit.json">逐例轨迹覆盖</a> · <a href="band_sensitivity.json">高度分界敏感性</a> · <a href="release_metrics.json">分母明确的指标</a>。</p>'''
    page=(OUT/'report.html').read_text(encoding='utf-8').replace('<a href="None">阶段视频</a>','<span>未初始化：无阶段视频</span>').replace('</html>',extra+'</html>')
    (OUT/'report.html').write_text(page,encoding='utf-8')
    (OUT/'README.txt').write_text('''守门员完整首次扑救阶段与高度探索包。先打开report.html。
正式方向/人工方向/高度/整条QA未被覆盖；summary.json中的phase_height_candidate和silhouette_height_proxy_candidate为不同证据等级的新增字段。
严格高度：可见肩髋姿态、稳定预踢地面参考、落地前伸展峰值，缺口/深度偏移拒绝。
较弱轮廓高度：连续SAM2人物轮廓、变扁/下边缘趋稳落地候选、接地锚点之间直线投影地面路径假设；不是真实髋点、精确离地高度或地面真值。
高度低/中/高按预踢身体像素高度归一化，非米制；未做视频人工高度真值标注，accuracy_estimate=null。
相机运动为排除人物的LK+RANSAC背景二维仿射；不能解决所有景深运动和遮挡。
跟踪至踢球后2秒或实际视频结尾；遇掩膜QC、镜头突变/ROI出界停止身份连接；不自动桥接到另一个人物。首次扑救完整候选不等于整个长视频都跟完。
代码复现（在原项目环境，权重和上游缓存必须存在）：
python -X utf8 scripts/run_goalkeeper_height_phase.py --output-dir outputs/new_phase_run
默认只处理23条高度unknown；使用--clips传入规范39条clip_id可复现全批次。
--cache-run仅在视频JPEG字节、窗口、ROI、初始提示和模型哈希一致时复用本地实验缓存；正式结果不作为高度标签输入。
SAM2：Meta官方sam2.1_hiera_tiny.pt，512分辨率CPU实验（非默认1024等效精度声明）。
本地依赖隔离：.deps_goalkeeper_segmentation；姿态Ultralytics由原项目.venv导入。
包含局部JPEG、掩膜、候选姿态、阶段视频、源码快照与哈希。完整原视频、150MB权重和上游全部缓存未打包。
模型来源：https://github.com/facebookresearch/sam2
阶段分类/阈值/角色筛选/质量门控为本项目工程实现，不声称来自官方SAM2或SoccerNet。
''',encoding='utf-8')
    shutil.copy2(ROOT/'outputs/goalkeeper_sam2_update_2026-10-11/results_v4_release/requirements-sam2.txt',OUT/'requirements-sam2.txt')
    shutil.copy2(ROOT/'notebooks/goalkeeper_height_phase_review.ipynb',OUT/'goalkeeper_height_phase_review.ipynb')
    recovery=ROOT/'outputs/goalkeeper_height_phase_2026-10-11/expanded_roi_pilot_v1'
    if (recovery/'summary.json').exists():
        comparison=[]; bytrial={r['clip_id']:r for r in summary['experiments']}
        for candidate in read(recovery/'summary.json')['experiments']:
            original=bytrial[candidate['clip_id']]
            comparison.append(dict(clip_id=candidate['clip_id'],original_connected_mask_frames=original['phase_evidence'].get('connected_mask_frames'),
                                   expanded_connected_mask_frames=candidate['phase_evidence'].get('connected_mask_frames'),
                                   original_stop=original['phase_evidence'].get('tracking_stop_reason'),expanded_stop=candidate['phase_evidence'].get('tracking_stop_reason'),
                                   selected_automatically=False,height_not_promoted=True))
        shutil.copytree(recovery,OUT/'expanded_roi_ablation',ignore=shutil.ignore_patterns('phase_staging.mp4'))
        save(OUT/'expanded_roi_comparison.json',comparison)
        page=(OUT/'report.html').read_text(encoding='utf-8').replace('</html>',
            '<h2>扩大ROI对照</h2><p>另做2例1.6倍ROI扩大实验，身体归一化尺度不变。不是普遍修复，也不自动选择标签。</p><pre>'+
            html.escape(json.dumps(comparison,ensure_ascii=False,indent=2))+'</pre><p><a href="expanded_roi_ablation/report.html">对照视频/结果</a></p></html>')
        (OUT/'report.html').write_text(page,encoding='utf-8')
    for name in ['scripts/finalize_goalkeeper_height_phase.py','tests/test_goalkeeper_dive_phase.py','scripts/run_goalkeeper_role_experiment.py',
                 'src/goalkeeper_height_fusion.py','tests/test_goalkeeper_height_fusion.py','scripts/recover_goalkeeper_phase_tracking.py']:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    bundle=ROOT/'deliverables/VisionII_Goalkeeper_Height_Phases_2026-10-11.zip'
    if bundle.exists():raise FileExistsError('Bundle already exists')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('CRC failed')
    save(OUT/'bundle_validation.json',dict(path=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size))
    print(json.dumps(dict(metrics=metrics,validation=validation,bundle=str(bundle)),ensure_ascii=False))


if __name__=='__main__':main()
