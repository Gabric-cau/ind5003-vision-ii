"""Freeze SAM2 release candidates and preserve the prior 39-clip baseline."""
from pathlib import Path
from collections import Counter
import sys,shutil,io,unittest,json,html,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha
BASE=ROOT/'outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final'
RUN=ROOT/'outputs/goalkeeper_sam2_update_2026-10-11'
OUT=RUN/'results_v4_release'


def main():
    if OUT.exists():raise FileExistsError('Frozen release exists')
    OUT.mkdir(parents=True);baseline=read(BASE/'summary.json');experiments=[];hashes=[]
    for row in baseline['rows']:
        p=BASE/row['result_file'];hashes.append(dict(path=str(p),sha256=sha(p)))
        if row['dive_direction']!='unknown':continue
        cid=row['clip_id']
        folder=('cuadrado_integrated1024_v4' if cid=='col_eng_p03_cuadrado_col11' else
                'aspas_censored_v2' if cid=='esp_rus_p09_aspas_esp17' else 'all_unknown_512_v1')
        source=RUN/folder/'clips'/cid;record=read(source/'result.json')
        dest=OUT/'clips'/cid
        shutil.copytree(source,dest,ignore=shutil.ignore_patterns('review_staging.mp4'))
        for p,h in read(source/'input_hashes.json').items():
            if sha(Path(p))!=h:raise ValueError('Input changed')
        for p in [source/'result.json',source/'segmentation.json',source/'direction_evidence.json']:
            hashes.append(dict(path=str(p),sha256=sha(p)))
        record['experiment_source_directory']=str(source);experiments.append(record)
    byid={r['clip_id']:r for r in experiments};rows=[]
    for row in baseline['rows']:
        item=dict(row);r=byid.get(row['clip_id'])
        if r and r['candidate_direction']!='unknown':
            item.update(previous_v3_direction=row['dive_direction'],dive_direction=r['candidate_direction'],
                sam2_direction_evidence=r['direction_evidence'],direction_support_status='sam2_candidate_review_required',
                whole_record_review_required=True,human_verified=False,identity_verified=False)
        full=read(BASE/row['result_file'])
        full['source_v3_result']=str(BASE/row['result_file'])
        if r and r['candidate_direction']!='unknown':
            full.update(dive_direction=r['candidate_direction'],dive_direction_support_status='sam2_candidate_review_required',
                sam2_direction_evidence=r['direction_evidence'],direction_policy='sam2_anatomical_fallback_v4_final',
                sam2_feature_provenance=dict(direction_evidence_file=f'clips/{row["clip_id"]}/direction_evidence.json',
                                            old_selected_candidate_trajectory_retained=True),
                review_required=True,human_verified=False,identity_verified=False)
        item['result_file']=f'results_39/{row["clip_id"]}.json'
        save(OUT/item['result_file'],full)
        rows.append(item)
    summary=dict(policy='sam2_anatomical_fallback_v4_final',clips=39,attempted_unknowns=6,previous_unknown=6,
        new_unknown=sum(r['dive_direction']=='unknown' for r in rows),directions=dict(Counter(r['dive_direction'] for r in rows)),
        recovered_candidates=[r['clip_id'] for r in experiments if r['candidate_direction']!='unknown'],
        unchanged_known_direction_candidates=33,unchanged_height_labels=39,
        whole_record_QA_not_promoted=True,human_ground_truth_records=0,accuracy_estimate=None,
        sam2_image_size='Cuadrado body1024/head512, other body512',original_v3_retained=True,rows=rows,experiments=experiments)
    save(OUT/'summary.json',summary);save(OUT/'goalkeeper_results_39.json',dict(policy=summary['policy'],rows=rows))
    save(OUT/'input_hashes.json',hashes)
    files=['src/goalkeeper_sam2.py','src/goalkeeper_sam2_head_tracking.py',
        'scripts/run_goalkeeper_sam2_update.py','scripts/probe_goalkeeper_sam2_head_part.py',
        'scripts/finalize_goalkeeper_sam2_update.py','tests/test_goalkeeper_sam2.py']
    for n in files:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    # Copy matching historical source, because an unused head helper was appended
    # during the original body-only batch. Snapshot hash proves exact body source.
    shutil.copytree(RUN/'all_unknown_512_v1/source_snapshot',OUT/'body_batch_source_snapshot')
    proto=read(RUN/'all_unknown_512_v1/protocol.json')
    for n,h in proto['code_sha256'].items():
        if sha(OUT/'body_batch_source_snapshot'/n)!=h:raise ValueError('Historical source snapshot hash mismatch')
    save(OUT/'protocol.json',dict(policy=summary['policy'],sam2_source='https://github.com/facebookresearch/sam2',
        sam2_commit='2b90b9f5ceec907a1c18123530e92e794ad901a4',
        model_sha256=sha(ROOT/'models/weights/sam2.1_hiera_tiny.pt'),
        pose_model_sha256=sha(ROOT/'models/weights/yolo11m_pose.pt'),
        sam2_image_size=summary['sam2_image_size'],original_default_image_size=1024,
        note='512 is an explicit CPU-speed setting evaluated here, not claimed equal to 1024 accuracy.',
        source_sha256={n:sha(ROOT/n) for n in files},no_human_labels_read=True,accuracy_estimate=None))
    suites=['tests.test_goalkeeper_sam2','tests.test_goalkeeper_mask_experiment','tests.test_goalkeeper_role_association',
        'tests.test_goalkeeper_pose_direction','tests.test_goalkeeper_direction_evidence','tests.test_goalkeeper_observation_guard']
    stream=io.StringIO();test=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    (OUT/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    video_checks=[read(OUT/'clips'/r['clip_id']/'video_validation.json') for r in experiments]
    validation=dict(tests_run=test.testsRun,tests_pass=test.wasSuccessful(),all_review_videos_decoded=all(r['passed'] for r in video_checks),
        total_decoded_frames=sum(r['decoded_frames'] for r in video_checks),expected_frames=sum(r['expected_frames'] for r in video_checks),
        all_inputs_unchanged=all(sha(Path(r['path']))==r['sha256'] for r in hashes),
        historical_source_hashes_verified=True,all_39_result_links_resolve=all((OUT/r['result_file']).is_file() for r in rows),
        all_39_record_directions_match_summary=all(read(OUT/r['result_file'])['dive_direction']==r['dive_direction'] for r in rows),accuracy_estimate=None)
    save(OUT/'validation.json',validation)
    if not all(validation[k] for k in ['tests_pass','all_review_videos_decoded','all_inputs_unchanged']):raise ValueError('Validation failed')
    visual=read(RUN/'ai_visual_review.json');save(OUT/'ai_visual_review.json',visual)
    notes={
        'col_eng_p03_cuadrado_col11':'头部单独分割补足左右端点；跳变处停止，不使用之后漂移片段。',
        'cro_den_p04_kramaric_cro9':'新预踢推理恢复角色初始化；能看到门将，但未形成连续侧向头/躯干转变。',
        'esp_rus_p02_smolov_rus10':'SAM2延续原漏检段；头/躯干与身体长轴一致，仍需复核落地后的姿态。',
        'esp_rus_p06_golovin_rus17':'持续身体分割提供新的头/躯干方向候选。',
        'esp_rus_p08_cheryshev_rus6':'分割得到身体，但头/躯干位置及侧向姿态没有满足连续支持条件。',
        'esp_rus_p09_aspas_esp17':'源片段只有120帧，实际窗口90–119；姿态不够稳定，且身体接近ROI边缘。'}
    table=''.join(f'<tr><td>{html.escape(r["clip_id"])}</td><td>{r["candidate_direction"]}</td><td>{notes[r["clip_id"]]}</td><td><a href="clips/{r["clip_id"]}/sam2_keeper_review.mp4">视频</a></td></tr>' for r in experiments)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAM2门将接入报告</title>
<style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#fafafa;color:#17232b}}td,th{{border:1px solid #ccd4da;padding:10px;text-align:left}}table{{border-collapse:collapse;width:100%}}video{{width:100%}}.warn{{background:#fff0d3;padding:15px}}</style>
<h1>SAM2 门将跟踪接入：v4</h1><p>已对剩余6条 unknown 实际测试。方向候选由33条增加到{39-summary['new_unknown']}条，unknown 从6条降到{summary['new_unknown']}条。</p>
<p class="warn">这些是自动方向候选，不是准确率。没有人工 ground truth；自身左/右仍需相机朝向依据。原33条方向、39条扑救高度和原整条QA均保留，没有覆盖历史v3。</p>
<table><tr><th>案例</th><th>新候选（画面坐标）</th><th>效果/剩余问题</th><th>复核</th></tr>{table}</table>
<h2>Cuadrado 接入后的复核视频</h2><video controls src="clips/col_eng_p03_cuadrado_col11/sam2_keeper_review.mp4"></video>
<h2>方法</h2><ol><li>用预踢脚到门线的多帧角色筛选自动选择门将；排除裁判等其他人物；不读取人工身份、进球标签。</li>
<li>自动扩大人物局部ROI，SAM2.1 Hiera tiny持续分割。1024是原官方配置；另5例试用512降低CPU开销，不能称为等效精度。Cuadrado低分辨率掩膜损失头部重叠支持，因此保留1024身体掩膜，头部用经像素/模型校验的512缓存。</li>
<li>检查掩膜面积、逐帧位移和与被排除人物躯干的重叠。失败片段停止，不桥接到后来的另一个人。</li>
<li>对实际掩膜前景保留原像素、背景调暗，YOLO11m-pose在0/90/270度视图重提关键点，映射回原视频坐标。头、肩、髋需置信度≥0.5且位于掩膜；相反方向候选拒绝。</li>
<li>方向符号来自头/躯干，掩膜长轴只佐证形状，不能独自定左右。要求可靠预踢直立参考与连续≥3帧、跨度≥0.06秒侧向转变。</li>
<li>Cuadrado关键点仍漏检，因此追加SAM2头部子区域跟踪：初帧鼻子/可见头部点自动正提示，肩/髋负提示。检查头部面积、身体内重叠≥65%、端点位置与每帧位移；头部跳变停止。不是人工点击头部。</li>
<li>Aspas源视频仅120帧，自动截到真实结尾，不造帧、不将未观察末段当成已完成。结果39条汇总单独保存。</li></ol>
<p>底层来源：<a href="https://github.com/facebookresearch/sam2">Meta官方SAM2</a>。角色筛选、方向规则、阈值和整合脚本为项目工程实现，未声称来自SoccerNet。</p>
<h2>验证边界</h2><p>{test.testsRun}项所选测试通过；6个复核视频共{validation['total_decoded_frames']}帧全部解码通过；输入和历史v3哈希未变。AI静态查看记录在<a href="ai_visual_review.json">视觉复核记录</a>，不等于用户人工复核。Smolov等部分支持在落地阶段，仍需要完整动作/身份独立复核；固定ROI在大幅镜头运动下会失效。</p>
<p><a href="goalkeeper_results_39.json">新版39条结果</a> · <a href="summary.json">汇总</a> · <a href="README.txt">复现说明</a> · <a href="requirements-sam2.txt">依赖</a> · <a href="validation.json">验证记录</a></p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    (OUT/'README.txt').write_text('''SAM2门将v4接入包。先打开 report.html。
这是方向候选更新，不是人工真值/模型准确率；height/whole-record QA不因方向恢复而提升。
新增主入口：scripts/run_goalkeeper_sam2_update.py；--head-part-fallback 开启自动头部子区域续跟。
默认SAM2分辨率1024；六例均试过512，但Cuadrado保留1024身体+512头部，其他5例为512身体。CPU推理，CUDA扩展关闭。
在原项目根目录运行，必须选择新的输出目录：
python -X utf8 scripts/run_goalkeeper_sam2_update.py --image-size 512 --head-part-fallback --output-dir outputs/my_sam2_run
--clips 可只测指定案例；不传时只处理旧v3的6条unknown。已知33条不重新分类。
--resume 只允许协议/源码一致且输入哈希未变的已有运行，不应绕过校验。
本次Cuadrado执行头部fallback；另5条实际仅运行身体+姿态。不要把此发布声称为所有6条头部fallback都已跑过。
原v3保留在 outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final。
上游缓存、原视频和本地权重为依赖，包中不附150MB模型或完整原视频；原视频 data/raw_clips。
Meta权重：https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
官方SAM2提交：2b90b9f5ceec907a1c18123530e92e794ad901a4；隔离依赖 .deps_goalkeeper_segmentation。
所有新方法源码在source_snapshot，原身体批次源码在body_batch_source_snapshot（与当时协议哈希一致）。
源片段截尾/ROI边缘/裁判遮挡/落地滚动都可能导致unknown或错误方向，仍需复核。
''',encoding='utf-8')
    shutil.copy2(ROOT/'outputs/goalkeeper_mask_comparison_2026-10-11/requirements-mask-experiment.txt',OUT/'requirements-sam2.txt')
    bundle=ROOT/'deliverables/VisionII_Goalkeeper_SAM2_Update_2026-10-11.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
    save(OUT/'bundle_validation.json',dict(path=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size))
    print(json.dumps(dict(unknown=summary['new_unknown'],recovered=summary['recovered_candidates'],tests=test.testsRun,bundle=str(bundle)),ensure_ascii=False))


if __name__=='__main__':main()
