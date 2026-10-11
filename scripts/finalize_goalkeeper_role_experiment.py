"""Freeze identity-filter evidence and source-frame/video QA, without promoting directions."""
from pathlib import Path
from copy import deepcopy
import sys,html,io,unittest,importlib.util,ast,subprocess,shutil,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha,RUN
from scripts.run_goalkeeper_role_experiment import compute,geometries_for
from src.goalkeeper_role_association import select_goalkeeper_role,deduplicate_pose_people
OUT=ROOT/'outputs/goalkeeper_role_association_2026-10-11/final_v4'
CID='col_eng_p03_cuadrado_col11'


def main():
    cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():raise FileExistsError('Final output already exists')
    sources=['src/goalkeeper_role_association.py','scripts/run_goalkeeper_role_experiment.py',
             'scripts/refine_goalkeeper_actor_pose.py','scripts/finalize_goalkeeper_role_experiment.py',
             'tests/test_goalkeeper_role_association.py','src/goal_geometry.py','src/person_appearance.py',
             'src/goalkeeper_pose_direction.py','src/goalkeeper_direction_evidence.py','src/goalkeeper_pipeline.py']
    protocol=dict(code_sha256={n:sha(ROOT/n) for n in sources},no_manual_identity_or_outcome_inputs=True,
                  default_v3_results_not_overwritten=True,identity_verified=False,accuracy_estimate=None,
                  engineering_role_weights={'feet':.6,'goal_mouth':.2,'wrists':.1,'pre_visibility':.1})
    save(OUT/'protocol.json',protocol)
    for name in sources:
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        ast.parse((ROOT/name).read_text(encoding='utf-8'))
    base=ROOT/'outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final';summary=read(base/'summary.json');roles=[];inputs=[]
    for row in summary['rows']:
        cid=row['clip_id'];original=read(RUN/'goal_appearance_v2'/f'{cid}.json');path=Path(original.get('source_pose_path','missing'))
        baseline=read(base/row['result_file']);inputs.append(dict(path=str(base/row['result_file']),sha256=sha(base/row['result_file'])))
        if not path.is_file():roles.append(dict(clip_id=cid,status='abstain',reason='missing_cached_pose'));continue
        pose=read(path);g=geometries_for(pose,baseline['kick_frame']);role=select_goalkeeper_role(pose,g,baseline['kick_frame'])
        save(OUT/'pre_roles'/f'{cid}.json',role);save(OUT/'geometry'/f'{cid}.json',g)
        oldseed=original.get('goal_identity_evidence',{}).get('seed_track_id')
        roles.append(dict(clip_id=cid,status=role['status'],seed=role['seed_track_id'],previous_seed=oldseed,
                          excluded_pre_person_ids=role['excluded_pre_person_ids'],reason=role['reason']))
        inputs.append(dict(path=str(path),sha256=sha(path)))
        if cid==CID:
            keeper_pose=pose;keeper_baseline=baseline;cached=compute(pose,baseline['kick_frame'],g,baseline)
            save(OUT/'cached_role_lock.json',cached)
    variants=[dict(variant='cached_nano_pose_plus_role_filter',**{k:cached[k] for k in ('experimental_direction',) },last_frame=cached['association'].get('last_observed_frame'))]
    from src.step_pipeline import _build_tracks
    for label,path in [('medium_model_goal_roi',ROOT/f'outputs/goalkeeper_role_association_2026-10-11/experiment_v1/{CID}_fresh_pose.json'),
                       ('medium_model_rotated_goal_roi',ROOT/f'outputs/goalkeeper_role_association_2026-10-11/experiment_rotated_v2/{CID}_fresh_pose.json'),
                       ('medium_model_actor_roi_pilot',ROOT/f'outputs/goalkeeper_role_association_2026-10-11/actor_pilot_v3/{CID}_pose.json')]:
        p=read(path)
        for frame in p['pose_frames']:frame['detections']=deduplicate_pose_people(frame['detections'])
        p['tracks']=list(_build_tracks(p['pose_frames']).values())
        result=compute(p,keeper_baseline['kick_frame'],geometries_for(p,keeper_baseline['kick_frame']),keeper_baseline)
        save(OUT/f'{label}_role_lock.json',result)
        variants.append(dict(variant=label,experimental_direction=result['experimental_direction'],last_frame=result['association'].get('last_observed_frame'),
                             seed=result['role']['seed_track_id'],reasons=result['pose_direction_evidence']['reasons']))
        inputs.append(dict(path=str(path),sha256=sha(path)))
    save(OUT/'input_hashes.json',inputs)
    # The demonstration shows real detections only, never a stale goalkeeper box.
    cap=cv2.VideoCapture(keeper_pose['video_path']);kick=keeper_baseline['kick_frame'];first=kick-15;end=kick+36
    fullvideo=OUT/'review_staging.mp4';video=OUT/'cuadrado_role_filter_review.mp4'
    writer=cv2.VideoWriter(str(fullvideo),cv2.VideoWriter_fourcc(*'mp4v'),keeper_pose['fps'],(1280,720))
    if not writer.isOpened():raise ValueError('Video writer unavailable')
    observations={o['frame']:o for o in cached['association']['observations']};excluded=set(cached['role']['excluded_pre_person_ids'])
    source={}
    for t in keeper_pose['tracks']:
        for o in t['observations']:source.setdefault(o['frame'],[]).append((t['track_id'],o))
    chosen={kick-10,kick+9,kick+18,kick+30};panels=[];cap.set(1,first)
    for f in range(first,end+1):
        ok,img=cap.read()
        if not ok:raise ValueError(f'Decode/{f}')
        for tid,o in source.get(f,[]):
            if tid not in excluded:continue
            b=o['bbox'];cv2.rectangle(img,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(40,40,230),2)
            cv2.putText(img,f'EXCLUDED other person {tid}',(int(b[0]),int(b[1])-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,40,230),2)
        if f in observations:
            o=observations[f];b=o['bbox'];cv2.rectangle(img,tuple(map(round,b[:2])),tuple(map(round,b[2:])),(50,230,50),2)
            cv2.putText(img,f'GK candidate {o["association_source_track_id"]}',(int(b[0]),int(b[1])-10),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,230,50),2)
            for j in (9,10,15,16):
                if o['keypoint_confidences'][j]>=.4:cv2.circle(img,tuple(map(round,o['keypoints'][j])),4,(0,220,255),-1)
        else:cv2.putText(img,'NO GK DETECTION - do not substitute referee',(24,100),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,220,255),2)
        cv2.putText(img,f'f{f}  original time {f/keeper_pose["fps"]:.2f}s',(24,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
        writer.write(img)
        if f in chosen:panels.append(cv2.resize(img[200:580,150:650],(650,450)))
    cap.release();writer.release()
    ffmpeg=ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    subprocess.run([str(ffmpeg),'-y','-i',str(fullvideo),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(video)],check=True,capture_output=True)
    cap=cv2.VideoCapture(str(video));count=0
    while True:
        ok,img=cap.read()
        if not ok:break
        count+=1
    cap.release()
    if count!=end-first+1:raise ValueError('Review video decode mismatch')
    cv2.imwrite(str(OUT/'role_filter_frames.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
    stream=io.StringIO();suite=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_goalkeeper*')
    spec=importlib.util.spec_from_file_location('extra_features',ROOT/'tests/test_goalkeeper_analysis.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in vars(module):
        if name.startswith('test_') and callable(getattr(module,name)):suite.addTest(unittest.FunctionTestCase(getattr(module,name)))
    test=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);(OUT/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    if not test.wasSuccessful():raise ValueError('Tests failed')
    unchanged=all(sha(r['path'])==r['sha256'] for r in inputs)
    if not unchanged:raise ValueError('Source/previous output changed')
    audit=dict(tests_run=test.testsRun,tests_passed=True,input_hashes_unchanged=True,canonical_clips=39,
               role_candidates=sum(r['status']=='candidate_supported' for r in roles),
               changed_pre_seed=[r['clip_id'] for r in roles if r.get('seed') is not None and r.get('seed')!=r.get('previous_seed')],
               cached_no_excluded_person_selected=not(set(cached['association']['source_track_ids'])&excluded),
               review_video_decoded_frames=count,review_video_sha256=sha(video),accuracy_estimate=None,
               current_v3_unknown_unchanged=summary['new_unknown'],variants=variants)
    save(OUT/'validation.json',audit);save(OUT/'roles_summary_39.json',roles)
    selected=next(c for c in cached['role']['candidates'] if c['track_id']==cached['role']['seed_track_id'])
    ref=next(c for c in cached['role']['candidates'] if c['track_id']==3)
    vtable=''.join(f'<tr><td>{v["variant"]}</td><td>{v["experimental_direction"]}</td><td>{v.get("last_frame")}</td></tr>' for v in variants)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>门将与裁判分离实验</title>
<style>body{{max-width:1100px;margin:30px auto;padding:20px;font:16px/1.7 system-ui;color:#203b4b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd;padding:8px}}video,img{{width:100%}}.note{{background:#fff1da;padding:16px}}</style>
<h1>先分人、按门线选门将、过滤前景裁判</h1><p>2026-10-11 · Cuadrado重点实验与39条身份筛选检查</p>
<p>已实现多帧角色加权：脚到门线距离60%，身体在球门区域20%，手腕准备姿态10%，触球前可见性10%。使用脚踝关键点而非框底；没有手腕证据不强行补值。手腕不是手套识别。缺少门线或脚踝，或两个人同样符合，均弃权。</p>
<p>靠得近的两个人按各自肩、髋、脚踝区分，不仅按框重叠去重。选中门将后，记录其他人物的排除ID；之后运动跟踪不能接到这些人身上。门将的身体变低不会仅因框高度骤减被当成身份改变；短时换ID需位置、外观连续证据，长断轨不硬接。</p>
<h2>这段实测</h2><p>缓存轨迹4为门将候选，轨迹3对应前景裁判。门将脚到门线的多帧中位距离为{selected['median_foot_distance_goal_height']:.3f}个图像球门高度，裁判为{ref['median_foot_distance_goal_height']:.3f}；工程加权得分约{selected['score']:.3f}与{ref['score']:.3f}，不是身份概率。门将被保留，裁判被排除。</p>
<video controls preload="metadata" src="cuadrado_role_filter_review.mp4"></video><p>绿框：当帧门将检测；红框：排除的人物；黄点：手腕/脚踝。后段没有门将检测时明确显示缺失，不把绿框贴到裁判上。视频是源时间2.90–4.60秒的标注预览，非新生成动作。</p>
<img src="role_filter_frames.jpg" alt="四个实际源帧的身份过滤证据">
<div class="note">这段的方向仍unknown。触球前原版其实已选对门将；关键问题是第111帧以后门将横卧并部分被裁判遮挡，检测模型漏掉了身体。身份加权能防止误跟裁判，但不能补出不存在的检测。本轮没有把unknown强行改成左右，现有v3总表仍33条方向候选、6条unknown。</div>
<h2>对照实验</h2><table><tr><th>方法</th><th>方向</th><th>最后有效门将帧</th></tr>{vtable}</table>
<p>额外测试了更大的本地YOLO11m姿态模型、局部画面0/90/270度旋转及更紧的人物ROI。旋转后坐标映射回原图，重复检测须身体关键点也重合才融合；未改善这条后段漏检。人物ROI是3帧试探，不是全量新推理。</p>
<h2>39条筛选检查</h2><p>36条得到门线身份候选，选中的起始人物与旧版相同；Kramaric无缓存，Vida与Rakitic的俄罗斯—克罗地亚片段未得到足够门线/脚踝证据。这是身份候选覆盖，不是36/39的准确率，也没有因此清空原总表已有方向。</p>
<h2>验证与文件</h2><p>{test.testsRun}项相关测试通过；旧缓存、旧结果哈希一致；标注视频{count}帧解码通过。<a href="validation.json">验证与实验结果</a> · <a href="roles_summary_39.json">39条身份筛选</a> · <a href="cached_role_lock.json">逐帧选择与排除记录</a> · <a href="tests.log">测试日志</a> · <a href="README.txt">复现说明</a></p>
<p>静态图由AI查看，不作为用户人工真值；预测身份与方向仍需独立视频标注验证。旧结果与原视频均未覆盖。</p></html>'''
    (OUT/'report.html').write_text(report,encoding='utf-8')
    (OUT/'README.txt').write_text('门将/裁判分离实验：report.html，绿色门将、红色排除人物。\n当前方向总表没有替换，仍6条unknown；本轮解决身份筛选与防误跟，不宣称解决遮挡下的漏检。\n源码快照在source_snapshot/；逐帧身份选择cached_role_lock.json；39条角色检查roles_summary_39.json。\n复现身份及模型实验（必须使用新的输出目录）：\npython -X utf8 scripts/run_goalkeeper_role_experiment.py --all-roles --fresh --output-dir outputs/goalkeeper_role_new_run\n旋转消融加 --rotations 0 90 270。\n已有本地模型models/weights/yolo11m_pose.pt，无新模型下载；沿用requirements-evening-experiment.txt的观测环境。\n原推理输入、历史对照与源码哈希见protocol.json及input_hashes.json。没有人工身份种子或外部赛果输入。\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
