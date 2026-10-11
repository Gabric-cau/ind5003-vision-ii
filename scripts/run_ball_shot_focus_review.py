"""All-39 shot-focus replay with finite candidate extent and bounded context."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.deps_ball_mask'))
import cv2
import numpy as np
from src.ball_shot_focus import ShotFocusConfig,focus_shot
from src.goal_plane_rectification import project_image_point
from scripts.run_ball_mask_seed_experiment import crop,draw_text,FFMPEG

cv2.setNumThreads(2)


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def goal_polygon(c):
    quads=[]
    for f in range(c['kick_frame']-5,c['kick_frame']+1):
        q=c['goal_proposals'].get(str(f),{}).get('selected')
        t=c['camera']['transforms'].get(str(f))
        if q and t is not None:
            pts=[project_image_point(t,p) for p in q['corners']]
            if all(p is not None for p in pts):quads.append(pts)
    return np.median(quads,axis=0).tolist() if len(quads)>=2 else None


def render(c,r,path,dest):
    cid=c['clip_id'];phase=r['phase'];end=phase['focus_end_frame']
    expected=end-c['start_frame']+1
    assert expected>0
    video=dest/'media'/f'{cid}.mp4'
    cmd=[str(FFMPEG),'-hide_banner','-loglevel','error','-n','-f','rawvideo',
         '-pix_fmt','bgr24','-s','1120x500','-r',str(c['fps']/3),'-i','pipe:0',
         '-an','-c:v','libx264','-threads','2','-crf','21','-preset','fast',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    cap=cv2.VideoCapture(c['video_path']);cap.set(cv2.CAP_PROP_POS_FRAMES,c['start_frame'])
    points={p['frame']:p for p in path};seeds={p['frame']:p for p in r['seed']}
    main=phase['main_end_frame'] if phase['main_end_frame'] is not None else c['kick_frame']
    samples=[min(end,max(c['start_frame'],f)) for f in
             [c['kick_frame']-3,c['kick_frame']+4,main-3,main,main+2,end]]
    tiles={};scale=np.array([640/c['width'],360/c['height']])
    try:
        for f in range(c['start_frame'],end+1):
            ok,fr=cap.read();assert ok
            im=np.full((500,1120,3),25,np.uint8);scene=cv2.resize(fr,(640,360))
            p=points.get(f) or seeds.get(f)
            context=f>main
            color=(25,185,255) if context else (70,255,70)
            proposal=c['goal_proposals'].get(str(f),{}).get('selected')
            if proposal:
                cv2.polylines(scene,[np.rint(np.array(proposal['corners'])*scale).astype(np.int32)],True,(255,145,60),1)
            if p:
                center=np.array([p['x'],p['y']])*scale
                ext=np.array([p.get('width',8),p.get('height',8)])*scale/2
                cv2.rectangle(scene,tuple(np.rint(center-ext).astype(int)),tuple(np.rint(center+ext).astype(int)),color,1)
                cv2.circle(scene,tuple(np.rint(center).astype(int)),max(6,round(min(ext))),color,2)
                im[75:235,665:825]=crop(fr,p)
            im[40:400,:640]=scene
            draw_text(im,f'{cid} | source frame {f} | kick {c["kick_frame"]}',(10,23),scale=.48)
            draw_text(im,'CONTACT CONTEXT: excluded from incoming analysis' if context else 'INCOMING SHOT: candidate + finite bbox',(10,426),color,scale=.47)
            draw_text(im,f'Main end {main} | Context end {end} | outcome UNKNOWN',(10,452),scale=.48)
            draw_text(im,'Goal outline is a 2-D candidate; net bounce cannot establish a miss',(10,477),scale=.43)
            draw_text(im,'Contact context candidate' if context else 'Incoming candidate',(655,54),color,scale=.44)
            draw_text(im,'none' if p is None else p['detection_id'],(655,260),scale=.46)
            draw_text(im,phase['reason'],(655,305),scale=.34)
            draw_text(im,'Ball depth / physical contact: unknown',(655,335),scale=.35)
            draw_text(im,'Late tail retained in the original report',(655,365),scale=.35)
            proc.stdin.write(im.tobytes())
            if f in samples:tiles[f]=im.copy()
    finally:
        cap.release();proc.stdin.close()
    error=proc.stderr.read().decode(errors='replace')
    if proc.wait():raise RuntimeError(error)
    cap=cv2.VideoCapture(str(video));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(500,1120,3);count+=1
    cap.release();assert count==expected
    sheet=np.vstack([np.hstack([tiles[samples[j]],tiles[samples[j+1]]]) for j in (0,2,4)])
    cv2.imwrite(str(dest/'media'/f'{cid}_sheet.jpg'),sheet)
    return {'video':video.name,'sheet':f'{cid}_sheet.jpg','decoded_frames':count,
            'expected_frames':expected,'source_start_frame':c['start_frame'],
            'source_end_frame_exclusive':end+1,'video_sha256':sha(video)}


REASONS={
 'goal_reference_area_reached':'候选到达球门参考范围，主要射门段结束',
 'supported_large_turn_before_goal_area':'到球门参考范围前出现连续大转向：没进/误跟的候选线索',
 'candidate_outside_reachable_range':'候选超出连续运动可达范围，停止关联分析',
 'continuity_lost':'连续观测中断，停止延长轨迹',
 'bounded_horizon_without_established_event':'达到射门时间上限，结果仍未知',
 'track_ended_without_established_event':'当前轨迹结束，接触或结果未确认',
 'missing_track_or_seed':'缺少起点或轨迹'}


def report(out,rows):
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>球：射门范围与短观察窗口</title><style>body{max-width:1200px;margin:24px auto;padding:0 16px;background:#15191f;color:#e4eaf0;font:16px/1.65 system-ui}a{color:#a6d4ff}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border:1px solid #52616f;padding:8px}video,img{width:100%}details{border:1px solid #52616f;margin:15px 0;padding:12px}summary{cursor:pointer;font-weight:650}.note{color:#ffd3a4}</style>
<h1>球：射门范围与短观察窗口</h1>
<p>按你的新范围：主要关注离脚后的射门，首次到达球门参考范围、出现连续大转向或失去连续观测时结束主要分析段；接触附近最多再看约0.2秒。未触发事件时，主要窗口上限为自动触球后0.85秒。</p>
<p>绿色框/圈：主要射门候选；橙色框/圈：接触观察区候选，不参与来球方向判断；蓝色轮廓：自动球门范围。范围采用检测框尺寸，模糊和误识别会影响它，不能当成球的真实三维体积。</p>
<p class="note">大角度转向只作为候选线索，至少需要3个连续点支持；到达球门参考范围后的反弹不用于判没进。二维球门投影无法确认球的深度，当前进球/没进仍是unknown，也没有确认最终停球位置。</p>
<p>这是同一批自动球候选的分析范围调整，没有重新运行神经检测。截去后段不能当作准确率提升。旧完整轨迹保留供对照，当前范围仍可能提前结束或保留误跟，需要核对接触前后原画面。</p>
<p><a href="protocol.json">运行协议</a> · <a href="validation.json">检查记录</a> · <a href="tests.json">新增边界规则检查</a> · <a href="README.md">复现与限制</a></p>'''
    doc+=f'<p>已复测 {len(rows)}/39 条；分析边界均为自动候选，未经人工逐帧确认。</p><table><tr><th>片段</th><th>主要段结束</th><th>短观察结束</th><th>边界原因</th><th>后段未参与点数</th></tr>'
    for rr in rows:
        cid=rr['clip_id'];d=rr['phase']
        doc+=f'<tr><td><a href="#{cid}">{cid}</a></td><td>{d["main_end_frame"]}</td><td>{d["focus_end_frame"]}</td><td>{REASONS[d["reason"]]}</td><td>{d["excluded_observation_count"]}</td></tr>'
    doc+='</table>'
    for rr in rows:
        cid=rr['clip_id'];d=rr['phase'];m=rr['media']
        doc+=f'<details id="{cid}"><summary>{cid}</summary><p>{REASONS[d["reason"]]}。这只是分析边界，结果未知。</p>'
        if d['supported_large_turn']:
            turn=d['supported_large_turn']
            doc+=f'<p>大转向候选帧 {turn["turn_frame"]}，角度 {turn["angle_degrees"]:.1f}°；支持帧 {turn["support_frames"]}。仍可能是身份切换，不能据此自动判没进。</p>'
        doc+=f'<video controls preload="none" src="media/{m["video"]}"></video><img loading="lazy" src="media/{m["sheet"]}"><p><a href="results/{cid}.json">有限范围与边界证据</a></p><details><summary>此前完整窗口（后段对照）</summary><p><a href="../ball_mask_review_ready_2026-10-11/report.html#{cid}">完整轨迹及此前视觉核对</a></p><video controls preload="none" src="../ball_mask_review_ready_2026-10-11/media/{cid}.mp4"></video></details></details>'
    doc+='<script>function openTarget(){const e=document.getElementById(location.hash.slice(1));if(e){e.open=true;e.scrollIntoView();}}window.addEventListener("hashchange",openTarget);openTarget();</script></html>'
    (out/'report.html').write_text(doc,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve();parent=ROOT/'outputs/ball_mask_review_ready_2026-10-11'
    if out.exists():raise FileExistsError(out)
    for folder in ['media','results','source_snapshot']:(out/folder).mkdir(parents=True)
    protocol=load(parent/'protocol.json');ids=protocol['canonical_clips'];assert len(ids)==len(set(ids))==39
    cfg=ShotFocusConfig();files=[Path(__file__),ROOT/'src/ball_shot_focus.py',ROOT/'tests/test_ball_shot_focus.py',ROOT/'src/goal_plane_rectification.py']
    for file in files:(out/'source_snapshot'/file.name).write_bytes(file.read_bytes())
    save(out/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),
      'canonical_clips':ids,'config':asdict(cfg),'parent_report':str(parent),'parent_protocol_sha256':sha(parent/'protocol.json'),
      'policy_scope':'automatic analysis boundaries and short playback; unchanged ball candidates',
      'neural_inference_repeated':False,'tracker_reassociated':False,'outcome_labels_used':False,
      'visual_review_used_for_boundary':False,'manual_ball_points_used':False,'full_raw_clip_coverage':False,
      'depth_or_crossing_established':False,'accuracy_validated':False,'human_verified':False,
      'goal_polygon_source':'median stabilized automatic goal corners from kick-5 through kick, at least two proposals',
      'code_sha256':{str(file.relative_to(ROOT)):sha(file) for file in files}})
    rows=[]
    for i,cid in enumerate(ids,1):
        old_file=parent/'results'/f'{cid}.json';inp_file=parent/'inputs'/f'{cid}.json'
        old=load(old_file);inp=load(inp_file);v=old['variants']['mask_and_radius_35']
        cache_file=ROOT/'outputs/ball_detection_all_2026-10-11/detections'/f'{cid}.json';c=load(cache_file)
        assert sha(c['video_path'])==old['source_sha256']==inp['source_sha256']
        before=json.dumps(v['trajectory'],sort_keys=True)
        polygon=goal_polygon(c)
        kept,phase=focus_shot(v['trajectory'],v['effective_stationary_seed'],inp['goal_center'],
          c['kick_frame'],c['fps'],c['end_frame_exclusive'],goal_polygon=polygon,config=cfg)
        repeat,d2=focus_shot(v['trajectory'],v['effective_stationary_seed'],inp['goal_center'],
          c['kick_frame'],c['fps'],c['end_frame_exclusive'],goal_polygon=polygon,config=cfg)
        assert kept==repeat and phase==d2 and before==json.dumps(v['trajectory'],sort_keys=True)
        assert kept==v['trajectory'][:len(kept)]
        r={'clip_id':cid,'phase':phase,'trajectory_in_focus':kept,
           'seed':v['effective_stationary_seed'],'goal_reference_polygon':polygon,
           'parent_result_sha256':sha(old_file),'parent_inputs_sha256':sha(inp_file),
           'source_sha256':c['source_sha256'],'source_video_path':c['video_path'],
           'outcome':'unknown','final_position':None,'human_verified':False,
           'two_runs_equal':True,'parent_track_unchanged':True,'kept_track_is_prefix':True}
        r['media']=render(c,r,kept,out);save(out/'results'/f'{cid}.json',r)
        rows.append(r);report(out,rows)
        save(out/'summary.json',{'processed_clips':len(rows),'rows':rows,'accuracy_validated':False,'human_verified':False})
        save(out/'validation.json',{'processed_clips':len(rows),'rows':[
          {'clip_id':q['clip_id'],'two_runs_equal':q['two_runs_equal'],'parent_track_unchanged':True,
           'kept_track_is_prefix':True,**q['media']} for q in rows],
          'browser_interaction_verified':False,'accuracy_validated':False,'human_verified':False})
        print(f'[{i}/39] {cid}: main={phase["main_end_frame"]} context={phase["focus_end_frame"]} {phase["reason"]}',flush=True)


if __name__=='__main__':main()
