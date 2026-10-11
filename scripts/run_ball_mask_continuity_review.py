"""Reproducible all-39 pixel-mask/departure/continuity comparison.

Neural candidates come from the previous fresh all-39 inference. No earlier
ball trajectory, sparse visual review, outcome, or manually selected ball point
enters this run. Source-pixel masks and motion candidates are calculated anew.
"""
from scripts.run_ball_mask_seed_experiment import (
    ROOT, load, save, sha, extract_support, source_time_indices, motion_blobs,
    collision_geometry, MaskSeedConfig, choose_seed, associate_with_seed, render,
    cv2, np, asdict, datetime, timezone, Path, argparse, json, time, hashlib)
from src.ball_continuity_v4 import ContinuityConfig, track_ball_continuity, motion_forecasts
from src.ball_continuity import track_ball_continuity as control_tracker, ContinuityConfig as ControlConfig


def make_report(out, rows, complete=False):
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>球：掩膜与连续运动优化复测</title><style>
body{max-width:1200px;margin:25px auto;padding:0 15px;background:#15191f;color:#e2e7ed;font:16px/1.65 system-ui}a{color:#9bd0ff}video,img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #55616f}details{margin:20px 0;padding:14px;border:1px solid #55616f}summary{cursor:pointer;font-weight:650}.seeds{display:flex;gap:15px;flex-wrap:wrap}.seeds figure{width:300px;margin:0}.warn{color:#ffc4a0}.review{white-space:pre-wrap}</style>
<h1>球：掩膜与连续运动优化复测</h1>
<p>原始39条主镜头窗口全部纳入，同上轮窗口，没有按已知进球／没进球截掉尾段。复用上轮新推理的神经检测，以及原像素掩膜、运动差分候选；重新执行关联。本轮未重新运行神经模型。</p>
<p>起点加入白色、低饱和度与绿色偏色限制，并收紧脚附近范围；掩膜只影响静止起点，不删除飞行神经检测。离脚判断允许首次可见球已经离罚球点较远，用连续实测候选回推起跑区间。关联同时比较方向、预测位置及候选大小，减少选择细小网格干扰。</p>
<p>绿色：新版本选中的候选；蓝色：上轮候选；黄色：新静止起点；橙色加号：预测。右侧放大显示原像素和掩膜。绿色圈只是算法选择，必须看原像素判断是否是真球。</p>
<p class="warn">球门附近的遮挡、反弹、身份切换仍需核对；没有自动判定进球或最终位置。没有人工逐帧真值，不能把候选数、完整运行或视频解码当成准确率。</p>
<p id="review-status">AI抽样视觉复核待完成。</p>'''
    body += f'<p>运行完成 {len(rows)}/39；总状态：{"完成" if complete else "进行中"}。<a href="protocol.json">运行协议</a> · <a href="validation.json">检查记录</a></p>'
    body += '<table><tr><th>片段</th><th>原种子点数→新种子点数</th><th>旧／新关联结束帧</th><th>新状态</th></tr>'
    for r in rows:
        body += f'<tr><td><a href="#{r["clip_id"]}">{r["clip_id"]}</a></td><td>{r["old_seed_n"]}→{r["seed_n"]}</td><td>{r["old_end"]} / {r["end"]}</td><td>{r["status"]}</td></tr>'
    body += '</table>'
    for r in rows:
        cid = r['clip_id']
        body += f'<details id="{cid}"><summary>{cid}</summary><p class="review" id="review-{cid}">抽样核对待完成。</p><div class="seeds"><figure><img loading="lazy" src="media/{cid}_mask_and_radius_35_seed.jpg"><figcaption>新起点代表帧</figcaption></figure></div><video controls preload="none" src="media/{cid}.mp4"></video><img loading="lazy" src="media/{cid}_sheet.jpg"><a href="results/{cid}.json">新候选、对照与诊断数据</a></details>'
    body += '<script>const openTarget=()=>{const el=document.getElementById(location.hash.slice(1));if(el&&el.tagName==="DETAILS"){el.open=true;el.scrollIntoView();}};window.addEventListener("hashchange",openTarget);openTarget();</script></html>'
    (out/'report.html').write_text(body, encoding='utf-8')
    save(out/'summary.json', {'processed_clips':len(rows),'rows':rows,'accuracy_validated':False,'human_verified':False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--candidate-cache',type=Path)
    args=parser.parse_args()
    out=args.output_dir.resolve(); inp=ROOT/'outputs/ball_detection_all_2026-10-11'
    if out.exists(): raise FileExistsError(out)
    for folder in ['media','results','inputs','source_snapshot']:(out/folder).mkdir(parents=True,exist_ok=True)
    ids=load(inp/'protocol.json')['canonical_clips']
    assert len(ids)==len(set(ids))==39
    mask_cfg=MaskSeedConfig(maximum_saturation=95, maximum_green_excess=25,seed_after_kick_frames=3)
    cfg=ContinuityConfig()
    files=[Path(__file__),ROOT/'scripts/run_ball_mask_seed_experiment.py',ROOT/'src/ball_mask_seed.py',
           ROOT/'src/ball_continuity_v4.py',ROOT/'src/ball_continuity.py',ROOT/'src/event_ball_tracking.py',
           ROOT/'src/goal_event_tracker_v2.py',ROOT/'scripts/probe_goal_event_tracking.py',
           ROOT/'scripts/run_ball_continuity_probe.py',ROOT/'scripts/build_ball_detection_seed_evidence.py',
           ROOT/'src/goal_plane_rectification.py']
    for f in files:(out/'source_snapshot'/f.name).write_bytes(f.read_bytes())
    save(out/'protocol.json',{'created_at':datetime.now(timezone.utc).isoformat(),'canonical_clips':ids,
      'mask_config':asdict(mask_cfg),'continuity_config':asdict(cfg),'control_config':asdict(ControlConfig()),
      'input_neural_candidates':str(inp/'detections'),'neural_protocol_sha256':sha(inp/'protocol.json'),
      'new_neural_inference':False,'fresh_pixel_masks':args.candidate_cache is None,
      'fresh_motion_candidates':args.candidate_cache is None,
      'candidate_cache':str(args.candidate_cache.resolve()) if args.candidate_cache else None,
      'old_ball_path_used_for_inference':False,'old_result_usage':'automatic pose anchors and goal center; old paths for display only',
      'outcome_labels_used':False,'manual_ball_points_used':False,'full_raw_clip_coverage':False,
      'endpoint_policy_unchanged':True,'accuracy_validated':False,'human_verified':False,
      'opencv':cv2.__version__,'numpy':np.__version__,'code_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files}})
    rows=[]; validation=[]
    priority=['esp_rus_p09_aspas_esp17','esp_rus_p01_iniesta_esp6','esp_rus_p03_pique_esp3',
              'esp_rus_p07_ramos_esp15','esp_rus_p08_cheryshev_rus6','rus_cro_p08_vida_cro21']
    for i,cid in enumerate(priority+[x for x in ids if x not in priority],1):
        begun=time.monotonic();print(f'[{i}/39] {cid}',flush=True)
        c=load(inp/'detections'/f'{cid}.json');b=load(inp/'results'/f'{cid}.json')
        assert sha(c['video_path'])==c['source_sha256']
        if args.candidate_cache:
            prior=load(args.candidate_cache/'protocol.json')
            old_mask=dict(prior['mask_config']);new_mask=asdict(mask_cfg)
            old_mask.pop('seed_after_kick_frames');new_mask.pop('seed_after_kick_frames')
            # These are raw measured candidates and seed pixel counts, not
            # previous trajectories; association settings do not affect them.
            assert old_mask==new_mask
            cached=load(args.candidate_cache/'inputs'/f'{cid}.json')
            assert cached['source_sha256']==c['source_sha256']
            assert cached['raw_neural_cache_sha256']==sha(inp/'detections'/f'{cid}.json')
            e=load(args.candidate_cache/'results'/f'{cid}.json')['mask_evidence']
            pool=cached['candidates']
        else:
            e=extract_support(c,mask_cfg); timing=source_time_indices(c); blobs=motion_blobs(c)
            pool={f:[{**d,**timing[int(f)]} for d in ds+blobs.get(f,[])] for f,ds in c['detections'].items()}
        pool_hash=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()
        if args.candidate_cache: assert cached['candidate_hash']==pool_hash
        save(out/'inputs'/f'{cid}.json',{'source_sha256':c['source_sha256'],'candidates':pool,
          'camera':c['camera'],'anchors':b['anchors'],'goal_center':b['goal_reference_center'],
          'candidate_hash':pool_hash,'raw_neural_cache_sha256':sha(inp/'detections'/f'{cid}.json')})
        seed,sd=choose_seed(c['detections'],c['kick_frame'],b['anchors'],e,use_mask=True,radius=35,config=mask_cfg)
        meta={f:{'coordinate_supported':c['camera']['transforms'].get(f) is not None} for f in pool}
        regions=collision_geometry(c,{})
        kw=dict(frame_metadata=meta,collision_regions=regions)
        p,d=associate_with_seed(pool,c['kick_frame'],b['anchors'],b['goal_reference_center'],seed,sd,tracker=track_ball_continuity,config=cfg,**kw)
        repeat,rd=associate_with_seed(pool,c['kick_frame'],b['anchors'],b['goal_reference_center'],seed,sd,tracker=track_ball_continuity,config=cfg,**kw)
        assert p==repeat and d==rd
        cp,cd=associate_with_seed(pool,c['kick_frame'],b['anchors'],b['goal_reference_center'],seed,sd,tracker=control_tracker,config=ControlConfig(),**kw)
        allowed={(x['frame'],x['detection_id']) for ds in pool.values() for x in ds}
        assert all((x['frame'],x['detection_id']) in allowed for x in p)
        assert hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()==pool_hash
        effective_seed=[q for q in seed if q['frame']<=p[0]['frame']] if p and d['tracking_status']=='candidate' else seed
        result={'clip_id':cid,'mask_evidence':e,'source_video_path':c['video_path'],'source_sha256':c['source_sha256'],
         'variants':{'mask_and_radius_35':{'seed':seed,'seed_diagnostics':sd,'trajectory':p,'tracking':d,
          'effective_stationary_seed':effective_seed,'stationary_support_after_departure_not_displayed':True,
          'forecasts':motion_forecasts(p,c['start_frame'],c['end_frame_exclusive'],config=cfg,model_states=d.get('motion_states'))},
          'same_seed_previous_association_control':{'seed':seed,'trajectory':cp,'tracking':cd}},
         'candidate_hash':pool_hash,'deterministic_replay':True,'input_candidates_unchanged':True,
         'flight_neural_candidates_retained':True,'manual_ball_points_used':False,'old_ball_path_used_for_inference':False,
         'outcome':'unknown','accuracy_validated':False,'human_verified':False}
        # The render function uses this name for the new selected track.
        result['media']=render(c,b,result,out,mask_config=mask_cfg)
        save(out/'results'/f'{cid}.json',result)
        old=b['variants']['detector_and_motion']['trajectory']
        row={'clip_id':cid,'seed_n':len(seed),'old_seed_n':len(b['stationary_seed']),
          'path_n':len(p),'end':p[-1]['frame'] if p else None,'old_end':old[-1]['frame'] if old else None,
          'status':d['tracking_status'],'control_path_n':len(cp),'control_status':cd['tracking_status'],
          'elapsed_seconds':round(time.monotonic()-begun,2),**result['media']}
        rows.append(row);print(json.dumps(row),flush=True)
        validation.append({'clip_id':cid,'deterministic_replay':True,'candidate_membership':True,
          'candidate_pool_unchanged':True,'source_hash_matches':sha(c['video_path'])==c['source_sha256'],
          'decoded_video_frames':result['media']['decoded_frames'], 'expected_video_frames':c['end_frame_exclusive']-c['start_frame']})
        save(out/'validation.json',{'rows':validation,'processed_clips':len(rows),'accuracy_validated':False})
        make_report(out,rows,complete=len(rows)==39)


if __name__=='__main__':main()
