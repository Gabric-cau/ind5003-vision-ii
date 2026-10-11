"""Paired partial-pose ablation with temporal and different-event controls."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import argparse,copy,hashlib,html,io,json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.camera_partial_alignment import AlignmentConfig,partial_angles,usable,align
from src.camera_timing_v2 import playback_evidence
from scripts.run_camera_partial_pose_experiment import BASE,DEFAULT,PILOT,read,save,sha


def sequences(measurement):
    result={};counts=Counter()
    for segment in measurement['segments']:
        roles={}
        for role in ('kicker','keeper'):
            seq=[]
            for frame in segment['frames']:
                counts['role_frame_checks']+=1
                people=[p for p in frame['people'] if p['role']==role+'_candidate']
                if len(people)!=1:
                    counts['no_unique_kit_role' if not people else 'multiple_same_kit_people']+=1
                    continue
                feature=partial_angles(people[0]);row={'time_sec':frame['time_sec'],'frame':frame['frame'],'feature':feature}
                if usable(row):
                    seq.append(row);counts['usable_partial_lower_body']+=1
                    if all(v is not None for v in feature):counts['usable_complete_six_angles']+=1
                    else:counts['rescued_partial_not_complete']+=1
                else:counts['unique_role_but_insufficient_lower_body']+=1
            roles[role]=seq
        result[str(segment['segment_id'])]=roles
    return result,counts


def brief(result):
    return {k:v for k,v in result.items() if k!='config'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=DEFAULT)
    p.add_argument('--name',default='evaluation_v1');a=p.parse_args();out=a.output_dir.resolve();dest=out/a.name
    if dest.exists():raise FileExistsError(dest)
    protocol=read(out/'extraction_protocol.json')
    missing=[cid for cid in protocol['canonical_clips'] if not (out/'measurements'/f'{cid}.json').exists()]
    if missing:raise ValueError('Measurements incomplete: '+','.join(missing))
    dest.mkdir()
    cfg=AlignmentConfig()
    config={'alignment':asdict(cfg),'code_sha256':sha(ROOT/'src/camera_partial_alignment.py'),
      'evaluation_script_sha256':sha(__file__),'extraction_protocol_sha256':sha(out/'extraction_protocol.json'),
      'reference':'https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_DiagonalMatching.html',
      'implementation':'project-written masked lower-body affine time-grid matching; not DTW, not 3D invariant',
      'selection':'parameters fixed before this real-data evaluation; data are development data, not independent holdout',
      'negative_controls':'time reversal + 3 seeded shuffles with identical search budget; different-event same-match references',
      'release_policy':'do not promote any result to release without independent identity/action/speed validation'}
    save(dest/'protocol.json',config)
    files={cid:read(out/'measurements'/f'{cid}.json') for cid in protocol['canonical_clips']}
    seqs={};instrument=Counter()
    for cid,m in files.items():
        s,c=sequences(m);seqs[cid]=s;instrument.update(c)
    pairs=[];controls=[];reference_controls=[]
    for cid,m in files.items():
        refid=str(m['reference_segment_id'])
        if not m['segments']:continue
        old=read(BASE/'action_v3'/f'{cid}.json')
        for role in ('kicker','keeper'):
            ref=seqs[cid][refid][role]
            if len(ref)<cfg.min_samples:continue
            for kind,factor in (('same_timestamps',1),('synthetic_2x_timestamps',2),('synthetic_reversed_2x',2)):
                q=[{**r,'time_sec':10+factor*(r['time_sec']-ref[0]['time_sec'])} for r in ref]
                if kind=='synthetic_reversed_2x':
                    q=[{**r,'feature':list(oldrow['feature'])} for r,oldrow in zip(q,ref[::-1])]
                reference_controls.append({'clip_id':cid,'role':role,'control':kind,'result':brief(align(ref,q,cfg)),
                                           'is_synthetic_control_not_new_video':True})
        for sid in m['query_segment_ids']:
            for role in ('kicker','keeper'):
                ref=seqs[cid][refid][role];q=seqs[cid][str(sid)][role]
                full_ref=[r for r in ref if all(v is not None for v in r['feature'])]
                full_q=[r for r in q if all(v is not None for v in r['feature'])]
                nearest=playback_evidence([r['time_sec'] for r in full_ref],[r['feature'] for r in full_ref],
                                         [r['time_sec'] for r in full_q],[r['feature'] for r in full_q])
                original=old['evidence'].get(str(sid),{}).get('role_results',{}).get(role)
                row={'clip_id':cid,'segment_id':sid,'role':role,'original_cohort':original is not None,
                     'pilot':cid in PILOT,'reference_complete_samples':len(full_ref),'query_complete_samples':len(full_q),
                     'reference_partial_samples':len(ref),'query_partial_samples':len(q),
                     'old_7_5hz_result':original,'new_10hz_complete_nearest':nearest,
                     'new_10hz_complete_affine':brief(align(full_ref,full_q,cfg)),
                     'new_10hz_partial_affine':brief(align(ref,q,cfg))}
                pairs.append(row)
                # Next DIFFERENT event in the same match, selected independently of fit.
                pool=[other for other in protocol['canonical_clips'] if other!=cid and other.split('_p')[0]==cid.split('_p')[0]]
                if pool:
                    pivot=pool.index(next((o for o in pool if o>cid),pool[0]))
                    pool=pool[pivot:]+pool[:pivot]
                for other in pool:
                    om=files[other];otherref=seqs[other].get(str(om['reference_segment_id']),{}).get(role,[])
                    if len(otherref)>=cfg.min_samples:
                        controls.append({'reference_clip_id':other,'query_clip_id':cid,'segment_id':sid,'role':role,
                             'valid_sample_count_eligible':len(q)>=cfg.min_samples,
                             'result':brief(align(otherref,q,cfg)),'known_different_event':True});break
        print(json.dumps({'stage':'evaluate','clip':cid,'pairs':len(m['query_segment_ids'])*2}),flush=True)
    save(dest/'pairs.json',pairs);save(dest/'different_event_controls.json',controls);save(dest/'synthetic_time_controls.json',reference_controls)
    patterns=('test_camera_partial_alignment.py','test_camera_timing_v2.py','test_goal_geometry.py',
              'test_goal_plane_rectification.py','test_goal_event_tracker.py')
    suite=unittest.TestSuite(unittest.TestLoader().discover(str(ROOT/'tests'),pattern=p) for p in patterns)
    stream=io.StringIO();test=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (dest/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    by_variant={key:dict(Counter(r[key]['status'] for r in pairs)) for key in
                  ('new_10hz_complete_nearest','new_10hz_complete_affine','new_10hz_partial_affine')}
    common=[r for r in pairs if r['original_cohort']]
    candidates=[{'clip_id':r['clip_id'],'segment_id':r['segment_id'],'role':r['role'],**r['new_10hz_partial_affine']['best_alignment']}
                for r in pairs if r['new_10hz_partial_affine']['status']=='relative_slowdown_candidate']
    summary={'clips_measured':len(files),'pose_frames':sum(len(s['frames']) for m in files.values() for s in m['segments']),
      'query_shots':len(pairs)//2,'role_pairs':len(pairs),'old_cohort_role_pairs':len(common),'instrumentation':dict(instrument),
      'partial_sample_eligible_pairs':sum(min(r['reference_partial_samples'],r['query_partial_samples'])>=10 for r in pairs),
      'complete_sample_eligible_pairs':sum(min(r['reference_complete_samples'],r['query_complete_samples'])>=10 for r in pairs),
      'common_cohort_sample_eligible':{'complete':sum(min(r['reference_complete_samples'],r['query_complete_samples'])>=10 for r in common),
                                     'partial':sum(min(r['reference_partial_samples'],r['query_partial_samples'])>=10 for r in common)},
      'variants':by_variant,'partial_reasons':dict(Counter(r['new_10hz_partial_affine']['reason'] for r in pairs)),
      'slowdown_candidates':candidates,'different_event_controls':len(controls),
      'different_event_sample_eligible':sum(r['valid_sample_count_eligible'] for r in controls),
      'different_event_accepted':sum(r['result']['status']=='relative_slowdown_candidate' for r in controls),
      'synthetic_control_status_counts':{kind:dict(Counter(r['result']['status'] for r in reference_controls if r['control']==kind))
                                      for kind in ('same_timestamps','synthetic_2x_timestamps','synthetic_reversed_2x')},
      'tests_run':test.testsRun,'tests_passed':test.wasSuccessful(),'source_videos_hash_matched':39,
      'human_ground_truth_records':0,'accuracy_estimate':None,'independent_holdout':False,'release_replaced':False,
      'limitation':'A time-stretch candidate is not verified same event, identity, replay, camera category or absolute slow-motion speed.'}
    save(dest/'summary.json',summary)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>部分姿态与时间对齐实验</title>
<style>body{{max-width:1100px;margin:36px auto;font:16px/1.7 system-ui;padding:0 20px;color:#17363d}}pre{{white-space:pre-wrap;background:#eff4f5;padding:18px}}.warn{{background:#fff4d4;padding:18px}}</style>
<h1>Vision II：部分姿态与有序时间对齐实验</h1><p>2026-10-11 · 单独实验，不覆盖2026-10-10成果包</p>
<p class="warn">候选更多不代表准确率更高。本次没有人工真值，也没有把普通视频合成成虚构新机位。所有源视频保持不变。</p>
<h2>修改了什么</h2><ol><li>从要求六个角度全部可用，改为分别保存可见角度；腿部至少两项有效才进入候选序列。缺失值不填0、不插值。</li>
<li>用有序的时间缩放和平移搜索代替每帧独立最近邻；不允许动作倒退，也不跨越超过0.35秒的观测缺口。</li>
<li>以倒序、三次固定随机打乱、正常速度及不同点球作对照。左右交换按整个序列尝试，不能逐帧挑选最便宜的一侧。</li>
<li>待查镜头允许已有三个采样中至少一个产生相关机位线索；新增镜头只进入实验，不改写原机位标签。</li></ol>
<p>算法是项目编写的掩码关节角恒定时间缩放搜索，不是DTW，也不是跨视角3D动作恢复。方法思路参考 <a href="{config['reference']}">FMP时间缩放/对角子序列匹配</a>，将音频匹配思想用于姿态是本项目实验，不是该来源对足球有效性的证明。</p>
<h2>结果统计</h2><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>
<h2>如何解读</h2><p>旧方案38段，新搜索范围和10Hz采样都变化了，因此应先看原38段中的完整/部分姿态对照，不把扩大搜索范围当成准确率进步。所有阈值是未标定工程选择，同一批视频此前已经被观察过，不称作独立测试集。</p>
<p>不同点球对照如果也得到减速候选，说明身体动作相似性不足以确认同一次事件。即使这些对照全拒绝，也不等于已测得真实准确率。完整原始记录见 <a href="pairs.json">配对结果</a>、<a href="different_event_controls.json">不同事件对照</a>、<a href="synthetic_time_controls.json">合成时间对照</a>、<a href="tests.log">测试日志</a>。</p></html>'''
    (dest/'report.html').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
