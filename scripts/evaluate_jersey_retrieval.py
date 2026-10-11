"""Evaluate same-input OCR ablation and apply conservative text evidence guards."""
from pathlib import Path
import sys,json,argparse,re
from collections import Counter
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import read,save,sha,DEFAULT as OLD
from scripts.build_jersey_gallery import DEFAULT
from src.jersey_evidence import consolidate,query_evidence,parse_number,iou
from src.camera_view_intervals import supported_intervals
from src.camera_view_guard import closeup_guard


def expected_queries():
    manifest=yaml.safe_load((ROOT/'configs/penalty_slicing_manifest.yaml').read_text(encoding='utf-8'));queries={}
    for match,group in manifest.items():
        for penalty in group['penalties']:
            key=penalty['kicker'];number=int(re.search(r'(\d+)$',key)[1])
            queries[f'{match}_p{penalty["number"]:02}_{key}']={'number':number,'surname':key.rsplit('_',1)[0],'team':penalty['team']}
    return queries


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=DEFAULT);a=p.parse_args();out=a.output_dir
    dest=out/'evaluation_v1';dest.mkdir(exist_ok=False)
    queries=expected_queries();index=[];counts=Counter();per_clip=[];metadata={}
    save(dest/'protocol.json',{'evaluator_sha256':sha(__file__),'evidence_code_sha256':sha(ROOT/'src/jersey_evidence.py'),
          'manifest_sha256':sha(ROOT/'configs/penalty_slicing_manifest.yaml'),'minimum_OCR_confidence':.85,
          'minimum_number_frames':2,'minimum_number_span_sec':.2,'minimum_vote_consensus':.75,
          'selection':'engineering evaluation on previously inspected development clips, not independent test set',
          'query_metadata_warning':'Expected jersey/name come from slicing manifest, not new human visual labels',
          'unknown_speed_preserved':True,'promotion_policy':'closeup-only; existing kit candidate and unique-context gates remain; no cross-shot identity assertion'})
    for path in sorted((out/'gallery').glob('*.json')):
        cid=path.stem;g=read(path);ocr=read(out/'ocr'/path.name);upright=read(out/'ocr_upright'/path.name)
        if sha(path)!=ocr['gallery_sha256'] or sha(path)!=upright['gallery_sha256']:raise ValueError('Inputs differ')
        q=queries[cid];m={(s['segment_id'],t['tracklet_id']):t for s in g['shots'] for t in s['tracklets']};metadata[cid]=m
        b={(r['segment_id'],r['tracklet_id']):r for r in ocr['tracklets']};u={(r['segment_id'],r['tracklet_id']):r for r in upright['tracklets']}
        if set(b)!=set(m) or set(u)!=set(m):raise ValueError('Tracklet cohorts differ')
        for key,track in m.items():
            obs_b=b[key]['observations'];obs_u=u[key]['observations']
            if [r['image'] for r in obs_b]!=[r['image'] for r in obs_u]:raise ValueError('Crop cohorts differ')
            eb=consolidate(obs_b);eu=consolidate(obs_u)
            qb=query_evidence(eb,q['number'],q['surname']);qu=query_evidence(eu,q['number'],q['surname'])
            for variant,obs,e,query in (('default',obs_b,eb,qb),('upright',obs_u,eu,qu)):
                counts[variant+'_tracklet_'+e['status']]+=1;counts[variant+'_query_'+query['status']]+=1
                counts[variant+'_frames_with_high_conf_numeric_token']+=sum(any(parse_number(t['text']) is not None and t['confidence']>=.85 and t['within_torso'] for t in o['tokens']) for o in obs)
                # Single temporal-middle selected crop, target blind. This is a
                # reading-availability ablation, NOT an equal-certainty estimate.
                middle=obs[len(obs)//2] if obs else {'tokens':[]}
                nums={parse_number(t['text']) for t in middle['tokens'] if t['confidence']>=.85 and t['within_torso'] and parse_number(t['text']) is not None}
                counts[variant+'_single_middle_crop_unambiguous_number']+=len(nums)==1
            record={'clip_id':cid,'segment_id':key[0],'tracklet_id':key[1],'start_time_sec':track['start_frame']/g['fps'],
              'end_time_sec':track['end_frame_exclusive']/g['fps'],'expected_query':q,'keyframes':track['keyframes'],
              'default_evidence':eb,'upright_evidence':eu,'default_query':qb,'upright_query':qu,
              'upright_observations':obs_u,'identity_verified':False,'human_verified':False}
            index.append(record)
        rows=[r for r in index if r['clip_id']==cid]
        per_clip.append({'clip_id':cid,'tracklets':len(rows),'default_matching_tracks':sum('match_candidate' in r['default_query']['status'] or r['default_query']['status']=='number_and_name_candidate' for r in rows),
                         'upright_matching_tracks':sum('match_candidate' in r['upright_query']['status'] or r['upright_query']['status']=='number_and_name_candidate' for r in rows)})
    lookup={(r['clip_id'],r['segment_id'],r['tracklet_id']):r for r in index}
    mutations=[];intervals=[];sample_records=[]
    for path in sorted((OLD/'guarded_views_v2/clips').glob('*.json')):
        cid=path.stem;old=read(path);g=read(out/'gallery'/path.name)
        for segment in old['segments']:
            sid=segment['segment_id'];votes=list(segment['guarded_classification']['sample_votes']);sample_updates=[]
            for k,sample in enumerate(segment['samples']):
                old_vote=votes[k];original=segment['dense_classification']['sample_votes'][k]
                if original!='kicker_closeup':continue
                people=sample['people'];kickers=[p for p in people if p.get('role')=='kicker_candidate']
                if not kickers:continue
                dominant=max(kickers,key=lambda p:p['area_fraction']);candidates=[]
                for key,t in metadata[cid].items():
                    if key[0]!=sid:continue
                    obs=next((o for o in t['observations'] if o['frame']==sample['frame']),None)
                    if obs is not None and iou(obs['bbox'],dominant['bbox'])>=.9:candidates.append(key[1])
                if len(candidates)!=1:continue
                track_id=candidates[0];e=lookup[(cid,sid,track_id)];status=e['upright_query']['status'];reason=None
                if status=='number_conflict':
                    if old_vote=='kicker_closeup':votes[k]='person_closeup_identity_unresolved';reason='repeated_number_conflict'
                elif status in ('number_match_candidate','number_and_name_candidate','name_match_candidate'):
                    gate=closeup_guard(people,sample.get('closeup_context',{}))
                    if old_vote=='person_closeup_identity_unresolved' and gate['reason']=='insufficient_field_context_not_proof_of_spectator':
                        votes[k]='kicker_closeup';reason='repeated_text_support_relaxes_grass_only_gate'
                if reason:
                    change={'clip_id':cid,'segment_id':sid,'frame':sample['frame'],'before':old_vote,'after':votes[k],
                          'reason':reason,'evidence_tracklet_id':track_id,'query_status':status,'recognized_number':e['upright_evidence']['number'],
                          'identity_verified':False,'human_verified':False}
                    mutations.append(change);sample_updates.append(change)
            original_shot=next(s for s in g['shots'] if s['segment_id']==sid)
            found=supported_intervals([s['frame'] for s in segment['samples']],votes,
                 original_shot['start_frame'],original_shot['end_frame_exclusive'],g['fps'])
            intervals.extend({**r,'clip_id':cid,'segment_id':sid,'identity_verified':False} for r in found)
            sample_records.append({'clip_id':cid,'segment_id':sid,'frames':[s['frame'] for s in segment['samples']],
                          'before_votes':segment['guarded_classification']['sample_votes'],'after_votes':votes,'changes':sample_updates})
    save(dest/'retrieval_index.json',{'tracklets':index,'human_ground_truth_records':0,'accuracy_estimate':None})
    save(dest/'camera_changes.json',{'changes':mutations,'sample_records':sample_records,'intervals':intervals,
         'release_replaced':False,'human_verified':False,'identity_verified':False})
    summary={'clips':len(queries),'shots':read(out/'gallery_summary.json')['shots_indexed'],'tracklets':len(index),
       'same_input_crops':sum(len(r['keyframes']) for r in index),'counts':dict(counts),'per_clip':per_clip,
       'default_clips_with_matching_text':sum(r['default_matching_tracks']>0 for r in per_clip),
       'upright_clips_with_matching_text':sum(r['upright_matching_tracks']>0 for r in per_clip),
       'camera_vote_changes':dict(Counter(r['reason'] for r in mutations)),
       'before_guarded_intervals':38,'after_text_guard_intervals':len(intervals),
       'after_views':dict(Counter(r['view_candidate'] for r in intervals)),
       'human_ground_truth_records':0,'accuracy_estimate':None,'release_replaced':False,
       'limitation':'Read/match coverage is not identity accuracy; spectators can wear matching jerseys; no same-action or playback-speed verification.'}
    save(dest/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='per_clip'},indent=2))


if __name__=='__main__':main()
