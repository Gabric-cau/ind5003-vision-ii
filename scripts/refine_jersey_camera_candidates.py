"""Preserve OCR conflicts as warnings, rerun integration without redoing OCR."""
from pathlib import Path
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_partial_pose_experiment import read,save,sha,DEFAULT as OLD
from scripts.build_jersey_gallery import DEFAULT
from src.jersey_evidence import iou
from src.jersey_camera_guard import text_camera_decision
from src.camera_view_intervals import supported_intervals


def main():
    out=DEFAULT/'camera_integration_v2';out.mkdir(exist_ok=False)
    index=read(DEFAULT/'evaluation_v1/retrieval_index.json')['tracklets'];lookup={(r['clip_id'],r['segment_id'],r['tracklet_id']):r for r in index}
    save(out/'protocol.json',{'runner_sha256':sha(__file__),'decision_code_sha256':sha(ROOT/'src/jersey_camera_guard.py'),
         'index_sha256':sha(DEFAULT/'evaluation_v1/retrieval_index.json'),
         'decision':'OCR number conflict is review warning only; never automatically delete based on it.',
         'reason':'Development inspection found systematic 11->71 / 14->74 errors. No target-biased correction is applied.',
         'release_replaced':False})
    rows=[];changes=[];warnings=[]
    for path in sorted((OLD/'guarded_views_v2/clips').glob('*.json')):
        cid=path.stem;d=read(path);g=read(DEFAULT/'gallery'/path.name)
        for segment in d['segments']:
            sid=segment['segment_id'];shot=next(s for s in g['shots'] if s['segment_id']==sid)
            votes=list(segment['guarded_classification']['sample_votes'])
            for k,sample in enumerate(segment['samples']):
                original=segment['dense_classification']['sample_votes'][k]
                people=sample['people'];kickers=[p for p in people if p.get('role')=='kicker_candidate']
                if original!='kicker_closeup' or not kickers:continue
                target=max(kickers,key=lambda p:p['area_fraction']);tracks=[]
                for t in shot['tracklets']:
                    obs=next((o for o in t['observations'] if o['frame']==sample['frame']),None)
                    if obs and iou(obs['bbox'],target['bbox'])>=.9:tracks.append(t['tracklet_id'])
                if len(tracks)!=1:continue
                evidence=lookup[(cid,sid,tracks[0])]
                decision=text_camera_decision(votes[k],original,evidence['upright_query'],people,sample.get('closeup_context',{}))
                item={'clip_id':cid,'segment_id':sid,'frame':sample['frame'],'tracklet_id':tracks[0],
                      'previous_vote':votes[k],'query_status':evidence['upright_query']['status'],**decision}
                if decision['warning']:warnings.append(item)
                if decision['changed']:changes.append(item)
                votes[k]=decision['vote']
            found=supported_intervals([s['frame'] for s in segment['samples']],votes,shot['start_frame'],shot['end_frame_exclusive'],g['fps'])
            rows.extend({**r,'clip_id':cid,'segment_id':sid,'identity_verified':False} for r in found)
    old=read(OLD/'guarded_views_v2/summary.json')['intervals']
    key=lambda r:(r['clip_id'],r['segment_id'],r['view_candidate'],r['start_frame'],r['end_frame_exclusive'])
    before={key(r) for r in old};after={key(r) for r in rows}
    summary={'before_intervals':len(old),'after_intervals':len(rows),'new_or_changed_intervals':[r for r in rows if key(r) not in before],
        'old_intervals_not_retained_exactly':[r for r in old if key(r) not in after],
        'changed_samples':len(changes),'warning_samples':len(warnings),'changes_by_reason':dict(Counter(r['reason'] for r in changes)),
        'human_ground_truth_records':0,'accuracy_estimate':None,'release_replaced':False}
    save(out/'camera_candidates.json',{'intervals':rows,'changes':changes,'warnings':warnings,'identity_verified':False,'human_verified':False})
    save(out/'summary.json',summary);print(summary)


if __name__=='__main__':main()
