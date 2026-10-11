"""Partial-pose, order-preserving, constant-speed subsequence matching.

This is a bounded affine time search, NOT unconstrained DTW or 3D pose.
q_time = offset + stretch * (reference_time - reference_start).
Reference: FMP diagonal/tempo-scaled matching (Muller et al.); code here is
project-written. Missing joints remain missing. No temporal interpolation.
The engineering thresholds are uncalibrated and outputs are candidates only.
"""
from dataclasses import dataclass, asdict
import numpy as np

TRIPLES=((5,7,9),(6,8,10),(5,11,13),(6,12,14),(11,13,15),(12,14,16))


@dataclass(frozen=True)
class AlignmentConfig:
    min_samples:int=10
    min_shared_lower_angles:int=2
    min_variation:float=.06
    min_reference_span_sec:float=.8
    min_coverage:float=.8
    max_pose_gap_sec:float=.35
    time_tolerance_sec:float=.075
    max_angle_rmse:float=.14
    min_null_margin:float=.015
    min_normal_speed_margin:float=.015
    slow_stretch:float=1.5
    max_near_optimal_stretch_width:float=.5
    near_optimal_cost_delta:float=.005
    stretches:tuple=tuple(float(x) for x in np.arange(.75,4.001,.125))


def partial_angles(person, confidence=.4):
    k=np.asarray(person.get('keypoints',[]),float)
    c=np.asarray(person.get('keypoint_confidences',[]),float)
    if k.shape!=(17,2) or c.shape!=(17,):return [None]*6
    values=[]
    for a,b,d in TRIPLES:
        if not np.isfinite(k[[a,b,d]]).all() or not np.isfinite(c[[a,b,d]]).all() or min(c[[a,b,d]])<confidence:
            values.append(None);continue
        u=k[a]-k[b];v=k[d]-k[b];norm=np.linalg.norm(u)*np.linalg.norm(v)
        values.append(float(np.arccos(np.clip(np.dot(u,v)/norm,-1,1))/np.pi) if norm>=1 else None)
    return values


def usable(row):
    return sum(v is not None and np.isfinite(v) for v in row['feature'][2:])>=2


def contiguous(rows, gap):
    """Never bridge a large observation gap. Return all eligible blocks."""
    blocks=[]
    for r in rows:
        if not usable(r):continue
        if not blocks or r['time_sec']-blocks[-1][-1]['time_sec']>gap:
            blocks.append([])
        blocks[-1].append(r)
    return blocks


def _arrays(rows):
    return np.array([r['time_sec'] for r in rows]),np.array([r['feature'][2:] for r in rows],float)


def _search(rt,a,qt,b,cfg):
    """Search all offsets/scales equally for forward and null templates."""
    matches=[]
    if len(rt)<cfg.min_samples or len(qt)<cfg.min_samples:return matches
    active=np.array([np.isfinite(a[:,j]).sum()>=cfg.min_samples and
                     np.nanstd(a[:,j])>=cfg.min_variation for j in range(a.shape[1])])
    if active.sum()<cfg.min_shared_lower_angles:return matches
    a=a.copy();a[:,~active]=np.nan
    for swap in (False,True):
        query=b[:,[1,0,3,2]] if swap else b
        for stretch in cfg.stretches:
            starts=qt[qt+stretch*(rt[-1]-rt[0])<=qt[-1]+cfg.time_tolerance_sec]
            if not len(starts):continue
            predicted=starts[:,None]+stretch*(rt-rt[0])[None,:]
            right=np.clip(np.searchsorted(qt,predicted),0,len(qt)-1)
            left=np.maximum(0,right-1)
            index=np.where(abs(qt[left]-predicted)<=abs(qt[right]-predicted),left,right)
            paired=query[index]
            finite=np.isfinite(paired)&np.isfinite(a[None,:,:])
            shared=finite.sum(axis=2)
            supported=(abs(qt[index]-predicted)<=cfg.time_tolerance_sec)&(shared>=cfg.min_shared_lower_angles)
            costs=np.where(finite,(paired-a[None,:,:])**2,0).sum(axis=2)/np.maximum(shared,1)
            for off in np.where((supported.mean(axis=1)>=cfg.min_coverage)&supported[:,0]&supported[:,-1])[0]:
                good=supported[off];ri=np.flatnonzero(good);qi=index[off,good]
                if len(np.unique(qi))<cfg.min_samples:continue
                if np.max(np.diff(rt[ri]),initial=0)>cfg.max_pose_gap_sec:continue
                # Require actual observed movement also in the matched query span.
                moving=sum(np.isfinite(paired[off,good,j]).sum()>=cfg.min_samples and
                           np.nanstd(paired[off,good,j])>=cfg.min_variation for j in np.flatnonzero(active))
                if moving<cfg.min_shared_lower_angles:continue
                matches.append({'cost':float(np.sqrt(np.mean(costs[off,good]))),
                    'stretch':float(stretch),'offset_sec':float(starts[off]),'left_right_swap':swap,
                    'coverage':float(good.mean()),'matched_samples':int(good.sum()),
                    'reference_indices':ri.tolist(),'query_indices':qi.tolist(),
                    'query_start_sec':float(qt[qi[0]]),'query_end_sec':float(qt[qi[-1]])})
    return sorted(matches,key=lambda r:(r['cost'],r['stretch'],r['offset_sec'],r['left_right_swap']))


def align(reference,query,config=None):
    cfg=config or AlignmentConfig()
    result={'status':'unknown','reason':None,'human_verified':False,'identity_verified':False,
            'reference_speed_verified':False,'absolute_speed_factor':None,'accuracy_validated':False,
            'viewpoint_invariant':False,'review_required':True,'config':asdict(cfg)}
    def stop(reason,**more):return {**result,'reason':reason,**more}
    for sequence in (reference,query):
        times=np.array([r['time_sec'] for r in sequence])
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):return stop('invalid_time_order')
        if any(len(r.get('feature',[]))!=6 for r in sequence):return stop('invalid_feature_shape')
    rb=[b for b in contiguous(reference,cfg.max_pose_gap_sec) if len(b)>=cfg.min_samples]
    qb=[b for b in contiguous(query,cfg.max_pose_gap_sec) if len(b)>=cfg.min_samples]
    if not rb or not qb:return stop('insufficient_contiguous_partial_pose')
    # Longest reference block is selected before looking at query fit.
    ref=max(rb,key=lambda b:(len(b),b[-1]['time_sec']-b[0]['time_sec']))
    rt,a=_arrays(ref)
    result.update(reference_samples=len(ref),reference_span_sec=float(rt[-1]-rt[0]),
                  reference_start_sec=float(rt[0]),reference_end_sec=float(rt[-1]))
    if rt[-1]-rt[0]<cfg.min_reference_span_sec:return stop('insufficient_reference_span')
    all_matches=[]
    for block_id,block in enumerate(qb):
        qt,b=_arrays(block)
        all_matches.extend({**r,'query_block_id':block_id} for r in _search(rt,a,qt,b,cfg))
    if not all_matches:return stop('insufficient_motion_or_supported_alignment')
    all_matches.sort(key=lambda r:r['cost']);best=all_matches[0]
    result['best_alignment']=best
    if best['cost']>cfg.max_angle_rmse:return stop('pose_mismatch')
    normal=min((r['cost'] for r in all_matches if r['stretch']<=1.25),default=None)
    result['best_normal_speed_cost']=normal
    near=[r['stretch'] for r in all_matches if r['cost']<=best['cost']+cfg.near_optimal_cost_delta]
    result['near_optimal_stretch_range']=[min(near),max(near)]
    # Null templates have the same search budget, masks and sample values.
    null_costs=[]
    rng=np.random.default_rng(20261011)
    for name,order in [('reversed',np.arange(len(a))[::-1])]+[(f'shuffle_{i}',rng.permutation(len(a))) for i in range(3)]:
        null=[]
        for block in qb:
            qt,b=_arrays(block);null.extend(_search(rt,a[order],qt,b,cfg))
        null_costs.append({'control':name,'cost':min((r['cost'] for r in null),default=None)})
    result['null_controls']=null_costs
    finite_null=[n['cost'] for n in null_costs if n['cost'] is not None]
    if finite_null and min(finite_null)-best['cost']<cfg.min_null_margin:return stop('ambiguous_vs_reversed_or_shuffled')
    if max(near)-min(near)>cfg.max_near_optimal_stretch_width:return stop('unstable_stretch')
    if best['stretch']<cfg.slow_stretch:
        return {**result,'status':'no_relative_slowdown_evidence','reason':'ordered_affine_alignment_without_sufficient_time_stretch'}
    if normal is not None and normal-best['cost']<cfg.min_normal_speed_margin:return stop('ambiguous_vs_normal_speed')
    return {**result,'status':'relative_slowdown_candidate','reason':'partial_lower_body_ordered_affine_match_beats_internal_nulls; not verified same event'}
