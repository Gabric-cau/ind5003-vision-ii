"""SAM2 fallback for unknown keeper directions, with visible-anatomy gating.

New versioned outputs only. The 33 existing direction candidates, heights, and
whole-record QA are retained. No manual roles/outcomes are input.
"""
from pathlib import Path
import sys,argparse,time,json,shutil,gc
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_direction_update import read,save,sha,RUN
from scripts.run_goalkeeper_role_experiment import geometries_for,fresh_pose
from src.goalkeeper_role_association import select_goalkeeper_role,unrotate_pose
from src.goalkeeper_mask_experiment import pose_seed,mask_info
from src.goalkeeper_sam2 import mask_qc,anatomy_sample,sam2_direction_evidence
from src.project_paths import resolve_project_file, resolve_project_video
from src.runtime import add_runtime_arguments, resolve_device, resolve_ffmpeg, inference_provenance
BASE=ROOT/'outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final'
DEFAULT=ROOT/'outputs/goalkeeper_sam2_update_2026-10-11/results_v4'


def prepare(cid,out,device='cpu',source_dir=None,baseline_dir=BASE,appearance_dir=None,pose_model=None):
    appearance_dir=appearance_dir or RUN/'goal_appearance_v2'
    pose_model=pose_model or ROOT/'models/weights/yolo11m_pose.pt'
    baseline_path=baseline_dir/'clips'/f'{cid}.json'
    appearance_path=appearance_dir/f'{cid}.json'
    baseline=read(baseline_path);kick=baseline['kick_frame']
    original=read(appearance_path)
    try:
        pp=resolve_project_file(ROOT,original.get('source_pose_path','missing'));pose=read(pp)
    except FileNotFoundError:
        pp=None
    if pp is None:
        vp=resolve_project_video(ROOT,'',cid,source_dir);cap=cv2.VideoCapture(str(vp))
        pose=dict(clip_id=cid,video_path=str(vp),fps=float(cap.get(cv2.CAP_PROP_FPS)),
                  frame_width=int(cap.get(3)),frame_height=int(cap.get(4)),tracks=[])
        cap.release();g=geometries_for(pose,kick)
        if not g:return baseline,None,dict(status='abstain',reason='missing_pose_and_no_supported_goal_geometry')
        pose=fresh_pose(pose,kick,g,out/'fresh_initialization_pose.json',pose_model,640,[0],device=device)
        pp=out/'fresh_initialization_pose.json'
    pose['video_path']=str(resolve_project_video(ROOT,pose['video_path'],cid,source_dir))
    if pose.get('source_sha256') and sha(pose['video_path']) != pose['source_sha256']:
        raise ValueError('Keeper pose belongs to different source video bytes')
    g=geometries_for(pose,kick);role=select_goalkeeper_role(pose,g,kick)
    save(out/'role.json',role)
    if role['seed_track_id'] is None:return baseline,None,role
    seed=next(t for t in pose['tracks'] if t['track_id']==role['seed_track_id'])
    first=max(0,kick-round(.5*pose['fps']));last=kick+round(1.2*pose['fps'])
    byframe={o['frame']:o for o in seed['observations']}
    candidates=sorted(f for f in byframe if first<=f<=first+round(.1*pose['fps']))
    if not candidates:return baseline,None,dict(status='abstain',reason='no_supported_initial_frame',role=role)
    first=candidates[0]
    pre=[o for f,o in byframe.items() if first<=f<kick-round(.2*pose['fps'])]
    if len(pre)<3:return baseline,None,dict(status='abstain',reason='too_few_pre_pose_boxes',role=role)
    b=np.median([o['bbox'] for o in pre],axis=0);h=b[3]-b[1];cx=(b[0]+b[2])/2
    roi=[max(0,int(cx-1.8*h)),max(0,int(b[1]-.35*h)),
         min(pose['frame_width'],int(cx+1.8*h)),min(pose['frame_height'],int(b[3]+.8*h))]
    frames={};cap=cv2.VideoCapture(pose['video_path'])
    source_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    requested_last=last
    if source_frames>0:last=min(last,source_frames-1)
    cap.set(1,first)
    for f in range(first,last+1):
        ok,img=cap.read()
        if not ok:raise ValueError(f'Decode/{cid}/{f}')
        frames[f]=img[roi[1]:roi[3],roi[0]:roi[2]]
    cap.release()
    others={};ids=set(role['excluded_pre_person_ids'])
    for t in pose['tracks']:
        if t['track_id'] in ids:
            for o in t['observations']:others.setdefault(o['frame'],[]).append(o)
    return baseline,dict(pose=pose,pose_path=pp,role=role,roi=roi,scale=float(h),frames=frames,
                         initial=byframe[first],keeper=byframe,others=others,first=first,last=last,
                         requested_last=requested_last,right_censored=last<requested_last,
                         source_inputs=[baseline_path,appearance_path]),role


def segment(data,out,predictor,image_size):
    first=data['first'];roi=data['roi'];frames=data['frames'];started=time.monotonic()
    maskdir=out/'masks';maskdir.mkdir(exist_ok=True);masks={}
    import torch
    jpg=out/'sam2_input';jpg.mkdir(exist_ok=True)
    for i,img in enumerate(frames.values()):
        if not cv2.imwrite(str(jpg/f'{i:05}.jpg'),img):raise ValueError('JPEG write failed')
    with torch.inference_mode():
        state=predictor.init_state(str(jpg),offload_video_to_cpu=True,offload_state_to_cpu=True)
        box=np.asarray(data['initial']['bbox'],np.float32)-np.tile(roi[:2],2)
        predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,box=box)
        for idx,ids,logits in predictor.propagate_in_video(state):
            f=first+idx;mask=(logits[0,0].cpu().numpy()>0).astype(np.uint8);masks[f]=mask
            if not cv2.imwrite(str(maskdir/f'{f:05}.png'),mask*255):raise ValueError('Mask write failed')
            if idx%10==0:print(f'{data["pose"]["clip_id"]} SAM2 f{f}/{data["last"]} {time.monotonic()-started:.1f}s',flush=True)
    del state;gc.collect()
    save(out/'segmentation.json',dict(frame_window=[first,data['last']],roi=roi,
        image_size=image_size,reused_identical_prior_masks=False,new_model_inference=True,elapsed_seconds=time.monotonic()-started,
        source_right_censored=data['right_censored'],requested_last=data['requested_last'],
        initial_prompt=dict(frame=first,bbox=data['initial']['bbox'],seed_track_id=data['role']['seed_track_id']),
        masks=[mask_info(m,f) for f,m in masks.items()],no_late_manual_prompt=True))
    return masks


def refine_pose(data,masks,out,model,device='cpu'):
    roi=data['roi'];results={};qc={};samples=[];previous=None;initial_area=int(masks[data['first']].sum())
    for f,img in data['frames'].items():
        mask=masks[f];excluded=np.zeros(mask.shape,np.uint8)
        for o in data['others'].get(f,[]):excluded|=pose_seed(o,img.shape,roi[:2])
        q=mask_qc(mask,previous,data['scale'],initial_area,excluded);qc[f]=q
        if q['valid']:previous=mask
        candidates=[]
        if f in data['keeper']:
            o=dict(data['keeper'][f],source_pose='original_role_seed_track');candidates.append(o)
        if q['valid'] and q.get('geometry'):
            b=q['geometry']['bbox'];pad=max(12,round(.20*data['scale']))
            x0=max(0,b[0]-pad);y0=max(0,b[1]-pad);x1=min(img.shape[1],b[2]+pad);y1=min(img.shape[0],b[3]+pad)
            # Actual foreground pixels unchanged; only surrounding background dimmed.
            crop=img[y0:y1,x0:x1].copy();cmask=mask[y0:y1,x0:x1]>0
            faded=(crop*.20).astype(np.uint8);faded[cmask]=crop[cmask]
            h,w=faded.shape[:2];views=[faded,cv2.rotate(faded,cv2.ROTATE_90_CLOCKWISE),cv2.rotate(faded,cv2.ROTATE_90_COUNTERCLOCKWISE)]
            preds=model.predict(views,imgsz=512,device=device,conf=.08,iou=.80,max_det=10,verbose=False)
            for angle,p in zip([0,90,270],preds):
                if p.boxes is None or p.keypoints is None:continue
                boxes,kp=unrotate_pose(p.boxes.xyxy.cpu().numpy(),p.keypoints.xy.cpu().numpy(),w,h,angle)
                conf=p.keypoints.conf.cpu().numpy();boxes[:,[0,2]]+=roi[0]+x0;boxes[:,[1,3]]+=roi[1]+y0
                kp[:,:,0]+=roi[0]+x0;kp[:,:,1]+=roi[1]+y0
                for j in range(len(boxes)):
                    candidates.append(dict(frame=f,bbox=boxes[j].tolist(),keypoints=kp[j].tolist(),
                        keypoint_confidences=conf[j].tolist(),source_pose=f'sam2_foreground_dimmed_rotation_{angle}',
                        box_confidence=float(p.boxes.conf[j])))
        valid=[]
        if q['valid']:
            for o in candidates:
                s=anatomy_sample(o,mask,roi[:2])
                if s is not None:valid.append((s,o))
        # Ambiguous opposite signed anatomical candidates cannot be majority-voted away.
        signs={s['sign'] for s,o in valid if s['sign']}
        selected=None
        if len(signs)<=1 and valid:
            selected,observation=max(valid,key=lambda pair:pair[0]['min_anatomical_confidence'])
            samples.append(selected)
        results[f]=dict(candidates=candidates,selected_anatomy=selected,
                        rejection='conflicting_rotated_anatomy' if len(signs)>1 else None)
        if (f-data['first'])%15==0:print(f'{data["pose"]["clip_id"]} anatomical refinement f{f}/{data["last"]}',flush=True)
    save(out/'refined_poses.json',results);save(out/'mask_qc.json',qc)
    e=sam2_direction_evidence(samples,qc,data['kick'],data['pose']['fps'])
    save(out/'direction_evidence.json',e);return e,qc,results


def review_video(data,masks,qc,poses,e,out,ffmpeg=None):
    import subprocess
    images=[];writer=cv2.VideoWriter(str(out/'review_staging.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),data['pose']['fps'],(960,600))
    if not writer.isOpened():raise ValueError('Review writer failed')
    for f,img in data['frames'].items():
        panel=img.copy();mask=masks[f]>0
        panel[mask]=(.55*panel[mask]+.45*np.array([30,190,255])).astype(np.uint8)
        s=poses[f]['selected_anatomy']
        if s:
            g=qc[f]['geometry'];c=np.asarray(g['center']);head=np.asarray(s['head_point'])
            cv2.circle(panel,tuple(np.rint(head).astype(int)),3,(255,0,255),-1)
            cv2.arrowedLine(panel,tuple(np.rint(c).astype(int)),tuple(np.rint(c+np.asarray(s['torso_vector'])*2).astype(int)),(255,0,255),2)
        partpath=out/'head_part/head_masks'/f'{f:05}.png'
        if e.get('anatomy_source')=='separate SAM2 part prompt' and partpath.is_file():
            prefix=e.get('observed_mask_prefix_frames',[])
            if prefix and f<=prefix[-1] and e.get('head_part_qc',{}).get(f,{}).get('valid'):
                hm=cv2.imread(str(partpath),0)>0
                panel[hm]=(.3*panel[hm]+.7*np.array([220,40,230])).astype(np.uint8)
        panel=cv2.resize(panel,(960,500));header=np.zeros((100,960,3),np.uint8)
        lines=[f'{data["pose"]["clip_id"]} f{f} {f/data["pose"]["fps"]:.2f}s',
               f'SAM2 mask QC: {qc[f]["reason"]} | direction candidate: {e["direction"]}',
               'yellow=visible silhouette; magenta=head/torso evidence; not human-validated']
        for i,line in enumerate(lines):cv2.putText(header,line,(12,25+30*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        full=np.vstack([header,panel]);writer.write(full)
        if f in [data['kick']-8,data['kick']+9,data['kick']+18,data['kick']+30] or f in e.get('supporting_frames',[]):
            cv2.imwrite(str(out/f'review_f{f}.jpg'),full);images.append(f)
    writer.release();video=out/'sam2_keeper_review.mp4'
    ff=resolve_ffmpeg(ffmpeg,ROOT)
    subprocess.run([str(ff),'-y','-i',str(out/'review_staging.mp4'),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(video)],check=True,capture_output=True)
    cap=cv2.VideoCapture(str(video));count=0
    while cap.read()[0]:count+=1
    cap.release()
    save(out/'video_validation.json',dict(decoded_frames=count,expected_frames=len(data['frames']),passed=count==len(data['frames']),
                                         static_review_frames=images,video_sha256=sha(video)))
    if count!=len(data['frames']):raise ValueError('Video decode failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=DEFAULT)
    p.add_argument('--clips',nargs='*');p.add_argument('--image-size',type=int,choices=[512,1024],default=1024)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--head-part-fallback',action='store_true',help='Track an automatically prompted head part if pose anatomy is unsupported')
    p.add_argument('--head-cache-run',type=Path,help='Reuse previously generated automatic head masks only after matching crop/video/checkpoint checks')
    add_runtime_arguments(p)
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/raw_clips')
    p.add_argument('--baseline-dir',type=Path,default=BASE)
    p.add_argument('--appearance-dir',type=Path,default=RUN/'goal_appearance_v2')
    p.add_argument('--pose-model',type=Path,default=ROOT/'models/weights/yolo11m_pose.pt')
    p.add_argument('--sam2-model',type=Path,default=ROOT/'models/weights/sam2.1_hiera_tiny.pt')
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'.deps_goalkeeper_segmentation'))
    import torch;torch.set_num_threads(3);cv2.setNumThreads(2)
    device=resolve_device(a.device,torch);ffmpeg=resolve_ffmpeg(a.ffmpeg,ROOT)
    from sam2.build_sam import build_sam2_video_predictor
    from ultralytics import YOLO
    summary=read(a.baseline_dir/'summary.json');unknown=[r['clip_id'] for r in summary['rows'] if r['dive_direction']=='unknown']
    ids=a.clips if a.clips else unknown
    if any(cid not in unknown for cid in ids):raise ValueError('Only existing unknown directions are in this fallback scope')
    checkpoint=a.sam2_model
    protocol=dict(policy='sam2_anatomical_fallback_v4',clips=ids,image_size=a.image_size,head_part_fallback=a.head_part_fallback,
        head_cache_run=str(a.head_cache_run.resolve()) if a.head_cache_run else None,model_sha256=sha(checkpoint),
        pose_model_sha256=sha(a.pose_model),
        runtime=inference_provenance(a.device,device,[checkpoint,a.pose_model],torch_module=torch),ffmpeg=ffmpeg,
        source_dir=str(a.source_dir.resolve()),appearance_dir=str(a.appearance_dir.resolve()),
        default_results_not_overwritten=True,no_human_roles_or_outcomes_read=True,accuracy_estimate=None,
        code_sha256={n:sha(ROOT/n) for n in ['src/goalkeeper_sam2.py','src/goalkeeper_sam2_head_tracking.py','scripts/run_goalkeeper_sam2_update.py']},
        baseline_summary_sha256=sha(a.baseline_dir/'summary.json'),sam2_source='https://github.com/facebookresearch/sam2',
        sam2_commit='2b90b9f5ceec907a1c18123530e92e794ad901a4')
    if (out/'protocol.json').exists():
        if not a.resume or read(out/'protocol.json')!=protocol:raise FileExistsError('Use a fresh output dir for changed protocol')
    else:save(out/'protocol.json',protocol)
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device=device,
        apply_postprocessing=False,hydra_overrides_extra=[f'++model.image_size={a.image_size}'])
    model=YOLO(str(a.pose_model));records=[]
    for cid in ids:
        clipout=out/'clips'/cid;clipout.mkdir(parents=True,exist_ok=True)
        target=clipout/'result.json'
        if a.resume and target.is_file():
            manifest=clipout/'input_hashes.json'
            if manifest.is_file() and any(sha(Path(p))!=h for p,h in read(manifest).items()):
                raise ValueError('Resume input hashes changed')
            records.append(read(target));continue
        baseline,data,role=prepare(cid,clipout,device,a.source_dir,a.baseline_dir,a.appearance_dir,a.pose_model)
        if data is None:
            r=dict(clip_id=cid,status='abstain_initialization',reason=role.get('reason'),previous_direction='unknown',
                   candidate_direction='unknown',identity_verified=False,human_verified=False)
        else:
            data['kick']=baseline['kick_frame']
            save(clipout/'input_hashes.json',{str(p):sha(p) for p in [*data['source_inputs'],data['pose_path'],Path(data['pose']['video_path'])]})
            masks=segment(data,clipout,predictor,a.image_size);e,qc,poses=refine_pose(data,masks,clipout,model,device)
            if a.head_part_fallback and not e['supported']:
                from src.goalkeeper_sam2_head_tracking import track_head_part
                if a.head_cache_run:
                    from src.goalkeeper_sam2 import head_part_direction_evidence
                    cache=a.head_cache_run.resolve();cp=read(cache/'protocol.json');cd=cache/'clips'/cid
                    cs=read(cd/'segmentation.json')
                    if (not cp.get('head_part_fallback') or cp['model_sha256']!=sha(checkpoint)
                        or cs['roi']!=data['roi'] or cs['frame_window']!=[data['first'],data['last']]):
                        raise ValueError('Head cache prompt/crop/model protocol mismatch')
                    # Automatic head inputs must be pixel-identical to this actual clip crop.
                    for idx,img in enumerate(data['frames'].values()):
                        ok,encoded=cv2.imencode('.jpg',img)
                        if not ok or encoded.tobytes()!=(cd/'head_part/input_jpg'/f'{idx:05}.jpg').read_bytes():
                            raise ValueError('Head cache video crop pixels changed')
                    heads={f:(cv2.imread(str(cd/'head_part/head_masks'/f'{f:05}.png'),0)>0).astype(np.uint8) for f in masks}
                    if any(m.shape!=masks[f].shape for f,m in heads.items()):raise ValueError('Head cache mask shape mismatch')
                    shutil.copytree(cd/'head_part',clipout/'head_part')
                    part=head_part_direction_evidence(heads,masks,qc,data['kick'],data['pose']['fps'],data['scale'])
                    save(clipout/'head_cache_validation.json',dict(source=str(cd),all_crop_JPEG_bytes_identical=True,
                        model_hash_identical=True,head_mask_sha256={str(f):sha(cd/'head_part/head_masks'/f'{f:05}.png') for f in heads},
                        head_inference_image_size=cp['image_size'],body_inference_image_size=a.image_size))
                else:
                    part=track_head_part(data,masks,qc,predictor,clipout/'head_part')
                save(clipout/'head_part_evidence.json',part)
                if part['supported']:
                    save(clipout/'pose_only_direction_evidence.json',e);e=part
                    save(clipout/'direction_evidence.json',e)
            review_video(data,masks,qc,poses,e,clipout,ffmpeg)
            r=dict(clip_id=cid,status='candidate_supported' if e['supported'] else 'mask_observed_direction_unknown',
                candidate_direction=e['direction'],previous_direction='unknown',pose_seed=role['seed_track_id'],
                mask_valid_frames=sum(q['valid'] for q in qc.values()),total_mask_frames=len(qc),
                direction_evidence=e,identity_verified=False,human_verified=False,accuracy_estimate=None,
                review_video=f'clips/{cid}/sam2_keeper_review.mp4')
        r['runtime']=protocol['runtime']
        save(target,r);records.append(r);save(out/'progress.json',dict(completed=len(records),total=len(ids),rows=records))
        print(json.dumps({k:r[k] for k in ['clip_id','status','candidate_direction']},ensure_ascii=False),flush=True)
    updates={r['clip_id']:r for r in records};rows=[]
    for row in summary['rows']:
        item=dict(row);r=updates.get(item['clip_id'])
        if r and r['candidate_direction']!='unknown':
            item.update(previous_v3_direction=item['dive_direction'],dive_direction=r['candidate_direction'],
                direction_support_status='sam2_anatomical_candidate_review_required',sam2_direction_evidence=r['direction_evidence'],
                whole_record_review_required=True,human_verified=False,identity_verified=False)
        rows.append(item)
    result=dict(policy=protocol['policy'],clips=len(summary['rows']),attempted=len(records),previous_unknown=len(unknown),
        new_unknown=sum(r['dive_direction']=='unknown' for r in rows),
        recovered_candidates=[r['clip_id'] for r in records if r['candidate_direction']!='unknown'],
        unchanged_known_candidates=len(summary['rows'])-len(unknown),unchanged_height_labels=len(summary['rows']),whole_record_QA_not_promoted=True,
        accuracy_estimate=None,default_v3_summary_unchanged=sha(a.baseline_dir/'summary.json')==protocol['baseline_summary_sha256'],
        rows=rows,experiments=records)
    save(out/'summary.json',result);save(out/'goalkeeper_results_39.json',dict(policy=result['policy'],rows=rows))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','experiments']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
