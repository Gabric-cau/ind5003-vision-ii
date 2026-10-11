"""Bounded SAM2 full first-dive tracking, phase-aware image-height experiment."""
from pathlib import Path
from collections import Counter
import argparse, sys, time, json, gc, subprocess, html, shutil
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_goalkeeper_sam2_update import prepare, segment
from scripts.run_goalkeeper_direction_update import read, save, sha
from src.goalkeeper_sam2 import mask_qc, mask_geometry
from src.goalkeeper_mask_experiment import pose_seed, camera_affine
from src.goalkeeper_role_association import unrotate_pose
from src.goalkeeper_dive_phase import pose_measurement, analyse_phases, mask_phase_candidates
BASE = ROOT / 'outputs/goalkeeper_sam2_update_2026-10-11/results_v4_release'
DISPLAY = ROOT / 'outputs/goalkeeper_user_review_2026-10-11/goalkeeper_display_results_39.json'


def extend(data, kick, seconds):
    """Larger static ROI and actual source frames; never pad a truncated clip."""
    pre = [o['bbox'] for f, o in data['keeper'].items() if data['first'] <= f < kick-3]
    b = np.median(pre, axis=0); h = data['scale']; cx = (b[0]+b[2])/2
    roi = [max(0, int(cx-2.5*h)), max(0, int(b[1]-.5*h)),
           min(data['pose']['frame_width'], int(cx+2.5*h)), min(data['pose']['frame_height'], int(b[3]+h))]
    cap = cv2.VideoCapture(data['pose']['video_path'])
    available = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); requested = kick+round(seconds*data['pose']['fps'])
    last = min(requested, available-1); cap.set(1, data['first'])
    frames = {}; fullgray = {}; excluded = {}; shrink = min(1., 640/data['pose']['frame_width'])
    for f in range(data['first'], last+1):
        ok, img = cap.read()
        if not ok:
            raise ValueError(f'Source decode failed at {f}')
        frames[f] = img[roi[1]:roi[3], roi[0]:roi[2]]
        fullgray[f] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=shrink, fy=shrink)
        ex = np.zeros(img.shape[:2], np.uint8)
        # All cached persons are excluded from background motion estimation.
        for t in data['pose'].get('tracks', []):
            for o in t['observations']:
                if o['frame'] == f:
                    a = np.asarray(o['bbox'], int); x0,y0,x1,y1 = a
                    cv2.rectangle(ex, (max(0,x0-8),max(0,y0-8)), (x1+8,y1+8), 1, -1)
        excluded[f] = cv2.resize(ex, fullgray[f].shape[::-1], interpolation=cv2.INTER_NEAREST)
    cap.release()
    data.update(roi=roi, frames=frames, last=last, requested_last=requested,
                right_censored=last < requested, kick=kick)
    return fullgray, excluded, shrink


def motion_and_qc(data, masks, gray, excluded, shrink, out):
    matrices = {}; qcs = {}; camera = {}; forward = np.eye(3); previous_mask = None
    initial_area = int(masks[data['first']].sum()); prefix = True; stop_reason = None
    for f in data['frames']:
        reason = None
        if f == data['first']:
            camera[f] = dict(valid=True, status='reference_frame')
        else:
            # Explicitly exclude SAM keeper silhouette even when original poses disappear.
            ex = excluded[f-1].copy(); roi = data['roi']; h,w = ex.shape
            fullmask = np.zeros((data['pose']['frame_height'],data['pose']['frame_width']),np.uint8)
            fullmask[roi[1]:roi[3],roi[0]:roi[2]] = masks[f-1]
            ex |= cv2.resize(fullmask,(w,h),interpolation=cv2.INTER_NEAREST)
            mat, diag = camera_affine(gray[f-1],gray[f],ex)
            diff = float(np.mean(cv2.absdiff(gray[f-1],gray[f])))
            valid = mat is not None
            if valid:
                factor = np.sqrt(abs(np.linalg.det(mat[:,:2])))
                valid = .95 <= factor <= 1.05 and np.linalg.norm(mat[:,2]) < 25
            camera[f] = dict(valid=bool(valid), frame_difference=diff, **diag)
            if not valid and diff > 25:
                reason = 'possible_camera_cut_or_large_uncompensated_motion'
            if valid:
                m = np.eye(3); m[:2] = mat
                # Convert the estimated downsampled affine to full-image coordinates.
                d = np.diag([shrink,shrink,1]); m = np.linalg.inv(d) @ m @ d
                forward = m @ forward
        matrices[f] = np.linalg.inv(forward)
        ex_torso = np.zeros(masks[f].shape,np.uint8)
        for o in data['others'].get(f,[]):
            ex_torso |= pose_seed(o,masks[f].shape,data['roi'][:2])
        q = mask_qc(masks[f], previous_mask, data['scale'], initial_area, ex_torso)
        if q['valid']:
            b = q['geometry']['bbox']; mh,mw = masks[f].shape
            if b[0] <= 2 or b[1] <= 2 or b[2] >= mw-2 or b[3] >= mh-2:
                reason = reason or 'silhouette_reaches_roi_boundary'
        if reason:
            q.update(valid=False,reason=reason)
        if not q['valid'] and prefix:
            stop_reason = q['reason']; prefix = False
        q['connected_identity_candidate_prefix'] = prefix
        if q['valid']:
            previous_mask = masks[f]
        qcs[f] = q
    save(out/'camera_motion.json',camera); save(out/'camera_inverse_matrices.json',{f:m.tolist() for f,m in matrices.items()}); save(out/'mask_qc.json',qcs)
    return matrices,qcs,camera,stop_reason


def silhouette_rows(data,masks,qc,matrices):
    rows=[]
    for f in data['frames']:
        if not qc[f]['connected_identity_candidate_prefix']: break
        y,x=np.nonzero(masks[f]); points=np.c_[x+data['roi'][0],y+data['roi'][1],np.ones(len(x))] @ matrices[f].T
        points=points[:,:2]/points[:,2:3]; center=points.mean(0)
        vals,vec=np.linalg.eigh(np.cov(points.T));axis=vec[:,-1]
        near=points[:,1]>=np.percentile(points[:,1],95);lower=np.median(points[near],axis=0)
        rows.append(dict(frame=f,center=center.tolist(),lower_edge=lower.tolist(),
                         axis_lean=float(np.degrees(np.arctan2(abs(axis[0]),abs(axis[1])))),
                         vertical_extent_body=float(np.ptp(points[:,1])/data['scale']),
                         elongation=float(np.sqrt(vals[-1]/max(vals[0],1)))))
    return rows


def infer_poses(data,masks,qc,model,out):
    results = {}; jobs = []; contexts = []
    for f,img in data['frames'].items():
        if not qc[f]['connected_identity_candidate_prefix']:
            continue
        candidates = []
        if f in data['keeper']:
            o = dict(data['keeper'][f],source_pose='original_seed_track'); candidates.append(o)
        results[f] = candidates
        # Actual SAM2 is at source FPS; pose re-inference is every other frame.
        if (f-data['first']) % 2:
            continue
        b = qc[f]['geometry']['bbox']; pad = max(12,round(.25*data['scale']))
        x0,y0 = max(0,b[0]-pad),max(0,b[1]-pad)
        x1,y1 = min(img.shape[1],b[2]+pad),min(img.shape[0],b[3]+pad)
        crop = img[y0:y1,x0:x1].copy(); foreground = masks[f][y0:y1,x0:x1]>0
        faded = (crop*.2).astype(np.uint8); faded[foreground] = crop[foreground]
        # 0 and both sideways views retained, not majority-voted.
        for angle in (0,90,270):
            view = faded if angle == 0 else cv2.rotate(faded,cv2.ROTATE_90_CLOCKWISE if angle == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
            jobs.append(view); contexts.append((f,angle,faded.shape[1],faded.shape[0],x0,y0))
    for start in range(0,len(jobs),12):
        predictions = model.predict(jobs[start:start+12],imgsz=512,device='cpu',conf=.08,iou=.8,max_det=10,verbose=False)
        for context,p in zip(contexts[start:start+12],predictions):
            f,angle,w,h,x0,y0 = context
            if p.boxes is None or p.keypoints is None:
                continue
            boxes,kp = unrotate_pose(p.boxes.xyxy.cpu().numpy(),p.keypoints.xy.cpu().numpy(),w,h,angle)
            cf = p.keypoints.conf.cpu().numpy(); dx,dy = data['roi'][0]+x0,data['roi'][1]+y0
            boxes[:,[0,2]] += dx; boxes[:,[1,3]] += dy; kp[:,:,0] += dx; kp[:,:,1] += dy
            for i in range(len(boxes)):
                results[f].append(dict(frame=f,bbox=boxes[i].tolist(),keypoints=kp[i].tolist(),
                    keypoint_confidences=cf[i].tolist(),source_pose=f'sam2_roi_rotation_{angle}',box_confidence=float(p.boxes.conf[i])))
        if start%48 == 0:
            print(f'{data["pose"]["clip_id"]} pose views {min(start+12,len(jobs))}/{len(jobs)}',flush=True)
    save(out/'pose_candidates.json',results)
    return results


def measurements(data,masks,qc,matrices,poses,out):
    footrows = []
    for f,obs in data['keeper'].items():
        if not data['first'] <= f < data['kick'] or f not in matrices:
            continue
        kp = np.asarray(obs['keypoints']); cf = np.asarray(obs['keypoint_confidences'])
        if np.all(cf[[15,16]] >= .4):
            foot = kp[[15,16]][np.argmax(kp[[15,16],1])]
            stable = matrices[f] @ np.r_[foot,1.]; footrows.append((f,stable[:2]/stable[2]))
    if len(footrows) < 3:
        return None, dict(reason='missing_pre_kick_planted_foot_reference')
    # Local goal-line slope is a perspective correction, not a 3D calibration.
    geometries = data['role'].get('geometry_sample_frames',[])
    from scripts.run_goalkeeper_role_experiment import geometries_for
    geo = geometries_for(data['pose'],data['kick']); slopes = []
    for f,g in geo.items():
        if f not in matrices:
            continue
        corners = np.asarray(g['selected']['corners']); bl,br = corners[3],corners[2]
        p = np.c_[np.array([bl,br]),np.ones(2)] @ matrices[f].T; p = p[:,:2]/p[:,2:3]
        if abs(p[1,0]-p[0,0]) > 20:
            slopes.append(float((p[1,1]-p[0,1])/(p[1,0]-p[0,0])))
    if len(slopes) < 2 or np.std(slopes) > .08:
        return None, dict(reason='unstable_goal_groundline_slope')
    slope = float(np.median(slopes)); intercept = float(np.median([p[1]-slope*p[0] for f,p in footrows]))
    scale = data['scale']; samples = []
    for f,candidates in poses.items():
        valid = [s for o in candidates if (s:=pose_measurement(o,masks[f],data['roi'][:2],matrices[f],scale,(slope,intercept))) is not None]
        if not valid:
            continue
        # Conflicting augmentations are rejected, not chosen to meet a category.
        if max(s['hip_clearance_body'] for s in valid)-min(s['hip_clearance_body'] for s in valid) > .20:
            continue
        s = max(valid,key=lambda s:s['min_confidence'])
        ys,xs = np.nonzero(masks[f]); near = ys >= np.percentile(ys,95)
        low = np.array([np.median(xs[near])+data['roi'][0],np.median(ys[near])+data['roi'][1],1.])
        low = matrices[f] @ low; low = low[:2]/low[2]
        s['bottom_clearance_body'] = float((slope*low[0]+intercept-low[1])/scale)
        samples.append(s)
    calibration = dict(ground_line=[slope,intercept],body_scale_px=scale,
                       source='pre-kick ankles and goal-line slope in stabilized image coordinates',
                       physical_ground_plane_calibrated=False,reference_frames=[f for f,p in footrows])
    save(out/'height_reference.json',calibration); save(out/'measurements.json',samples)
    return samples,calibration


def render(data,masks,qc,e,out):
    path = out/'phase_staging.mp4'; writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),data['pose']['fps'],(960,600))
    if not writer.isOpened(): raise ValueError('Video writer unavailable')
    contacts = {s['frame']:s for s in e.get('samples',[])}
    montage = []
    for f,img in data['frames'].items():
        panel = img.copy(); good = qc[f]['connected_identity_candidate_prefix']
        if good:
            m = masks[f]>0; panel[m] = (.6*panel[m]+.4*np.array([30,190,255])).astype(np.uint8)
        phase = 'unresolved'
        phases = e.get('silhouette_phase',{}).get('phases') or e.get('phases',[])
        for p in phases:
            if p['start_frame'] <= f <= p['end_frame']: phase = p['name']
        panel = cv2.resize(panel,(960,490)); header = np.zeros((110,960,3),np.uint8)
        lines = [f'{data["pose"]["clip_id"]} | f{f} {f/data["pose"]["fps"]:.2f}s',
                 f'{phase} | pose-height={e["height_candidate"]} mask-proxy={e.get("silhouette_phase",{}).get("height_proxy_candidate","unknown")}',
                 'Image-plane body-height proxy, not metres; yellow=SAM2; no direction edits']
        for i,line in enumerate(lines): cv2.putText(header,line,(10,25+30*i),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        canvas = np.vstack([header,panel]); writer.write(canvas)
        se = e.get('silhouette_phase',{})
        landmarks = [data['kick']-5,e.get('onset_frame'),e.get('landing_frame'),se.get('onset_frame'),se.get('landing_frame')]
        landmarks += e.get('height_supporting_frames',[])[:1]
        landmarks += se.get('supporting_frames',[])[:1]
        if f in landmarks:
            cv2.imwrite(str(out/f'phase_f{f}.jpg'),canvas); montage.append(canvas)
    writer.release(); video = out/'goalkeeper_phase_review.mp4'
    ff = ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    subprocess.run([str(ff),'-y','-i',str(path),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(video)],check=True,capture_output=True)
    cap = cv2.VideoCapture(str(video)); count = 0
    while cap.read()[0]: count += 1
    cap.release()
    if count != len(data['frames']): raise ValueError('Review decode mismatch')
    save(out/'video_validation.json',dict(decoded_frames=count,expected_frames=len(data['frames']),passed=True))
    if montage:
        cv2.imwrite(str(out/'phase_montage.jpg'),np.vstack([cv2.resize(i,(768,480)) for i in montage]))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--clips',nargs='*'); parser.add_argument('--post-sec',type=float,default=2.)
    parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cache-run',type=Path)
    a = parser.parse_args(); out = a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    baseline = read(BASE/'summary.json'); byid = {r['clip_id']:r for r in baseline['rows']}
    ids = a.clips or [r['clip_id'] for r in baseline['rows'] if r['dive_height']=='unknown']
    if any(cid not in byid for cid in ids): raise ValueError('Noncanonical clip')
    code = ['src/goalkeeper_dive_phase.py','scripts/run_goalkeeper_height_phase.py','scripts/run_goalkeeper_sam2_update.py',
            'src/goalkeeper_sam2.py','src/goalkeeper_role_association.py','src/goalkeeper_mask_experiment.py']
    protocol = dict(policy='phase_height_experiment_v2',clips=ids,post_sec=a.post_sec,image_size=512,cache_run=str(a.cache_run) if a.cache_run else None,
                    model_sha256=sha(ROOT/'models/weights/sam2.1_hiera_tiny.pt'),
                    pose_model_sha256=sha(ROOT/'models/weights/yolo11m_pose.pt'),code_sha256={n:sha(ROOT/n) for n in code},
                    baseline_sha256=sha(BASE/'summary.json'),no_human_labels_used_for_inference=True,
                    directions_preserved=True,accuracy_estimate=None)
    if (out/'protocol.json').exists():
        if not a.resume or read(out/'protocol.json') != protocol: raise FileExistsError('Changed protocol requires new output folder')
    else:
        save(out/'protocol.json',protocol)
        for n in code:
            p = out/'source_snapshot'/n; p.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/n,p)
    sys.path.insert(0,str(ROOT/'.deps_goalkeeper_segmentation')); sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch
    torch.set_num_threads(3); cv2.setNumThreads(2)
    from sam2.build_sam import build_sam2_video_predictor
    from ultralytics import YOLO
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'models/weights/sam2.1_hiera_tiny.pt'),
                    device='cpu',apply_postprocessing=False,hydra_overrides_extra=['++model.image_size=512'])
    model = YOLO(str(ROOT/'models/weights/yolo11m_pose.pt')); results = []
    for cid in ids:
        dest = out/'clips'/cid; dest.mkdir(parents=True,exist_ok=True)
        if a.resume and (dest/'result.json').exists():
            if any(sha(Path(p)) != h for p,h in read(dest/'input_hashes.json').items()): raise ValueError('Changed input')
            results.append(read(dest/'result.json')); continue
        print('START '+cid,flush=True); started = time.monotonic()
        baseline_record,data,role = prepare(cid,dest)
        inputhashes = {str(BASE/byid[cid]['result_file']):sha(BASE/byid[cid]['result_file'])}
        if data is None:
            e = dict(height_candidate='unknown',height_supported=False,complete_first_dive_observed=False,
                     reasons=[role.get('reason')],identity_verified=False,accuracy_estimate=None)
        else:
            inputhashes.update({str(data['pose_path']):sha(data['pose_path']),str(Path(data['pose']['video_path'])):sha(Path(data['pose']['video_path']))})
            gray,excluded,shrink = extend(data,baseline_record['kick_frame'],a.post_sec)
            cache = a.cache_run/'clips'/cid if a.cache_run else None
            reused = False
            if cache and (cache/'segmentation.json').exists():
                cs = read(cache/'segmentation.json'); cp = read(a.cache_run/'protocol.json')
                if cs['roi'] != data['roi'] or cs['frame_window'] != [data['first'],data['last']] or cp['model_sha256'] != protocol['model_sha256']:
                    raise ValueError('Cache video-window/crop/model mismatch')
                if cs['initial_prompt']['bbox'] != data['initial']['bbox']: raise ValueError('Cache prompt mismatch')
                for idx,img in enumerate(data['frames'].values()):
                    ok,encoded = cv2.imencode('.jpg',img)
                    if not ok or encoded.tobytes() != (cache/'sam2_input'/f'{idx:05}.jpg').read_bytes(): raise ValueError('Cache frame pixels mismatch')
                for name in ['masks','sam2_input']: shutil.copytree(cache/name,dest/name,dirs_exist_ok=True)
                masks = {f:(cv2.imread(str(dest/'masks'/f'{f:05}.png'),0)>0).astype(np.uint8) for f in data['frames']}
                save(dest/'segmentation.json',dict(cs,reused_pixel_identical_cache=str(cache))); reused = True
            else:
                masks = segment(data,dest,predictor,512)
            matrices,qc,camera,stop = motion_and_qc(data,masks,gray,excluded,shrink,dest)
            if reused and cp['pose_model_sha256'] == protocol['pose_model_sha256']:
                poses = {int(f):v for f,v in read(cache/'pose_candidates.json').items()}
                save(dest/'pose_candidates.json',poses)
            else:
                poses = infer_poses(data,masks,qc,model,dest)
            samples,cal = measurements(data,masks,qc,matrices,poses,dest)
            if samples is None:
                e = dict(height_candidate='unknown',height_supported=False,complete_first_dive_observed=False,
                         reasons=[cal['reason']],samples=[],identity_verified=False,accuracy_estimate=None)
            else:
                prefix = [f for f,q in qc.items() if q['connected_identity_candidate_prefix']]
                supported = bool(prefix) and sum(camera[f]['valid'] for f in prefix)/len(prefix) >= .9
                e = analyse_phases(samples,data['kick'],data['pose']['fps'],data['scale'],tracking_stop_reason=stop,
                                   tracking_last=max(prefix) if prefix else None,source_censored=data['right_censored'],camera_supported=supported)
                e['reference'] = cal
            silhouette = silhouette_rows(data,masks,qc,matrices)
            se = mask_phase_candidates(silhouette,data['kick'],data['pose']['fps'],data['scale'])
            prefix = [f for f,q in qc.items() if q['connected_identity_candidate_prefix']]
            if not prefix or sum(camera[f]['valid'] for f in prefix)/len(prefix)<.9:
                se.update(supported=False,height_proxy_candidate='unknown',complete_first_dive_candidate=False)
                se['reasons'].append('camera_motion_unreliable')
            e['silhouette_phase'] = se; save(dest/'silhouette_measurements.json',silhouette)
            e.update(frame_window=[data['first'],data['last']],requested_end_frame=data['requested_last'],
                     connected_mask_frames=sum(q['connected_identity_candidate_prefix'] for q in qc.values()),
                     total_mask_frames=len(masks),tracking_stop_reason=stop,source_censored=data['right_censored'])
            save(dest/'phase_evidence.json',e); render(data,masks,qc,e,dest)
            del masks,data,gray,excluded,poses; gc.collect()
        result = dict(clip_id=cid,previous_height=byid[cid]['dive_height'],height_candidate=e['height_candidate'],
                      height_supported=e['height_supported'],complete_first_dive_observed=e['complete_first_dive_observed'],
                      silhouette_height_proxy_candidate=e.get('silhouette_phase',{}).get('height_proxy_candidate','unknown'),
                      complete_silhouette_phase_candidate=e.get('silhouette_phase',{}).get('complete_first_dive_candidate',False),
                      phase_evidence=e,review_required=True,accuracy_estimate=None,elapsed_seconds=time.monotonic()-started,
                      review_video=f'clips/{cid}/goalkeeper_phase_review.mp4' if (dest/'goalkeeper_phase_review.mp4').exists() else None)
        save(dest/'input_hashes.json',inputhashes); save(dest/'result.json',result); results.append(result)
        save(out/'progress.json',dict(completed=len(results),total=len(ids),rows=results))
        print(json.dumps({k:v for k,v in result.items() if k!='phase_evidence'},ensure_ascii=False),flush=True)
    # Display feedback is read only AFTER inference, solely to preserve direction.
    display = read(DISPLAY); updates = {r['clip_id']:r for r in results}; rows = []
    for original in display['rows']:
        row = dict(original); trial = updates.get(row['clip_id'])
        row.update(previous_height=row['dive_height'],phase_height_candidate=trial['height_candidate'] if trial else None,
                   silhouette_height_proxy_candidate=trial['silhouette_height_proxy_candidate'] if trial else None,
                   phase_height_supported=trial['height_supported'] if trial else False,
                   phase_height_evidence_file=f'clips/{row["clip_id"]}/phase_evidence.json' if trial and trial['review_video'] else None)
        rows.append(row)
    summary = dict(clips=39,experiment_clips=len(results),phase_height_counts=dict(Counter(r['height_candidate'] for r in results)),
                   silhouette_height_proxy_counts=dict(Counter(r['silhouette_height_proxy_candidate'] for r in results)),
                   complete_silhouette_phase_candidates=sum(r['complete_silhouette_phase_candidate'] for r in results),
                   complete_first_dive_candidates=sum(r['complete_first_dive_observed'] for r in results),
                   supported_new_height_candidates=[r['clip_id'] for r in results if r['previous_height']=='unknown' and r['height_supported']],
                   automatic_height_not_overwritten=True,directions_unchanged=True,whole_record_QA_not_promoted=True,
                   accuracy_estimate=None,rows=rows,experiments=results)
    save(out/'summary.json',summary)
    table = ''.join(f'<tr><td>{r["clip_id"]}</td><td>{r["previous_height"]}</td><td>{r["height_candidate"]} / {r["silhouette_height_proxy_candidate"]}</td>'
                    f'<td>{r["complete_first_dive_observed"]}</td><td>{html.escape(", ".join(str(x) for x in r["phase_evidence"]["reasons"]))}</td>'
                    f'<td><a href="{r["review_video"]}">阶段视频</a></td></tr>' for r in results)
    (out/'report.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>门将扑救阶段与高度实验</title>
<style>body{{font:16px/1.65 system-ui;max-width:1200px;margin:30px auto;padding:20px}}table{{border-collapse:collapse}}td,th{{padding:10px;border:1px solid #ccc}}</style>
<h1>完整首次扑救跟踪与落地前高度</h1><p>本次实际测试{len(results)}条。严格姿态阶段完整候选{summary['complete_first_dive_candidates']}条；轮廓阶段完整候选{summary['complete_silhouette_phase_candidates']}条。严格姿态高度{summary['phase_height_counts']}；轮廓高度代理{summary['silhouette_height_proxy_counts']}。不是准确率，也不自动改整条QA。</p>
<p>原方法的最大身体下移可能来自落地。新方法：自动门线角色选择→扩大ROI→SAM2连续分割至踢球后2秒或实际结尾→排除人物的背景相机运动补偿→旋转重提肩髋关键点→持续启动/伸展/落地状态→仅取落地前连续三样本髋部离地高度中值峰值。</p>
<p>高度是局部门线斜率和预踢脚位置作为地面参考、除以预踢身体像素高度的二维代理量，不是三维米制高度。低&lt;0.35、中0.35–0.75、高&gt;0.75为未经真值验证的工程类别，边界±0.04拒绝。手腕高度另存，不冒充球的高度或手部接球点。</p>
<p>关键点缺失时另试轮廓阶段：横向变扁且下边缘连续5帧趋稳作为落地候选，仅取落地前轮廓中心离地代理。其地面轨迹假设连接最后直立点与落地候选点，修正前跨导致的画面下移；这是较弱的探索性证据，不是精确接地、髋部或物理高度，不替换正式dive_height。两条路线分开记录，不把轮廓候选当成关键点验证成功。</p>
<p>阶段在可靠落地前停止时，不推断未观察峰值；长关键点缺口、ROI出界、人物混入或镜头突变不桥接。首个扑救阶段可完整，但不代表之后回放、起身全部跟踪。未读取人工方向做高度推理；已确认方向保持不变。</p>
<table><tr><th>视频</th><th>原高度</th><th>严格姿态 / 轮廓高度代理</th><th>严格首次阶段完整候选</th><th>保留问题</th><th>复核</th></tr>{table}</table>
<p><a href="summary.json">39条字段及实验结果</a> · <a href="protocol.json">协议/源码哈希</a>。模型来源：<a href="https://github.com/facebookresearch/sam2">Meta SAM2</a>；角色、阶段和高度规则为本项目工程实现。</p></html>''',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','experiments']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
