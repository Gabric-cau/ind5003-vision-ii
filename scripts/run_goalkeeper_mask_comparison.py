"""Cuadrado local segmentation ablations: no baseline output is mutated."""
from pathlib import Path
import argparse, hashlib, json, sys, subprocess, time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.goalkeeper_mask_experiment import (pose_seed, box_mask, train_lab_model,
    color_score, nearest_component, warp_mask, grabcut_mask, lk_tracks,
    camera_affine, mask_info)
CID = 'col_eng_p03_cuadrado_col11'
BASE = ROOT/'outputs/goalkeeper_role_association_2026-10-11/final_v4'
DEFAULT = ROOT/'outputs/goalkeeper_mask_comparison_2026-10-11/experiment_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(out):
    lock = read(BASE/'cached_role_lock.json')
    source = ROOT/'outputs/optimization_evening_2026-10-09/goal_pose_all'/f'{CID}_pose.json'
    pose = read(source)
    keeper = {o['frame']: o for o in lock['association']['observations']}
    excluded = set(lock['role']['excluded_pre_person_ids'])
    others = {}
    for t in pose['tracks']:
        if t['track_id'] in excluded:
            for o in t['observations']:
                others.setdefault(o['frame'], []).append(o)
    pre = np.asarray([o['bbox'] for f, o in keeper.items() if f < 96])
    b = np.median(pre, axis=0); h = b[3]-b[1]; cx = (b[0]+b[2])/2
    roi = [max(0, int(cx-1.8*h)), max(0, int(b[1]-.35*h)),
           min(pose['frame_width'], int(cx+1.8*h)), min(pose['frame_height'], int(b[3]+.8*h))]
    frames = {}; cap = cv2.VideoCapture(pose['video_path']); cap.set(1, 87)
    for f in range(87, 139):
        ok, img = cap.read()
        if not ok:
            raise ValueError(f'Decode failed/{f}')
        frames[f] = img[roi[1]:roi[3], roi[0]:roi[2]]
    cap.release()
    return keeper, others, roi, frames, pose, source


def run_classical(out, chroma_gate=False):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('Use a new experiment directory')
    keeper, others, roi, frames, pose, source = setup(out)
    save(out/'protocol.json', dict(clip_id=CID, frame_window_inclusive=[87,138],
        kick_frame=102, roi=roi, fps=pose['fps'], automatic_identity_source=str(BASE/'cached_role_lock.json'),
        no_human_labels_read=True, no_outcome_labels_read=True, accuracy_estimate=None,
        no_default_results_modified=True, chroma_gate=chroma_gate,
        chroma_gate_rule='HSV saturation >= 60 suppresses white goal lines; heuristic, not ground truth' if chroma_gate else None,
        foreground_training_frames=list(range(87,96)),
        reliable_pose_reanchors_through_frame=111,
        heuristics=dict(color_log_likelihood_ratio=1.0, max_component_distance_px=60,
                        LK_forward_backward_error_px=1.5, GrabCut_iterations=3),
        sources={'GrabCut':'https://docs.opencv.org/4.13.0/d8/d83/tutorial_py_grabcut.html',
                 'LK':'https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html',
                 'SAM2':'https://github.com/facebookresearch/sam2'},
        input_hashes={str(p):sha(p) for p in [source, BASE/'cached_role_lock.json', Path(pose['video_path']),
             ROOT/'outputs/goalkeeper_pose_direction_2026-10-11/results_v3_final/summary.json']},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'src/goalkeeper_mask_experiment.py']}))
    offset = roi[:2]; seed = {}; exclusion = {}
    for f, img in frames.items():
        seed[f] = pose_seed(keeper[f], img.shape, offset) if f in keeper else np.zeros(img.shape[:2], np.uint8)
        exclusion[f] = np.zeros(img.shape[:2], np.uint8)
        for o in others.get(f, []) + ([keeper[f]] if f in keeper else []):
            exclusion[f] |= box_mask(o['bbox'], img.shape, offset, 8)
    training = list(range(87,96))
    models, stats = train_lab_model([frames[f] for f in training], [seed[f] for f in training],
                                   [exclusion[f] for f in training])
    save(out/'color_model.json', stats)
    history = {k:[] for k in ['color', 'grabcut', 'optical_flow']}
    prev_mask = {}; prev_gray = None; points = None; cum_camera = np.eye(3)
    initial_gray = None; started = time.monotonic()
    for f, img in frames.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); score = color_score(img, models)
        chroma = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1] >= 60
        cam_info = dict(status='initial_frame'); mat = None
        if prev_gray is not None:
            mat, cam_info = camera_affine(prev_gray, gray, exclusion[f-1])
            if mat is not None:
                mat3 = np.vstack([mat,[0,0,1]]); cum_camera = mat3 @ cum_camera
            back = cv2.calcOpticalFlowFarneback(gray, prev_gray, None, .5, 3, 15, 3, 5, 1.2, 0)
        else:
            back = np.zeros((*gray.shape,2), np.float32); initial_gray = gray
        masks = {}
        for method in ('color', 'grabcut'):
            prior = (box_mask(keeper[f]['bbox'], img.shape, offset, 3) if f in keeper
                     else warp_mask(prev_mask[method], back))
            if method == 'color':
                raw = (score > 1.).astype(np.uint8)
                if chroma_gate: raw *= chroma
                raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                mask, reason = nearest_component(raw, prior)
            else:
                mask, reason = grabcut_mask(img, prior, score, seed[f] if f in keeper else None)
            masks[method] = mask; prev_mask[method] = mask
            info = mask_info(mask, f); info.update(status=reason, pose_anchor=f in keeper)
            # Rectangle overlap is only a warning, NOT proof of referee identity.
            other_boxes = np.zeros(gray.shape, np.uint8)
            for o in others.get(f, []): other_boxes |= box_mask(o['bbox'], img.shape, offset)
            info['excluded_person_bbox_overlap_fraction'] = float(np.count_nonzero(mask & other_boxes)/max(1,mask.sum()))
            history[method].append(info)
            dest = out/'masks'/method/f'{f:05}.png'; dest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest), mask*255): raise ValueError('Mask write failed')
        old, new = lk_tracks(prev_gray, gray, points) if prev_gray is not None else (np.empty((0,1,2)),np.empty((0,1,2)))
        relative = None
        if len(new) >= 3 and mat is not None:
            background_pred = cv2.transform(old.astype(np.float32), mat)
            relative = np.median((new-background_pred).reshape(-1,2), axis=0).tolist()
        points = new
        if chroma_gate and len(points):
            xy=np.rint(points.reshape(-1,2)).astype(int)
            inside=(xy[:,0]>=0)&(xy[:,0]<gray.shape[1])&(xy[:,1]>=0)&(xy[:,1]<gray.shape[0])
            allowed=np.zeros(len(points),bool)
            allowed[inside]=(score[xy[inside,1],xy[inside,0]]>1)&chroma[xy[inside,1],xy[inside,0]]
            points=points[allowed]
        measured_count = len(new)
        if f in keeper:
            # Refresh only from reliable original pose-anchored foreground. No late manual seeds.
            feature_mask = masks['grabcut']*255
            if chroma_gate: feature_mask *= ((score>1)&chroma).astype(np.uint8)
            fresh = cv2.goodFeaturesToTrack(gray, 80, .01, 3, mask=feature_mask)
            if fresh is not None: points = fresh
        valid = points is not None and len(points)>=3
        row = dict(frame=f, status='feature_candidate' if valid else 'lost',
                   forward_backward_survivors=measured_count, tracked_points=[] if points is None else points.reshape(-1,2).tolist(),
                   camera_model=cam_info, camera_relative_step=relative, pose_reanchor=f in keeper,
                   centroid=None if not valid else points.reshape(-1,2).mean(0).tolist())
        history['optical_flow'].append(row)
        prev_gray = gray
        if (f-87)%10 == 0: print(f'classical frame {f}/138, elapsed={time.monotonic()-started:.1f}s', flush=True)
    save(out/'classical_results.json', dict(methods=history, elapsed_seconds=time.monotonic()-started,
         identity_verified=False, accuracy_validated=False,
         notes=['Mask pixel count/feature survival is NOT successful identity tracking or accuracy.',
                'Each method uses automatic pose reanchors through 111; afterwards no new pose anchor.',
                'Referee bbox overlap warns of occlusion; it is not pixel-level contamination ground truth.',
                'Do not infer keeper-relative left/right from image motion alone.']))


def run_sam(out, checkpoint):
    sys.path.insert(0,str(ROOT/'.deps_goalkeeper_segmentation'))
    sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch
    torch.set_num_threads(3)
    from sam2.build_sam import build_sam2_video_predictor
    keeper, others, roi, frames, pose, source = setup(out)
    jpg = out/'sam2_input'; jpg.mkdir(exist_ok=True)
    for i, img in enumerate(frames.values()):
        if not cv2.imwrite(str(jpg/f'{i:05}.jpg'), img): raise ValueError('JPEG write failed')
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',
        str(checkpoint), device='cpu', apply_postprocessing=False)
    history = []; started = time.monotonic()
    with torch.inference_mode():
        state = predictor.init_state(str(jpg), offload_video_to_cpu=True, offload_state_to_cpu=True)
        initial = np.asarray(keeper[87]['bbox'],np.float32)-np.tile(roi[:2],2)
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=initial)
        for idx, ids, logits in predictor.propagate_in_video(state):
            f = 87+idx; mask = (logits[0,0].cpu().numpy()>0).astype(np.uint8)
            path = out/'masks/sam2'/f'{f:05}.png'; path.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(path), mask*255): raise ValueError('SAM mask write failed')
            history.append(mask_info(mask,f))
            if idx%5 == 0: print(f'SAM2 frame {f}/138 elapsed={time.monotonic()-started:.1f}s',flush=True)
    save(out/'sam2_results.json',dict(status='inference_completed', frames=history, elapsed_seconds=time.monotonic()-started,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),initial_prompt_source='automatic pre-kick role bbox at frame 87',
        no_late_manual_prompt=True,model='SAM2.1 Hiera tiny',device='cpu',identity_verified=False,accuracy_validated=False))


def render(out):
    keeper, others, roi, frames, pose, source = setup(out)
    classical = read(out/'classical_results.json')['methods']
    methods = ['color','grabcut','optical_flow','sam2']; records = []
    staging = out/'comparison_staging.mp4'
    writer = cv2.VideoWriter(str(staging),cv2.VideoWriter_fourcc(*'mp4v'),pose['fps'],(1280,800))
    if not writer.isOpened(): raise ValueError('Writer failed')
    for f,img in frames.items():
        panels=[]
        for method in methods:
            panel=img.copy(); text=method
            path=out/'masks'/method/f'{f:05}.png'
            if path.exists():
                mask=cv2.imread(str(path),0)>0
                panel[mask]=(.55*panel[mask]+.45*np.array([20,190,255])).astype(np.uint8)
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel,contours,-1,(0,255,255),1)
            elif method=='optical_flow':
                row=classical[method][f-87]
                for point in row['tracked_points']: cv2.circle(panel,tuple(np.rint(point).astype(int)),2,(0,255,255),-1)
                text+=f' points={len(row["tracked_points"])}'
            else: text+=' NOT RUN'
            # Red rectangles are only referee/other detections, not true pixel masks.
            for o in others.get(f,[]):
                b=np.asarray(o['bbox'])-np.tile(roi[:2],2)
                cv2.rectangle(panel,tuple(np.rint(b[:2]).astype(int)),tuple(np.rint(b[2:]).astype(int)),(50,50,230),1)
            panel=cv2.resize(panel,(640,350)); banner=np.zeros((50,640,3),np.uint8)
            cv2.putText(banner,f'{text} f{f} t={f/pose["fps"]:.2f}s',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(banner,'candidate only - identity/direction not validated',(10,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,200,255),1)
            panels.append(np.vstack([banner,panel]))
        mosaic=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]); writer.write(mosaic)
        if f in [92,111,120,132,138]:
            cv2.imwrite(str(out/f'comparison_f{f}.jpg'),mosaic); records.append(f)
    writer.release()
    ffmpeg=ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    video=out/'cuadrado_mask_comparison.mp4'
    subprocess.run([str(ffmpeg),'-y','-i',str(staging),'-c:v','libx264','-pix_fmt','yuv420p',
                    '-movflags','+faststart','-an',str(video)],check=True,capture_output=True)
    cap=cv2.VideoCapture(str(video)); count=0
    while cap.read()[0]: count+=1
    cap.release()
    protocol=read(out/'protocol.json')
    unchanged={p:sha(Path(p))==v for p,v in protocol['input_hashes'].items()}
    save(out/'validation.json',dict(decoded_frames=count,expected_frames=len(frames),decode_pass=count==len(frames),
         all_baseline_inputs_unchanged=all(unchanged.values()),input_hash_checks=unchanged,video_sha256=sha(video),
         static_visual_review_frames=records,accuracy_estimate=None))
    if count!=len(frames) or not all(unchanged.values()): raise ValueError('Validation failed')
    print(dict(video=str(video),decoded_frames=count,inputs_unchanged=all(unchanged.values())),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=DEFAULT)
    p.add_argument('--mode',choices=['classical','sam2','render'],default='classical')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'models/weights/sam2.1_hiera_tiny.pt')
    p.add_argument('--chroma-gate',action='store_true',help='Ablation: suppress low saturation paint and filter flow to learned jersey color')
    a=p.parse_args();cv2.setNumThreads(2);cv2.setRNGSeed(0);np.random.seed(0)
    if a.mode=='classical': run_classical(a.output_dir,a.chroma_gate)
    elif a.mode=='sam2': run_sam(a.output_dir,a.checkpoint)
    render(a.output_dir)


if __name__=='__main__': main()
