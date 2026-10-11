"""Separate bounded experiment: SAM2 initialized head part, not whole-person pose."""
from pathlib import Path
import sys,time,json,subprocess
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_goalkeeper_sam2_update import prepare
from scripts.run_goalkeeper_direction_update import read,save,sha
from src.goalkeeper_sam2 import head_part_sample,sam2_direction_evidence,mask_qc


def main():
    out=ROOT/'outputs/goalkeeper_sam2_update_2026-10-11/cuadrado_head_part_512_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Experiment already exists')
    cid='col_eng_p03_cuadrado_col11';baseline,data,role=prepare(cid,out)
    first=data['first'];roi=data['roi'];kp=np.asarray(data['initial']['keypoints']);cf=np.asarray(data['initial']['keypoint_confidences'])
    if cf[0]<.5:raise ValueError('No reliable automatic nose prompt')
    visible=kp[:5][cf[:5]>=.5]-roi[:2]
    box=np.r_[visible.min(0)-5,visible.max(0)+5].astype(np.float32)
    # Nose foreground and shoulder/hip centers background distinguish head from shirt.
    points=np.asarray([kp[0],kp[[5,6]].mean(0),kp[[11,12]].mean(0)],np.float32)-roi[:2]
    save(out/'protocol.json',dict(clip_id=cid,initial_frame=first,head_box=box.tolist(),points=points.tolist(),
        labels=[1,0,0],prompt_source='automatic pre-kick COCO nose and visible head points, no human prompt',
        image_size=512,body_masks_source='prior Cuadrado SAM2 at 1024',direction_promotion=False,
        code_sha256={n:sha(ROOT/n) for n in ['scripts/probe_goalkeeper_sam2_head_part.py','src/goalkeeper_sam2.py']}))
    sys.path.insert(0,str(ROOT/'.deps_goalkeeper_segmentation'));sys.path.append(str(ROOT/'.venv/Lib/site-packages'))
    import torch;torch.set_num_threads(3);cv2.setNumThreads(2)
    from sam2.build_sam import build_sam2_video_predictor
    model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'models/weights/sam2.1_hiera_tiny.pt'),
        device='cpu',apply_postprocessing=False,hydra_overrides_extra=['++model.image_size=512'])
    jpg=out/'input_jpg';jpg.mkdir(exist_ok=True)
    for idx,img in enumerate(data['frames'].values()):cv2.imwrite(str(jpg/f'{idx:05}.jpg'),img)
    masks={};started=time.monotonic()
    with torch.inference_mode():
        state=model.init_state(str(jpg),offload_state_to_cpu=True,offload_video_to_cpu=True)
        model.add_new_points_or_box(state,frame_idx=0,obj_id=2,box=box,points=points,labels=np.array([1,0,0]))
        for idx,ids,logits in model.propagate_in_video(state):
            f=first+idx;masks[f]=(logits[0,0].cpu().numpy()>0).astype(np.uint8)
            dest=out/'head_masks'/f'{f:05}.png';dest.parent.mkdir(exist_ok=True);cv2.imwrite(str(dest),masks[f]*255)
            if idx%10==0:print(f'head part f{f}/138 {time.monotonic()-started:.1f}s',flush=True)
    bodyroot=ROOT/'outputs/goalkeeper_mask_comparison_2026-10-11/experiment_v1/masks/sam2'
    samples=[];qc={};head_area=int(masks[first].sum());previous=None
    writer=cv2.VideoWriter(str(out/'review_staging.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,600))
    for f,img in data['frames'].items():
        body=(cv2.imread(str(bodyroot/f'{f:05}.png'),0)>0).astype(np.uint8)
        q=mask_qc(body,previous,data['scale'],int((cv2.imread(str(bodyroot/f'{first:05}.png'),0)>0).sum()))
        previous=body;s=head_part_sample(masks[f],body,f,head_area)
        qc[f]=dict(valid=q['valid'] and s is not None,reason='valid_head_part_candidate' if s is not None else 'head_part_lost_or_ambiguous')
        if s is not None:samples.append(s)
        panel=img.copy();panel[body>0]=(.65*panel[body>0]+.35*np.array([30,190,255])).astype(np.uint8)
        panel[masks[f]>0]=(.35*panel[masks[f]>0]+.65*np.array([220,40,230])).astype(np.uint8)
        panel=cv2.resize(panel,(960,500));header=np.zeros((100,960,3),np.uint8)
        cv2.putText(header,f'SAM head-part pilot f{f}: {qc[f]["reason"]}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.putText(header,'Magenta=HEAD CANDIDATE, yellow=body. No direction promotion.',(10,65),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        full=np.vstack([header,panel]);writer.write(full)
        if f in [92,111,114,120,132,138]:cv2.imwrite(str(out/f'head_review_f{f}.jpg'),full)
    writer.release();e=sam2_direction_evidence(samples,qc,baseline['kick_frame'],30)
    e['policy']='sam2_head_part_tracker_pilot_v1';e['anatomy_source']='separate SAM2 head prompt';e['promoted']=False
    save(out/'result.json',dict(evidence=e,head_initial_pixels=head_area,elapsed_seconds=time.monotonic()-started))
    save(out/'qc.json',qc)
    ff=ROOT/'outputs/review_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    subprocess.run([str(ff),'-y','-i',str(out/'review_staging.mp4'),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'head_part_review.mp4')],check=True,capture_output=True)
    print(dict(direction=e['direction'],reasons=e['reasons']),flush=True)


if __name__=='__main__':main()
